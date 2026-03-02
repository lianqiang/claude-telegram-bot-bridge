"""
Streaming message handler for progressive draft message updates.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional, List

from telegram import Bot
from telegram.error import TelegramError

from telegram_bot.utils.config import config

logger = logging.getLogger(__name__)


@dataclass
class DraftState:
    """State for a single draft message"""
    message_id: int
    text: str
    last_update_time: float
    char_count_since_update: int = 0
    draft_id: Optional[str] = None


class StreamingMessageHandler:
    """
    Handles progressive streaming of AI responses using Telegram draft messages.

    Manages draft message lifecycle: creation, updates, finalization, and cancellation.
    Supports multi-message handling for content exceeding 4000 characters.
    """

    def __init__(self, bot: Bot, chat_id: int, user_id: int):
        self.bot = bot
        self.chat_id = chat_id
        self.user_id = user_id
        self.drafts: List[DraftState] = []
        self.accumulated_text: str = ""
        self.min_chars = config.draft_update_min_chars
        self.min_interval = config.draft_update_interval
        self._finalized = False
        self._draft_seq = 0

    def _next_draft_id(self) -> str:
        self._draft_seq += 1
        return f"{self.user_id}-{int(time.time() * 1000)}-{self._draft_seq}"

    async def _send_draft_compat(self, text: str) -> tuple[Optional[Any], Optional[str]]:
        """
        Try Telegram draft API first, then gracefully fall back.

        Some runtime environments expose ExtBot.send_message_draft with a required
        `draft_id` parameter. Older/other environments may not support it.
        """
        if not hasattr(self.bot, "send_message_draft"):
            return None, None

        draft_id = self._next_draft_id()
        try:
            message = await self.bot.send_message_draft(
                chat_id=self.chat_id,
                draft_id=draft_id,
                text=text,
            )
            return message, draft_id
        except TypeError as e:
            logger.warning(f"send_message_draft signature mismatch, fallback to send_message: {e}")
            return None, None
        except TelegramError as e:
            logger.warning(f"send_message_draft failed, fallback to send_message: {e}")
            return None, None
        except Exception as e:
            logger.warning(f"send_message_draft unexpected error, fallback to send_message: {e}")
            return None, None

    async def create_draft(self, text: str) -> Optional[DraftState]:
        """Send initial draft message"""
        content = text or "..."
        try:
            message, draft_id = await self._send_draft_compat(content)
            if message is None:
                message = await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=content,
                )
                draft_id = None

            draft = DraftState(
                message_id=message.message_id,
                text=text,
                last_update_time=time.time(),
                char_count_since_update=0,
                draft_id=draft_id,
            )
            self.drafts.append(draft)
            logger.debug(f"Created draft message {draft.message_id} for user {self.user_id}")
            return draft
        except TelegramError as e:
            logger.error(f"Failed to create draft message: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to create draft message: {e}")
            return None

    async def update_draft(self, draft: DraftState, new_text: str) -> bool:
        """Update existing draft with new text"""
        try:
            await self.bot.edit_message_text(
                chat_id=self.chat_id,
                message_id=draft.message_id,
                text=new_text
            )
            draft.text = new_text
            draft.last_update_time = time.time()
            draft.char_count_since_update = 0
            logger.debug(f"Updated draft {draft.message_id} ({len(new_text)} chars)")
            return True
        except TelegramError as e:
            logger.error(f"Failed to update draft {draft.message_id}: {e}")
            return False

    def should_update(self, draft: DraftState, new_char_count: int) -> bool:
        """Check if draft should be updated based on thresholds"""
        time_elapsed = time.time() - draft.last_update_time
        return (new_char_count >= self.min_chars or time_elapsed >= self.min_interval)

    async def finalize_draft(self, draft: DraftState) -> bool:
        """Convert draft to regular message"""
        try:
            await self.bot.edit_message_text(
                chat_id=self.chat_id,
                message_id=draft.message_id,
                text=draft.text
            )
            logger.debug(f"Finalized draft {draft.message_id}")
            return True
        except TelegramError as e:
            logger.error(f"Failed to finalize draft {draft.message_id}: {e}")
            return False

    def _find_split_boundary(self, text: str, max_length: int = 4000) -> int:
        """Find smart boundary for text splitting (paragraph > line > hard cut)"""
        if len(text) <= max_length:
            return len(text)

        # Try paragraph boundary (double newline)
        search_start = max(0, max_length - 200)
        para_idx = text.rfind("\n\n", search_start, max_length)
        if para_idx > search_start:
            return para_idx + 2

        # Try line boundary (single newline)
        line_idx = text.rfind("\n", search_start, max_length)
        if line_idx > search_start:
            return line_idx + 1

        # Hard cut at max_length
        return max_length

    async def handle_overflow(self) -> bool:
        """Handle 4000 character boundary by finalizing current draft and creating new one"""
        if not self.drafts:
            return False

        current_draft = self.drafts[-1]
        split_point = self._find_split_boundary(self.accumulated_text)

        # Finalize current draft with text up to split point
        finalize_text = self.accumulated_text[:split_point]
        current_draft.text = finalize_text
        await self.finalize_draft(current_draft)

        # Create new draft with remaining text
        remaining_text = self.accumulated_text[split_point:]
        self.accumulated_text = remaining_text

        if remaining_text:
            await self.create_draft(remaining_text)
            logger.debug(f"Created overflow draft, remaining {len(remaining_text)} chars")

        return True

    async def update_if_needed(self, new_text_chunk: str) -> bool:
        """
        Main entry point: accumulate text and update draft if thresholds met.
        Handles overflow to new drafts when exceeding 4000 chars.
        """
        if self._finalized:
            return False

        self.accumulated_text += new_text_chunk

        # Check for overflow
        if len(self.accumulated_text) >= 4000:
            await self.handle_overflow()
            return True

        # Create first draft if needed
        if not self.drafts:
            await self.create_draft(self.accumulated_text)
            return True

        # Update existing draft if thresholds met
        current_draft = self.drafts[-1]
        chars_since_update = len(self.accumulated_text) - len(current_draft.text)
        current_draft.char_count_since_update = chars_since_update

        if self.should_update(current_draft, chars_since_update):
            await self.update_draft(current_draft, self.accumulated_text)
            return True

        return False

    async def finalize_all(self) -> bool:
        """Finalize all active draft messages"""
        if self._finalized:
            return False

        self._finalized = True

        # Update last draft with final accumulated text
        if self.drafts and self.accumulated_text:
            current_draft = self.drafts[-1]
            if current_draft.text != self.accumulated_text:
                current_draft.text = self.accumulated_text

        # Finalize all drafts
        for draft in self.drafts:
            await self.finalize_draft(draft)

        logger.debug(f"Finalized {len(self.drafts)} draft(s) for user {self.user_id}")
        return True

    async def cancel(self) -> bool:
        """Delete all unfinished draft messages and clean up state"""
        if self._finalized:
            return False

        self._finalized = True

        # Delete all draft messages
        for draft in self.drafts:
            try:
                await self.bot.delete_message(
                    chat_id=self.chat_id,
                    message_id=draft.message_id
                )
                logger.debug(f"Deleted draft {draft.message_id}")
            except TelegramError as e:
                logger.error(f"Failed to delete draft {draft.message_id}: {e}")

        self.drafts.clear()
        self.accumulated_text = ""
        logger.debug(f"Cancelled streaming for user {self.user_id}")
        return True
