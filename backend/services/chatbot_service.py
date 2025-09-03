from __future__ import annotations

import logging
from typing import Optional, List, Dict

from backend.config import Settings
from backend.services.swing_chatbot import EnhancedSwingChatBot
from simple_chatbot import SimpleChatBot


logger = logging.getLogger("uvicorn.error")


class ChatbotService:
    """Manages chatbot state and interactions.

    - Single Responsibility: chat logic only.
    - Depends on analysis service via explicit methods to fetch data when
      initializing the swing chatbot.
    """

    def __init__(self, enabled: bool = Settings.ENABLE_CHATBOT, lazy_init: bool = Settings.LAZY_CHATBOT_INIT):
        self.enabled = enabled
        self.lazy_init = lazy_init

        # Swing chatbot (keypoint-aware)
        self._swing_bot: Optional[EnhancedSwingChatBot] = None
        self._swing_messages: List[Dict[str, str]] = []

        # General-purpose chatbot
        self._general_bot: Optional[SimpleChatBot] = None
        self._general_messages: List[Dict[str, str]] = []

    # ------------------------ General-purpose chat ------------------------
    def general_messages(self) -> List[Dict[str, str]]:
        return self._general_messages

    def general_ask(self, message: str) -> str:
        if self._general_bot is None:
            self._general_bot = SimpleChatBot(
                model_name=Settings.CHAT_MODEL,
                gguf_filename=Settings.CHAT_GGUF_FILENAME,
                backend=Settings.LLM_BACKEND,
            )
        self._general_messages.append({"role": "user", "content": message})
        reply = self._general_bot.ask(message)
        self._general_messages.append({"role": "assistant", "content": reply})
        return reply

    def general_clear(self) -> None:
        self._general_messages.clear()

    # --------------------------- Swing chatbot ----------------------------
    def is_enabled(self) -> bool:
        return self.enabled

    def is_initialized(self) -> bool:
        return self._swing_bot is not None

    def clear_swing(self) -> None:
        self._swing_bot = None
        self._swing_messages.clear()

    def try_init(self, ref_keypoints, cur_keypoints, score: Optional[float], analysis_running: bool) -> bool:
        if not self.enabled:
            self.clear_swing()
            return False
        if self._swing_bot is not None:
            return True
        if analysis_running:
            return False
        if ref_keypoints is None or cur_keypoints is None or score is None:
            return False
        try:
            self._swing_bot = EnhancedSwingChatBot(ref_keypoints, cur_keypoints, score)
            initial = self._swing_bot.initial_message()
            self._swing_messages.append({"role": "assistant", "content": initial})
            return True
        except Exception as exc:
            logger.exception("Failed to initialize swing chatbot: %s", exc)
            self.clear_swing()
            return False

    def swing_messages(self) -> List[Dict[str, str]]:
        return self._swing_messages

    def swing_ask(self, message: str, max_messages: int = 20) -> str:
        if self._swing_bot is None:
            raise RuntimeError("Swing chatbot not initialized")
        self._swing_messages.append({"role": "user", "content": message})
        # Trim history
        if len(self._swing_messages) > max_messages:
            self._swing_messages[:] = self._swing_messages[-max_messages:]
        reply = self._swing_bot.ask(message)
        self._swing_messages.append({"role": "assistant", "content": reply})
        if len(self._swing_messages) > max_messages:
            self._swing_messages[:] = self._swing_messages[-max_messages:]
        return reply
