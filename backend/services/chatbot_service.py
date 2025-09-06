from __future__ import annotations

import logging
from typing import Optional, List, Dict, Iterable

from backend.config import Settings
from backend.services.swing_chatbot import EnhancedSwingChatBot
from backend.simple_chatbot import SimpleChatBot
from backend.simple_chatbot.config import ChatbotConfig


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

    def general_ask_stream(self, message: str, max_messages: int | None = None) -> Iterable[str]:
        if self._general_bot is None:
            cfg = ChatbotConfig()
            self._general_bot = SimpleChatBot(
                model_name=cfg.model_name,
                gguf_filename=cfg.gguf_filename,
                backend=cfg.backend,
            )
        self._general_messages.append({"role": "user", "content": message})
        parts: List[str] = []
        for ch in self._general_bot.ask_stream(message):
            parts.append(ch)
            yield ch
        self._general_messages.append({"role": "assistant", "content": "".join(parts)})
        if max_messages is not None and len(self._general_messages) > max_messages:
            self._general_messages[:] = self._general_messages[-max_messages:]

    def general_ask(self, message: str, max_messages: int | None = None) -> str:
        return "".join(self.general_ask_stream(message, max_messages=max_messages))

    def general_clear(self) -> None:
        self._general_messages.clear()

    def general_reset(self, backend: Optional[str] = None) -> None:
        """Reset the general chatbot instance and optionally set backend.

        Clears the in-memory conversation and forces the next call to
        instantiate a new model, picking up any env/config changes such as
        ``LLM_BACKEND`` or device variables.
        """
        if backend:
            try:
                from backend.config import Settings as _S
                # Update Settings for in-process readers
                _S.LLM_BACKEND = backend
                # Also update environment so new ChatbotConfig() picks it up
                import os as _os
                _os.environ["LLM_BACKEND"] = backend
            except Exception:
                pass
        self._general_bot = None
        self._general_messages.clear()
        try:
            eff_backend = backend or getattr(Settings, "LLM_BACKEND", "")
            logger.info("[chat] General chatbot reset. Pending reinit with backend=%s", eff_backend)
        except Exception:
            pass

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

    def swing_ask_stream(self, message: str, max_messages: int = 20) -> Iterable[str]:
        if self._swing_bot is None:
            raise RuntimeError("Swing chatbot not initialized")
        self._swing_messages.append({"role": "user", "content": message})
        if len(self._swing_messages) > max_messages:
            self._swing_messages[:] = self._swing_messages[-max_messages:]

        parts: List[str] = []
        ask_stream = getattr(self._swing_bot, "ask_stream", None)
        if callable(ask_stream):
            iterator = ask_stream(message)
        else:
            iterator = iter(self._swing_bot.ask(message))
        for ch in iterator:
            parts.append(ch)
            yield ch
        self._swing_messages.append({"role": "assistant", "content": "".join(parts)})
        if len(self._swing_messages) > max_messages:
            self._swing_messages[:] = self._swing_messages[-max_messages:]

    def swing_ask(self, message: str, max_messages: int = 20) -> str:
        return "".join(self.swing_ask_stream(message, max_messages=max_messages))
