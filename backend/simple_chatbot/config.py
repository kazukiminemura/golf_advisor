"""Configuration for the simple chatbot module.

This module now sources its defaults from ``backend.config.Settings`` so that
changing environment variables or updating Settings at runtime (e.g., via
``ChatbotService.general_reset``) is reflected consistently across the app.
"""

from dataclasses import dataclass
import os
from backend.config import Settings


@dataclass
class ChatbotConfig:
    """Settings for the simple chatbot.

    Values can be overridden via environment variables.
    """

    # Use Settings (which itself reads env vars) to allow both env-driven and
    # programmatic overrides through a single source of truth.
    model_name: str = os.getenv("CHAT_MODEL", Settings.CHAT_MODEL)
    gguf_filename: str = os.getenv("CHAT_GGUF_FILENAME", Settings.CHAT_GGUF_FILENAME)
    backend: str = os.getenv("LLM_BACKEND", Settings.LLM_BACKEND or "openvion")
