"""Configuration for the simple chatbot module."""

from dataclasses import dataclass
import os


@dataclass
class ChatbotConfig:
    """Settings for the simple chatbot.

    Values can be overridden via environment variables.
    """

    model_name: str = os.getenv("CHAT_MODEL", "bartowski/Qwen2.5-1.5B-Instruct-GGUF")
    gguf_filename: str = os.getenv("CHAT_GGUF_FILENAME", "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf")
    backend: str = os.getenv("LLM_BACKEND", "openvion")
