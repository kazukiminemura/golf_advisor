from __future__ import annotations

import os
from pathlib import Path


def env_flag(name: str, default: str = "") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes"}


def env_str(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


class Settings:
    """Application settings and defaults.

    Centralizes environment configuration and default paths to support DI and
    Single-Responsibility separation for the API vs. business logic.
    """

    DATA_DIR = Path("data")
    STATIC_DIR = Path("static")

    # Default OpenVINO IR model (OpenPose-like)
    MODEL_XML = "intel/human-pose-estimation-0001/FP16/human-pose-estimation-0001.xml"
    # Default YOLOv8 pose checkpoint (Ultralytics will auto-download if needed)
    YOLOV8_MODEL = env_str("YOLOV8_MODEL", "yolov8n-pose.pt")
    # Frontend-selectable pose model: 'openvino' or 'yolo'
    POSE_MODEL = env_str("POSE_MODEL", "openvino").lower() or "openvino"
    DEVICE = env_str("DEVICE", "CPU").upper() or "CPU"

    ENABLE_CHATBOT = env_flag("ENABLE_CHATBOT", "1")
    LAZY_CHATBOT_INIT = env_flag("LAZY_CHATBOT_INIT", "true")

    # LLM configuration
    CHAT_MODEL = env_str("CHAT_MODEL", "bartowski/Qwen2.5-1.5B-Instruct-GGUF")
    CHAT_GGUF_FILENAME = env_str("CHAT_GGUF_FILENAME", "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf")
    # Backend selector: auto | openvino | llama | transformers
    LLM_BACKEND = env_str("LLM_BACKEND", "auto")


