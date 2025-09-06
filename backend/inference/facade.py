"""Facade for selecting and preloading pose extractors.

- Single Responsibility: selection and lifecycle of default extractor.
"""

from __future__ import annotations

from pathlib import Path
import logging

from .base import TARGET_SIZE, PoseExtractor
from .openvino_extractor import OpenPoseExtractor
from .yolov8_extractor import YoloV8PoseExtractor, is_yolov8_model, _to_torch_device


logger = logging.getLogger("uvicorn.error")

_DEFAULT_EXTRACTOR: PoseExtractor | None = None


def preload_openpose_model(model_xml: str, device: str = "CPU") -> None:
    global _DEFAULT_EXTRACTOR
    if (
        not isinstance(_DEFAULT_EXTRACTOR, OpenPoseExtractor)
        or _DEFAULT_EXTRACTOR.model_xml != model_xml
        or _DEFAULT_EXTRACTOR.device != device
    ):
        logger.info("Preloading OpenPose model %s on %s", model_xml, device)
        _DEFAULT_EXTRACTOR = OpenPoseExtractor(model_xml, device)


def preload_yolov8_model(model_path: str, device: str | None = None, conf: float = 0.25, imgsz: int | None = None) -> None:
    global _DEFAULT_EXTRACTOR
    if (
        not isinstance(_DEFAULT_EXTRACTOR, YoloV8PoseExtractor)
        or _DEFAULT_EXTRACTOR.model_path != model_path
        or _DEFAULT_EXTRACTOR.device != _to_torch_device(device)
    ):
        logger.info(
            "Preloading YOLOv8 model %s on %s",
            model_path,
            _to_torch_device(device),
        )
        _DEFAULT_EXTRACTOR = YoloV8PoseExtractor(model_path, device=device, conf=conf, imgsz=imgsz)


def extract_keypoints(
    video_path: Path,
    model_path: str,
    device: str,
    target_size: tuple[int, int] | None = TARGET_SIZE,
):
    """Auto-select backend based on provided model path and extract keypoints."""
    global _DEFAULT_EXTRACTOR
    if is_yolov8_model(model_path):
        if (
            not isinstance(_DEFAULT_EXTRACTOR, YoloV8PoseExtractor)
            or getattr(_DEFAULT_EXTRACTOR, "model_path", None) != model_path
            or getattr(_DEFAULT_EXTRACTOR, "device", None) != _to_torch_device(device)
        ):
            preload_yolov8_model(model_path, device)
    else:
        if (
            not isinstance(_DEFAULT_EXTRACTOR, OpenPoseExtractor)
            or getattr(_DEFAULT_EXTRACTOR, "model_xml", None) != model_path
            or getattr(_DEFAULT_EXTRACTOR, "device", None) != device
        ):
            preload_openpose_model(model_path, device)

    assert _DEFAULT_EXTRACTOR is not None
    logger.info(
        "Extracting keypoints from %s using %s on %s",
        video_path,
        type(_DEFAULT_EXTRACTOR).__name__,
        _DEFAULT_EXTRACTOR.device,
    )
    return _DEFAULT_EXTRACTOR.extract(video_path, target_size=target_size)


__all__ = [
    "preload_openpose_model",
    "preload_yolov8_model",
    "extract_keypoints",
]

