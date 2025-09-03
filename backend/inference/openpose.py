"""OpenPose/OpenVINO-based keypoint extraction with SOLID-friendly design.

- Single Responsibility: this module handles pose extraction only.
- Open/Closed: new extractors can implement `PoseExtractor` without changing clients.
- Liskov: `PoseExtractor` implementations are interchangeable.
- Interface Segregation: small, focused protocol.
- Dependency Inversion: clients depend on `PoseExtractor` abstraction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Tuple, Iterable


# ------------------------ common normalization helpers -----------------------

TARGET_SIZE = (1280, 720)
BASE_RESOLUTION = 720


def normalize_coords(keypoints, width: int, height: int):
    return [(kp[0] / width, kp[1] / height, kp[2]) for kp in keypoints]


def scale_score(score: float, height: int, base: int = BASE_RESOLUTION) -> float:
    return score * (base / max(height, 1))


# ----------------------------- abstraction layer ----------------------------


class PoseExtractor(ABC):
    @abstractmethod
    def extract(self, video_path: Path, *, target_size: tuple[int, int] | None = TARGET_SIZE):
        """Return a sequence of normalized keypoints per frame."""
        raise NotImplementedError


_MODEL_CACHE: Dict[Tuple[str, str], Tuple[object, object]] = {}


class OpenVinoPoseExtractor(PoseExtractor):
    def __init__(self, model_xml: str, device: str = "CPU") -> None:
        self.model_xml = model_xml
        self.device = device
        self._compiled_model, self._output_layer = self._load_model(model_xml, device)

    @staticmethod
    def _load_model(model_xml: str, device: str):
        key = (str(model_xml), str(device))
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached
        from openvino.runtime import Core

        core = Core()
        model = core.read_model(model=model_xml)
        compiled_model = core.compile_model(model=model, device_name=device)
        output_layer = compiled_model.output(0)
        _MODEL_CACHE[key] = (compiled_model, output_layer)
        return compiled_model, output_layer

    @staticmethod
    def _preprocess(frame, input_shape):
        import cv2
        import numpy as np

        _, _, h, w = input_shape
        image = cv2.resize(frame, (w, h))
        image = image.transpose((2, 0, 1))
        image = image[np.newaxis, :]
        return image

    @staticmethod
    def _postprocess(results, frame_size):
        import cv2
        import numpy as np

        heatmaps = np.squeeze(results, axis=0)
        points = []
        num_kp = heatmaps.shape[0]
        frame_w, frame_h = frame_size
        for i in range(num_kp):
            heatmap = heatmaps[i]
            _, conf, _, point = cv2.minMaxLoc(heatmap)
            x = point[0] * frame_w / heatmap.shape[1]
            y = point[1] * frame_h / heatmap.shape[0]
            points.append((x, y, conf))
        return points

    def extract(self, video_path: Path, *, target_size: tuple[int, int] | None = TARGET_SIZE):
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        keypoints = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if target_size is not None:
                frame = cv2.resize(frame, target_size)
            h, w = frame.shape[:2]
            inp = self._preprocess(frame, self._compiled_model.input(0).shape)
            results = self._compiled_model([inp])[self._output_layer]
            pts = self._postprocess(results, (w, h))
            keypoints.append(normalize_coords(pts, w, h))
        cap.release()
        return keypoints


# ------------------------------- Facade funcs -------------------------------

_DEFAULT_EXTRACTOR: OpenVinoPoseExtractor | None = None


def preload_openpose_model(model_xml: str, device: str = "CPU") -> None:
    global _DEFAULT_EXTRACTOR
    if _DEFAULT_EXTRACTOR is None or _DEFAULT_EXTRACTOR.model_xml != model_xml or _DEFAULT_EXTRACTOR.device != device:
        _DEFAULT_EXTRACTOR = OpenVinoPoseExtractor(model_xml, device)


def extract_keypoints(video_path: Path, model_xml: str, device: str, target_size: tuple[int, int] | None = TARGET_SIZE):
    if _DEFAULT_EXTRACTOR is None or _DEFAULT_EXTRACTOR.model_xml != model_xml or _DEFAULT_EXTRACTOR.device != device:
        preload_openpose_model(model_xml, device)
    assert _DEFAULT_EXTRACTOR is not None
    return _DEFAULT_EXTRACTOR.extract(video_path, target_size=target_size)


__all__ = [
    "PoseExtractor",
    "OpenVinoPoseExtractor",
    "extract_keypoints",
    "preload_openpose_model",
    "normalize_coords",
    "scale_score",
]

