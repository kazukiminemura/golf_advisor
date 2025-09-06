"""YOLOv8-pose based keypoint extraction.

- Single Responsibility: this module handles YOLOv8 pose extraction only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

from .base import PoseExtractor, TARGET_SIZE, normalize_coords


_YOLO_MODEL_CACHE: Dict[Tuple[str, str], object] = {}


def _to_torch_device(device: str | None) -> str:
    if not device:
        return "cpu"
    d = device.strip().lower()
    if d in {"cpu", "cuda", "cuda:0", "cuda:1"}:
        return d
    if d == "gpu":
        try:
            import torch  # noqa: F401

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return "cpu"


def is_yolov8_model(model_path: str) -> bool:
    mp = str(model_path).lower()
    return mp.endswith(".pt") or "yolov8" in mp or mp.endswith(".onnx")


class YoloV8PoseExtractor(PoseExtractor):
    def __init__(self, model_path: str, device: str | None = None, conf: float = 0.25, imgsz: int | None = None) -> None:
        self.model_path = model_path
        self.device = _to_torch_device(device)
        self.conf = conf
        self.imgsz = 640 if imgsz is None else imgsz
        self.model = self._load_model(model_path, self.device)

    @staticmethod
    def _load_model(model_path: str, device: str):
        key = (str(model_path), str(device))
        cached = _YOLO_MODEL_CACHE.get(key)
        if cached is not None:
            return cached
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "ultralytics is required for YOLOv8-pose. Add 'ultralytics' to requirements and install it."
            ) from e
        model = YOLO(model_path)
        try:
            model.to(device)
        except Exception:
            pass
        _YOLO_MODEL_CACHE[key] = model
        return model

    def extract(self, video_path: Path, *, target_size: tuple[int, int] | None = TARGET_SIZE):
        import cv2
        import numpy as np

        cap = cv2.VideoCapture(str(video_path))
        keypoints_per_frame = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if target_size is not None:
                frame = cv2.resize(frame, target_size)
            h, w = frame.shape[:2]

            infer_kwargs = {
                "verbose": False,
                "device": self.device,
                "conf": self.conf,
                "imgsz": int(self.imgsz),
            }
            results = self.model(frame, **infer_kwargs)
            if not results:
                keypoints_per_frame.append([])
                continue
            r = results[0]

            if r.keypoints is None or (getattr(r.boxes, "shape", [0])[0] if hasattr(r, "boxes") else 0) == 0:
                keypoints_per_frame.append([])
                continue

            try:
                confs = r.boxes.conf.detach().cpu().numpy() if hasattr(r.boxes, "conf") else None
                idx = int(np.argmax(confs)) if confs is not None and len(confs) > 0 else 0
            except Exception:
                idx = 0

            xy = r.keypoints.xy[idx].detach().cpu().numpy()  # (17,2)
            kc = (
                r.keypoints.conf[idx].detach().cpu().numpy()
                if hasattr(r.keypoints, "conf")
                else np.ones((xy.shape[0],), dtype=float)
            )

            def mk(i):
                return (float(xy[i, 0]), float(xy[i, 1]), float(kc[i]))

            def mid(i, j):
                x = 0.5 * (xy[i, 0] + xy[j, 0])
                y = 0.5 * (xy[i, 1] + xy[j, 1])
                c = float(min(kc[i], kc[j]))
                return (float(x), float(y), c)

            mapped = [
                mk(0),               # 0 nose
                mid(5, 6),           # 1 neck
                mk(6),               # 2 right shoulder
                mk(8),               # 3 right elbow
                mk(10),              # 4 right wrist
                mk(5),               # 5 left shoulder
                mk(7),               # 6 left elbow
                mk(9),               # 7 left wrist
                mid(11, 12),         # 8 mid hip
                mk(12),              # 9 right hip
                mk(14),              # 10 right knee
                mk(16),              # 11 right ankle
                mk(11),              # 12 left hip
                mk(13),              # 13 left knee
                mk(15),              # 14 left ankle
                mk(2),               # 15 right eye
                mk(1),               # 16 left eye
                mk(4),               # 17 right ear
                mk(3),               # 18 left ear
            ]

            keypoints_per_frame.append(normalize_coords(mapped, w, h))

        cap.release()
        return keypoints_per_frame


__all__ = [
    "YoloV8PoseExtractor",
    "_to_torch_device",
    "is_yolov8_model",
]

