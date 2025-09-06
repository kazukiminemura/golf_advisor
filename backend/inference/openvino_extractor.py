"""OpenVINO-based keypoint extraction and IR utilities.

- Single Responsibility: this module handles OpenVINO pose extraction only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

from .base import PoseExtractor, TARGET_SIZE, normalize_coords


_MODEL_CACHE: Dict[Tuple[str, str], Tuple[object, object]] = {}


class OpenPoseExtractor(PoseExtractor):
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
        _ensure_openvino_model_files(model_xml)
        from openvino import Core

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


def _ensure_openvino_model_files(model_xml: str) -> None:
    """Best-effort model file fetch for Open Model Zoo IRs.

    Only attempts download for 'human-pose-estimation-0001' model layout.
    """
    xml_path = Path(model_xml)
    if xml_path.exists():
        bin_path = xml_path.with_suffix('.bin')
        if bin_path.exists():
            return
    else:
        xml_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        parts = list(xml_path.parts)
        if "human-pose-estimation-0001" not in parts:
            return
        model_idx = parts.index("human-pose-estimation-0001")
        precision = parts[model_idx + 1] if len(parts) > model_idx + 1 else "FP16"
        base_url = (
            "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.3/models_bin/3/"
            "human-pose-estimation-0001/"
        )
        xml_url = f"{base_url}{precision}/human-pose-estimation-0001.xml"
        bin_url = f"{base_url}{precision}/human-pose-estimation-0001.bin"

        def _download(url: str, dst: Path) -> None:
            try:
                import urllib.request

                with urllib.request.urlopen(url, timeout=30) as resp:  # nosec - controlled URL
                    data = resp.read()
                with open(dst, "wb") as f:
                    f.write(data)
            except Exception:
                return

        _download(xml_url, xml_path)
        _download(bin_url, xml_path.with_suffix(".bin"))
    except Exception:
        return


def is_valid_openvino_ir(xml_path: str) -> bool:
    """Lightweight sanity check for OpenVINO IR files.

    Ensures XML exists and looks like XML, and the corresponding BIN exists
    and doesn't look like an HTML error page.
    """
    try:
        p = Path(xml_path)
        if not p.exists() or p.suffix.lower() != ".xml":
            return False
        with open(p, "rb") as f:
            head = f.read(64).lstrip()
        if not head.startswith(b"<"):
            return False

        bin_path = p.with_suffix(".bin")
        if not bin_path.exists():
            return False
        with open(bin_path, "rb") as f:
            bhead = f.read(256)
        lower = bhead.lower()
        if lower.startswith(b"<") or b"<html" in lower or b"storage.openvinotoolkit.org" in lower:
            return False
        return True
    except Exception:
        return False


__all__ = [
    "OpenPoseExtractor",
    "is_valid_openvino_ir",
]

