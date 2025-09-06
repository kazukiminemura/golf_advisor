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
        # Ensure model files exist (attempt first-run download for Open Model Zoo models)
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


# ------------------------------- Facade funcs -------------------------------

# Cache for default, auto-selected backend
_DEFAULT_EXTRACTOR: PoseExtractor | None = None


# ------------------------------ YOLOv8 backend ------------------------------

_YOLO_MODEL_CACHE: Dict[Tuple[str, str], object] = {}


def _to_torch_device(device: str | None) -> str:
    if not device:
        return "cpu"
    d = device.strip().lower()
    if d in {"cpu", "cuda", "cuda:0", "cuda:1"}:
        return d
    # Map project-specific values
    if d == "gpu":
        try:
            import torch  # noqa: F401

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    # Unknown/NPU -> fallback to CPU
    return "cpu"


def _is_yolov8_model(model_path: str) -> bool:
    mp = str(model_path).lower()
    return mp.endswith(".pt") or "yolov8" in mp or mp.endswith(".onnx")


class YoloV8PoseExtractor(PoseExtractor):
    def __init__(self, model_path: str, device: str | None = None, conf: float = 0.25, imgsz: int | None = None) -> None:
        self.model_path = model_path
        self.device = _to_torch_device(device)
        self.conf = conf
        # Some Ultralytics versions require a concrete imgsz; default to 640
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
        except Exception as e:  # pragma: no cover - import-time error path
            raise ImportError(
                "ultralytics is required for YOLOv8-pose. Add 'ultralytics' to requirements and install it."
            ) from e
        # This will auto-download weights if 'model_path' is a known model name (e.g., 'yolov8n-pose.pt')
        model = YOLO(model_path)
        # The .to() is optional; we pass device per-call too
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

            # Run pose inference
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

            # No detections
            if r.keypoints is None or (getattr(r.boxes, "shape", [0])[0] if hasattr(r, "boxes") else 0) == 0:
                keypoints_per_frame.append([])
                continue

            # Select primary person (highest confidence)
            try:
                import numpy as np

                confs = r.boxes.conf.detach().cpu().numpy() if hasattr(r.boxes, "conf") else None
                idx = int(np.argmax(confs)) if confs is not None and len(confs) > 0 else 0
            except Exception:
                idx = 0

            xy = r.keypoints.xy[idx].detach().cpu().numpy()  # (17,2) for COCO
            kc = (
                r.keypoints.conf[idx].detach().cpu().numpy()
                if hasattr(r.keypoints, "conf")
                else np.ones((xy.shape[0],), dtype=float)
            )

            # Map YOLOv8 COCO-17 layout to OpenPose-like indices used downstream
            # YOLO idx: 0 nose,1 leye,2 reye,3 lear,4 rear,5 lsho,6 rsho,7 lelbow,8 relbow,9 lwrist,10 rwrist,
            #          11 lhip,12 rhip,13 lknee,14 rknee,15 lankle,16 rankle
            def mk(i):
                return (float(xy[i, 0]), float(xy[i, 1]), float(kc[i]))

            def mid(i, j):
                x = 0.5 * (xy[i, 0] + xy[j, 0])
                y = 0.5 * (xy[i, 1] + xy[j, 1])
                c = float(min(kc[i], kc[j]))
                return (float(x), float(y), c)

            mapped = [
                mk(0),               # 0 nose
                mid(5, 6),           # 1 neck (mid shoulders)
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

            pts = mapped
            keypoints_per_frame.append(normalize_coords(pts, w, h))

        cap.release()
        return keypoints_per_frame


def preload_openpose_model(model_xml: str, device: str = "CPU") -> None:
    global _DEFAULT_EXTRACTOR
    # Reset only if model/device changed or backend differs
    if not isinstance(_DEFAULT_EXTRACTOR, OpenPoseExtractor) or _DEFAULT_EXTRACTOR.model_xml != model_xml or _DEFAULT_EXTRACTOR.device != device:
        _DEFAULT_EXTRACTOR = OpenPoseExtractor(model_xml, device)


def preload_yolov8_model(model_path: str, device: str | None = None, conf: float = 0.25, imgsz: int | None = None) -> None:
    global _DEFAULT_EXTRACTOR
    if not isinstance(_DEFAULT_EXTRACTOR, YoloV8PoseExtractor) or _DEFAULT_EXTRACTOR.model_path != model_path or _DEFAULT_EXTRACTOR.device != _to_torch_device(device):
        _DEFAULT_EXTRACTOR = YoloV8PoseExtractor(model_path, device=device, conf=conf, imgsz=imgsz)


def extract_keypoints(video_path: Path, model_xml: str, device: str, target_size: tuple[int, int] | None = TARGET_SIZE):
    """Auto-select backend based on the provided model path.

    - OpenVINO backend when `model_xml` ends with .xml
    - YOLOv8-pose backend when path suggests a YOLOv8 model (.pt/.onnx or name contains 'yolov8')
    """
    global _DEFAULT_EXTRACTOR
    if _is_yolov8_model(model_xml):
        if not isinstance(_DEFAULT_EXTRACTOR, YoloV8PoseExtractor) or getattr(_DEFAULT_EXTRACTOR, "model_path", None) != model_xml or getattr(_DEFAULT_EXTRACTOR, "device", None) != _to_torch_device(device):
            preload_yolov8_model(model_xml, device)
    else:
        if not isinstance(_DEFAULT_EXTRACTOR, OpenPoseExtractor) or getattr(_DEFAULT_EXTRACTOR, "model_xml", None) != model_xml or getattr(_DEFAULT_EXTRACTOR, "device", None) != device:
            preload_openpose_model(model_xml, device)

    assert _DEFAULT_EXTRACTOR is not None
    return _DEFAULT_EXTRACTOR.extract(video_path, target_size=target_size)


# ------------------------- First-run download helpers ------------------------

def _ensure_openvino_model_files(model_xml: str) -> None:
    """If an OpenVINO IR XML is missing, try to download it (and its .bin).

    Logic is conservative: only attempts download for Open Model Zoo
    'human-pose-estimation-0001' layout when placed under 'intel/'. Otherwise
    it leaves the path untouched and lets OpenVINO raise a clear error.
    """
    xml_path = Path(model_xml)
    if xml_path.exists():
        # If XML present but BIN missing, attempt to fetch BIN only
        bin_path = xml_path.with_suffix('.bin')
        if bin_path.exists():
            return
    else:
        # Parent directory must exist for safe write
        xml_path.parent.mkdir(parents=True, exist_ok=True)

    # Only auto-download for our known default model structure
    try:
        # Expect .../human-pose-estimation-0001/<PRECISION>/human-pose-estimation-0001.xml
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
                # Silently skip on failure; OpenVINO will later raise a clear error
                return

        _download(xml_url, xml_path)
        _download(bin_url, xml_path.with_suffix(".bin"))
    except Exception:
        # Best-effort only
        return


__all__ = [
    "PoseExtractor",
    "OpenPoseExtractor",
    "YoloV8PoseExtractor",
    "extract_keypoints",
    "preload_openpose_model",
    "preload_yolov8_model",
    "normalize_coords",
    "scale_score",
]

def is_valid_openvino_ir(xml_path: str) -> bool:
    """Lightweight sanity check for OpenVINO IR files.

    Returns True only when the XML exists and looks like XML, and the
    corresponding .bin exists and does not look like an HTML error page.

    This prevents noisy startup failures on first run when models are not
    downloaded yet or placeholders are present.
    """
    try:
        p = Path(xml_path)
        if not p.exists() or p.suffix.lower() != ".xml":
            return False
        # XML should look like XML
        with open(p, "rb") as f:
            head = f.read(64).lstrip()
        if not head.startswith(b"<"):
            return False

        bin_path = p.with_suffix(".bin")
        if not bin_path.exists():
            return False
        # BIN should be binary, not an HTML page (common when a mirror returns HTML)
        with open(bin_path, "rb") as f:
            bhead = f.read(256)
        # Heuristic: HTML pages begin with '<' or contain an HTML title
        lower = bhead.lower()
        if lower.startswith(b"<") or b"<html" in lower or b"storage.openvinotoolkit.org" in lower:
            return False
        return True
    except Exception:
        return False

