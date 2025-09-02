"""Utilities for extracting keypoints using OpenPose/OpenVINO models."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Normalization utilities
# ---------------------------------------------------------------------------

# Fixed resolution used during keypoint extraction.  Resizing frames to a common
# size before inference helps make scores less sensitive to the input video
# resolution.
TARGET_SIZE = (1280, 720)

# Baseline height used when optionally scaling scores to account for different
# input resolutions.  Exposed so callers can reuse the same logic.
BASE_RESOLUTION = 720


def normalize_coords(keypoints, width: int, height: int):
    """Return keypoints as 0-1 normalized coordinates.

    Args:
        keypoints: Iterable of ``(x, y, conf)`` tuples in pixel coordinates.
        width: Frame width in pixels.
        height: Frame height in pixels.
    """

    return [
        (kp[0] / width, kp[1] / height, kp[2])
        for kp in keypoints
    ]


def scale_score(score: float, height: int, base: int = BASE_RESOLUTION) -> float:
    """Scale ``score`` according to ``height`` relative to ``base`` resolution."""

    return score * (base / max(height, 1))


def load_model(model_xml: str, device: str = "CPU"):
    """Load an OpenVINO pose estimation model."""
    from openvino.runtime import Core

    core = Core()
    model = core.read_model(model=model_xml)
    compiled_model = core.compile_model(model=model, device_name=device)
    output_layer = compiled_model.output(0)
    return compiled_model, output_layer


def preprocess(frame, input_shape):
    import cv2
    import numpy as np

    _, _, h, w = input_shape
    image = cv2.resize(frame, (w, h))
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    return image


def postprocess(results, frame_size):
    import cv2
    import numpy as np

    heatmaps = np.squeeze(results, axis=0)
    points = []
    num_kp = heatmaps.shape[0]
    frame_w, frame_h = frame_size
    for i in range(num_kp):
        heatmap = heatmaps[i]
        _, conf, _, point = cv2.minMaxLoc(heatmap)
        # Convert heatmap coordinates to pixel coordinates of the processed frame
        x = point[0] * frame_w / heatmap.shape[1]
        y = point[1] * frame_h / heatmap.shape[0]
        points.append((x, y, conf))
    return points


def extract_keypoints(
    video_path: Path,
    model_xml: str,
    device: str,
    target_size: tuple[int, int] | None = TARGET_SIZE,
):
    """Extract keypoints from a video using the specified model.

    Frames are optionally resized to ``target_size`` before inference and the
    returned keypoints are normalized to the range ``[0, 1]``.
    """
    import cv2

    compiled_model, output_layer = load_model(model_xml, device)
    cap = cv2.VideoCapture(str(video_path))
    keypoints = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if target_size is not None:
            frame = cv2.resize(frame, target_size)
        h, w = frame.shape[:2]
        inp = preprocess(frame, compiled_model.input(0).shape)
        results = compiled_model([inp])[output_layer]
        pts = postprocess(results, (w, h))
        keypoints.append(normalize_coords(pts, w, h))
    cap.release()
    return keypoints

