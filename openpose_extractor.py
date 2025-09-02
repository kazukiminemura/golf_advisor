"""Utilities for extracting keypoints using OpenPose/OpenVINO models."""

from pathlib import Path


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


def postprocess(results):
    import cv2
    import numpy as np

    heatmaps = np.squeeze(results, axis=0)
    points = []
    num_kp = heatmaps.shape[0]
    for i in range(num_kp):
        heatmap = heatmaps[i]
        _, conf, _, point = cv2.minMaxLoc(heatmap)
        x = point[0] / heatmap.shape[1]
        y = point[1] / heatmap.shape[0]
        points.append((x, y, conf))
    return points


def extract_keypoints(video_path: Path, model_xml: str, device: str):
    """Extract keypoints from a video using the specified model."""
    import cv2

    compiled_model, output_layer = load_model(model_xml, device)
    cap = cv2.VideoCapture(str(video_path))
    keypoints = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        inp = preprocess(frame, compiled_model.input(0).shape)
        results = compiled_model([inp])[output_layer]
        points = postprocess(results)
        keypoints.append(points)
    cap.release()
    return keypoints

