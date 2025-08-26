import argparse
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
    image = image.transpose((2, 0, 1))  # HWC -> CHW
    image = image[np.newaxis, :]
    return image


def postprocess(results, frame_height, frame_width):
    import cv2
    import numpy as np

    # The network output is expected to be a 4D tensor of shape
    # (1, num_keypoints, height, width). Remove the batch dimension so the
    # heatmaps are indexed as (num_keypoints, height, width).
    heatmaps = np.squeeze(results, axis=0)
    points = []
    num_kp = heatmaps.shape[0]
    for i in range(num_kp):
        heatmap = heatmaps[i]
        _, conf, _, point = cv2.minMaxLoc(heatmap)
        x = int(frame_width * point[0] / heatmap.shape[1])
        y = int(frame_height * point[1] / heatmap.shape[0])
        points.append((x, y, conf))
    return points


def extract_keypoints(video_path: Path, model_xml: str, device: str):
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    compiled_model, output_layer = load_model(model_xml, device)
    keypoints = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        inp = preprocess(frame, compiled_model.input(0).shape)
        results = compiled_model([inp])[output_layer]
        points = postprocess(results, frame.shape[0], frame.shape[1])
        keypoints.append(points)
    cap.release()
    return keypoints


def compare_swings(ref_kp, test_kp):
    import numpy as np

    length = min(len(ref_kp), len(test_kp))
    if length == 0:
        return float("inf")
    diff = 0.0
    for i in range(length):
        ref = np.array([p[:2] for p in ref_kp[i]])
        test = np.array([p[:2] for p in test_kp[i]])
        diff += np.linalg.norm(ref - test) / ref.size
    return diff / length


def main():
    parser = argparse.ArgumentParser(description="Compare golf swings using OpenVINO OpenPose")
    parser.add_argument("--reference", required=True, help="Reference swing video path")
    parser.add_argument("--test", required=True, help="Test swing video path")
    parser.add_argument("--model", required=True, help="Path to OpenVINO pose model (.xml)")
    parser.add_argument("--device", default="CPU", help="Device name for inference")
    args = parser.parse_args()

    ref_kp = extract_keypoints(Path(args.reference), args.model, args.device)
    test_kp = extract_keypoints(Path(args.test), args.model, args.device)
    score = compare_swings(ref_kp, test_kp)
    print(f"Swing difference score: {score:.4f}")


if __name__ == "__main__":
    main()
