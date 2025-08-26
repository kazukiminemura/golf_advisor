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


# Pairs of keypoints that make up the skeletal connections. The indices
# correspond to the standard OpenPose output ordering. When a pair index is
# outside the number of detected keypoints it is ignored so the code can work
# with different model variants.
POSE_PAIRS = [
    (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10),
    (8, 12), (12, 13), (13, 14),
    (0, 1), (0, 15), (15, 17), (0, 16), (16, 18),
]


def draw_skeleton(frame, keypoints):
    """Draw detected keypoints and skeleton on a frame."""
    import cv2

    for x, y, conf in keypoints:
        if conf > 0.3:
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
    for a, b in POSE_PAIRS:
        if a < len(keypoints) and b < len(keypoints):
            x1, y1, c1 = keypoints[a]
            x2, y2, c2 = keypoints[b]
            if c1 > 0.3 and c2 > 0.3:
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)


def show_comparison(
    ref_path: Path,
    test_path: Path,
    ref_kp,
    test_kp,
    score,
    start_paused: bool = False,
):
    """Display the reference and test swings side by side with skeletons.

    Press space to pause/resume. While paused, use the left/right arrow keys
    to step through frames. Press ``q`` to exit.
    """
    import cv2
    import numpy as np

    cap_ref = cv2.VideoCapture(str(ref_path))
    cap_test = cv2.VideoCapture(str(test_path))
    frame_idx = 0
    paused = start_paused
    combined = None
    while True:
        # Seek to the current frame index so we can step forwards/backwards.
        cap_ref.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        cap_test.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret_ref, frame_ref = cap_ref.read()
        ret_test, frame_test = cap_test.read()
        if (
            not ret_ref
            or not ret_test
            or frame_idx >= len(ref_kp)
            or frame_idx >= len(test_kp)
        ):
            break
        draw_skeleton(frame_ref, ref_kp[frame_idx])
        draw_skeleton(frame_test, test_kp[frame_idx])
        combined = cv2.hconcat([frame_ref, frame_test])
        cv2.imshow("Swing Comparison", combined)

        key = cv2.waitKey(0 if paused else 30) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            paused = not paused
        elif key == 83 and paused:  # Right arrow
            frame_idx += 1
            continue
        elif key == 81 and paused:  # Left arrow
            frame_idx = max(0, frame_idx - 1)
            continue
        elif not paused:
            frame_idx += 1

    if combined is None:
        combined = np.zeros((480, 2 * 640, 3), dtype=np.uint8)
    cv2.putText(
        combined,
        f"Score: {score:.4f}",
        (10, combined.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
    )
    cv2.imshow("Swing Comparison", combined)
    cv2.waitKey(0)
    cap_ref.release()
    cap_test.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Compare golf swings using OpenVINO OpenPose")
    parser.add_argument("--reference", required=True, help="Reference swing video path")
    parser.add_argument("--test", required=True, help="Test swing video path")
    parser.add_argument("--model", required=True, help="Path to OpenVINO pose model (.xml)")
    parser.add_argument("--device", default="CPU", help="Device name for inference")
    parser.add_argument(
        "--step",
        action="store_true",
        help="Start playback paused for frame-by-frame stepping",
    )
    args = parser.parse_args()

    ref_kp = extract_keypoints(Path(args.reference), args.model, args.device)
    test_kp = extract_keypoints(Path(args.test), args.model, args.device)
    score = compare_swings(ref_kp, test_kp)
    print(f"Swing difference score: {score:.4f}")
    show_comparison(
        Path(args.reference),
        Path(args.test),
        ref_kp,
        test_kp,
        score,
        start_paused=args.step,
    )


if __name__ == "__main__":
    main()
