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


def postprocess(results):
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
        x = point[0] / heatmap.shape[1]
        y = point[1] / heatmap.shape[0]
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
        points = postprocess(results)
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


def analyze_differences(ref_kp, test_kp):
    """Compute average per-keypoint differences between two swings."""
    import numpy as np

    length = min(len(ref_kp), len(test_kp))
    if length == 0:
        return {}
    num_kp = min(len(ref_kp[0]), len(test_kp[0]))
    diff_sum = np.zeros(num_kp)
    for i in range(length):
        ref = np.array([p[:2] for p in ref_kp[i][:num_kp]])
        test = np.array([p[:2] for p in test_kp[i][:num_kp]])
        diff_sum += np.linalg.norm(ref - test, axis=1)
    diff_avg = diff_sum / length
    names = {
        0: "nose",
        1: "neck",
        2: "right shoulder",
        3: "right elbow",
        4: "right wrist",
        5: "left shoulder",
        6: "left elbow",
        7: "left wrist",
        8: "mid hip",
        9: "right hip",
        10: "right knee",
        11: "right ankle",
        12: "left hip",
        13: "left knee",
        14: "left ankle",
    }
    return {names.get(i, str(i)): diff_avg[i] for i in range(num_kp)}


class SwingChatBot:
    """Simple wrapper around a small language model for swing advice."""

    def __init__(self, ref_kp, test_kp, score):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        diffs = analyze_differences(ref_kp, test_kp)
        significant = sorted(diffs.items(), key=lambda x: x[1], reverse=True)[:3]
        diff_text = ", ".join(f"{name} ({dist:.1f})" for name, dist in significant)

        model_name = "ELYZA/Llama-3-ELYZA-JP-8B"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        self.history = (
            "あなたは役立つゴルフスイングコーチのチャットボットです。\n"
            f"スイングの全体的な差分スコア: {score:.2f}。\n"
            f"主な差分: {diff_text}。\n"
            "簡潔なアドバイスをしてください。\nコーチ:"
        )

    def initial_message(self):
        input_ids = self.tokenizer.encode(self.history, return_tensors="pt")
        output_ids = self.model.generate(
            input_ids, max_new_tokens=60, do_sample=True, top_p=0.95, top_k=50
        )
        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        reply = response[len(self.history) :].strip()
        self.history += " " + reply
        return reply

    def ask(self, user):
        self.history += f"\nユーザー: {user}\nコーチ:"
        input_ids = self.tokenizer.encode(self.history, return_tensors="pt")
        output_ids = self.model.generate(
            input_ids, max_new_tokens=60, do_sample=True, top_p=0.95, top_k=50
        )
        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        reply = response[len(self.history) :].strip()
        self.history += " " + reply
        return reply


def run_chatbot(ref_kp, test_kp, score):
    """Provide swing advice using a small LLM chatbot in the terminal."""
    bot = SwingChatBot(ref_kp, test_kp, score)
    print(f"コーチ: {bot.initial_message()}")
    while True:
        try:
            user = input("あなた: ")
        except EOFError:
            break
        if user.strip().lower() in {"quit", "exit", "終了"}:
            break
        print(f"コーチ: {bot.ask(user)}")


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
    frame_count = min(len(ref_kp), len(test_kp))
    paused = start_paused
    combined = None
    while True:
        # Ensure the frame index wraps around for continuous looping.
        frame_idx %= frame_count
        # Seek to the current frame index so we can step forwards/backwards.
        cap_ref.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        cap_test.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret_ref, frame_ref = cap_ref.read()
        ret_test, frame_test = cap_test.read()
        if not ret_ref or not ret_test:
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
            frame_idx = (frame_idx + 1) % frame_count
            continue
        elif key == 81 and paused:  # Left arrow
            frame_idx = (frame_idx - 1) % frame_count
            continue
        elif not paused:
            frame_idx = (frame_idx + 1) % frame_count

    if combined is None:
        combined = np.zeros((480, 2 * 640, 3), dtype=np.uint8)
    cv2.putText(
        combined,
        f"差分スコア: {score:.4f}",
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


def show_comparison_with_chat(
    ref_path: Path,
    test_path: Path,
    ref_kp,
    test_kp,
    score,
    start_paused: bool = False,
):
    """Display swings alongside a chat panel in a single window."""
    import cv2
    import numpy as np
    import tkinter as tk
    from PIL import Image, ImageTk

    bot = SwingChatBot(ref_kp, test_kp, score)

    root = tk.Tk()
    root.title("スイング比較")
    video_label = tk.Label(root)
    video_label.pack()
    chat_box = tk.Text(root, height=10)
    chat_box.pack(fill=tk.BOTH, expand=True)
    entry = tk.Entry(root)
    entry.pack(fill=tk.X)

    def send_message(event=None):
        user = entry.get()
        entry.delete(0, tk.END)
        chat_box.insert(tk.END, f"あなた: {user}\n")
        reply = bot.ask(user)
        chat_box.insert(tk.END, f"コーチ: {reply}\n")
        chat_box.see(tk.END)

    entry.bind("<Return>", send_message)
    send_button = tk.Button(root, text="送信", command=send_message)
    send_button.pack()
    chat_box.insert(tk.END, f"コーチ: {bot.initial_message()}\n")

    cap_ref = cv2.VideoCapture(str(ref_path))
    cap_test = cv2.VideoCapture(str(test_path))
    frame_idx = 0
    frame_count = min(len(ref_kp), len(test_kp))
    paused = start_paused

    def update_frame():
        nonlocal frame_idx
        cap_ref.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        cap_test.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret_ref, frame_ref = cap_ref.read()
        ret_test, frame_test = cap_test.read()
        if ret_ref and ret_test:
            draw_skeleton(frame_ref, ref_kp[frame_idx])
            draw_skeleton(frame_test, test_kp[frame_idx])
            combined = cv2.hconcat([frame_ref, frame_test])
            cv2.putText(
                combined,
                f"差分スコア: {score:.4f}",
                (10, combined.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )
            img = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
            imgtk = ImageTk.PhotoImage(Image.fromarray(img))
            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)
        if not paused:
            frame_idx = (frame_idx + 1) % frame_count
        root.after(30, update_frame)

    def toggle_pause(event=None):
        nonlocal paused
        paused = not paused

    def step(delta):
        nonlocal frame_idx
        frame_idx = (frame_idx + delta) % frame_count

    root.bind("<space>", toggle_pause)
    root.bind("<Left>", lambda e: step(-1))
    root.bind("<Right>", lambda e: step(1))

    update_frame()
    root.mainloop()
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
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Show chat panel alongside comparison",
    )
    args = parser.parse_args()

    ref_kp = extract_keypoints(Path(args.reference), args.model, args.device)
    test_kp = extract_keypoints(Path(args.test), args.model, args.device)
    score = compare_swings(ref_kp, test_kp)
    print(f"Swing difference score: {score:.4f}")
    if args.chat:
        show_comparison_with_chat(
            Path(args.reference),
            Path(args.test),
            ref_kp,
            test_kp,
            score,
            start_paused=args.step,
        )
    else:
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
