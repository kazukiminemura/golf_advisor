from flask import Flask, request, jsonify, render_template, send_from_directory
from pathlib import Path

from golf_swing_compare import compare_swings, draw_skeleton, extract_keypoints

app = Flask(__name__)
messages = []  # 簡易的にメッセージをメモリに保持

# Paths and model configuration for OpenPose processing
MODEL_XML = "human-pose-estimation-0001.xml"
DEVICE = "CPU"
REF_VIDEO = Path("data/reference.mp4")
CUR_VIDEO = Path("data/current.mp4")
OUT_REF = Path("static/reference_annotated.mp4")
OUT_CUR = Path("static/current_annotated.mp4")

score = None


def _annotate_video(src: Path, keypoints, dst: Path) -> None:
    """Render keypoints on frames and save to a new video."""
    import cv2

    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(dst), fourcc, fps, (width, height))

    for kp in keypoints:
        ret, frame = cap.read()
        if not ret:
            break
        draw_skeleton(frame, kp)
        writer.write(frame)

    cap.release()
    writer.release()


def prepare_videos() -> None:
    """Generate annotated videos and compute the swing score once."""
    global score
    if score is not None and OUT_REF.exists() and OUT_CUR.exists():
        return

    ref_kp = extract_keypoints(REF_VIDEO, MODEL_XML, DEVICE)
    cur_kp = extract_keypoints(CUR_VIDEO, MODEL_XML, DEVICE)
    score = compare_swings(ref_kp, cur_kp)
    _annotate_video(REF_VIDEO, ref_kp, OUT_REF)
    _annotate_video(CUR_VIDEO, cur_kp, OUT_CUR)


@app.route("/")
def index():
    prepare_videos()
    return render_template("index.html", score=score)


@app.route("/messages", methods=["GET", "POST"])
def message_handler():
    if request.method == "POST":
        data = request.get_json()
        messages.append(data["message"])
        return "", 204  # 成功時は空でOK
    else:
        return jsonify(messages)


@app.route("/videos/<path:filename>")
def serve_video(filename):
    """Serve video files from the data directory."""
    return send_from_directory("data", filename)


if __name__ == "__main__":
    app.run(debug=True)
