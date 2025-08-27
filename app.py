from flask import Flask, request, jsonify, render_template, send_from_directory
from pathlib import Path
import json
from werkzeug.utils import secure_filename
from transformers import AutoModelForCausalLM, AutoTokenizer

from golf_swing_compare import compare_swings, draw_skeleton, extract_keypoints

app = Flask(__name__)
LLAMA_MODEL = "ELYZA/Llama-3-ELYZA-JP-8B"
tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL)
model = AutoModelForCausalLM.from_pretrained(LLAMA_MODEL)
# 簡易的なチャットボット用メッセージ履歴をメモリに保持
messages = [{"role": "assistant", "content": "ゴルフスイングについてご質問ください。"}]


def _generate_reply(_: str) -> str:
    """Return an LLM-generated reply for the chatbot."""
    prompt = "あなたは役立つゴルフスイングコーチです。\n"
    for m in messages:
        role = "ユーザー" if m["role"] == "user" else "コーチ"
        prompt += f"{role}: {m['content']}\n"
    prompt += "コーチ:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output_ids = model.generate(
        input_ids, max_new_tokens=60, do_sample=True, top_p=0.95, top_k=50
    )
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    reply = response[len(prompt):].strip()
    return reply

# Paths and model configuration for OpenPose processing
MODEL_XML = "human-pose-estimation-0001.xml"
DEVICE = "CPU"
REF_VIDEO = Path("data/reference.mp4")
CUR_VIDEO = Path("data/current.mp4")
OUT_REF = Path("static/reference_annotated.mp4")
OUT_CUR = Path("static/current_annotated.mp4")
REF_KP_JSON = Path("static/reference_keypoints.json")
CUR_KP_JSON = Path("static/current_keypoints.json")

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
        scaled = [(x * width, y * height, c) for x, y, c in kp]
        draw_skeleton(frame, scaled)
        writer.write(frame)

    cap.release()
    writer.release()


def _save_keypoints_json(keypoints, fps, dst: Path) -> None:
    """Save keypoints and fps to a JSON file."""
    serializable = [[list(map(float, kp)) for kp in frame] for frame in keypoints]
    with dst.open("w") as f:
        json.dump({"fps": fps, "keypoints": serializable}, f)


def prepare_videos() -> None:
    """Generate annotated videos, keypoint JSONs and compute the swing score."""
    global score
    if (
        score is not None
        and OUT_REF.exists()
        and OUT_CUR.exists()
        and REF_KP_JSON.exists()
        and CUR_KP_JSON.exists()
    ):
        return

    import cv2

    ref_kp = extract_keypoints(REF_VIDEO, MODEL_XML, DEVICE)
    cur_kp = extract_keypoints(CUR_VIDEO, MODEL_XML, DEVICE)

    ref_cap = cv2.VideoCapture(str(REF_VIDEO))
    ref_fps = ref_cap.get(cv2.CAP_PROP_FPS) or 30.0
    ref_cap.release()
    cur_cap = cv2.VideoCapture(str(CUR_VIDEO))
    cur_fps = cur_cap.get(cv2.CAP_PROP_FPS) or 30.0
    cur_cap.release()

    score = compare_swings(ref_kp, cur_kp)
    _annotate_video(REF_VIDEO, ref_kp, OUT_REF)
    _annotate_video(CUR_VIDEO, cur_kp, OUT_CUR)
    _save_keypoints_json(ref_kp, ref_fps, REF_KP_JSON)
    _save_keypoints_json(cur_kp, cur_fps, CUR_KP_JSON)


@app.route("/")
def index():
    prepare_videos()
    return render_template(
        "index.html",
        score=score,
        ref_video_name=REF_VIDEO.name,
        cur_video_name=CUR_VIDEO.name,
    )


@app.route("/messages", methods=["GET", "POST"])
def message_handler():
    if request.method == "POST":
        data = request.get_json() or {}
        user_msg = data.get("message", "")
        messages.append({"role": "user", "content": user_msg})
        reply = _generate_reply(user_msg)
        messages.append({"role": "assistant", "content": reply})
        return jsonify({"reply": reply})
    else:
        return jsonify(messages)


@app.route("/videos/<path:filename>")
def serve_video(filename):
    """Serve video files from the data directory."""
    return send_from_directory("data", filename)


@app.route("/list_videos")
def list_videos():
    """Return available mp4 files in the data directory."""
    files = sorted(p.name for p in Path("data").glob("*.mp4"))
    return jsonify(files)


@app.route("/upload_videos", methods=["POST"])
def upload_videos():
    """Upload video files to the data directory."""
    data_dir = Path("data")
    saved = {}
    ref = request.files.get("reference")
    cur = request.files.get("current")
    if ref:
        name = secure_filename(ref.filename)
        ref.save(data_dir / name)
        saved["reference_file"] = name
    if cur:
        name = secure_filename(cur.filename)
        cur.save(data_dir / name)
        saved["current_file"] = name
    return jsonify(saved)


@app.route("/set_videos", methods=["POST"])
def set_videos():
    """Select videos from local files and re-run analysis."""
    data = request.get_json() or {}
    ref_file = data.get("reference_file")
    cur_file = data.get("current_file")

    global REF_VIDEO, CUR_VIDEO
    if ref_file:
        REF_VIDEO = Path("data") / ref_file
    if cur_file:
        CUR_VIDEO = Path("data") / cur_file

    global score
    score = None
    for p in (OUT_REF, OUT_CUR, REF_KP_JSON, CUR_KP_JSON):
        if p.exists():
            p.unlink()
    prepare_videos()
    return jsonify({"score": score})


if __name__ == "__main__":
    app.run(debug=True)
