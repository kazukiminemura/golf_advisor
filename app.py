from flask import Flask, request, jsonify, render_template, send_from_directory
from pathlib import Path
import json
import os
import platform
import subprocess
from werkzeug.utils import secure_filename
import psutil

from golf_swing_compare import (
    compare_swings,
    draw_skeleton,
    extract_keypoints,
    analyze_differences,
)

app = Flask(__name__)

ENABLE_CHATBOT = os.environ.get("ENABLE_CHATBOT", "").lower() in {
    "1",
    "true",
    "yes",
}

tokenizer = model = None  # Lazy-initialized chatbot model

if ENABLE_CHATBOT:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        modeling_utils,
    )
    import torch

    if getattr(modeling_utils, "ALL_PARALLEL_STYLES", None) is None:  # pragma: no cover - defensive
        modeling_utils.ALL_PARALLEL_STYLES = []

    QWEN_MODEL = "Qwen/Qwen3-8B"
    QWEN_QUANT_CONFIG = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )

    def _ensure_chatbot_model() -> None:
        """Load the LLM and tokenizer on demand to conserve memory."""
        global tokenizer, model
        if tokenizer is None or model is None:
            tokenizer = AutoTokenizer.from_pretrained(
                QWEN_MODEL, trust_remote_code=True
            )
            device_map = {"": "cuda:0"} if torch.cuda.is_available() else {"": "cpu"}
            model = AutoModelForCausalLM.from_pretrained(
                QWEN_MODEL,
                device_map=device_map,
                quantization_config=QWEN_QUANT_CONFIG,
                trust_remote_code=True,
            )
            print("LLM platform: ", next(model.parameters()).device)
else:  # pragma: no cover - simple fallback

    def _ensure_chatbot_model() -> None:  # pragma: no cover - no-op when disabled
        return

MAX_MESSAGES = 20
messages = []  # Store conversation history for the chatbot
# Optional debug mode that returns a simple echo instead of invoking the
# heavy language model.  Enable with ``CHATBOT_DEBUG=1`` when running the
# server.
CHATBOT_DEBUG = os.environ.get("CHATBOT_DEBUG", "").lower() in {
    "1",
    "true",
    "yes",
}


def _generate_reply() -> str:
    """Return an LLM-generated reply for the chatbot."""

    if not ENABLE_CHATBOT:
        return "チャットボットは無効化されています。"

    if CHATBOT_DEBUG:  # Simple echo reply for debugging conversation flow
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return f"(デバッグ) {last_user}"

    _ensure_chatbot_model()
    prompt = "あなたは役立つゴルフスイングコーチです。\n"
    for m in messages:
        role = "ユーザー" if m["role"] == "user" else "コーチ"
        prompt += f"{role}: {m['content']}\n"
    prompt += "コーチ:"

    inputs = tokenizer(prompt, return_tensors="pt")
    try:
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                top_k=50,
            )
        gen_ids = output_ids[0, inputs["input_ids"].shape[1] :]
        reply = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        return reply or "返答を生成できませんでした。"
    except Exception as exc:  # pragma: no cover - best effort logging
        app.logger.exception("Chatbot reply generation failed: %s", exc)
        return "返答の生成中にエラーが発生しました。"

# Paths and model configuration for OpenPose processing
# Use the INT8 variant of the model for faster inference by default.
MODEL_XML = "intel/human-pose-estimation-0001/FP16/human-pose-estimation-0001.xml"
DEVICE = "CPU"
REF_VIDEO = Path("data/reference.mp4")
CUR_VIDEO = Path("data/current.mp4")
OUT_REF = Path("static/reference_annotated.mp4")
OUT_CUR = Path("static/current_annotated.mp4")
REF_KP_JSON = Path("static/reference_keypoints.json")
CUR_KP_JSON = Path("static/current_keypoints.json")

score = None


def get_gpu_usage() -> float:
    """Return GPU utilization percentage if available."""
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_util = float(util.gpu)
        pynvml.nvmlShutdown()
        return gpu_util
    except Exception:
        try:  # Fallback to nvidia-smi on systems including Windows
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                encoding="utf-8",
                stderr=subprocess.DEVNULL,
            )
            return float(out.strip().splitlines()[0])
        except Exception:
            return 0.0


def get_npu_usage() -> float:
    """Best-effort NPU utilization percentage.

    Windows 11 exposes NPU usage through the ``AI Accelerator`` performance
    counters. When unavailable this function returns ``0``.
    """
    if platform.system() == "Windows":
        try:
            cmd = (
                "Get-Counter '\\AI Accelerator(*)\\Usage Percentage' "
                "| Select -First 1 -ExpandProperty CounterSamples "
                "| Select -ExpandProperty CookedValue"
            )
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", cmd],
                encoding="utf-8",
                stderr=subprocess.DEVNULL,
            )
            return float(out.strip())
        except Exception:
            return 0.0
    return 0.0


def _annotate_video(src: Path, keypoints, dst: Path) -> None:
    """Render keypoints on frames and save to a new video."""
    import cv2

    cap = cv2.VideoCapture(str(src))  # Open the source video
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0  # Use FPS from video or fallback
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Frame width in pixels
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Frame height in pixels
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for mp4 output
    writer = cv2.VideoWriter(str(dst), fourcc, fps, (width, height))  # Video writer

    for kp in keypoints:
        ret, frame = cap.read()  # Read frame-by-frame
        if not ret:
            break  # Stop if video ends
        scaled = [(x * width, y * height, c) for x, y, c in kp]  # Scale to pixels
        draw_skeleton(frame, scaled)  # Draw pose skeleton
        writer.write(frame)  # Write annotated frame

    cap.release()  # Close video reader
    writer.release()  # Finalize video writer


def _save_keypoints_json(keypoints, fps, dst: Path) -> None:
    """Save keypoints and fps to a JSON file."""
    serializable = [[list(map(float, kp)) for kp in frame] for frame in keypoints]
    # Write out a dictionary containing fps and keypoints
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
        return  # Skip processing if results already exist

    import cv2

    # Extract keypoints for reference and current videos
    ref_kp = extract_keypoints(REF_VIDEO, MODEL_XML, DEVICE)
    cur_kp = extract_keypoints(CUR_VIDEO, MODEL_XML, DEVICE)

    # Retrieve frame rates for later JSON output
    ref_cap = cv2.VideoCapture(str(REF_VIDEO))
    ref_fps = ref_cap.get(cv2.CAP_PROP_FPS) or 30.0
    ref_cap.release()
    cur_cap = cv2.VideoCapture(str(CUR_VIDEO))
    cur_fps = cur_cap.get(cv2.CAP_PROP_FPS) or 30.0
    cur_cap.release()

    # Compute similarity score and produce annotated videos/JSON files
    score = compare_swings(ref_kp, cur_kp)
    _annotate_video(REF_VIDEO, ref_kp, OUT_REF)
    _annotate_video(CUR_VIDEO, cur_kp, OUT_CUR)
    _save_keypoints_json(ref_kp, ref_fps, REF_KP_JSON)
    _save_keypoints_json(cur_kp, cur_fps, CUR_KP_JSON)
    diffs = analyze_differences(ref_kp, cur_kp)
    significant = sorted(diffs.items(), key=lambda x: x[1], reverse=True)[:3]
    diff_text = ", ".join(f"{name} ({dist:.1f})" for name, dist in significant)
    prompt = (
        "あなたは役立つゴルフスイングコーチです。\n"
        f"スイングの全体的な差スコア: {score:.2f}。\n"
        f"主な差分: {diff_text}。\n"
        "まず何が良くて何が悪いのか簡潔に教えてください。\nコーチ:"
    )
    global messages
    if ENABLE_CHATBOT:
        _ensure_chatbot_model()
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                top_k=50,
            )
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        initial = response[len(prompt) :].strip()
        messages = [{"role": "assistant", "content": initial}]
    else:
        messages.clear()


@app.route("/")
def index():
    has_results = (
        score is not None
        and OUT_REF.exists()
        and OUT_CUR.exists()
        and REF_KP_JSON.exists()
        and CUR_KP_JSON.exists()
    )  # Determine if analysis results exist
    return render_template(
        "index.html",
        score=score,
        ref_video_name=REF_VIDEO.name,
        cur_video_name=CUR_VIDEO.name,
        has_results=has_results,
        chatbot_enabled=ENABLE_CHATBOT,
    )

@app.route("/messages", methods=["GET", "POST"])
def message_handler():
    if request.method == "POST":
        if not ENABLE_CHATBOT:
            return jsonify({"reply": _generate_reply()})
        data = request.get_json() or {}
        user_msg = data.get("message", "")
        messages.append({"role": "user", "content": user_msg})
        if len(messages) > MAX_MESSAGES:
            del messages[:-MAX_MESSAGES]
        reply = _generate_reply()
        messages.append({"role": "assistant", "content": reply})
        if len(messages) > MAX_MESSAGES:
            del messages[:-MAX_MESSAGES]
        return jsonify({"reply": reply})
    else:
        return jsonify(messages if ENABLE_CHATBOT else [])


@app.route("/videos/<path:filename>")
def serve_video(filename):
    """Serve video files from the data directory."""
    return send_from_directory("data", filename)  # Flask helper to send files


@app.route("/list_videos")
def list_videos():
    """Return available mp4 files in the data directory."""
    files = sorted(p.name for p in Path("data").glob("*.mp4"))  # Find mp4 files
    return jsonify(files)


@app.route("/upload_videos", methods=["POST"])
def upload_videos():
    """Upload video files to the data directory."""
    data_dir = Path("data")  # Directory where videos are stored
    saved = {}
    ref = request.files.get("reference")  # Reference video from request
    cur = request.files.get("current")  # Current video from request
    if ref:
        name = secure_filename(ref.filename)  # Sanitize filename
        ref.save(data_dir / name)  # Save uploaded file
        saved["reference_file"] = name
    if cur:
        name = secure_filename(cur.filename)  # Sanitize filename
        cur.save(data_dir / name)  # Save uploaded file
        saved["current_file"] = name
    return jsonify(saved)


@app.route("/system_usage")
def system_usage():
    """Return current CPU, GPU and NPU utilization percentages."""
    cpu = psutil.cpu_percent()
    gpu = get_gpu_usage()
    npu = get_npu_usage()
    return jsonify({"cpu": cpu, "gpu": gpu, "npu": npu})


@app.route("/analyze", methods=["POST"])
def analyze():
    """Run pose analysis and return the score."""
    prepare_videos()  # Ensure videos are processed
    return jsonify({"score": score})  # Send back computed score


@app.route("/set_videos", methods=["POST"])
def set_videos():
    """Select videos from local files and clear previous analysis."""
    data = request.get_json() or {}
    ref_file = data.get("reference_file")
    cur_file = data.get("current_file")

    global REF_VIDEO, CUR_VIDEO
    if ref_file:
        REF_VIDEO = Path("data") / ref_file  # Update reference video path
    if cur_file:
        CUR_VIDEO = Path("data") / cur_file  # Update current video path

    global score
    score = None  # Clear previous score
    for p in (OUT_REF, OUT_CUR, REF_KP_JSON, CUR_KP_JSON):
        if p.exists():
            p.unlink()  # Remove previous output files
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=False)
