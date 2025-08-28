import os
import json
import psutil
import asyncio
import platform
import subprocess
from pathlib import Path
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, render_template, send_from_directory

from golf_swing_compare import (
    compare_swings,
    draw_skeleton,
    extract_keypoints,
    SwingChatBot,
)
from simple_chatbot import SimpleChatBot

app = Flask(__name__)

ENABLE_CHATBOT = os.environ.get("ENABLE_CHATBOT", "").lower() in {
    "1",
    "true",
    "yes",
}

# Option to initialize chatbot only when first used (to save memory during analysis)
LAZY_CHATBOT_INIT = os.environ.get("LAZY_CHATBOT_INIT", "true").lower() in {
    "1",
    "true",
    "yes",
}

bot = None
MAX_MESSAGES = 20
messages = []  # Store conversation history for the swing chatbot

# General purpose chatbot
general_bot = None
general_messages = []

# Cached keypoints for chatbot initialization
ref_keypoints = None
cur_keypoints = None

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


def _init_chatbot_sync() -> bool:
    """Initialize chatbot using cached keypoints if enabled.
    
    Returns:
        bool: True if chatbot was successfully initialized, False otherwise.
    """
    global bot, messages, ref_keypoints, cur_keypoints, score
    
    if not ENABLE_CHATBOT:
        app.logger.info("Chatbot is disabled by configuration")
        bot = None
        messages.clear()
        return False
        
    if bot is not None:
        app.logger.info("Chatbot already initialized")
        return True  # Already initialized
        
    # Check if all required data is available
    if ref_keypoints is None:
        app.logger.warning("Reference keypoints not available for chatbot initialization")
        return False
    if cur_keypoints is None:
        app.logger.warning("Current keypoints not available for chatbot initialization")
        return False
    if score is None:
        app.logger.warning("Score not available for chatbot initialization")
        return False
    
    try:
        app.logger.info("Initializing chatbot with keypoints and score")
        bot = SwingChatBot(ref_keypoints, cur_keypoints, score)
        initial_msg = bot.initial_message()
        messages = [{"role": "assistant", "content": initial_msg}]
        app.logger.info("Chatbot initialized successfully")
        return True
    except Exception as exc:
        app.logger.exception("Failed to initialize chatbot: %s", exc)
        bot = None
        messages.clear()
        return False


async def init_chatbot() -> bool:
    """Asynchronously initialize the chatbot.
    
    Returns:
        bool: True if chatbot was successfully initialized, False otherwise.
    """
    return await asyncio.to_thread(_init_chatbot_sync)


def _prepare_videos_sync() -> None:
    """Generate annotated videos, keypoint JSONs and compute the swing score."""
    global score, ref_keypoints, cur_keypoints
    if (
        score is not None
        and OUT_REF.exists()
        and OUT_CUR.exists()
        and REF_KP_JSON.exists()
        and CUR_KP_JSON.exists()
    ):
        app.logger.info("Videos already processed, skipping")
        return  # Skip processing if results already exist

    try:
        import cv2
        app.logger.info("Starting video analysis...")
        app.logger.info(f"ENABLE_CHATBOT setting: {ENABLE_CHATBOT}")
        app.logger.info(f"Reference video: {REF_VIDEO}, exists: {REF_VIDEO.exists()}")
        app.logger.info(f"Current video: {CUR_VIDEO}, exists: {CUR_VIDEO.exists()}")
        app.logger.info(f"Model path: {MODEL_XML}")
        app.logger.info(f"Device: {DEVICE}")
        
        # Check video files exist
        if not REF_VIDEO.exists():
            raise FileNotFoundError(f"Reference video not found: {REF_VIDEO}")
        if not CUR_VIDEO.exists():
            raise FileNotFoundError(f"Current video not found: {CUR_VIDEO}")
        
        # Extract keypoints for reference and current videos
        app.logger.info("Extracting keypoints from reference video...")
        try:
            ref_keypoints = extract_keypoints(REF_VIDEO, MODEL_XML, DEVICE)
            app.logger.info(f"Reference keypoints extracted: {len(ref_keypoints)} frames")
        except Exception as e:
            app.logger.error(f"Failed to extract reference keypoints: {e}")
            raise
            
        app.logger.info("Extracting keypoints from current video...")
        try:
            cur_keypoints = extract_keypoints(CUR_VIDEO, MODEL_XML, DEVICE)
            app.logger.info(f"Current keypoints extracted: {len(cur_keypoints)} frames")
        except Exception as e:
            app.logger.error(f"Failed to extract current keypoints: {e}")
            raise

        # Retrieve frame rates for later JSON output
        app.logger.info("Getting video frame rates...")
        ref_cap = cv2.VideoCapture(str(REF_VIDEO))
        ref_fps = ref_cap.get(cv2.CAP_PROP_FPS) or 30.0
        ref_cap.release()
        cur_cap = cv2.VideoCapture(str(CUR_VIDEO))
        cur_fps = cur_cap.get(cv2.CAP_PROP_FPS) or 30.0
        cur_cap.release()
        app.logger.info(f"Frame rates - Reference: {ref_fps}, Current: {cur_fps}")

        # Compute similarity score and produce annotated videos/JSON files
        app.logger.info("Computing swing similarity score...")
        try:
            score = compare_swings(ref_keypoints, cur_keypoints)
            app.logger.info(f"Computed score: {score}")
        except Exception as e:
            app.logger.error(f"Failed to compute swing score: {e}")
            raise
        
        app.logger.info("Creating annotated videos...")
        try:
            _annotate_video(REF_VIDEO, ref_keypoints, OUT_REF)
            app.logger.info("Reference video annotated")
            _annotate_video(CUR_VIDEO, cur_keypoints, OUT_CUR)
            app.logger.info("Current video annotated")
            _save_keypoints_json(ref_keypoints, ref_fps, REF_KP_JSON)
            app.logger.info("Reference keypoints JSON saved")
            _save_keypoints_json(cur_keypoints, cur_fps, CUR_KP_JSON)
            app.logger.info("Current keypoints JSON saved")
        except Exception as e:
            app.logger.error(f"Failed to create annotated videos: {e}")
            raise
        
        app.logger.info("Video analysis completed successfully")
        
    except Exception as e:
        app.logger.exception(f"Critical error during video analysis: {e}")
        # Clean up partial results
        score = None
        ref_keypoints = None
        cur_keypoints = None
        for p in (OUT_REF, OUT_CUR, REF_KP_JSON, CUR_KP_JSON):
            if p.exists():
                try:
                    p.unlink()
                    app.logger.info(f"Cleaned up partial result: {p}")
                except Exception:
                    pass
        raise


async def prepare_videos() -> None:
    """Asynchronously generate annotated videos and compute the swing score."""
    await asyncio.to_thread(_prepare_videos_sync)


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
        device=DEVICE,
    )


@app.route("/chat")
def chat():
    """Render simple general-purpose chat interface."""
    return render_template("chat.html")


@app.route("/chat_messages", methods=["GET", "POST"])
def chat_messages_handler():
    """Handle messages for the general-purpose chatbot."""
    global general_bot, general_messages
    if request.method == "POST":
        data = request.get_json() or {}
        user_msg = data.get("message", "").strip()
        if not user_msg:
            return jsonify({"reply": ""})
        if general_bot is None:
            general_bot = SimpleChatBot()
        general_messages.append({"role": "user", "content": user_msg})
        if len(general_messages) > MAX_MESSAGES:
            general_messages[:] = general_messages[-MAX_MESSAGES:]
        reply = general_bot.ask(user_msg)
        general_messages.append({"role": "assistant", "content": reply})
        if len(general_messages) > MAX_MESSAGES:
            general_messages[:] = general_messages[-MAX_MESSAGES:]
        return jsonify({"reply": reply})
    else:
        return jsonify(general_messages)

@app.route("/messages", methods=["GET", "POST"])
def message_handler():
    if not ENABLE_CHATBOT:
        if request.method == "POST":
            return jsonify({"reply": "チャットボットは無効化されています。"})
        return jsonify([])

    if request.method == "POST":
        data = request.get_json() or {}
        user_msg = data.get("message", "").strip()
        
        if not user_msg:
            return jsonify({"reply": "メッセージを入力してください。"})

        # Try to initialize the chatbot if it's not ready
        if bot is None:
            success = _init_chatbot_sync()
            if not success:
                return jsonify({"reply": "チャットボットは準備中です。まず動画を分析してください。"})

        try:
            # Add user message to conversation history
            messages.append({"role": "user", "content": user_msg})
            if len(messages) > MAX_MESSAGES:
                messages[:] = messages[-MAX_MESSAGES:]
            
            # Get bot response
            reply = bot.ask(user_msg)
            
            # Add bot response to conversation history
            messages.append({"role": "assistant", "content": reply})
            if len(messages) > MAX_MESSAGES:
                messages[:] = messages[-MAX_MESSAGES:]
            
            return jsonify({"reply": reply})
            
        except Exception as exc:
            app.logger.exception("Error in chatbot conversation: %s", exc)
            return jsonify({"reply": "申し訳ございませんが、エラーが発生しました。もう一度お試しください。"})
    else:
        # GET request - return conversation history or initialization message
        if bot is not None:
            return jsonify(messages)
        else:
            # Check if video analysis is complete but chatbot not initialized
            analysis_complete = all(x is not None for x in [ref_keypoints, cur_keypoints, score])
            if analysis_complete:
                return jsonify([
                    {
                        "role": "assistant", 
                        "content": "動画分析が完了しました。チャットボットの準備ができました！質問をどうぞ。"
                    }
                ])
            else:
                return jsonify([
                    {
                        "role": "assistant",
                        "content": "チャットボットは準備中です。まず動画を分析してください。",
                    }
                ])


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
    data_dir.mkdir(exist_ok=True)  # Ensure directory exists
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
    
    # Add memory usage information
    memory = psutil.virtual_memory()
    memory_percent = memory.percent
    memory_available_gb = memory.available / (1024**3)
    memory_used_gb = memory.used / (1024**3)
    memory_total_gb = memory.total / (1024**3)
    
    return jsonify({
        "cpu": cpu, 
        "gpu": gpu, 
        "npu": npu,
        "memory_percent": memory_percent,
        "memory_available_gb": round(memory_available_gb, 2),
        "memory_used_gb": round(memory_used_gb, 2),
        "memory_total_gb": round(memory_total_gb, 2)
    })


@app.route("/analyze", methods=["POST"])
async def analyze():
    """Run pose analysis and return the score."""
    try:
        app.logger.info(f"Starting analysis with ENABLE_CHATBOT={ENABLE_CHATBOT}")
        
        # Clear any existing chatbot to free memory
        global bot, messages
        if bot is not None:
            app.logger.info("Clearing existing chatbot to free memory")
            bot = None
            messages.clear()
            
        await prepare_videos()  # Ensure videos are processed
        
        # After video processing, try to initialize chatbot automatically
        # Only if chatbot is enabled and video processing succeeded
        if ENABLE_CHATBOT and score is not None and not LAZY_CHATBOT_INIT:
            app.logger.info("Attempting to initialize chatbot after successful analysis")
            try:
                success = await init_chatbot()
                if success:
                    app.logger.info("Chatbot automatically initialized after analysis")
                else:
                    app.logger.warning("Failed to automatically initialize chatbot after analysis")
            except Exception as chatbot_exc:
                app.logger.error(f"Chatbot initialization failed: {chatbot_exc}")
                # Don't fail the whole analysis if chatbot init fails
        
        if score is None:
            return jsonify({"error": "動画分析に失敗しました。ログを確認してください。"}), 500
            
        return jsonify({"score": score, "analysis_complete": True})  # Send back computed score with status
        
    except Exception as exc:
        app.logger.exception("Error during analysis: %s", exc)
        error_msg = str(exc)
        if "FileNotFoundError" in error_msg:
            return jsonify({"error": f"動画ファイルが見つかりません: {error_msg}"}), 400
        elif "extract_keypoints" in error_msg:
            return jsonify({"error": "キーポイント抽出に失敗しました。モデルファイルを確認してください。"}), 500
        else:
            return jsonify({"error": f"分析中にエラーが発生しました: {error_msg}"}), 500


@app.route("/init_chatbot", methods=["POST"])
async def init_chatbot_route():
    """Initialize the chatbot after videos are processed."""
    try:
        success = await init_chatbot()  # Initialize chatbot separately
        if success:
            return jsonify({"status": "ok", "message": "チャットボットの準備ができました。"})
        else:
            return jsonify({"status": "error", "message": "チャットボットの初期化に失敗しました。"}), 400
    except Exception as exc:
        app.logger.exception("Error initializing chatbot: %s", exc)
        return jsonify({"status": "error", "message": "チャットボットの初期化中にエラーが発生しました。"}), 500


@app.route("/set_videos", methods=["POST"])
def set_videos():
    """Select videos from local files and clear previous analysis."""
    data = request.get_json() or {}
    ref_file = data.get("reference_file")
    cur_file = data.get("current_file")
    device = (data.get("device") or "").upper()

    global REF_VIDEO, CUR_VIDEO, DEVICE
    if ref_file:
        REF_VIDEO = Path("data") / ref_file  # Update reference video path
    if cur_file:
        CUR_VIDEO = Path("data") / cur_file  # Update current video path
    if device in {"CPU", "GPU", "NPU"}:
        DEVICE = device

    # Clear all previous analysis data
    global score, bot, messages, ref_keypoints, cur_keypoints
    score = None  # Clear previous score
    bot = None
    messages.clear()
    ref_keypoints = None
    cur_keypoints = None
    
    # Remove previous output files
    for p in (OUT_REF, OUT_CUR, REF_KP_JSON, CUR_KP_JSON):
        if p.exists():
            p.unlink()  # Remove previous output files
            
    app.logger.info(
        f"Videos set to: ref={REF_VIDEO.name}, cur={CUR_VIDEO.name}, device={DEVICE}"
    )
    return jsonify({"status": "ok", "device": DEVICE})


@app.route("/chatbot_status")
def chatbot_status():
    """Get current chatbot status."""
    analysis_complete = all(x is not None for x in [ref_keypoints, cur_keypoints, score])
    return jsonify({
        "enabled": ENABLE_CHATBOT,
        "initialized": bot is not None,
        "ready_to_init": analysis_complete,
        "analysis_complete": analysis_complete,
        "status_message": (
            "チャットボットの準備ができました。" if bot is not None 
            else "動画分析が完了しました。チャットボットの準備ができました！" if analysis_complete
            else "チャットボットは準備中です。まず動画を分析してください。"
        )
    })


if __name__ == "__main__":
    # Ensure required directories exist
    Path("data").mkdir(exist_ok=True)
    Path("static").mkdir(exist_ok=True)
    
    app.run(debug=True)