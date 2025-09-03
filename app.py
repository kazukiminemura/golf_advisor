import os
import json
import psutil
import asyncio
import platform
import subprocess
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, UploadFile, File, Response
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
import base64
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from golf_swing_compare import (
    compare_swings,
    draw_skeleton,
    extract_keypoints,
    EnhancedSwingChatBot,
)
from simple_chatbot import SimpleChatBot

app = FastAPI()

# Work around Windows Proactor event loop issues causing noisy
# ConnectionResetError stack traces when clients disconnect abruptly.
# The Selector policy is more compatible with common libraries on Windows.
if os.name == "nt":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Serve a tiny in-memory favicon to prevent 404s.

    Replace this with a real icon by placing `static/favicon.ico` and
    changing this route to return that file if desired.
    """
    tiny_png_b64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO3rN5cAAAAASUVORK5CYII="
    )
    return Response(content=base64.b64decode(tiny_png_b64), media_type="image/png")

# Mount static files (equivalent to Flask's static folder)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

logger = logging.getLogger("uvicorn.error")


def _safe_filename(name: str) -> str:
    """Return a filesystem-safe filename (basename, no path separators)."""
    # Keep only the final component and strip dangerous characters
    base = Path(name).name
    # Replace path-unfriendly characters
    return "".join(c for c in base if c.isalnum() or c in {"_", "-", ".", " "}).strip()


def _env_flag(name: str, default: str = "") -> bool:
    """Return True if the given environment variable looks truthy.

    This helper normalizes common truthy values ("1", "true", "yes") and allows
    the application to respect changes to the environment without requiring a
    restart.  Previously ``ENABLE_CHATBOT`` was evaluated only once at import
    time which could lead to the frontend not rendering the chat panel even when
    the variable was set later in the execution environment.
    """

    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes"}


def is_chatbot_enabled() -> bool:
    """Return True unless ``ENABLE_CHATBOT`` explicitly disables the chatbot.

    The web UI now shows the chat panel by default. Set ``ENABLE_CHATBOT=0``
    to hide the chatbot entirely.
    """

    return _env_flag("ENABLE_CHATBOT", "1")


# Option to initialize chatbot only when first used (to save memory during analysis)
LAZY_CHATBOT_INIT = _env_flag("LAZY_CHATBOT_INIT", "true")

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
analysis_running = False  # Guard to prevent chatbot init during analysis


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
    
    if not is_chatbot_enabled():
        logger.info("Chatbot is disabled by configuration")
        bot = None
        messages.clear()
        return False
        
    if bot is not None:
        logger.info("Chatbot already initialized")
        return True  # Already initialized
        
    # Check if all required data is available
    if ref_keypoints is None:
        logger.warning("Reference keypoints not available for chatbot initialization")
        return False
    if cur_keypoints is None:
        logger.warning("Current keypoints not available for chatbot initialization")
        return False
    if score is None:
        logger.warning("Score not available for chatbot initialization")
        return False
    # Avoid initializing the chatbot while analysis is still producing outputs
    if analysis_running:
        logger.info("Deferring chatbot initialization until analysis completes")
        return False
    
    try:
        logger.info("Initializing chatbot with keypoints and score")
        bot = EnhancedSwingChatBot(ref_keypoints, cur_keypoints, score)
        initial_msg = bot.initial_message()
        messages = [{"role": "assistant", "content": initial_msg}]
        logger.info("Chatbot initialized successfully")
        return True
    except Exception as exc:
        logger.exception("Failed to initialize chatbot: %s", exc)
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
    global score, ref_keypoints, cur_keypoints, analysis_running
    if (
        score is not None
        and OUT_REF.exists()
        and OUT_CUR.exists()
        and REF_KP_JSON.exists()
        and CUR_KP_JSON.exists()
    ):
        logger.info("Videos already processed, skipping")
        return  # Skip processing if results already exist

    analysis_running = True
    try:
        import cv2
        logger.info("Starting video analysis...")
        logger.info(
            f"ENABLE_CHATBOT setting: {is_chatbot_enabled()}"
        )
        logger.info(f"Reference video: {REF_VIDEO}, exists: {REF_VIDEO.exists()}")
        logger.info(f"Current video: {CUR_VIDEO}, exists: {CUR_VIDEO.exists()}")
        logger.info(f"Model path: {MODEL_XML}")
        logger.info(f"Device: {DEVICE}")
        
        # Check video files exist
        if not REF_VIDEO.exists():
            raise FileNotFoundError(f"Reference video not found: {REF_VIDEO}")
        if not CUR_VIDEO.exists():
            raise FileNotFoundError(f"Current video not found: {CUR_VIDEO}")
        
        # Extract keypoints for reference and current videos
        logger.info("Extracting keypoints from reference video...")
        try:
            ref_keypoints = extract_keypoints(REF_VIDEO, MODEL_XML, DEVICE)
            logger.info(f"Reference keypoints extracted: {len(ref_keypoints)} frames")
        except Exception as e:
            logger.error(f"Failed to extract reference keypoints: {e}")
            raise
            
        logger.info("Extracting keypoints from current video...")
        try:
            cur_keypoints = extract_keypoints(CUR_VIDEO, MODEL_XML, DEVICE)
            logger.info(f"Current keypoints extracted: {len(cur_keypoints)} frames")
        except Exception as e:
            logger.error(f"Failed to extract current keypoints: {e}")
            raise

        # Retrieve frame rates for later JSON output
        logger.info("Getting video frame rates...")
        ref_cap = cv2.VideoCapture(str(REF_VIDEO))
        ref_fps = ref_cap.get(cv2.CAP_PROP_FPS) or 30.0
        ref_cap.release()
        cur_cap = cv2.VideoCapture(str(CUR_VIDEO))
        cur_fps = cur_cap.get(cv2.CAP_PROP_FPS) or 30.0
        cur_cap.release()
        logger.info(f"Frame rates - Reference: {ref_fps}, Current: {cur_fps}")

        # Compute similarity score and produce annotated videos/JSON files
        logger.info("Computing swing similarity score...")
        try:
            score, _ = compare_swings(ref_keypoints, cur_keypoints)
            logger.info(f"Computed score: {score}")
        except Exception as e:
            logger.error(f"Failed to compute swing score: {e}")
            raise
        
        logger.info("Creating annotated videos...")
        try:
            _annotate_video(REF_VIDEO, ref_keypoints, OUT_REF)
            logger.info("Reference video annotated")
            _annotate_video(CUR_VIDEO, cur_keypoints, OUT_CUR)
            logger.info("Current video annotated")
            _save_keypoints_json(ref_keypoints, ref_fps, REF_KP_JSON)
            logger.info("Reference keypoints JSON saved")
            _save_keypoints_json(cur_keypoints, cur_fps, CUR_KP_JSON)
            logger.info("Current keypoints JSON saved")
        except Exception as e:
            logger.error(f"Failed to create annotated videos: {e}")
            raise
        
        logger.info("Video analysis completed successfully")
        
    except Exception as e:
        logger.exception(f"Critical error during video analysis: {e}")
        # Clean up partial results
        score = None
        ref_keypoints = None
        cur_keypoints = None
        for p in (OUT_REF, OUT_CUR, REF_KP_JSON, CUR_KP_JSON):
            if p.exists():
                try:
                    p.unlink()
                    logger.info(f"Cleaned up partial result: {p}")
                except Exception:
                    pass
        raise


async def prepare_videos() -> None:
    """Asynchronously generate annotated videos and compute the swing score."""
    await asyncio.to_thread(_prepare_videos_sync)


def _results_ready() -> bool:
    """Return True when analysis artifacts are ready for UI/chat.

    Prefer JSON keypoints existence since the web UI overlays skeletons
    client-side and the chatbot only needs keypoints and score.
    """
    # Prefer in-memory values first
    if score is not None and ref_keypoints is not None and cur_keypoints is not None:
        return True
    # Fallback to persisted artifacts (JSONs) plus score
    return score is not None and REF_KP_JSON.exists() and CUR_KP_JSON.exists()


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    has_results = _results_ready()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "score": score,
            "ref_video_name": REF_VIDEO.name,
            "cur_video_name": CUR_VIDEO.name,
            "ref_kp_name": REF_KP_JSON.name,
            "cur_kp_name": CUR_KP_JSON.name,
            "has_results": has_results,
            "chatbot_enabled": is_chatbot_enabled(),
            "device": DEVICE,
        },
    )

@app.api_route("/chat_messages", methods=["GET", "POST"])
async def chat_messages_handler(request: Request):
    """Handle messages for the general-purpose chatbot."""
    global general_bot, general_messages
    if request.method == "POST":
        data = (await request.json()) or {}
        user_msg = data.get("message", "").strip()
        if not user_msg:
            return JSONResponse({"reply": ""})
        if general_bot is None:
            general_bot = SimpleChatBot()
        general_messages.append({"role": "user", "content": user_msg})
        if len(general_messages) > MAX_MESSAGES:
            general_messages[:] = general_messages[-MAX_MESSAGES:]
        reply = general_bot.ask(user_msg)
        general_messages.append({"role": "assistant", "content": reply})
        if len(general_messages) > MAX_MESSAGES:
            general_messages[:] = general_messages[-MAX_MESSAGES:]
        return JSONResponse({"reply": reply})
    else:
        return JSONResponse(general_messages)

@app.api_route("/messages", methods=["GET", "POST"])
async def message_handler(request: Request):
    if not is_chatbot_enabled():
        if request.method == "POST":
            return JSONResponse({"reply": "チャットボットは無効化されています。"})
        return JSONResponse([])

    if request.method == "POST":
        data = (await request.json()) or {}
        user_msg = data.get("message", "").strip()
        
        if not user_msg:
            return JSONResponse({"reply": "メッセージを入力してください。"})

        # Try to initialize the chatbot if it's not ready
        if bot is None:
            success = _init_chatbot_sync()
            if not success:
                return JSONResponse({"reply": "チャットボットは準備中です。まず動画を分析してください。"})

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
            
            return JSONResponse({"reply": reply})
            
        except Exception as exc:
            logger.exception("Error in chatbot conversation: %s", exc)
            return JSONResponse({"reply": "申し訳ございませんが、エラーが発生しました。もう一度お試しください。"})
    else:
        # GET request - return conversation history or initialization message
        if bot is not None:
            return JSONResponse(messages)
        else:
            # Check if video analysis is complete but chatbot not initialized
            analysis_complete = (not analysis_running) and _results_ready()
            if analysis_complete:
                return JSONResponse([
                    {
                        "role": "assistant", 
                        "content": "動画分析が完了しました。チャットボットの準備ができました！質問をどうぞ。"
                    }
                ])
            else:
                return JSONResponse([
                    {
                        "role": "assistant",
                        "content": "チャットボットは準備中です。まず動画を分析してください。",
                    }
                ])


@app.get("/videos/{filename:path}")
def serve_video(filename: str):
    """Serve video files from the data directory."""
    safe = Path("data") / Path(filename)
    # Prevent path traversal outside data dir
    try:
        safe_path = safe.resolve(strict=True)
        data_root = Path("data").resolve()
        if data_root not in safe_path.parents and safe_path != data_root:
            raise RuntimeError("Invalid path")
    except Exception:
        return JSONResponse({"error": "file not found"}, status_code=404)
    return FileResponse(str(safe_path))


@app.get("/list_videos")
def list_videos():
    """Return available mp4 files in the data directory."""
    files = sorted(p.name for p in Path("data").glob("*.mp4"))  # Find mp4 files
    return JSONResponse(files)


@app.post("/upload_videos")
async def upload_videos(reference: Optional[UploadFile] = File(None), current: Optional[UploadFile] = File(None)):
    """Upload video files to the data directory."""
    data_dir = Path("data")  # Directory where videos are stored
    data_dir.mkdir(exist_ok=True)  # Ensure directory exists
    saved = {}
    if reference is not None:
        name = _safe_filename(reference.filename or "reference.mp4")
        dest = data_dir / name
        with dest.open("wb") as f:
            f.write(await reference.read())
        saved["reference_file"] = name
    if current is not None:
        name = _safe_filename(current.filename or "current.mp4")
        dest = data_dir / name
        with dest.open("wb") as f:
            f.write(await current.read())
        saved["current_file"] = name
    return JSONResponse(saved)


@app.get("/system_usage")
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
    
    return JSONResponse({
        "cpu": cpu, 
        "gpu": gpu, 
        "npu": npu,
        "memory_percent": memory_percent,
        "memory_available_gb": round(memory_available_gb, 2),
        "memory_used_gb": round(memory_used_gb, 2),
        "memory_total_gb": round(memory_total_gb, 2)
    })


@app.post("/analyze")
async def analyze():
    """Run pose analysis and return the score."""
    try:
        logger.info(
            f"Starting analysis with ENABLE_CHATBOT={is_chatbot_enabled()}"
        )
        
        # Clear any existing chatbot to free memory
        global bot, messages
        if bot is not None:
            logger.info("Clearing existing chatbot to free memory")
            bot = None
            messages.clear()
            
        await prepare_videos()  # Ensure videos are processed
        
        # After video processing, try to initialize chatbot automatically
        # Only if chatbot is enabled and video processing succeeded
        if is_chatbot_enabled() and score is not None and not LAZY_CHATBOT_INIT:
            logger.info("Attempting to initialize chatbot after successful analysis")
            try:
                success = await init_chatbot()
                if success:
                    logger.info("Chatbot automatically initialized after analysis")
                else:
                    logger.warning("Failed to automatically initialize chatbot after analysis")
            except Exception as chatbot_exc:
                logger.error(f"Chatbot initialization failed: {chatbot_exc}")
                # Don't fail the whole analysis if chatbot init fails
        
        if score is None:
            return JSONResponse({"error": "動画分析に失敗しました。ログを確認してください。"}, status_code=500)
            
        return JSONResponse({"score": score, "analysis_complete": True})  # Send back computed score with status
        
    except Exception as exc:
        logger.exception("Error during analysis: %s", exc)
        error_msg = str(exc)
        if "FileNotFoundError" in error_msg:
            return JSONResponse({"error": f"動画ファイルが見つかりません: {error_msg}"}, status_code=400)
        elif "extract_keypoints" in error_msg:
            return JSONResponse({"error": "キーポイント抽出に失敗しました。モデルファイルを確認してください。"}, status_code=500)
        else:
            return JSONResponse({"error": f"分析中にエラーが発生しました: {error_msg}"}, status_code=500)
    finally:
        # Ensure running flag is cleared even if exceptions occur
        global analysis_running
        analysis_running = False


@app.post("/init_chatbot")
async def init_chatbot_route():
    """Initialize the chatbot after videos are processed."""
    try:
        # Prevent initialization during analysis to avoid contention
        global analysis_running
        if analysis_running:
            # Respond with 202 so clients know to retry later without logging
            # a misleading Bad Request.
            return JSONResponse(
                {
                    "status": "pending",
                    "message": "動画分析中のため、チャットボットの初期化は完了後に実行してください。",
                },
                status_code=202,
            )
        success = await init_chatbot()  # Initialize chatbot separately
        if success:
            return JSONResponse({"status": "ok", "message": "チャットボットの準備ができました。"})
        else:
            # Missing prerequisites (e.g. keypoints) shouldn't surface as a
            # 400 error in the logs; indicate initialization is pending.
            return JSONResponse(
                {
                    "status": "pending",
                    "message": "チャットボットの初期化に必要なデータが不足しています。",
                },
                status_code=202,
            )
    except Exception as exc:
        logger.exception("Error initializing chatbot: %s", exc)
        return JSONResponse({"status": "error", "message": "チャットボットの初期化中にエラーが発生しました。"}, status_code=500)


@app.post("/set_videos")
async def set_videos(request: Request):
    """Select videos from local files and clear previous analysis."""
    data = (await request.json()) or {}
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
            
    logger.info(
        f"Videos set to: ref={REF_VIDEO.name}, cur={CUR_VIDEO.name}, device={DEVICE}"
    )
    return JSONResponse({"status": "ok", "device": DEVICE})


@app.get("/chatbot_status")
def chatbot_status():
    """Get current chatbot status."""
    enabled = is_chatbot_enabled()
    analysis_complete = (not analysis_running) and _results_ready()
    status_message = (
        "チャットボットの準備ができました。" if bot is not None
        else "動画分析が完了しました。チャットボットの準備ができました！" if analysis_complete and enabled
        else "チャットボットは準備中です。まず動画を分析してください。" if enabled
        else "チャットボットは無効化されています。"
    )
    return JSONResponse({
        "enabled": enabled,
        "initialized": bot is not None,
        "ready_to_init": analysis_complete,
        "analysis_complete": analysis_complete,
        "status_message": status_message,
    })


if __name__ == "__main__":
    # Ensure required directories exist
    Path("data").mkdir(exist_ok=True)
    Path("static").mkdir(exist_ok=True)
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
