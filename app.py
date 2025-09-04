import os
import asyncio
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, UploadFile, File, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse, StreamingResponse
import base64
import json
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from backend.config import Settings
from backend.services.analysis_service import AnalysisService
from backend.services.chatbot_service import ChatbotService
from backend.services.system_service import SystemService
from backend.utils.files import safe_filename
from backend.simple_chatbot import preload_model
from backend.simple_chatbot.config import ChatbotConfig

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

# Services and configuration
settings = Settings()
analysis = AnalysisService(
    model_xml=settings.MODEL_XML,
    device=settings.DEVICE,
    data_dir=settings.DATA_DIR,
    static_dir=settings.STATIC_DIR,
)
chat = ChatbotService(enabled=settings.ENABLE_CHATBOT, lazy_init=settings.LAZY_CHATBOT_INIT)


def is_chatbot_enabled() -> bool:
    return chat.is_enabled()


MAX_MESSAGES = 20


@app.on_event("startup")
async def warm_models_background():
    """Kick off background warm-up for OpenPose and LLM models.

    This runs asynchronously so the server becomes available immediately while
    models load in the background.
    """
    # Warm OpenPose (OpenVINO) model
    async def _warm_openpose():
        try:
            from backend.inference import preload_openpose_model
            await asyncio.to_thread(preload_openpose_model, analysis.model_xml, analysis.device)
            logger.info("OpenPose model preloaded in background")
        except Exception as exc:
            logger.warning("OpenPose preload failed: %s", exc)

    # Warm LLM backend used by SimpleChatBot/EnhancedSwingChatBot
    async def _warm_llm():
        try:
            cfg = ChatbotConfig()
            await asyncio.to_thread(
                preload_model,
                cfg.model_name,
                gguf_filename=cfg.gguf_filename,
                backend=cfg.backend,
            )
            logger.info("LLM model preloaded in background")
        except Exception as exc:
            logger.warning("LLM preload failed: %s", exc)

    # Start both tasks without blocking startup
    tasks = [asyncio.create_task(_warm_openpose()), asyncio.create_task(_warm_llm())]
    # Keep references so they aren't GC'd; purely informational, not awaited.
    if not hasattr(app.state, "warmup_tasks"):
        app.state.warmup_tasks = []
    app.state.warmup_tasks.extend(tasks)


def get_gpu_usage() -> float:
    return SystemService.gpu_percent()


def get_npu_usage() -> float:
    return SystemService.npu_percent()


## analysis helpers moved to AnalysisService


def _init_chatbot_sync() -> bool:
    """Initialize chatbot using cached keypoints if enabled.
    
    Returns:
        bool: True if chatbot was successfully initialized, False otherwise.
    """
    return chat.try_init(analysis.ref_keypoints, analysis.cur_keypoints, analysis.score, analysis.analysis_running)


async def init_chatbot() -> bool:
    """Asynchronously initialize the chatbot.
    
    Returns:
        bool: True if chatbot was successfully initialized, False otherwise.
    """
    return await asyncio.to_thread(_init_chatbot_sync)


## analysis pipeline is implemented in backend.services.analysis_service.AnalysisService


async def prepare_videos() -> None:
    """Asynchronously generate annotated videos and compute the swing score."""
    await asyncio.to_thread(analysis.prepare)


def _results_ready() -> bool:
    """Return True when analysis artifacts are ready for UI/chat."""
    return analysis.results_ready()


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    has_results = _results_ready()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "score": analysis.score,
            "ref_video_name": analysis.ref_video.name,
            "cur_video_name": analysis.cur_video.name,
            "ref_kp_name": analysis.ref_kp_json.name,
            "cur_kp_name": analysis.cur_kp_json.name,
            "has_results": has_results,
            "chatbot_enabled": is_chatbot_enabled(),
            "device": analysis.device,
        },
    )

@app.api_route("/chat_messages", methods=["GET", "POST"])
async def chat_messages_handler(request: Request):
    """Handle messages for the general-purpose chatbot."""
    if request.method == "POST":
        data = (await request.json()) or {}
        user_msg = data.get("message", "").strip()
        if not user_msg:
            return JSONResponse({"reply": ""})
        if "text/event-stream" in request.headers.get("accept", ""):
            def stream():
                for ch in chat.general_ask_stream(user_msg, max_messages=MAX_MESSAGES):
                    yield ch

            async def event_stream():
                for ch in stream():
                    yield f"data: {ch}\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")
        reply = "".join(chat.general_ask_stream(user_msg, max_messages=MAX_MESSAGES))
        return JSONResponse({"reply": reply})
    else:
        return JSONResponse(chat.general_messages())

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
        if not chat.is_initialized():
            success = _init_chatbot_sync()
            if not success:
                return JSONResponse({"reply": "チャットボットは準備中です。まず動画を分析してください。"})

        try:
            if "text/event-stream" in request.headers.get("accept", ""):
                def stream():
                    yield from chat.swing_ask_stream(user_msg, max_messages=MAX_MESSAGES)

                async def event_stream():
                    for ch in stream():
                        yield f"data: {ch}\n\n"

                return StreamingResponse(event_stream(), media_type="text/event-stream")
            reply = "".join(chat.swing_ask_stream(user_msg, max_messages=MAX_MESSAGES))
            return JSONResponse({"reply": reply})
            
        except Exception as exc:
            logger.exception("Error in chatbot conversation: %s", exc)
            return JSONResponse({"reply": "申し訳ございませんが、エラーが発生しました。もう一度お試しください。"})
    else:
        # GET request - return conversation history or initialization message
        if chat.is_initialized():
            return JSONResponse(chat.swing_messages())
        else:
            # Check if video analysis is complete but chatbot not initialized
            analysis_complete = (not analysis.analysis_running) and _results_ready()
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


@app.websocket("/ws/messages")
async def websocket_message_handler(websocket: WebSocket):
    await websocket.accept()
    try:
        data = await websocket.receive_text()
        try:
            payload = json.loads(data)
            user_msg = payload.get("message", "").strip()
        except Exception:
            user_msg = data.strip()
        if not user_msg:
            await websocket.close()
            return
        if not is_chatbot_enabled():
            await websocket.send_text("チャットボットは無効化されています。")
            await websocket.send_text("[DONE]")
            return
        if not chat.is_initialized():
            success = _init_chatbot_sync()
            if not success:
                await websocket.send_text(
                    "チャットボットは準備中です。まず動画を分析してください。"
                )
                await websocket.send_text("[DONE]")
                return
        for ch in chat.swing_ask_stream(user_msg, max_messages=MAX_MESSAGES):
            await websocket.send_text(ch)
        await websocket.send_text("[DONE]")
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.exception("Error in websocket conversation: %s", exc)
        await websocket.send_text(
            "申し訳ございませんが、エラーが発生しました。もう一度お試しください。"
        )
        await websocket.send_text("[DONE]")
    finally:
        await websocket.close()


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
        name = safe_filename(reference.filename or "reference.mp4")
        dest = data_dir / name
        with dest.open("wb") as f:
            f.write(await reference.read())
        saved["reference_file"] = name
    if current is not None:
        name = safe_filename(current.filename or "current.mp4")
        dest = data_dir / name
        with dest.open("wb") as f:
            f.write(await current.read())
        saved["current_file"] = name
    return JSONResponse(saved)


@app.get("/system_usage")
def system_usage():
    """Return current CPU, GPU and NPU utilization percentages."""
    cpu = SystemService.cpu_percent()
    gpu = SystemService.gpu_percent()
    npu = SystemService.npu_percent()
    mem = SystemService.memory_stats()
    return JSONResponse({"cpu": cpu, "gpu": gpu, "npu": npu, **mem})


@app.post("/analyze")
async def analyze():
    """Run pose analysis and return the score."""
    try:
        logger.info(f"Starting analysis with ENABLE_CHATBOT={is_chatbot_enabled()}")

        # Clear any existing chatbot to free memory
        if chat.is_initialized():
            logger.info("Clearing existing chatbot to free memory")
            chat.clear_swing()

        await prepare_videos()  # Ensure videos are processed

        # After video processing, try to initialize chatbot automatically if allowed
        if is_chatbot_enabled() and analysis.score is not None and not chat.lazy_init:
            logger.info("Attempting to initialize chatbot after successful analysis")
            try:
                success = await init_chatbot()
                if success:
                    logger.info("Chatbot automatically initialized after analysis")
                else:
                    logger.warning("Failed to automatically initialize chatbot after analysis")
            except Exception as chatbot_exc:
                logger.error(f"Chatbot initialization failed: {chatbot_exc}")

        if analysis.score is None:
            return JSONResponse({"error": "動画分析に失敗しました。ログを確認してください。"}, status_code=500)
            
        return JSONResponse({"score": analysis.score, "analysis_complete": True})
        
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
        analysis.analysis_running = False


@app.post("/init_chatbot")
async def init_chatbot_route():
    """Initialize the chatbot after videos are processed."""
    try:
        # Prevent initialization during analysis to avoid contention
        if analysis.analysis_running:
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

    analysis.set_videos(ref_file, cur_file, device)
    chat.clear_swing()

    logger.info(
        f"Videos set to: ref={analysis.ref_video.name}, cur={analysis.cur_video.name}, device={analysis.device}"
    )
    return JSONResponse({"status": "ok", "device": analysis.device})


@app.get("/chatbot_status")
def chatbot_status():
    """Get current chatbot status."""
    enabled = is_chatbot_enabled()
    analysis_complete = (not analysis.analysis_running) and _results_ready()
    status_message = (
        "チャットボットの準備ができました。" if chat.is_initialized()
        else "動画分析が完了しました。チャットボットの準備ができました！" if analysis_complete and enabled
        else "チャットボットは準備中です。まず動画を分析してください。" if enabled
        else "チャットボットは無効化されています。"
    )
    return JSONResponse({
        "enabled": enabled,
        "initialized": chat.is_initialized(),
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
