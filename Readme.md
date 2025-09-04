# Golf Swing Comparator

This project compares two golf swing videos with OpenVINO and includes a web interface with a chatbot enabled by default.

## Requirements

- Python 3.8+
- OpenVINO Runtime (for pose extraction)
- Dependencies in `requirements.txt` (install via pip, see below)
- OpenPose model in OpenVINO IR format (e.g. `human-pose-estimation-0001` from Open Model Zoo)

Notes on models and backends:
- Pose model default path is `intel/human-pose-estimation-0001/FP16/human-pose-estimation-0001.xml` (see `backend/config.py:Settings.MODEL_XML`). Adjust if your IR is elsewhere or a different precision.
- General chatbot backends supported: `transformers` (HF), `llama.cpp` (GGUF), and `OpenVINO GenAI`. Select via `LLM_BACKEND` env var: `auto | transformers | llama | openvino` (default: `openvino`).
 - When using OpenVINO with GGUF, the app can auto-download the model (via `huggingface_hub`) and auto-convert the tokenizer (via `openvino-tokenizers`) when you pass a repo id like `Qwen/Qwen2.5-3B-Instruct-GGUF`.

## Installation

Install all Python dependencies:

```bash
pip install -r requirements.txt
```

Optional extras depending on your chosen LLM backend:
- Transformers backend: already covered by `transformers` and `torch` in `requirements.txt`.
- llama.cpp backend: `llama-cpp-python` (included). On some platforms you may need system compilers if wheels are unavailable.
- OpenVINO GenAI backend: `openvino-genai` (included) and optionally `openvino-tokenizers` if using GGUF + tokenizer IR.

## Web Chat and Video Demo

A minimal FastAPI application displays reference and test videos next to a chat window and shows the swing difference score.

1. Place your videos as `data/reference.mp4` and `data/current.mp4`.
2. Ensure the OpenVINO pose model exists. Default path: `intel/human-pose-estimation-0001/FP16/human-pose-estimation-0001.xml` (update `backend/config.py` if you use a different path).
3. Install the dependencies and start the server:
   ```bash
   pip install -r requirements.txt
   uvicorn app:app --reload
   ```
   4. Open `http://localhost:8000/` in your browser.


## Simple Web Chatbot

A lightweight general-purpose chatbot is available at `http://localhost:8000/chat`. It uses a small DialoGPT model and stores the conversation only for the current session.

Backend selection and env vars:
- `LLM_BACKEND`: `auto | transformers | llama | openvino` (default: `openvino`)
- `CHAT_MODEL`: HF repo/model id (e.g., `Qwen/Qwen2.5-3B-Instruct` or `Qwen/Qwen2.5-3B-Instruct-GGUF`)
- `CHAT_GGUF_FILENAME`: when using GGUF backends (e.g., `qwen2.5-3b-instruct-q4_k_m.gguf`)
- `TOKENIZER_ID`: optional override for tokenizer source when auto-converting (default: drop `-GGUF` from `CHAT_MODEL`).
- `DEVICE`: OpenVINO device for pose model (`CPU`, `GPU`, etc.; default `CPU`)

## Backend Architecture (SOLID Refactor)

- Services:
  - Analysis: `backend/services/analysis_service.py` handles keypoint extraction, scoring, and artifact generation.
  - Chatbot: `backend/services/chatbot_service.py` manages both swing-specific and general chatbots and their histories.
  - System: `backend/services/system_service.py` exposes CPU/GPU/NPU and memory metrics.
- Config: `backend/config.py` centralizes environment flags and default paths.
- Utils: `backend/utils/files.py` provides safe filename handling.

FastAPI routes in `app.py` delegate to these services, reducing global state and coupling. This improves testability and aligns with the Single Responsibility and Dependency Inversion principles while preserving existing HTTP API behavior.

## Model Assets

Use these commands to prepare local model files in the `models` directory.

- Convert tokenizer with detokenizer:

```
convert_tokenizer Qwen/Qwen2.5-3B-Instruct --with-detokenizer -o models
```

- Download a quantized GGUF model:

```
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct-GGUF qwen2.5-1.5b-instruct-q4_k_m.gguf --local-dir models
```

Notes:
- Ensure `models` directory exists (created already in this repo).
- `huggingface-cli` comes from `huggingface_hub` (install via `pip install huggingface_hub`). You may need `huggingface-cli login` for gated files.
- `convert_tokenizer` is provided by llama.cpp tooling; install per your environment if not present.
