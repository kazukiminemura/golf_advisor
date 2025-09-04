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
  - When using OpenVINO with GGUF, the app can auto-download the model (via `huggingface_hub`) and auto-convert the tokenizer (via `openvino-tokenizers`) when you pass a repo id like `Qwen/Qwen2.5-1.5B-Instruct-GGUF`.

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

A lightweight general-purpose chatbot is available at `http://localhost:8000/chat`. It uses the modular `backend.simple_chatbot` LLM wrapper (default: Qwen2.5 1.5B Instruct via OpenVINO) and stores the conversation only for the current session.

Backend selection and env vars:
- `LLM_BACKEND`: `auto | transformers | llama | openvino` (default: `openvino`)
- `CHAT_MODEL`: HF repo/model id (e.g., `Qwen/Qwen2.5-1.5B-Instruct` or `Qwen/Qwen2.5-1.5B-Instruct-GGUF`)
- `CHAT_GGUF_FILENAME`: when using GGUF backends (e.g., `qwen2.5-1.5b-instruct-q4_k_m.gguf`)
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

- Download LLM model:

```
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct-GGUF qwen2.5-1.5b-instruct-q4_k_m.gguf --local-dir models
```

- Convert tokenizer with detokenizer:

```
convert_tokenizer Qwen/Qwen2.5-1.5B-Instruct --with-detokenizer -o models
```

Or using the integrated helper in `backend.simple_chatbot` (no separate CLI required):

```
python -m backend.simple_chatbot --convert-tokenizer Qwen/Qwen2.5-1.5B-Instruct --with-detokenizer -o models
```

Notes:
- Ensure `models` directory exists (created already in this repo).
- `huggingface-cli` comes from `huggingface_hub` (install via `pip install huggingface_hub`). You may need `huggingface-cli login` for gated files.
- `convert_tokenizer` is provided by the `openvino-tokenizers` project. If the Python package is installed, you can also use the built-in helper in `backend.simple_chatbot` as shown above.

### Download via Scripts (bartowski/Qwen2.5-1.5B-Instruct-GGUF)

Use the provided scripts to download the community GGUF build from `bartowski/Qwen2.5-1.5B-Instruct-GGUF` into `models/`.

- PowerShell (Windows) with `huggingface-cli`:

```
./scripts/download_model.ps1 -RepoId "bartowski/Qwen2.5-1.5B-Instruct-GGUF" -Filename "qwen2.5-1.5b-instruct-q4_k_m.gguf" -LocalDir "models"
```

- Python (cross‑platform) with `huggingface_hub`:

```
python scripts/download_model.py --repo bartowski/Qwen2.5-1.5B-Instruct-GGUF --file Qwen2.5-1.5B-Instruct-Q4_K_M.gguf --out models
```

Or simply use defaults (no arguments):

```
python scripts/download_model.py
```

Prerequisites:
- PowerShell script: `huggingface-cli` available in `PATH` (`pip install -U huggingface-hub`).
- Python script: `huggingface_hub` Python package (`pip install -U huggingface-hub`).

Tips:
- You can swap `qwen2.5-1.5b-instruct-q4_k_m.gguf` for another quant (e.g., `q4_k_s`, `q5_k_m`, `q8_0`).
- The default app config expects `models/qwen2.5-1.5b-instruct-q4_k_m.gguf`. Update `CHAT_GGUF_FILENAME` if you choose a different file.
