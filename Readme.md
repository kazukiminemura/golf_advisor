# Golf Swing Comparator

This project compares two golf swing videos with OpenVINO and includes a web interface with a chatbot enabled by default.

## Requirements

- Python 3.8+
- OpenVINO Runtime
- OpenPose model converted to OpenVINO IR format (for example, `human-pose-estimation-0001` from Open Model Zoo)
- INT8 weights placed in `intel/human-pose-estimation-0001/INT8/`
- `opencv-python`
- `numpy`
- `transformers`
- `torch`
- `Pillow`
- `sentencepiece`

## Installation

Install the required packages:

```bash
pip install openvino opencv-python openvino-dev numpy transformers torch Pillow sentencepiece
```

## Web Chat and Video Demo

A minimal FastAPI application displays reference and test videos next to a chat window and shows the swing difference score.

1. Place your videos as `data/reference.mp4` and `data/current.mp4`.
2. Ensure the model file `intel/human-pose-estimation-0001/INT8/human-pose-estimation-0001.xml` is in the project root.
3. Install the dependencies and start the server:
   ```bash
   pip install -r requirements.txt
   uvicorn app:app --reload
   ```
   4. Open `http://localhost:8000/` in your browser.


## Simple Web Chatbot

A lightweight general-purpose chatbot is available at `http://localhost:8000/chat`. It uses a small DialoGPT model and stores the conversation only for the current session.

## Backend Architecture (SOLID Refactor)

- Services:
  - Analysis: `backend/services/analysis_service.py` handles keypoint extraction, scoring, and artifact generation.
  - Chatbot: `backend/services/chatbot_service.py` manages both swing-specific and general chatbots and their histories.
  - System: `backend/services/system_service.py` exposes CPU/GPU/NPU and memory metrics.
- Config: `backend/config.py` centralizes environment flags and default paths.
- Utils: `backend/utils/files.py` provides safe filename handling.

FastAPI routes in `app.py` delegate to these services, reducing global state and coupling. This improves testability and aligns with the Single Responsibility and Dependency Inversion principles while preserving existing HTTP API behavior.

