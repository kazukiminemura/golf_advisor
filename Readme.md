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

## Command-Line Usage

Compare two videos:

```bash
python golf_swing_compare.py --reference REF.mp4 --test TEST.mp4
```

The script defaults to the INT8 version of `human-pose-estimation-0001`. Use `--model` to override the path if necessary.

### Enable the Chatbot in the Command-Line Tool

1. Run the comparison with `--chat` to open a chat panel:
   ```bash
   python golf_swing_compare.py --reference REF.mp4 --test TEST.mp4 --chat
   ```
2. The chatbot uses the Qwen/Qwen3-8B model, speaks Japanese and offers basic advice.
3. To test without downloading the large model, enable debug mode:
   ```bash
   CHATBOT_DEBUG=1 python golf_swing_compare.py --reference REF.mp4 --test TEST.mp4 --chat
   ```
   Debug mode echoes messages prefixed with `(デバッグ)`.

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

### Chatbot in the Web App

The chat panel is enabled by default and uses the Qwen 8B model to provide golf swing tips in Japanese. Configure it with environment variables when starting `app.py`:

- `ENABLE_CHATBOT=0` disables the swing coach chatbot and hides the chat panel.
- `LAZY_CHATBOT_INIT=0` initializes the chatbot immediately after video analysis; the default (`1`) waits until the first chat request to load the model.
- `CHATBOT_DEBUG=1` returns echo responses for debugging.

## Simple Web Chatbot

A lightweight general-purpose chatbot is available at `http://localhost:8000/chat`. It uses a small DialoGPT model and stores the conversation only for the current session.

