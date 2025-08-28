# Golf Swing Comparator

This project provides a simple command-line tool to compare two golf swing videos using OpenVINO's implementation of the OpenPose model.

## Requirements
- Python 3.8+
- [OpenVINO Runtime](https://docs.openvino.ai/latest/openvino_docs_install_guides.html)
- OpenPose model converted to OpenVINO IR format (e.g. `human-pose-estimation-0001` from Open Model Zoo)
- The INT8 weights for `human-pose-estimation-0001` placed in
  `intel/human-pose-estimation-0001/INT8/`
- `opencv-python`
- `numpy`
- `transformers`
- `torch`
- `Pillow`
- `sentencepiece`

Install the requirements:

```bash
pip install openvino opencv-python openvino-dev numpy transformers torch Pillow sentencepiece
```

## Usage

```bash
python golf_swing_compare.py --reference REF.mp4 --test TEST.mp4
```

The script defaults to the INT8 version of `human-pose-estimation-0001`. Use
`--model` to override the path if needed.

Use `--chat` to display a chat panel alongside the video comparison. The chatbot
uses the Qwen/Qwen3-8B model, speaks Japanese and offers basic advice:

```bash
python golf_swing_compare.py --reference REF.mp4 --test TEST.mp4 --chat
```

Use `--step` to start playback paused for frame-by-frame analysis. During
playback, press the space bar to pause or resume and use the left/right arrow
keys to step through frames while paused. Playback loops automatically when
the end of the videos is reached; press `q` to exit.

The script outputs a single numerical score representing the average difference between the two swings. Lower scores indicate more similar swings.


## Web Chat and Video Demo

A minimal Flask application is included to display reference and test videos next to a chat window.
The server runs the INT8 OpenPose model (`intel/human-pose-estimation-0001/INT8/human-pose-estimation-0001.xml` by default) and exposes keypoint coordinates as JSON. The
browser draws the skeleton on top of each video using a canvas overlay, and the swing difference score is shown above the players.
Place your videos as `data/reference.mp4` and `data/current.mp4`, ensure the model XML file is available in the project root,
then run:

```bash
pip install -r requirements.txt
python app.py
```

Then open `http://localhost:5000/` in your browser to view the comparison.

The chat panel now uses the Qwen 8B model to provide golf swing tips in Japanese.
Messages are handled on the server and are not persisted; restart the server
to clear the conversation.

For debugging without downloading the large model, set the environment
variable ``CHATBOT_DEBUG=1`` when launching ``app.py``.  In this mode the
server simply echoes back user messages prefixed with ``(デバッグ)`` allowing
you to verify that message handling works correctly.

## Simple Web Chatbot

A lightweight general–purpose chatbot is also available at
`http://localhost:5000/chat`.  It uses a small DialoGPT model and stores the
conversation in memory only for the duration of the session.
