# Golf Swing Comparator

This project provides a simple command-line tool to compare two golf swing videos using OpenVINO's implementation of the OpenPose model.

## Requirements
- Python 3.8+
- [OpenVINO Runtime](https://docs.openvino.ai/latest/openvino_docs_install_guides.html)
- OpenPose model converted to OpenVINO IR format (e.g. `human-pose-estimation-0001` from Open Model Zoo)
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
python golf_swing_compare.py --reference REF.mp4 --test TEST.mp4 --model human-pose-estimation-0001.xml
```

Use `--chat` to display a chat panel alongside the video comparison. The chatbot speaks Japanese and offers basic advice:

```bash
python golf_swing_compare.py --reference REF.mp4 --test TEST.mp4 --model human-pose-estimation-0001.xml --chat
```

Use `--step` to start playback paused for frame-by-frame analysis. During
playback, press the space bar to pause or resume and use the left/right arrow
keys to step through frames while paused. Playback loops automatically when
the end of the videos is reached; press `q` to exit.

The script outputs a single numerical score representing the average difference between the two swings. Lower scores indicate more similar swings.


## Web Chat and Video Demo

A minimal Flask application is included to display reference and test videos next to a chat window.
The server runs the OpenPose model (`human-pose-estimation-0001.xml` by default) and exposes keypoint coordinates as JSON. The
browser draws the skeleton on top of each video using a canvas overlay, and the swing difference score is shown above the players.
Place your videos as `data/reference.mp4` and `data/current.mp4`, ensure the model XML file is available in the project root,
then run:

```bash
pip install -r requirements.txt
python app.py
```

Then open `http://localhost:5000/` in your browser to view the comparison.
