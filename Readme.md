# Golf Swing Comparator

This project provides a simple command-line tool to compare two golf swing videos using OpenVINO's implementation of the OpenPose model.

## Requirements
- Python 3.8+
- [OpenVINO Runtime](https://docs.openvino.ai/latest/openvino_docs_install_guides.html)
- OpenPose model converted to OpenVINO IR format (e.g. `human-pose-estimation-0001` from Open Model Zoo)
- `opencv-python`
- `numpy`

Install the requirements:

```bash
pip install openvino opencv-python numpy
```

## Usage

```bash
python golf_swing_compare.py --reference REF.mp4 --test TEST.mp4 --model human-pose-estimation-0001.xml
```

The script outputs a single numerical score representing the average difference between the two swings. Lower scores indicate more similar swings.
