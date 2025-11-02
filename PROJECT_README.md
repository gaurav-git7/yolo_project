# YOLOv5 People Detection Project

This project implements real-time people detection using a custom-trained YOLOv5n model.

## Features

- Real-time people detection from webcam or video files
- Customizable confidence thresholds
- Bounding box visualization with confidence scores
- Live people count display
- Pause/resume functionality
- Fullscreen display mode

## Main Files

- `filtered.py` - Real-time detection from webcam (camera index 1)
- `sample_video.py` - Detection from video file
- `my_hyp.yaml` - Custom hyperparameters for training
- `train.py` - Modified training script

## Requirements

```bash
pip install -r requirements.txt
```

Main dependencies:
- PyTorch
- OpenCV (cv2)
- YOLOv5

## Usage

### Real-time Detection (Webcam)
```bash
python filtered.py
```

### Video File Detection
```bash
python sample_video.py
```

### Controls
- `q` - Quit the application
- `p` - Pause/Resume detection

## Model

The project uses a custom-trained YOLOv5n model optimized for people detection.
Model weights should be placed at: `yolov5/runs/train/yolo-batch4/weights/best.pt`

## Training

To train your own model, use the modified `train.py` script with custom hyperparameters defined in `my_hyp.yaml`.

## Configuration

Key parameters in the detection scripts:
- `model.conf = 0.25` - Model confidence threshold
- `conf > 0.4` - Detection filtering threshold
- `area > 1000` - Minimum bounding box area filter

## Note

Model weights and training runs are not included in this repository due to file size.
Dataset files and videos are also excluded.
