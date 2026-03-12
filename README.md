# flir-adas-yolov8
YOLOv8x object detection and speed estimation pipeline for FLIR ADAS dataset. RPA automation for self-driving car research.
# FLIR ADAS YOLOv8 Object Detection Pipeline

An automated RPA (Robotic Process Automation) pipeline for self-driving car
object detection, multi-object tracking, and speed estimation using the
FLIR ADAS dataset.

## Project Overview

This project fine-tunes YOLOv8x on the FLIR ADAS RGB dashcam dataset
(Santa Barbara, CA) to detect road objects and estimate their speeds.

## Results

| Metric     | Value  |
|------------|--------|
| mAP50      | 0.368  |
| mAP50-95   | 0.206  |
| Precision  | 0.517  |
| Recall     | 0.350  |

### Per-class AP50
| Class         | AP50  |
|---------------|-------|
| car           | 0.701 |
| person        | 0.634 |
| motor         | 0.583 |
| sign          | 0.433 |
| hydrant       | 0.425 |
| light         | 0.294 |
| bike          | 0.234 |
| other vehicle | 0.221 |
| truck         | 0.158 |

## Dataset
- 10,319 training images
- 3,749 test images
- 15 classes: person, bike, car, motor, bus, train, truck, light,
  hydrant, sign, dog, skateboard, stroller, scooter, other vehicle
- Source: FLIR ADAS DualCapture RGB dashcam

## Model
- Architecture: YOLOv8x (68M parameters)
- Training: 50 epochs, batch=2, imgsz=640
- GPU: NVIDIA GeForce RTX 3050 6GB
- Best weights saved at epoch 21

## Tracking & Speed Estimation
- Tracker: ByteTrack
- 6,747 unique objects tracked
- Speed calibration: real car width (1.8m) / median pixel width (22px)
- Average speed: 10.1 km/h (urban streets)

## Pipeline Usage
```bash
# Full pipeline
python master_pipeline.py

# Skip training (use existing model)
python master_pipeline.py --skip-train

# Skip training and tracking
python master_pipeline.py --skip-train --skip-track

# Regenerate report only
python master_pipeline.py --report-only
```

## Tech Stack
- Python 3.12
- YOLOv8 (Ultralytics 8.2.27)
- PyTorch 2.5.1 + CUDA 12.1
- ByteTrack
