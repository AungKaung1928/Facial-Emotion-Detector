# Facial Emotion Detection System Project

## Overview

Real-time facial emotion detection with ROS 2 Humble, OpenCV and the `fer` library (a
pre-trained FER2013 CNN behind a Haar face detector). Maps the seven FER scores to five classes
(happy, sad, angry, surprised, neutral), smooths over 5 frames and publishes `/facial_emotion`.

**Accuracy, honestly:** FER2013 is a hard 48x48 grayscale dataset; the best published models
reach about 73 % on it and the `fer` package model sits around 65 %. On a webcam the useful
classes are HAPPY, SURPRISED and NEUTRAL; SAD and ANGRY are under-reported, so the decision
thresholds in `EmotionClassifier.decide()` are deliberately biased toward them (see the
constants at the top of the class). Expect confident smiles and open-mouth surprise to be right,
and sad/angry to flip with lighting and head pose. It is a demo of the ROS 2 pipeline, not a
calibrated classifier.

## Features

- **FER CNN-based classification**: pre-trained FER2013 model via the `fer` package
- **Real-time detection**: 15 Hz video processing with smooth emotion tracking
- **5 emotion classes**: Happy 😊, Sad 😢, Angry 😠, Surprised 😲, Neutral 😐
- **Live video display**: Webcam feed with face bounding boxes and emotion labels
- **Emoji overlays**: Large emoji indicator in video feed
- **ROS2 integration**: Publishes detected emotions to `/facial_emotion` topic
- **Unit-tested decision logic**: `decide()` is a pure function on the score dict; tests run without TensorFlow or a camera
- **Emotion smoothing**: 5-frame history-based filtering for stable detection

## System Requirements

- Ubuntu 22.04
- ROS2 Humble
- Python 3.10+
- Webcam (USB or built-in)

## Dependencies

### System Packages

```bash
sudo apt update
sudo apt install -y \
    python3-pip \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    python3-opencv
```

### Python Packages

`fer` pulls in TensorFlow (about 600 MB). Keep it out of the system Python: use a venv that can
still see the apt-installed `rclpy`. `--without-pip` sidesteps the missing `python3-venv`
package on stock Ubuntu (the venv reuses the system pip), and `--no-deps` on `fer` skips its
`facenet-pytorch` dependency, which would otherwise download a multi-GB CUDA torch build for the
MTCNN option this project never uses.

```bash
python3 -m venv --without-pip --system-site-packages ~/.venv-fer
~/.venv-fer/bin/python3 -m pip install --no-deps fer
~/.venv-fer/bin/python3 -m pip install tensorflow-cpu requests pillow
~/.venv-fer/bin/python3 -c "from fer.fer import FER; import numpy as np; print(FER(mtcnn=False).detect_emotions(np.zeros((480,640,3),np.uint8)))"
```
The last line must print `[]` (model loaded, no face in a black frame). Verified with fer 25.10.3
and tensorflow-cpu 2.21 on Python 3.10.

## Installation

### 1. Create Workspace

```bash
mkdir -p ~/facial_emotion_ws/src
cd ~/facial_emotion_ws/src
```

### 2. Clone/Create Package

```bash
# If using git
git clone <repository_url> facial_emotion_detector

# Or create manually and copy files
```

### 3. Build Package

```bash
cd ~/facial_emotion_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select facial_emotion_detector --symlink-install
```

### 4. Source Workspace

```bash
source install/setup.bash
```

## Usage

`ros2 run` uses the system interpreter, so start the node with the venv's Python instead:

```bash
source ~/facial_emotion_ws/install/setup.bash
~/.venv-fer/bin/python3 -m facial_emotion_detector.emotion_detector_node --ros-args -p camera_id:=0
```
The launch file works when `fer` is installed system-wide:
```bash
ros2 launch facial_emotion_detector emotion_detection.launch.py
```
A window shows the face box, emotion label, emoji and FPS. Press **Q** to quit.

Run only one camera client at a time: `emotion_display` is an alternative viewer that opens the
same device and will fail with "Camera not available" if the detector already holds it.

### How to test
1. **Logic, no hardware** (runs in CI):
   ```bash
   python3 -m pytest test/test_emotion_classifier.py
   ```
2. **Model + camera, one frame** (prints the raw FER scores so you can judge the thresholds):
   ```bash
   ~/.venv-fer/bin/python3 - <<'PY'
   import cv2
   from fer.fer import FER
   cap = cv2.VideoCapture(0); ok, frame = cap.read(); cap.release()
   assert ok, "camera 0 gave no frame"
   for face in FER(mtcnn=False).detect_emotions(frame):
       print(face["box"], {k: round(v, 2) for k, v in face["emotions"].items()})
   PY
   ```
   No output = no face found (light your face, look at the camera).
3. **ROS 2 pipeline**: start the node, then in another terminal
   ```bash
   ros2 topic echo /facial_emotion
   ```
   and act out the guide below. Pass = HAPPY and SURPRISED switch within a second; NEUTRAL at
   rest. SAD/ANGRY are best-effort.

## Emotion Detection Guide

### How to Express Each Emotion

| Emotion | Instructions |
|---------|--------------|
| 😊 **HAPPY** | Smile wide with visible teeth. Facial muscles relaxed and elevated. |
| 😢 **SAD** | Look DOWN at floor. Let face droop. Pout lower lip OUT. Slightly close eyes. |
| 😠 **ANGRY** | Stare FORWARD intensely. Squeeze eyebrows DOWN and TOGETHER. Clench jaw. |
| 😲 **SURPRISED** | Open mouth VERY WIDE (O-shape). Raise eyebrows UP HIGH. Open eyes wide. |
| 😐 **NEUTRAL** | Completely relax all facial muscles. Natural resting face. No tension. |

**Key tip:** For SAD look DOWN. For ANGRY stare FORWARD. This separates them easily.

## System Architecture

### Node Structure

```
emotion_detector_node
├── Camera Capture (15 Hz)
├── FER CNN Model
│   └── Pre-trained on FER2013 dataset
├── Emotion Classification
│   └── CNN softmax scores with tuned thresholds
├── Emotion Smoothing (5-frame history)
├── ROS2 Publisher (/facial_emotion)
└── Video Display with Overlays
```

### Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/facial_emotion` | `std_msgs/String` | Published emotion name |

- **Message values**: `"happy"`, `"sad"`, `"angry"`, `"surprised"`, `"neutral"`
- **Publish rate**: Variable (when face detected)

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `camera_id` | int | 0 | Camera device index |

### Package Structure

```
facial_emotion_ws/
├── src/
│   └── facial_emotion_detector/
│       ├── facial_emotion_detector/
│       │   ├── __init__.py                    # Package initialization
│       │   ├── emotion_classifier.py          # FER-based CNN emotion classifier
│       │   ├── emotion_detector_node.py       # ROS2 node with video display
│       │   └── emotion_display_node.py        # Alternative display node
│       ├── launch/
│       │   └── emotion_detection.launch.py    # Launch file for multiple nodes
│       ├── resource/
│       │   └── facial_emotion_detector        # Package resource marker
│       ├── test/
│       │   ├── test_copyright.py              # Copyright validation
│       │   ├── test_flake8.py                 # Code style checks
│       │   ├── test_pep257.py                 # Docstring validation
│       │   └── test_emotion_classifier.py     # Unit tests for emotion logic
│       ├── package.xml                        # ROS2 package dependencies
│       ├── setup.py                           # Python package setup
│       ├── setup.cfg                          # Package configuration
│       └── README.md                          # Documentation
├── build/                                     # Build artifacts
├── install/                                   # Installed files
└── log/                                       # Build and runtime logs
```

## Technical Details

### Emotion Classification Algorithm

The system uses FER library with pre-trained CNN:

1. **Face Detection**: OpenCV cascade classifier

2. **Emotion Classification**: 
   - FER library with TensorFlow Lite CNN model
   - Trained on FER2013 dataset (35,000+ labeled images)
   - Outputs softmax scores for 7 emotions

3. **Emotion Mapping** (FER → 5 classes):
   - `angry`, `disgust` → ANGRY
   - `fear`, `surprise` → SURPRISED
   - `happy` → HAPPY
   - `sad` → SAD
   - `neutral` → NEUTRAL

4. **Tuned Thresholds**:
   - SAD: `sad_score + fear_score * 0.3 > 0.25`
   - ANGRY: `angry_score + disgust_score * 0.5 > 0.15`
   - Separation: SAD wins if sad > angry, ANGRY wins if angry > sad

5. **Smoothing**: 5-frame majority voting for stable output

### Performance

| Metric | Value |
|--------|-------|
| Detection rate | 15 FPS (video display) |
| Latency | ~100ms end-to-end |
| Model accuracy | ~63% on FER2013 test set |

## Testing

### Run Unit Tests

```bash
cd ~/facial_emotion_ws
source install/setup.bash

# Run all tests
colcon test --packages-select facial_emotion_detector

# Run specific test
python3 src/facial_emotion_detector/test/test_emotion_classifier.py
```

### Test Coverage

- Emotion classification logic
- Threshold behavior
- Edge cases and defaults

## Troubleshooting

### Camera Issues

**Problem**: "Failed to open camera"

```bash
# Check available cameras
ls /dev/video*

# Test camera
ffplay /dev/video0

# Change camera_id parameter
ros2 run facial_emotion_detector emotion_detector --ros-args -p camera_id:=1
```

### Poor Detection

**Problem**: Emotions not changing or incorrect

- Ensure good lighting (front-lit face)
- Face camera directly
- Make exaggerated expressions initially
- Adjust distance (60-120cm optimal)
- For SAD: look DOWN
- For ANGRY: stare FORWARD intensely

### TensorFlow Warnings

**Problem**: TensorFlow deprecation warnings

These are harmless warnings from TensorFlow Lite. The system works correctly.

### Import Errors

**Problem**: "cannot import name 'FER' from 'fer'"

```bash
# Use correct import path
# In emotion_classifier.py, change:
from fer import FER
# To:
from fer.fer import FER
```

## References

- [FER Library](https://github.com/justinshenk/fer)
- [FER2013 Dataset](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge)
- [ROS2 Humble Documentation](https://docs.ros.org/en/humble/)
