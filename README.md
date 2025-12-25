# Facial Emotion Detection System

## Code Structure

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

## Overview

Real-time facial emotion detection system using ROS2 Humble, OpenCV, TensorFlow, and Python. Detects 5 emotions (happy, sad, angry, surprised, neutral) from webcam feed using FER library with pre-trained CNN model (trained on FER2013 dataset).

## Features

- **FER CNN-based classification**: Pre-trained deep learning model for accurate emotion detection
- **Real-time detection**: 15 Hz video processing with smooth emotion tracking
- **5 emotion classes**: Happy 😊, Sad 😢, Angry 😠, Surprised 😲, Neutral 😐
- **Live video display**: Webcam feed with face bounding boxes and emotion labels
- **Emoji overlays**: Large emoji indicator in video feed
- **ROS2 integration**: Publishes detected emotions to `/facial_emotion` topic
- **Production-grade architecture**: Type hints, clean code structure
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

```bash
pip3 install fer --break-system-packages
```

This installs FER library with TensorFlow and other dependencies.

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

### Basic Usage (Recommended)

```bash
# Run detector with live video display
ros2 run facial_emotion_detector emotion_detector
```

This opens a webcam window showing:

- Your face with green bounding box
- Current emotion label on face box
- Large emoji in top-right corner
- Emotion text with emoji at bottom
- FPS counter in top-left

Press **'Q'** to quit.

### Monitor Emotion Topic

In a separate terminal:

```bash
source ~/facial_emotion_ws/install/setup.bash
ros2 topic echo /facial_emotion
```

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
