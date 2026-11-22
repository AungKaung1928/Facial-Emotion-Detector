# Usage details

Long-form notes moved out of the README. Paths assume the workspace from the README quick start (`~/facial_emotion_ws`).

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
| `image_topic` | string | `''` | Non-empty: subscribe to this `sensor_msgs/Image` topic (bgr8) instead of opening `camera_id` |
| `show_window` | bool | true | Show the OpenCV window; false for headless runs |

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
   - pre-trained CNN from the FER library
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

