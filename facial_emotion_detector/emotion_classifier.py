"""
Emotion classifier using FER library (pre-trained CNN on FER2013).
Tuned for easier SAD and ANGRY detection.
"""
import cv2
import numpy as np
from enum import Enum
from typing import Dict, List, Tuple


class Emotion(Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISED = "surprised"


class EmotionClassifier:
    """
    Emotion classifier using FER library.
    Tuned thresholds for easier SAD/ANGRY detection.
    """

    EMOTION_EMOJIS = {
        Emotion.NEUTRAL: "😐",
        Emotion.HAPPY: "😊",
        Emotion.SAD: "😢",
        Emotion.ANGRY: "😠",
        Emotion.SURPRISED: "😲"
    }

    # Map FER output to our 5 emotions
    FER_TO_EMOTION = {
        "angry": Emotion.ANGRY,
        "disgust": Emotion.ANGRY,
        "fear": Emotion.SURPRISED,
        "happy": Emotion.HAPPY,
        "sad": Emotion.SAD,
        "surprise": Emotion.SURPRISED,
        "neutral": Emotion.NEUTRAL
    }

    # Decision thresholds on FER softmax scores. Deliberately biased toward SAD and ANGRY,
    # which the FER2013 model under-reports on webcam frames.
    HAPPY_MIN = 0.5
    SURPRISE_MIN = 0.4
    SAD_MIN = 0.25
    ANGRY_MIN = 0.15
    SAD_SECONDARY = 0.2
    ANGRY_SECONDARY = 0.1
    NEUTRAL_MIN = 0.3

    def __init__(self, cascade_path: str = None):
        """Load the FER detector (Haar face detector + FER2013 CNN). Imported lazily so the
        decision logic can be unit-tested without TensorFlow installed."""
        from fer.fer import FER
        self.detector = FER(mtcnn=False)

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces and return bounding boxes."""
        result = self.detector.detect_emotions(frame)
        faces = []
        for face in result:
            box = face["box"]
            faces.append((box[0], box[1], box[2], box[3]))
        return faces

    def classify_from_frame(self, frame: np.ndarray, face_rect: Tuple[int, int, int, int] = None) -> Emotion:
        """
        Classify emotion from frame with tuned thresholds.
        Makes SAD and ANGRY easier to detect.
        """
        result = self.detector.detect_emotions(frame)
        
        if not result:
            return Emotion.NEUTRAL
        
        # Get emotions dict
        if face_rect:
            x, y, w, h = face_rect
            best_match = None
            best_dist = float('inf')
            
            for detection in result:
                box = detection["box"]
                dist = abs(box[0] - x) + abs(box[1] - y)
                if dist < best_dist:
                    best_dist = dist
                    best_match = detection
            
            if best_match:
                emotions = best_match["emotions"]
            else:
                return Emotion.NEUTRAL
        else:
            emotions = result[0]["emotions"]
        
        return self.decide(emotions)

    @classmethod
    def decide(cls, emotions: Dict[str, float]) -> Emotion:
        """Map one FER score dict (angry, disgust, fear, happy, sad, surprise, neutral) to an
        Emotion. Pure function; this is what the unit tests exercise."""
        happy = emotions.get("happy", 0.0)
        surprise = emotions.get("surprise", 0.0)
        neutral = emotions.get("neutral", 0.0)
        # fear shares droopy features with sad, disgust shares tension with angry
        sad = emotions.get("sad", 0.0) + 0.3 * emotions.get("fear", 0.0)
        angry = emotions.get("angry", 0.0) + 0.5 * emotions.get("disgust", 0.0)

        if happy > cls.HAPPY_MIN:
            return Emotion.HAPPY
        if surprise > cls.SURPRISE_MIN:
            return Emotion.SURPRISED
        if sad > cls.SAD_MIN and angry < sad:
            return Emotion.SAD
        if angry > cls.ANGRY_MIN and sad < angry:
            return Emotion.ANGRY
        if sad > cls.SAD_SECONDARY and sad > angry:
            return Emotion.SAD
        if angry > cls.ANGRY_SECONDARY and angry > sad:
            return Emotion.ANGRY
        if neutral > cls.NEUTRAL_MIN:
            return Emotion.NEUTRAL
        fer_emotion = max(emotions, key=emotions.get) if emotions else "neutral"
        return cls.FER_TO_EMOTION.get(fer_emotion, Emotion.NEUTRAL)

    def get_emoji(self, emotion: Emotion) -> str:
        """Get emoji for emotion."""
        return self.EMOTION_EMOJIS.get(emotion, "😐")

    @staticmethod
    def draw_face_box(
        frame: np.ndarray,
        face_rect: Tuple[int, int, int, int],
        emotion: Emotion,
        color: Tuple[int, int, int] = (0, 255, 0)
    ) -> None:
        """Draw bounding box with emotion label."""
        x, y, w, h = face_rect
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        
        label = emotion.value.upper()
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(frame, (x, y - text_h - 15), (x + text_w, y), color, -1)
        cv2.putText(frame, label, (x, y - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)