"""
Emotion classifier using FER library (pre-trained CNN on FER2013).
Tuned for easier SAD and ANGRY detection.
"""
import cv2
import numpy as np
from enum import Enum
from typing import Tuple, Optional, List
from fer.fer import FER


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

    def __init__(self, cascade_path: str = None):
        """Initialize FER detector."""
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
        
        # Get scores
        angry_score = emotions.get("angry", 0)
        sad_score = emotions.get("sad", 0)
        happy_score = emotions.get("happy", 0)
        surprise_score = emotions.get("surprise", 0)
        neutral_score = emotions.get("neutral", 0)
        fear_score = emotions.get("fear", 0)
        disgust_score = emotions.get("disgust", 0)
        
        # Boost SAD detection - combine with fear (both have droopy features)
        sad_combined = sad_score + fear_score * 0.3
        
        # Boost ANGRY detection - combine with disgust
        angry_combined = angry_score + disgust_score * 0.5
        
        # Priority detection with lower thresholds
        
        # HAPPY - needs to be clearly dominant
        if happy_score > 0.5:
            return Emotion.HAPPY
        
        # SURPRISED - wide open features
        if surprise_score > 0.4:
            return Emotion.SURPRISED
        
        # SAD - lower threshold, easier to trigger
        # Triggers if sad is noticeable AND not clearly angry
        if sad_combined > 0.25 and angry_combined < sad_combined:
            return Emotion.SAD
        
        # ANGRY - lower threshold, easier to trigger
        # Triggers if angry is noticeable AND not clearly sad
        if angry_combined > 0.15 and sad_combined < angry_combined:
            return Emotion.ANGRY
        
        # Secondary check - if either is present at all
        if sad_combined > 0.2 and sad_combined > angry_combined:
            return Emotion.SAD
        
        if angry_combined > 0.1 and angry_combined > sad_combined:
            return Emotion.ANGRY
        
        # NEUTRAL - default
        if neutral_score > 0.3:
            return Emotion.NEUTRAL
        
        # Fallback - pick highest
        fer_emotion = max(emotions, key=emotions.get)
        return self.FER_TO_EMOTION.get(fer_emotion, Emotion.NEUTRAL)

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