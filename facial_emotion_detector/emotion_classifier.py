"""
Production-grade emotion classifier with geometric features.
Clean architecture with type safety and testable components.
"""
from typing import Tuple, Optional
from dataclasses import dataclass
import cv2
import numpy as np
from enum import Enum


class Emotion(Enum):
    """Supported emotion types."""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISED = "surprised"


@dataclass
class FacialFeatures:
    """Extracted facial geometry features for emotion classification."""
    ear: float  # Eye Aspect Ratio (0.0-1.0)
    mar: float  # Mouth Aspect Ratio (0.0-1.0)
    eyebrow_height: float  # Eyebrow position (0.0-1.0, higher = raised)
    smile_detected: bool  # Smile cascade detection
    mouth_down: bool  # Frown detection
    eyebrow_furrowed: bool  # Angry brow detection


class EmotionClassifier:
    """
    Geometric-based emotion classification using facial landmarks.
    No pre-trained deep learning models - pure computer vision.
    """

    EMOTION_EMOJIS = {
        Emotion.NEUTRAL: "😐",
        Emotion.HAPPY: "😊",
        Emotion.SAD: "😢",
        Emotion.ANGRY: "😠",
        Emotion.SURPRISED: "😲"
    }

    def __init__(
        self,
        cascade_path: str,
        ear_threshold: float = 0.55,
        mar_threshold: float = 0.65,
        eyebrow_raised_threshold: float = 120.0,
        eyebrow_furrow_ratio: float = 1.3
    ):
        """
        Initialize emotion classifier with configurable thresholds.
        
        Args:
            cascade_path: Path to Haar cascade XML
            ear_threshold: Eye aspect ratio threshold for wide eyes
            mar_threshold: Mouth aspect ratio threshold for open mouth
            eyebrow_raised_threshold: Brightness threshold for raised eyebrows
            eyebrow_furrow_ratio: Contrast ratio for furrowed brows
        """
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise FileNotFoundError(f"Cascade not found: {cascade_path}")
        
        eye_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
        self.eye_cascade = cv2.CascadeClassifier(eye_path)
        
        mouth_path = cv2.data.haarcascades + 'haarcascade_smile.xml'
        self.mouth_cascade = cv2.CascadeClassifier(mouth_path)
        
        # Configurable thresholds
        self._ear_threshold = ear_threshold
        self._mar_threshold = mar_threshold
        self._eyebrow_raised_threshold = eyebrow_raised_threshold
        self._eyebrow_furrow_ratio = eyebrow_furrow_ratio

    def detect_faces(self, frame: np.ndarray) -> list:
        """Detect faces in frame using Haar cascade."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )
        return faces

    def extract_features(self, frame: np.ndarray, face_rect: Tuple[int, int, int, int]) -> FacialFeatures:
        """
        Extract geometric features from face region.
        
        Args:
            frame: BGR image
            face_rect: Face bounding box (x, y, w, h)
            
        Returns:
            FacialFeatures dataclass with extracted measurements
        """
        x, y, w, h = face_rect
        
        # Extract face ROI
        face_roi = frame[y:y+h, x:x+w]
        gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        gray_roi = cv2.equalizeHist(gray_roi)
        
        roi_h, roi_w = gray_roi.shape
        
        # Extract features
        smile_detected = self._detect_smile(gray_roi)
        
        mouth_region = gray_roi[int(roi_h*0.55):, int(roi_w*0.25):int(roi_w*0.75)]
        mar = self._calculate_mouth_aspect_ratio(mouth_region)
        mouth_down = self._is_mouth_down(mouth_region)
        
        eye_region = gray_roi[:int(roi_h*0.5), :]
        eyes = self.eye_cascade.detectMultiScale(eye_region, 1.1, 3, minSize=(20, 20))
        ear = self._calculate_eye_aspect_ratio(eyes)
        
        eyebrow_region = gray_roi[int(roi_h*0.1):int(roi_h*0.35), :]
        eyebrow_height = self._calculate_eyebrow_height(eyebrow_region)
        eyebrow_furrowed = self._are_eyebrows_furrowed(eyebrow_region)
        
        return FacialFeatures(
            ear=ear,
            mar=mar,
            eyebrow_height=eyebrow_height,
            smile_detected=smile_detected,
            mouth_down=mouth_down,
            eyebrow_furrowed=eyebrow_furrowed
        )

    def classify(self, features: FacialFeatures) -> Emotion:
        """
        Classify emotion from extracted features.
        
        Args:
            features: FacialFeatures dataclass
            
        Returns:
            Detected emotion
        """
        # PRIORITY 1: HAPPY - Smile detected
        if features.smile_detected:
            return Emotion.HAPPY
        
        # PRIORITY 2: SURPRISED - Wide open mouth + raised eyebrows + wide eyes
        if (features.mar > self._mar_threshold and 
            features.eyebrow_height > self._eyebrow_raised_threshold and 
            features.ear > self._ear_threshold):
            return Emotion.SURPRISED
        
        # PRIORITY 3: ANGRY - Furrowed eyebrows + tense mouth + intense eyes
        if features.eyebrow_furrowed and features.ear > 0.5:
            return Emotion.ANGRY
        
        # PRIORITY 4: SAD - Mouth down + relaxed eyes
        if features.mouth_down and features.ear < 0.5:
            return Emotion.SAD
        
        # DEFAULT: NEUTRAL
        return Emotion.NEUTRAL

    def classify_from_frame(self, frame: np.ndarray, face_rect: Tuple[int, int, int, int]) -> Emotion:
        """
        Convenience method: extract features and classify in one step.
        
        Args:
            frame: BGR image
            face_rect: Face bounding box (x, y, w, h)
            
        Returns:
            Detected emotion
        """
        features = self.extract_features(frame, face_rect)
        return self.classify(features)

    # Private helper methods
    def _detect_smile(self, gray_roi: np.ndarray) -> bool:
        """Detect smile using Haar cascade."""
        smiles = self.mouth_cascade.detectMultiScale(
            gray_roi, scaleFactor=1.8, minNeighbors=20, minSize=(25, 25)
        )
        return len(smiles) > 0

    def _calculate_mouth_aspect_ratio(self, mouth_region: np.ndarray) -> float:
        """Calculate mouth openness (0.0-1.0)."""
        if mouth_region.size == 0:
            return 0.0
        
        std = np.std(mouth_region)
        mean = np.mean(mouth_region)
        ratio = std / (mean + 1e-6)
        return min(ratio, 1.0)

    def _is_mouth_down(self, mouth_region: np.ndarray) -> bool:
        """Detect frown (mouth corners down)."""
        if mouth_region.size == 0:
            return False
        
        h, w = mouth_region.shape
        top_half = mouth_region[:h//2, :]
        bottom_half = mouth_region[h//2:, :]
        
        top_mean = np.mean(top_half)
        bottom_mean = np.mean(bottom_half)
        
        return bottom_mean < (top_mean - 5)

    def _calculate_eye_aspect_ratio(self, eyes) -> float:
        """Calculate eye openness (0.0-1.0)."""
        if len(eyes) < 1:
            return 0.5
        
        ratios = []
        for (ex, ey, ew, eh) in eyes:
            ratios.append(eh / (ew + 1e-6))
        
        avg = np.mean(ratios)
        return min(avg / 0.5, 1.0)

    def _calculate_eyebrow_height(self, eyebrow_region: np.ndarray) -> float:
        """Calculate eyebrow position (brightness 0-255)."""
        if eyebrow_region.size == 0:
            return 0.0
        return float(np.mean(eyebrow_region))

    def _are_eyebrows_furrowed(self, eyebrow_region: np.ndarray) -> bool:
        """Detect angry eyebrows (high contrast in center)."""
        if eyebrow_region.size == 0:
            return False
        
        h, w = eyebrow_region.shape
        center = eyebrow_region[:, w//3:2*w//3]
        sides = np.concatenate([eyebrow_region[:, :w//3], eyebrow_region[:, 2*w//3:]], axis=1)
        
        center_std = np.std(center)
        sides_std = np.std(sides)
        
        return center_std > (sides_std * self._eyebrow_furrow_ratio)

    def get_emoji(self, emotion: Emotion) -> str:
        """Get emoji representation."""
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