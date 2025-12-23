"""
Unit tests for emotion classifier.
Production-grade testing for geometric feature extraction.
"""
import unittest
import numpy as np
from facial_emotion_detector.emotion_classifier import (
    EmotionClassifier,
    FacialFeatures,
    Emotion
)


class TestEmotionClassifier(unittest.TestCase):
    """Test emotion classification logic."""

    def test_happy_detection(self):
        """Test HAPPY emotion from smile."""
        classifier = EmotionClassifier(cascade_path="dummy")  # Won't use cascade
        
        features = FacialFeatures(
            ear=0.4,
            mar=0.7,
            eyebrow_height=100.0,
            smile_detected=True,
            mouth_down=False,
            eyebrow_furrowed=False
        )
        
        result = classifier.classify(features)
        self.assertEqual(result, Emotion.HAPPY)

    def test_surprised_detection(self):
        """Test SURPRISED emotion from wide open features."""
        classifier = EmotionClassifier(cascade_path="dummy")
        
        features = FacialFeatures(
            ear=0.7,  # Wide eyes
            mar=0.8,  # Open mouth
            eyebrow_height=140.0,  # Raised eyebrows
            smile_detected=False,
            mouth_down=False,
            eyebrow_furrowed=False
        )
        
        result = classifier.classify(features)
        self.assertEqual(result, Emotion.SURPRISED)

    def test_angry_detection(self):
        """Test ANGRY emotion from furrowed brows."""
        classifier = EmotionClassifier(cascade_path="dummy")
        
        features = FacialFeatures(
            ear=0.6,  # Intense eyes
            mar=0.2,  # Closed mouth
            eyebrow_height=90.0,
            smile_detected=False,
            mouth_down=False,
            eyebrow_furrowed=True  # Key: furrowed
        )
        
        result = classifier.classify(features)
        self.assertEqual(result, Emotion.ANGRY)

    def test_sad_detection(self):
        """Test SAD emotion from frown."""
        classifier = EmotionClassifier(cascade_path="dummy")
        
        features = FacialFeatures(
            ear=0.4,  # Droopy eyes
            mar=0.3,  # Closed mouth
            eyebrow_height=100.0,
            smile_detected=False,
            mouth_down=True,  # Key: frown
            eyebrow_furrowed=False
        )
        
        result = classifier.classify(features)
        self.assertEqual(result, Emotion.SAD)

    def test_neutral_detection(self):
        """Test NEUTRAL emotion as default."""
        classifier = EmotionClassifier(cascade_path="dummy")
        
        features = FacialFeatures(
            ear=0.5,
            mar=0.4,
            eyebrow_height=100.0,
            smile_detected=False,
            mouth_down=False,
            eyebrow_furrowed=False
        )
        
        result = classifier.classify(features)
        self.assertEqual(result, Emotion.NEUTRAL)

    def test_configurable_thresholds(self):
        """Test custom threshold configuration."""
        classifier = EmotionClassifier(
            cascade_path="dummy",
            ear_threshold=0.8,  # Very high threshold
            mar_threshold=0.9
        )
        
        # Should NOT trigger SURPRISED with normal values
        features = FacialFeatures(
            ear=0.7,
            mar=0.8,
            eyebrow_height=140.0,
            smile_detected=False,
            mouth_down=False,
            eyebrow_furrowed=False
        )
        
        result = classifier.classify(features)
        self.assertNotEqual(result, Emotion.SURPRISED)


if __name__ == '__main__':
    unittest.main()