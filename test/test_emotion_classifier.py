"""Unit tests for the FER-score decision logic (no TensorFlow or camera needed)."""
import unittest

from facial_emotion_detector.emotion_classifier import Emotion, EmotionClassifier


def scores(**kw):
    base = {k: 0.0 for k in ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")}
    base.update(kw)
    return base


class TestDecision(unittest.TestCase):
    def test_happy_dominant(self):
        self.assertEqual(EmotionClassifier.decide(scores(happy=0.8, neutral=0.2)), Emotion.HAPPY)

    def test_surprise(self):
        self.assertEqual(EmotionClassifier.decide(scores(surprise=0.5, fear=0.3)), Emotion.SURPRISED)

    def test_sad_boosted_by_fear(self):
        # sad 0.15 + 0.3 * fear 0.4 = 0.27 > SAD_MIN, angry 0
        self.assertEqual(EmotionClassifier.decide(scores(sad=0.15, fear=0.4, neutral=0.4)), Emotion.SAD)

    def test_angry_boosted_by_disgust(self):
        # angry 0.1 + 0.5 * disgust 0.2 = 0.2 > ANGRY_MIN, sad 0
        self.assertEqual(EmotionClassifier.decide(scores(angry=0.1, disgust=0.2, neutral=0.6)), Emotion.ANGRY)

    def test_angry_wins_over_weaker_sad(self):
        self.assertEqual(EmotionClassifier.decide(scores(angry=0.3, sad=0.2, neutral=0.5)), Emotion.ANGRY)

    def test_neutral_default(self):
        self.assertEqual(EmotionClassifier.decide(scores(neutral=0.9, happy=0.1)), Emotion.NEUTRAL)

    def test_fallback_to_argmax_mapping(self):
        # nothing crosses a threshold: argmax "fear" maps to SURPRISED
        self.assertEqual(EmotionClassifier.decide(scores(fear=0.3, neutral=0.29, happy=0.2)), Emotion.SURPRISED)

    def test_empty_scores(self):
        self.assertEqual(EmotionClassifier.decide({}), Emotion.NEUTRAL)


if __name__ == "__main__":
    unittest.main()
