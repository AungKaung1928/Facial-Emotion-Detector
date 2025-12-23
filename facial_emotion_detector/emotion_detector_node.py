#!/usr/bin/env python3
"""
ROS2 node for facial emotion detection with live video display.
"""
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from facial_emotion_detector.emotion_classifier import EmotionClassifier, Emotion


class EmotionDetectorNode(Node):
    """ROS2 node for emotion detection with video display."""

    def __init__(self):
        super().__init__('emotion_detector_node')
        
        # Parameters
        self.declare_parameter('camera_id', 0)
        self.declare_parameter('publish_rate', 10.0)
        
        camera_id = self.get_parameter('camera_id').value
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        # Initialize classifier with configurable thresholds
        self.classifier = EmotionClassifier(
            cascade_path=cascade_path,
            ear_threshold=0.55,
            mar_threshold=0.65,
            eyebrow_raised_threshold=120.0,
            eyebrow_furrow_ratio=1.3
        )
        self.get_logger().info('✓ Emotion classifier loaded')
        
        # Initialize camera
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            self.get_logger().error(f'✗ Failed to open camera {camera_id}')
            raise RuntimeError('Camera not available')
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.get_logger().info(f'✓ Camera {camera_id} opened')
        
        # Publisher
        self.publisher = self.create_publisher(String, '/facial_emotion', 10)
        
        # State
        self.current_emotion = Emotion.NEUTRAL
        self.emotion_history = []
        self.max_history = 5
        self.fps = 0.0
        self.last_time = self.get_clock().now()
        
        # Create window
        cv2.namedWindow('Facial Emotion Detection', cv2.WINDOW_NORMAL)
        
        # Timer (30 Hz)
        self.timer = self.create_timer(1.0/30.0, self.process_and_display)
        
        self.get_logger().info('✓ Emotion detector active - Press Q to quit')
        
        # Print guide
        print("\n" + "="*50)
        print("🎭 FACIAL EMOTION DETECTION")
        print("="*50)
        print("😊 HAPPY    - Smile wide!")
        print("😢 SAD      - Frown and look down")
        print("😠 ANGRY    - Furrow eyebrows, tense face")
        print("😲 SURPRISED - Open mouth wide, raise eyebrows")
        print("😐 NEUTRAL  - Relaxed face")
        print("="*50)
        print("Press 'Q' to quit\n")

    def smooth_emotion(self, emotion: Emotion) -> Emotion:
        """Smooth detection using 5-frame history."""
        self.emotion_history.append(emotion)
        if len(self.emotion_history) > self.max_history:
            self.emotion_history.pop(0)
        
        if len(self.emotion_history) >= 3:
            emotion_counts = {}
            for e in self.emotion_history:
                emotion_counts[e] = emotion_counts.get(e, 0) + 1
            return max(emotion_counts, key=emotion_counts.get)
        return emotion

    def process_and_display(self):
        """Process frame and display with overlays."""
        if not self.cap or not self.cap.isOpened():
            return
        
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return
        
        # Calculate FPS
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
        self.last_time = current_time
        
        # Detect faces
        faces = self.classifier.detect_faces(frame)
        
        if len(faces) > 0:
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            
            # NEW API: Use classify_from_frame
            detected_emotion = self.classifier.classify_from_frame(frame, tuple(largest_face))
            
            # Smooth
            self.current_emotion = self.smooth_emotion(detected_emotion)
            
            # Draw
            self.classifier.draw_face_box(frame, tuple(largest_face), 
                                         self.current_emotion, (0, 255, 0))
            
            # Publish
            msg = String()
            msg.data = self.current_emotion.value
            self.publisher.publish(msg)
            
            # Terminal output
            emoji = self.classifier.get_emoji(self.current_emotion)
            print(f"{emoji} {self.current_emotion.value.upper()}", flush=True, end='\r')
        
        # Draw overlays
        self._draw_big_emoji(frame)
        self._draw_emotion_text(frame)
        self._draw_fps(frame)
        
        # Display
        cv2.imshow('Facial Emotion Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.get_logger().info('Quitting...')
            raise KeyboardInterrupt

    def _draw_big_emoji(self, frame: np.ndarray):
        """Draw large emoji overlay."""
        emoji = self.classifier.get_emoji(self.current_emotion)
        height, width = frame.shape[:2]
        emoji_size = 120
        margin = 30
        
        x = width - emoji_size - margin
        y = margin
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (x - 20, y - 20),
                     (x + emoji_size + 20, y + emoji_size + 20),
                     (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        cv2.putText(frame, emoji, (x, y + 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 3.5, (0, 0, 0), 4)

    def _draw_emotion_text(self, frame: np.ndarray):
        """Draw emotion text at bottom."""
        emotion_text = f"{self.current_emotion.value.upper()}"
        emoji = self.classifier.get_emoji(self.current_emotion)
        
        height, width = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        text = f"{emoji} {emotion_text}"
        (text_w, text_h), _ = cv2.getTextSize(text, font, 1.5, 3)
        
        x = (width - text_w) // 2
        y = height - 40
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (x - 20, y - text_h - 20),
                     (x + text_w + 20, y + 20), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        cv2.putText(frame, text, (x, y), font, 1.5, (255, 255, 255), 3)

    def _draw_fps(self, frame: np.ndarray):
        """Draw FPS counter."""
        fps_text = f'FPS: {self.fps:.1f}'
        cv2.putText(frame, fps_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def destroy_node(self):
        """Clean up resources."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = EmotionDetectorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n✓ Shutting down emotion detector")
    except Exception as e:
        print(f"✗ Error: {e}")
    finally:
        cv2.destroyAllWindows()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()