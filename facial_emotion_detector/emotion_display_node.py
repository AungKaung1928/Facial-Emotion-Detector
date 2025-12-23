#!/usr/bin/env python3
"""
Emotion display node.
Subscribes to emotion messages and displays video feed with emoji overlay.
"""
from typing import Optional
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from facial_emotion_detector.emotion_classifier import EmotionClassifier, Emotion


class EmotionDisplayNode(Node):
    """
    ROS2 node for emotion display.
    Subscribes to emotion topic and displays video with emoji overlay.
    """

    def __init__(self):
        """Initialize emotion display node."""
        super().__init__('emotion_display_node')
        
        # Declare parameters
        self.declare_parameter('camera_id', 0)
        self.declare_parameter('window_name', 'Facial Emotion Detection')
        self.declare_parameter('display_fps', True)
        self.declare_parameter('cascade_path',
                             cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Get parameters
        camera_id = self.get_parameter('camera_id').value
        self.window_name = self.get_parameter('window_name').value
        self.display_fps_flag = self.get_parameter('display_fps').value
        cascade_path = self.get_parameter('cascade_path').value
        
        # Initialize emotion classifier
        try:
            self.classifier = EmotionClassifier(cascade_path)
        except Exception as e:
            self.get_logger().error(f'Failed to load classifier: {e}')
            raise
        
        # Initialize camera
        self.cap = cv2.VideoCapture(camera_id)
        
        if not self.cap.isOpened():
            self.get_logger().error(f'Failed to open camera {camera_id}')
            raise RuntimeError(f'Camera {camera_id} not available')
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.get_logger().info(f'Opened camera {camera_id} for display')
        
        # Create subscriber
        self.subscriber = self.create_subscription(
            String,
            '/facial_emotion',
            self.emotion_callback,
            10
        )
        
        # State
        self.current_emotion = Emotion.NEUTRAL
        self.frame_count = 0
        self.fps = 0.0
        self.last_time = self.get_clock().now()
        
        # Create display window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        # Create timer for display (30 Hz)
        self.timer = self.create_timer(1.0/30.0, self.display_frame)
        
        self.get_logger().info('Emotion display active')

    def emotion_callback(self, msg: String) -> None:
        """
        Handle emotion messages.
        
        Args:
            msg: String message containing emotion name
        """
        try:
            self.current_emotion = Emotion(msg.data)
        except ValueError:
            self.get_logger().warn(f'Unknown emotion: {msg.data}')
            self.current_emotion = Emotion.NEUTRAL

    def display_frame(self) -> None:
        """
        Display video frame with emotion overlay.
        """
        if not self.cap or not self.cap.isOpened():
            return
        
        # Capture frame
        ret, frame = self.cap.read()
        
        if not ret or frame is None:
            return
        
        self.frame_count += 1
        
        # Calculate FPS
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
        
        self.last_time = current_time
        
        # Detect faces
        faces = self.classifier.detect_faces(frame)
        
        # Draw face boxes and emotion labels
        for face in faces:
            self.classifier.draw_face_box(
                frame,
                tuple(face),
                self.current_emotion,
                color=(0, 255, 0)
            )
        
        # Draw emoji overlay
        emoji = self.classifier.get_emoji(self.current_emotion)
        self._draw_emoji_overlay(frame, emoji)
        
        # Draw FPS counter
        if self.display_fps_flag:
            self._draw_fps(frame)
        
        # Display frame
        cv2.imshow(self.window_name, frame)
        cv2.waitKey(1)

    def _draw_emoji_overlay(self, frame: np.ndarray, emoji: str) -> None:
        """
        Draw emoji overlay on frame.
        
        Args:
            frame: Input/output BGR image
            emoji: Unicode emoji character
        """
        # Emoji position (top-right corner)
        height, width = frame.shape[:2]
        emoji_size = 80
        margin = 20
        
        x = width - emoji_size - margin
        y = margin + emoji_size
        
        # Create semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (x - 10, y - emoji_size - 10),
            (x + emoji_size + 10, y + 10),
            (255, 255, 255),
            -1
        )
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw emotion text
        emotion_text = self.current_emotion.value.upper()
        font = cv2.FONT_HERSHEY_BOLD
        font_scale = 0.7
        thickness = 2
        
        (text_w, text_h), _ = cv2.getTextSize(
            emotion_text, 
            font, 
            font_scale, 
            thickness
        )
        
        text_x = x + (emoji_size - text_w) // 2
        text_y = y - emoji_size // 2 + text_h // 2
        
        # Draw emoji symbol
        cv2.putText(
            frame,
            emoji,
            (x, y - emoji_size // 4),
            font,
            2.0,
            (0, 0, 0),
            3
        )
        
        # Draw emotion text
        cv2.putText(
            frame,
            emotion_text,
            (text_x, text_y + emoji_size // 2),
            font,
            font_scale,
            (0, 0, 0),
            thickness
        )

    def _draw_fps(self, frame: np.ndarray) -> None:
        """
        Draw FPS counter on frame.
        
        Args:
            frame: Input/output BGR image
        """
        fps_text = f'FPS: {self.fps:.1f}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        cv2.putText(
            frame,
            fps_text,
            (10, 30),
            font,
            0.7,
            (0, 255, 0),
            2
        )

    def destroy_node(self):
        """Clean up resources."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)
    
    try:
        node = EmotionDisplayNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {e}')
    finally:
        cv2.destroyAllWindows()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()