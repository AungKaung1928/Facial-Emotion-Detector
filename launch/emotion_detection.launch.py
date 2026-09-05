"""
Launch the emotion detector. It captures, classifies, publishes /facial_emotion and draws the
window itself; emotion_display is a separate optional viewer and must not run at the same time
because both open the same camera device.
"""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description with both nodes."""
    
    # Emotion detector node
    detector_node = Node(
        package='facial_emotion_detector',
        executable='emotion_detector',
        name='emotion_detector_node',
        output='screen',
        parameters=[{'camera_id': 0}]
    )
    
    return LaunchDescription([detector_node])