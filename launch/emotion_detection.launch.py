"""
Launch file for facial emotion detection system.
Starts both detector and display nodes.
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
        parameters=[{
            'camera_id': 0,
            'publish_rate': 10.0
        }]
    )
    
    # Emotion display node
    display_node = Node(
        package='facial_emotion_detector',
        executable='emotion_display',
        name='emotion_display_node',
        output='screen',
        parameters=[{
            'camera_id': 0,
            'window_name': 'Facial Emotion Detection',
            'display_fps': True
        }]
    )
    
    return LaunchDescription([
        detector_node,
        display_node
    ])