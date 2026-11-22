"""Launch test: the installed emotion_detector reading an image topic instead of a camera, no window.

The test process publishes scikit-image's astronaut photo (NASA, public domain) as bgr8 on
/test/image_raw and waits for the node's smoothed decision on /facial_emotion.
Run: launch_test test/test_launch_image_topic.py (needs fer + tensorflow-cpu, see README).
"""
import time
import unittest

import cv2
import launch
import launch_ros.actions
import launch_testing
import launch_testing.actions
import pytest
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from skimage import data
from std_msgs.msg import String


@pytest.mark.launch_test
def generate_test_description():
    node = launch_ros.actions.Node(
        package='facial_emotion_detector', executable='emotion_detector', output='screen',
        parameters=[{'image_topic': '/test/image_raw', 'show_window': False}])
    return launch.LaunchDescription([node, launch_testing.actions.ReadyToTest()]), {'node': node}


class TestImageTopic(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def test_astronaut_reads_happy(self):
        probe = rclpy.create_node('emotion_probe')
        got = []
        probe.create_subscription(String, '/facial_emotion', lambda m: got.append(m.data), 10)
        pub = probe.create_publisher(Image, '/test/image_raw', qos_profile_sensor_data)
        bgr = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]
        msg = Image(height=h, width=w, encoding='bgr8', is_bigendian=0, step=w * 3, data=bgr.tobytes())
        try:
            start = time.monotonic()  # includes loading the FER model
            while len(got) < 5 and time.monotonic() - start < 90.0:
                pub.publish(msg)
                rclpy.spin_once(probe, timeout_sec=0.2)
        finally:
            probe.destroy_node()
        self.assertGreaterEqual(len(got), 5, f'decisions seen: {got}')
        self.assertEqual(got[-1], 'happy')
