from setuptools import setup
import os
from glob import glob

package_name = 'facial_emotion_detector'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), 
         glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='allkg',
    maintainer_email='allkg@example.com',
    description='Real-time facial emotion detection using ROS2 and OpenCV',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'emotion_detector = facial_emotion_detector.emotion_detector_node:main',
            'emotion_display = facial_emotion_detector.emotion_display_node:main',
        ],
    },
)