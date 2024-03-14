import os

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess

def generate_launch_description():
    # Make sure the docker container is running
    # This example assumes the docker container command is known and omitted for brevity

    # Launch sllidar_ros2 sllidar_a2m8_launch.py
    return LaunchDescription([
        ExecuteProcess(
            cmd=['ros2', 'launch', 'sllidar_ros2', 'sllidar_a2m8_launch.py'],
            output='screen',
            shell=True  # Use shell=True if the command does not execute correctly
        ),
    ])
