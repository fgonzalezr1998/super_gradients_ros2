import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    ld = LaunchDescription()

    config = os.path.join(
        get_package_share_directory('super_gradients_ros2'),
        'config',
        'config.yaml'
    )

    super_gradients_ros2_node = Node(
        package = "super_gradients_ros2",
        name = "super_gradients_ros2_node",
        executable = "super_gradients_ros2_node",
        parameters = [config],
        output='screen',
        arguments=[('__log_level:=debug')]
    )

    ld.add_action(super_gradients_ros2_node)

    return ld