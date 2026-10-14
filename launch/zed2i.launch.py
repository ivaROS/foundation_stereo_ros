import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory("foundation_stereo_ros")
    default_config = os.path.join(package_share, "config", "zed2i.yaml")
    default_model = os.path.expanduser(
        "~/ros2_ws/src/Fast-FoundationStereo/output_official_onnx/"
        "20_30_48_iters4_320x736/fast_foundationstereo.onnx"
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("config", default_value=default_config),
            DeclareLaunchArgument("model_path", default_value=default_model),
            Node(
                package="foundation_stereo_ros",
                executable="foundation_stereo_node",
                name="foundation_stereo",
                output="screen",
                parameters=[
                    LaunchConfiguration("config"),
                    {"model_path": LaunchConfiguration("model_path")},
                ],
            ),
        ]
    )
