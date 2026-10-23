import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    package_share = get_package_share_directory("foundation_stereo_ros")
    default_config = os.path.join(
        package_share, "config", "zed2i_open_capture.yaml"
    )
    rviz_config = os.path.join(
        package_share, "rviz", "foundation_stereo.rviz"
    )
    default_model = os.path.expanduser(
        "~/ros2_ws/src/Fast-FoundationStereo/output_official_onnx/"
        "20_30_48_iters4_320x736/fast_foundationstereo.onnx"
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("config", default_value=default_config),
            DeclareLaunchArgument("model_path", default_value=default_model),
            DeclareLaunchArgument("intra_op_threads", default_value="2"),
            DeclareLaunchArgument("max_inference_rate_hz", default_value="0.25"),
            DeclareLaunchArgument("max_disparity_model_px", default_value="192.0"),
            DeclareLaunchArgument(
                "rviz",
                default_value="false",
                description="Start RViz2 with the Foundation Stereo layout",
            ),
            Node(
                package="foundation_stereo_ros",
                executable="foundation_stereo_node",
                name="foundation_stereo",
                output="screen",
                parameters=[
                    LaunchConfiguration("config"),
                    {
                        "model_path": LaunchConfiguration("model_path"),
                        "intra_op_threads": ParameterValue(
                            LaunchConfiguration("intra_op_threads"), value_type=int
                        ),
                        "max_inference_rate_hz": ParameterValue(
                            LaunchConfiguration("max_inference_rate_hz"),
                            value_type=float,
                        ),
                        "max_disparity_model_px": ParameterValue(
                            LaunchConfiguration("max_disparity_model_px"),
                            value_type=float,
                        ),
                    },
                ],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="screen",
                arguments=["-d", rviz_config],
                condition=IfCondition(LaunchConfiguration("rviz")),
            ),
        ]
    )
