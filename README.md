# foundation_stereo_ros

ROS 2 node for stereo depth estimation with
[Fast-FoundationStereo](https://github.com/NVlabs/Fast-FoundationStereo).
It accepts synchronized rectified stereo images and publishes depth and
disparity. Inference uses ONNX Runtime on CPU.

Paper: [Fast-FoundationStereo: Real-Time Zero-Shot Stereo Matching](https://arxiv.org/abs/2512.11130)

## Requirements

- ROS 2 Jazzy
- `cv_bridge`, `message_filters`, `sensor_msgs`, and `stereo_msgs`
- Python 3 with NumPy, OpenCV, and ONNX Runtime
- A Fast-FoundationStereo ONNX model

Install ONNX Runtime in a virtual environment:

```bash
cd ~/ros2_ws
python3 -m venv --system-site-packages .venv-ffs-cpu
.venv-ffs-cpu/bin/python -m pip install onnxruntime
```

## Build

```bash
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
source .venv-ffs-cpu/bin/activate
colcon build --symlink-install --packages-select foundation_stereo_ros
source install/setup.bash
```

## Run

With `zed_open_capture_ros`:

```bash
ros2 launch zed_open_capture_ros zed2i.launch.py namespace:=zed2i/zed_node
ros2 launch foundation_stereo_ros zed2i_open_capture.launch.py
```

With ZED SDK topic names, including the recorded bag configuration:

```bash
ros2 launch foundation_stereo_ros zed2i.launch.py
```

The default model path is:

```text
~/ros2_ws/src/Fast-FoundationStereo/output_official_onnx/20_30_48_iters4_320x736/fast_foundationstereo.onnx
```

Use another model with:

```bash
ros2 launch foundation_stereo_ros zed2i_open_capture.launch.py \
  model_path:=/absolute/path/to/model.onnx
```

## Topics

Live ZED inputs:

- `/zed2i/zed_node/left/image_rect_color`
- `/zed2i/zed_node/right/image_rect_color`
- `/zed2i/zed_node/left/camera_info`
- `/zed2i/zed_node/right/camera_info`

Outputs:

- `/foundation_stereo/depth` — `sensor_msgs/Image`, `32FC1`, metres
- `/foundation_stereo/depth/camera_info` — left camera calibration
- `/foundation_stereo/disparity` — `stereo_msgs/DisparityImage`

Input images must be rectified, equal in size, and have matching timestamps.
The node reads focal length from the left `CameraInfo` and baseline from the
right projection matrix.
