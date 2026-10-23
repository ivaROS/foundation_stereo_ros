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

## Validated performance on this platform

The CPU tests below were run on the AMD Ryzen 5 7640HS with two ONNX Runtime
threads. Each model completed safely under bounded memory and CPU limits.

| Profile | Model input / refinement | Median speed | Peak RSS | Median disagreement vs. large model | Temporal MAD | Intended use |
|---|---:|---:|---:|---:|---:|---|
| **Fast (recommended)** | 192×352, 1 iteration, 64 px disparity | **3.93 FPS** | **0.76 GiB** | 6.57 cm | 8.09 mm | Best live CPU performance |
| Balanced | 256×512, 2 iterations, 192 px disparity | 0.86 FPS | 3.21 GiB | 3.09 cm | 7.22 mm | Better boundaries and thin objects |
| Large reference | 320×736, 4 iterations, 192 px disparity | 0.43 FPS | 5.61 GiB | Reference | 5.14 mm | Best tested CPU quality; offline or low-rate use |

The large model is the largest configuration validated on this platform. It
runs safely, but is not practical for responsive live depth. The fast model is
approximately 9.1× faster and uses 86.5% less memory, at the cost of larger
errors around boundaries, occlusions, thin structures, and distant surfaces.
These accuracy numbers measure agreement with the large learned model, not
metric ground truth.

Recommended fast model:

```text
~/ros2_ws/src/Fast-FoundationStereo/output_cpu_onnx/
20_30_48_iters1_192x352_disp64/fast_foundationstereo.onnx
```

The launch files currently retain the large model as their conservative
reference default. For practical CPU operation, pass the fast model explicitly
and pass `max_disparity_model_px:=64.0` to the launch file so the published
disparity metadata matches the model. Use two inference threads and increase
the output-rate limit gradually from 1 Hz to 3 Hz during live use.

```bash
FAST_MODEL=~/ros2_ws/src/Fast-FoundationStereo/output_cpu_onnx/20_30_48_iters1_192x352_disp64/fast_foundationstereo.onnx
ros2 launch foundation_stereo_ros zed2i_open_capture.launch.py \
  model_path:="$FAST_MODEL" intra_op_threads:=2 \
  max_inference_rate_hz:=3.0 max_disparity_model_px:=64.0 rviz:=true
```

The detailed measurements and visuals are in
`~/ros2_ws/results/2026-10-19_foundation_cpu_maxspeed/REPORT.md`.

## Depth-validation tools

The reproducible bag analyzers, CPU benchmarks, resource monitor, and model
comparison utilities live in `scripts/depth_validation/` and are installed with
this package. This keeps the evaluation code versioned beside the ROS node and
model integration it validates.

After building and sourcing the workspace, list the available command options
with, for example:

```bash
ros2 run foundation_stereo_ros analyze_sequential --help
ros2 run foundation_stereo_ros benchmark_foundation_cpu --help
ros2 run foundation_stereo_ros compare_foundation_onnx --help
```

The ONNX-based commands must run in the CPU virtual environment described under
Requirements. Generated bags, ONNX models, JSON metrics, plots, and reports stay
under `~/ros2_ws/results` or the model-output directory; they are intentionally
not installed or committed with the source package.

The default CPU configuration is deliberately conservative: two ONNX Runtime
threads and at most one inference start every four seconds. Incoming camera
frames still use a latest-frame queue, so throttling does not build latency.
Override the limit only after validating cooling and system stability:

```bash
ros2 launch foundation_stereo_ros zed2i_open_capture.launch.py \
  max_inference_rate_hz:=0.5 intra_op_threads:=2
```

Set `max_inference_rate_hz:=0.0` to remove the rate limit.

Depth output defaults to **0.3–20 m**, matching the supported range of the
wide-lens ZED 2i. The camera's ideal range is 0.3–12 m; values from 12–20 m are
retained but should be treated as lower-confidence stereo estimates. Override
`minimum_depth_m` or `maximum_depth_m` in the YAML configuration when an
application benefits from tighter filtering. Changing the far cutoff does not
change FoundationStereo inference speed.

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
