#!/usr/bin/env python3

import copy
import os
import signal
import threading
import time

import cv2
from cv_bridge import CvBridge
from message_filters import Subscriber, TimeSynchronizer
import numpy as np
import onnxruntime as ort
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from stereo_msgs.msg import DisparityImage


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resize_uniform_and_pad(rgb, target_height, target_width):
    source_height, source_width = rgb.shape[:2]
    scale = min(target_width / source_width, target_height / source_height)
    scaled_width = max(1, int(round(source_width * scale)))
    scaled_height = max(1, int(round(source_height * scale)))
    scaled = cv2.resize(
        rgb, (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR
    )
    padded = cv2.copyMakeBorder(
        scaled,
        0,
        target_height - scaled_height,
        0,
        target_width - scaled_width,
        cv2.BORDER_REPLICATE,
    )
    return padded, scaled_height, scaled_width


def normalize_nchw(rgb):
    normalized = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None, ...])


def nearest_resize_cpp(source, target_height, target_width):
    source_height, source_width = source.shape
    xs = np.rint(
        (np.arange(target_width, dtype=np.float32) + 0.5)
        * source_width
        / target_width
        - 0.5
    )
    ys = np.rint(
        (np.arange(target_height, dtype=np.float32) + 0.5)
        * source_height
        / target_height
        - 0.5
    )
    xs = np.clip(xs.astype(np.int64), 0, source_width - 1)
    ys = np.clip(ys.astype(np.int64), 0, source_height - 1)
    return source[ys[:, None], xs[None, :]]


def restore_disparity(
    raw_disparity,
    input_height,
    input_width,
    model_height,
    model_width,
    scaled_height,
    scaled_width,
):
    disparity = np.maximum(
        raw_disparity.reshape(model_height, model_width), 0.0
    ).astype(np.float32, copy=False)
    if input_height == model_height and input_width == model_width:
        return np.ascontiguousarray(disparity)
    cropped = disparity[:scaled_height, :scaled_width]
    restored = nearest_resize_cpp(cropped, input_height, input_width)
    return np.ascontiguousarray(
        restored * (float(input_width) / scaled_width), dtype=np.float32
    )


def disparity_to_depth(disparity, focal_length_px, baseline_m):
    output_disparity = disparity.astype(np.float32, copy=True)
    columns = np.arange(output_disparity.shape[1], dtype=np.float32)[None, :]
    output_disparity[(columns - output_disparity) < 0.0] = np.inf
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = np.float32(focal_length_px * baseline_m) / output_disparity
    return output_disparity, np.ascontiguousarray(depth, dtype=np.float32)


class FoundationStereoNode(Node):
    def __init__(self):
        super().__init__("foundation_stereo")
        backend = str(self.declare_parameter("backend", "onnx").value).lower()
        device = str(self.declare_parameter("device", "cpu").value).lower()
        model_path = self.declare_parameter("model_path", "").value
        left_topic = self.declare_parameter(
            "left_image_topic", "/zed2i/zed_node/left/color/rect/image"
        ).value
        right_topic = self.declare_parameter(
            "right_image_topic", "/zed2i/zed_node/right/color/rect/image"
        ).value
        left_info_topic = self.declare_parameter(
            "left_camera_info_topic",
            "/zed2i/zed_node/left/color/rect/camera_info",
        ).value
        right_info_topic = self.declare_parameter(
            "right_camera_info_topic",
            "/zed2i/zed_node/right/color/rect/camera_info",
        ).value
        queue_size = self.declare_parameter("sync_queue_size", 5).value
        self._baseline_override = float(
            self.declare_parameter("baseline_m", 0.0).value
        )
        self._publish_disparity = bool(
            self.declare_parameter("publish_disparity", True).value
        )
        self._max_disparity = float(
            self.declare_parameter("max_disparity_model_px", 192.0).value
        )
        intra_op_threads = int(
            self.declare_parameter("intra_op_threads", 6).value
        )

        if backend != "onnx":
            raise ValueError(
                f"unsupported backend {backend!r}; only 'onnx' is implemented"
            )
        if device != "cpu":
            raise ValueError(
                f"unsupported ONNX device {device!r}; only 'cpu' is implemented"
            )
        if not model_path or not os.path.isfile(model_path):
            raise ValueError(f"model_path is not a readable file: {model_path!r}")
        if queue_size < 1:
            raise ValueError("sync_queue_size must be at least 1")
        if intra_op_threads < 1:
            raise ValueError("intra_op_threads must be at least 1")

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.intra_op_num_threads = intra_op_threads
        session_options.inter_op_num_threads = 1
        session_options.log_severity_level = 3
        started = time.monotonic()
        self._session = ort.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        self._configure_model_contract()

        self._bridge = CvBridge()
        self._info_lock = threading.Lock()
        self._left_info = None
        self._right_info = None
        self._work_condition = threading.Condition()
        self._pending_pair = None
        self._stopping = False
        self._received_pairs = 0
        self._processed_pairs = 0
        self._replaced_pairs = 0

        sensor_qos = QoSProfile(
            depth=max(queue_size, 1),
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._depth_publisher = self.create_publisher(
            Image, "~/depth", sensor_qos
        )
        self._depth_info_publisher = self.create_publisher(
            CameraInfo, "~/depth/camera_info", sensor_qos
        )
        self._disparity_publisher = self.create_publisher(
            DisparityImage, "~/disparity", sensor_qos
        )
        self._left_info_subscription = self.create_subscription(
            CameraInfo, left_info_topic, self._set_left_info, sensor_qos
        )
        self._right_info_subscription = self.create_subscription(
            CameraInfo, right_info_topic, self._set_right_info, sensor_qos
        )
        self._left_subscription = Subscriber(
            self, Image, left_topic, qos_profile=sensor_qos
        )
        self._right_subscription = Subscriber(
            self, Image, right_topic, qos_profile=sensor_qos
        )
        self._synchronizer = TimeSynchronizer(
            [self._left_subscription, self._right_subscription], queue_size
        )
        self._synchronizer.registerCallback(self._queue_stereo_pair)
        self._worker = threading.Thread(
            target=self._worker_loop, name="foundation-stereo-onnx", daemon=True
        )
        self._worker.start()

        self.get_logger().info(
            "Loaded backend=%s device=%s model in %.3f s: %dx%d, threads=%d, "
            "left=%s, right=%s"
            % (
                backend,
                device,
                time.monotonic() - started,
                self._model_width,
                self._model_height,
                intra_op_threads,
                left_topic,
                right_topic,
            )
        )

    def _configure_model_contract(self):
        inputs = {value.name: value for value in self._session.get_inputs()}
        outputs = {value.name: value for value in self._session.get_outputs()}
        required_inputs = {"left_image", "right_image"}
        if not required_inputs.issubset(inputs) or "disparity" not in outputs:
            raise ValueError(
                "model must expose left_image, right_image, and disparity tensors"
            )
        left_shape = inputs["left_image"].shape
        right_shape = inputs["right_image"].shape
        output_shape = outputs["disparity"].shape
        if (
            left_shape != right_shape
            or len(left_shape) != 4
            or left_shape[0] != 1
            or left_shape[1] != 3
            or not isinstance(left_shape[2], int)
            or not isinstance(left_shape[3], int)
        ):
            raise ValueError(f"unsupported image tensor shapes: {left_shape}, {right_shape}")
        expected_output = [1, 1, left_shape[2], left_shape[3]]
        if output_shape != expected_output:
            raise ValueError(
                f"unsupported disparity shape {output_shape}; expected {expected_output}"
            )
        self._model_height = left_shape[2]
        self._model_width = left_shape[3]

    def _set_left_info(self, message):
        with self._info_lock:
            self._left_info = message

    def _set_right_info(self, message):
        with self._info_lock:
            self._right_info = message

    def _queue_stereo_pair(self, left, right):
        with self._work_condition:
            self._received_pairs += 1
            if self._pending_pair is not None:
                self._replaced_pairs += 1
            self._pending_pair = (left, right)
            self._work_condition.notify()

    def _worker_loop(self):
        while True:
            with self._work_condition:
                self._work_condition.wait_for(
                    lambda: self._pending_pair is not None or self._stopping
                )
                if self._stopping:
                    return
                left, right = self._pending_pair
                self._pending_pair = None
            try:
                self._process_pair(left, right)
            except Exception as error:  # Keep later frames alive after one failure.
                self.get_logger().error(f"Inference failed: {error}")

    def _process_pair(self, left, right):
        started = time.monotonic()
        if left.width != right.width or left.height != right.height:
            raise ValueError(
                "left/right dimensions differ: "
                f"{left.width}x{left.height} versus {right.width}x{right.height}"
            )
        with self._info_lock:
            left_info = self._left_info
            right_info = self._right_info
        if left_info is None or right_info is None:
            self.get_logger().warning("Waiting for left and right CameraInfo")
            return

        focal_length = left_info.p[0] if left_info.p[0] > 0.0 else left_info.k[0]
        baseline = self._baseline_override
        if baseline <= 0.0 and abs(right_info.p[0]) > 1e-9:
            baseline = abs(right_info.p[3] / right_info.p[0])
        if focal_length <= 0.0 or baseline <= 0.0:
            raise ValueError(
                f"invalid calibration: fx={focal_length:.6f}, baseline={baseline:.6f} m"
            )

        left_rgb = self._bridge.imgmsg_to_cv2(left, desired_encoding="rgb8")
        right_rgb = self._bridge.imgmsg_to_cv2(right, desired_encoding="rgb8")
        input_height, input_width = left_rgb.shape[:2]
        left_model, scaled_height, scaled_width = resize_uniform_and_pad(
            left_rgb, self._model_height, self._model_width
        )
        right_model, right_scaled_height, right_scaled_width = resize_uniform_and_pad(
            right_rgb, self._model_height, self._model_width
        )
        if (scaled_height, scaled_width) != (
            right_scaled_height,
            right_scaled_width,
        ):
            raise ValueError("left/right preprocessing produced different scales")

        raw_disparity = self._session.run(
            ["disparity"],
            {
                "left_image": normalize_nchw(left_model),
                "right_image": normalize_nchw(right_model),
            },
        )[0]
        disparity = restore_disparity(
            raw_disparity,
            input_height,
            input_width,
            self._model_height,
            self._model_width,
            scaled_height,
            scaled_width,
        )
        disparity, depth = disparity_to_depth(disparity, focal_length, baseline)

        depth_message = self._bridge.cv2_to_imgmsg(depth, encoding="32FC1")
        depth_message.header = left.header
        self._depth_publisher.publish(depth_message)

        output_info = copy.deepcopy(left_info)
        output_info.header = left.header
        self._depth_info_publisher.publish(output_info)

        if self._publish_disparity:
            disparity_message = DisparityImage()
            disparity_message.header = left.header
            disparity_message.image = self._bridge.cv2_to_imgmsg(
                disparity, encoding="32FC1"
            )
            disparity_message.image.header = left.header
            disparity_message.f = float(focal_length)
            disparity_message.t = float(baseline)
            disparity_message.min_disparity = 0.0
            resize_scale = float(scaled_width) / input_width
            disparity_message.max_disparity = self._max_disparity / resize_scale
            disparity_message.delta_d = 1.0 / resize_scale
            disparity_message.valid_window.x_offset = 0
            disparity_message.valid_window.y_offset = 0
            disparity_message.valid_window.width = input_width
            disparity_message.valid_window.height = input_height
            self._disparity_publisher.publish(disparity_message)

        self._processed_pairs += 1
        elapsed = time.monotonic() - started
        valid = np.isfinite(depth) & (depth > 0.0)
        if self._processed_pairs == 1 or self._processed_pairs % 10 == 0:
            valid_fraction = float(valid.mean()) if valid.size else 0.0
            self.get_logger().info(
                "Frame %d: %.3f s, valid_depth=%.1f%%, received=%d, replaced=%d"
                % (
                    self._processed_pairs,
                    elapsed,
                    100.0 * valid_fraction,
                    self._received_pairs,
                    self._replaced_pairs,
                )
            )

    def destroy_node(self):
        with self._work_condition:
            self._stopping = True
            self._work_condition.notify_all()
        if hasattr(self, "_worker"):
            self._worker.join()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = FoundationStereoNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as error:
        if node is not None:
            node.get_logger().fatal(str(error))
        else:
            print(f"foundation_stereo: {error}")
        raise
    finally:
        if node is not None:
            # rclpy's first SIGINT ends spin. Ignore repeated signals from a
            # supervising launch/timeout process while the ONNX worker finishes
            # its current frame, so the session cannot be destroyed in use.
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            try:
                node.destroy_node()
            except KeyboardInterrupt:
                pass
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    main()
