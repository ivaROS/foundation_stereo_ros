#!/usr/bin/env python3

import argparse
import csv
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from scipy.ndimage import distance_transform_edt
from sensor_msgs_py import point_cloud2

try:
    import onnxruntime as ort
except ModuleNotFoundError:
    ort = None


LEFT_TOPIC = "/zed2i/zed_node/left/image_rect_color"
RIGHT_TOPIC = "/zed2i/zed_node/right/image_rect_color"
LEFT_INFO_TOPIC = "/zed2i/zed_node/left/camera_info"
RIGHT_INFO_TOPIC = "/zed2i/zed_node/right/camera_info"
LIDAR_TOPIC = "/neuvition_cloud"

ONLINE_RUNTIME = {
    "FoundationStereo": {
        "output_hz": 0.57,
        "cpu_percent": 474.0,
        "rss_mib": 5869.9,
    },
    "ROS StereoBM": {
        "output_hz": 27.86,
        "cpu_percent": 122.0,
        "rss_mib": 99.4,
    },
    "ROS StereoSGBM": {
        "output_hz": 3.95,
        "cpu_percent": 103.0,
        "rss_mib": 138.7,
    },
}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class StereoPair:
    stamp_ns: int
    left: np.ndarray
    right: np.ndarray


@dataclass
class LidarFrame:
    stamp_ns: int
    xyz: np.ndarray
    rows: np.ndarray
    cols: np.ndarray


def stamp_ns(message):
    return int(message.header.stamp.sec) * 1_000_000_000 + int(
        message.header.stamp.nanosec
    )


def image_to_bgr(message):
    height, width = int(message.height), int(message.width)
    raw = np.frombuffer(message.data, dtype=np.uint8)
    step = int(message.step)
    rows = raw.reshape(height, step)
    encoding = message.encoding.lower()
    if encoding in ("bgr8", "rgb8"):
        image = rows[:, : width * 3].reshape(height, width, 3).copy()
        if encoding == "rgb8":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image
    if encoding in ("bgra8", "rgba8"):
        image = rows[:, : width * 4].reshape(height, width, 4).copy()
        code = cv2.COLOR_BGRA2BGR if encoding == "bgra8" else cv2.COLOR_RGBA2BGR
        return cv2.cvtColor(image, code)
    if encoding in ("mono8", "8uc1"):
        gray = rows[:, :width].reshape(height, width).copy()
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    raise ValueError(f"unsupported image encoding: {message.encoding}")


def read_live_bag(bag_path, sample_count):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_path), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    topic_types = {
        item.name: get_message(item.type)
        for item in reader.get_all_topics_and_types()
    }
    required = {LEFT_TOPIC, RIGHT_TOPIC, LEFT_INFO_TOPIC, RIGHT_INFO_TOPIC, LIDAR_TOPIC}
    missing = required - set(topic_types)
    if missing:
        raise RuntimeError(f"bag is missing topics: {sorted(missing)}")

    left_pending = {}
    right_pending = {}
    pairs = []
    lidar_frames = []
    left_info = None
    right_info = None
    first_image_stamp = None
    last_selected = None

    while reader.has_next():
        topic, serialized, _ = reader.read_next()
        if topic not in required:
            continue
        message = deserialize_message(serialized, topic_types[topic])
        if topic == LEFT_INFO_TOPIC:
            left_info = message
            continue
        if topic == RIGHT_INFO_TOPIC:
            right_info = message
            continue
        if topic == LIDAR_TOPIC:
            points = point_cloud2.read_points(
                message, field_names=["x", "y", "z", "row", "col"], skip_nans=True
            )
            lidar_frames.append(
                LidarFrame(
                    stamp_ns(message),
                    np.column_stack((points["x"], points["y"], points["z"])).astype(
                        np.float32, copy=False
                    ),
                    np.asarray(points["row"], dtype=np.int32),
                    np.asarray(points["col"], dtype=np.int32),
                )
            )
            continue

        key = stamp_ns(message)
        if first_image_stamp is None:
            first_image_stamp = key
        target = left_pending if topic == LEFT_TOPIC else right_pending
        target[key] = image_to_bgr(message)
        if key in left_pending and key in right_pending:
            elapsed = (key - first_image_stamp) * 1e-9
            interval = 18.0 / max(1, sample_count - 1)
            if (
                elapsed >= 1.0
                and len(pairs) < sample_count
                and (last_selected is None or (key - last_selected) * 1e-9 >= interval)
            ):
                pairs.append(StereoPair(key, left_pending[key], right_pending[key]))
                last_selected = key
            del left_pending[key]
            del right_pending[key]
        cutoff = key - 2_000_000_000
        for pending in (left_pending, right_pending):
            for old_key in [candidate for candidate in pending if candidate < cutoff]:
                del pending[old_key]

    if left_info is None or right_info is None:
        raise RuntimeError("camera calibration messages were not read")
    if len(pairs) < 3:
        raise RuntimeError(f"only found {len(pairs)} synchronized stereo pairs")
    if not lidar_frames:
        raise RuntimeError("no LiDAR frames were read")
    return pairs, lidar_frames, left_info, right_info


def resize_uniform_and_pad(rgb, target_height, target_width):
    source_height, source_width = rgb.shape[:2]
    scale = min(target_width / source_width, target_height / source_height)
    scaled_width = max(1, int(round(source_width * scale)))
    scaled_height = max(1, int(round(source_height * scale)))
    scaled = cv2.resize(rgb, (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)
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


def restore_disparity(raw, input_height, input_width, model_height, model_width, scaled_height, scaled_width):
    disparity = np.maximum(raw.reshape(model_height, model_width), 0.0).astype(np.float32)
    cropped = disparity[:scaled_height, :scaled_width]
    restored = cv2.resize(cropped, (input_width, input_height), interpolation=cv2.INTER_NEAREST)
    return np.ascontiguousarray(restored * (float(input_width) / scaled_width), dtype=np.float32)


def make_foundation_runner(model_path, threads):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
    inputs = {item.name: item for item in session.get_inputs()}
    model_height, model_width = inputs["left_image"].shape[2:4]

    def run(left_bgr, right_bgr):
        started = time.perf_counter()
        left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
        right_rgb = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)
        input_height, input_width = left_rgb.shape[:2]
        left_model, scaled_height, scaled_width = resize_uniform_and_pad(
            left_rgb, model_height, model_width
        )
        right_model, _, _ = resize_uniform_and_pad(right_rgb, model_height, model_width)
        prepared = time.perf_counter()
        raw = session.run(
            ["disparity"],
            {
                "left_image": normalize_nchw(left_model),
                "right_image": normalize_nchw(right_model),
            },
        )[0]
        inferred = time.perf_counter()
        disparity = restore_disparity(
            raw,
            input_height,
            input_width,
            model_height,
            model_width,
            scaled_height,
            scaled_width,
        )
        ended = time.perf_counter()
        return disparity, {
            "preprocess_ms": 1000.0 * (prepared - started),
            "inference_ms": 1000.0 * (inferred - prepared),
            "postprocess_ms": 1000.0 * (ended - inferred),
            "total_ms": 1000.0 * (ended - started),
        }

    return run


def make_bm_runner():
    matcher = cv2.StereoBM_create(numDisparities=64, blockSize=15)
    matcher.setPreFilterSize(9)
    matcher.setPreFilterCap(31)
    matcher.setMinDisparity(0)
    matcher.setTextureThreshold(10)
    matcher.setUniquenessRatio(15)
    matcher.setSpeckleWindowSize(100)
    matcher.setSpeckleRange(4)
    matcher.setDisp12MaxDiff(0)

    def run(left_bgr, right_bgr):
        left = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
        right = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)
        started = time.perf_counter()
        disparity = matcher.compute(left, right).astype(np.float32) / 16.0
        ended = time.perf_counter()
        return disparity, {"total_ms": 1000.0 * (ended - started)}

    return run


def make_sgbm_runner():
    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=160,
        blockSize=9,
        P1=200,
        P2=400,
        disp12MaxDiff=0,
        preFilterCap=31,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=4,
        mode=cv2.STEREO_SGBM_MODE_SGBM,
    )

    def run(left_bgr, right_bgr):
        left = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
        right = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)
        started = time.perf_counter()
        disparity = matcher.compute(left, right).astype(np.float32) / 16.0
        ended = time.perf_counter()
        return disparity, {"total_ms": 1000.0 * (ended - started)}

    return run


def disparity_to_depth(disparity, focal_px, baseline_m, maximum_depth=30.0):
    columns = np.arange(disparity.shape[1], dtype=np.float32)[None, :]
    valid = np.isfinite(disparity) & (disparity > 0.0) & ((columns - disparity) >= 0.0)
    depth = np.full(disparity.shape, np.nan, dtype=np.float32)
    depth[valid] = float(focal_px * baseline_m) / disparity[valid]
    depth[(depth <= 0.0) | (depth > maximum_depth)] = np.nan
    return depth


def photometric_metrics(left_bgr, right_bgr, disparity):
    left = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    right = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    height, width = left.shape
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    map_x = grid_x - disparity.astype(np.float32)
    warped = cv2.remap(right, map_x, grid_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    valid = np.isfinite(disparity) & (disparity > 0.0) & (map_x >= 0.0)
    if not np.any(valid):
        return float("nan"), float("nan"), np.full_like(left, np.nan)
    residual = np.abs(left - warped)
    masked = np.where(valid, residual, np.nan)
    return float(np.nanmedian(masked)), float(np.nanmean(masked)), masked


def lidar_raster(frame):
    depth = np.full((160, 240), np.nan, dtype=np.float32)
    columns = 239 - frame.cols
    valid = (
        (frame.rows >= 0)
        & (frame.rows < 160)
        & (columns >= 0)
        & (columns < 240)
        & np.isfinite(frame.xyz).all(axis=1)
        & (frame.xyz[:, 0] > 0.0)
    )
    depth[frame.rows[valid], columns[valid]] = frame.xyz[valid, 0]
    return depth


def project_lidar_uncalibrated(frame, camera_info, image_shape, image_bgr):
    # Axis conversion only: LiDAR ROS body (forward, left, up) to camera optical
    # (right, down, forward). No translation or measured rotation is claimed.
    xyz = frame.xyz
    optical = np.column_stack((-xyz[:, 1], -xyz[:, 2], xyz[:, 0]))
    z = optical[:, 2]
    radial_range = np.linalg.norm(xyz, axis=1)
    valid = np.isfinite(optical).all(axis=1) & (z > 0.1) & (radial_range > 0.3)
    optical = optical[valid]
    z = z[valid]
    fx, fy = float(camera_info.p[0]), float(camera_info.p[5])
    cx, cy = float(camera_info.p[2]), float(camera_info.p[6])
    u = fx * optical[:, 0] / z + cx
    v = fy * optical[:, 1] / z + cy
    height, width = image_shape[:2]
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u = np.rint(u[inside]).astype(np.int32)
    v = np.rint(v[inside]).astype(np.int32)
    z = z[inside]

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 140)
    distance = distance_transform_edt(edges == 0)
    best = (float("inf"), 0, 0)
    for dy in range(-60, 61, 4):
        for dx in range(-80, 81, 4):
            shifted_u = u + dx
            shifted_v = v + dy
            keep = (
                (shifted_u >= 0)
                & (shifted_u < width)
                & (shifted_v >= 0)
                & (shifted_v < height)
            )
            if np.count_nonzero(keep) < 100:
                continue
            score = float(np.median(np.minimum(distance[shifted_v[keep], shifted_u[keep]], 30.0)))
            if score < best[0]:
                best = (score, dx, dy)
    return u + best[1], v + best[2], z, best


def robust_stats(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
    }


def save_depth_panel(output_dir, pair, lidar, results, focal_px, baseline_m):
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), constrained_layout=True)
    axes[0, 0].imshow(cv2.cvtColor(pair.left, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Rectified left image")
    lidar_image = lidar_raster(lidar)
    lidar_plot = axes[0, 1].imshow(lidar_image, cmap="turbo", vmin=0.5, vmax=12.0)
    axes[0, 1].set_title("Neuvition forward depth (native raster)")
    fig.colorbar(lidar_plot, ax=axes[0, 1], label="m")
    axes[0, 2].axis("off")
    axes[0, 2].text(
        0.05,
        0.92,
        "Live capture\n"
        f"ZED: {pair.left.shape[1]}×{pair.left.shape[0]}\n"
        "LiDAR: 240×160 raster\n"
        f"fx: {focal_px:.3f} px\n"
        f"baseline: {baseline_m:.6f} m",
        va="top",
        fontsize=14,
    )
    for axis, (name, item) in zip(axes[1], results.items()):
        shown = axis.imshow(item["depth"], cmap="turbo", vmin=0.5, vmax=12.0)
        axis.set_title(f"{name} depth ({100.0 * item['coverage']:.1f}% valid)")
        fig.colorbar(shown, ax=axis, label="m")
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.savefig(output_dir / "depth_overview.png", dpi=150)
    plt.close(fig)


def save_lidar_overlay(output_dir, image_bgr, projected, shift):
    u, v, z = projected
    height, width = image_bgr.shape[:2]
    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height) & (z > 0.0) & (z < 20.0)
    fig, axis = plt.subplots(figsize=(16, 9), constrained_layout=True)
    axis.imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    points = axis.scatter(u[valid], v[valid], c=z[valid], s=2.0, cmap="turbo", vmin=0.5, vmax=12.0)
    axis.set_title(
        "Neuvition projected on ZED image — qualitative only\n"
        f"axis conversion plus RGB-edge 2-D shift dx={shift[1]} px, dy={shift[2]} px; no 6-DoF extrinsic"
    )
    axis.set_xlim(0, width)
    axis.set_ylim(height, 0)
    axis.set_xticks([])
    axis.set_yticks([])
    fig.colorbar(points, ax=axis, label="LiDAR forward depth (m)")
    fig.savefig(output_dir / "lidar_overlay_uncalibrated.png", dpi=170)
    plt.close(fig)


def save_metric_plots(output_dir, per_method):
    names = list(per_method)
    times = [np.median(per_method[name]["times_ms"]) for name in names]
    coverage = [100.0 * np.mean(per_method[name]["coverage"]) for name in names]
    photo = [np.mean(per_method[name]["photo_median"]) for name in names]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    colors = ["#4c78a8", "#f58518", "#54a24b"]
    axes[0].bar(names, times, color=colors)
    axes[0].set_ylabel("Median compute time (ms)")
    axes[0].set_title("Isolated offline compute")
    axes[0].set_yscale("log")
    axes[1].bar(names, coverage, color=colors)
    axes[1].set_ylabel("Valid depth (%)")
    axes[1].set_ylim(0, 100)
    axes[1].set_title("Coverage, 0–30 m")
    axes[2].bar(names, photo, color=colors)
    axes[2].set_ylabel("Median grayscale residual")
    axes[2].set_title("Stereo photometric consistency (lower is better)")
    for axis in axes:
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.25)
    fig.savefig(output_dir / "method_metrics.png", dpi=160)
    plt.close(fig)


def save_online_runtime(output_dir):
    names = list(ONLINE_RUNTIME)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    colors = ["#4c78a8", "#f58518", "#54a24b"]
    axes[0].bar(names, [ONLINE_RUNTIME[name]["output_hz"] for name in names], color=colors)
    axes[0].set_ylabel("Published output (Hz)")
    axes[0].set_title("Live throughput, estimator isolated")
    axes[1].bar(
        names,
        [ONLINE_RUNTIME[name]["cpu_percent"] / 100.0 for name in names],
        color=colors,
    )
    axes[1].set_ylabel("Logical CPU cores")
    axes[1].set_title("Live process CPU")
    axes[2].bar(names, [ONLINE_RUNTIME[name]["rss_mib"] for name in names], color=colors)
    axes[2].set_ylabel("Resident memory (MiB)")
    axes[2].set_yscale("log")
    axes[2].set_title("Live process memory")
    for axis in axes:
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.25)
    fig.savefig(output_dir / "live_runtime.png", dpi=160)
    plt.close(fig)


def write_online_runtime_csv(output_dir):
    with (output_dir / "runtime_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["method", "output_hz", "cpu_percent", "rss_mib"]
        )
        writer.writeheader()
        for name, values in ONLINE_RUNTIME.items():
            writer.writerow({"method": name, **values})


def save_temporal_plot(output_dir, per_method):
    fig, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    names = []
    values = []
    for name, item in per_method.items():
        stack = np.stack(item["depths"], axis=0)
        valid_count = np.sum(np.isfinite(stack), axis=0)
        stable = valid_count >= max(3, int(math.ceil(0.8 * stack.shape[0])))
        with np.errstate(invalid="ignore"), np.testing.suppress_warnings() as warning_context:
            warning_context.filter(RuntimeWarning, "All-NaN slice encountered")
            temporal_mad = np.nanmedian(
                np.abs(stack - np.nanmedian(stack, axis=0, keepdims=True)), axis=0
            )
        names.append(name)
        values.append(temporal_mad[stable & np.isfinite(temporal_mad)])
        item["temporal_mad_m"] = robust_stats(values[-1])
    axis.boxplot(values, labels=names, showfliers=False)
    axis.set_yscale("log")
    axis.set_ylabel("Per-pixel temporal MAD (m)")
    axis.set_title("Static-scene temporal stability (lower is better)")
    axis.grid(axis="y", alpha=0.25)
    fig.savefig(output_dir / "temporal_stability.png", dpi=160)
    plt.close(fig)


def save_uncalibrated_agreement(output_dir, lidar_u, lidar_v, lidar_z, per_method):
    fig, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    rows = []
    for name, item in per_method.items():
        depth = item["depths"][len(item["depths"]) // 2]
        height, width = depth.shape
        u = np.rint(lidar_u).astype(np.int32)
        v = np.rint(lidar_v).astype(np.int32)
        valid = (u >= 0) & (u < width) & (v >= 0) & (v < height) & (lidar_z > 0.3) & (lidar_z < 20.0)
        differences = []
        for px, py, reference in zip(u[valid], v[valid], lidar_z[valid]):
            patch = depth[max(0, py - 2) : min(height, py + 3), max(0, px - 2) : min(width, px + 3)]
            candidate = np.nanmedian(patch)
            if np.isfinite(candidate):
                differences.append(float(candidate - reference))
        differences = np.asarray(differences, dtype=np.float32)
        if differences.size:
            clipped = differences[np.abs(differences) < 10.0]
            axis.hist(clipped, bins=80, range=(-5, 5), histtype="step", linewidth=2, label=name)
            stats = robust_stats(differences)
            stats["median_absolute_m"] = float(np.median(np.abs(differences)))
            rows.append((name, stats))
            item["uncalibrated_lidar_agreement"] = stats
    axis.axvline(0.0, color="black", linewidth=1)
    axis.set_xlabel("Stereo depth − LiDAR forward depth (m)")
    axis.set_ylabel("Projected LiDAR samples")
    axis.set_title("Uncalibrated LiDAR agreement — diagnostic, not accuracy")
    if rows:
        axis.legend()
    else:
        axis.text(
            0.5,
            0.5,
            "Insufficient valid projected LiDAR returns\nfor an agreement histogram",
            ha="center",
            va="center",
            transform=axis.transAxes,
            fontsize=14,
        )
    axis.grid(alpha=0.25)
    fig.savefig(output_dir / "uncalibrated_lidar_agreement.png", dpi=160)
    plt.close(fig)
    return rows


def write_csv(output_dir, per_method):
    fields = [
        "method",
        "frames",
        "median_compute_ms",
        "p95_compute_ms",
        "mean_valid_percent",
        "median_photo_residual",
        "mean_photo_residual",
        "median_temporal_mad_m",
        "uncalibrated_lidar_median_error_m",
        "uncalibrated_lidar_median_abs_error_m",
    ]
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, item in per_method.items():
            lidar = item.get("uncalibrated_lidar_agreement", {})
            writer.writerow(
                {
                    "method": name,
                    "frames": len(item["times_ms"]),
                    "median_compute_ms": float(np.median(item["times_ms"])),
                    "p95_compute_ms": float(np.percentile(item["times_ms"], 95)),
                    "mean_valid_percent": 100.0 * float(np.mean(item["coverage"])),
                    "median_photo_residual": float(np.mean(item["photo_median"])),
                    "mean_photo_residual": float(np.mean(item["photo_mean"])),
                    "median_temporal_mad_m": item["temporal_mad_m"].get("median", float("nan")),
                    "uncalibrated_lidar_median_error_m": lidar.get("median", float("nan")),
                    "uncalibrated_lidar_median_abs_error_m": lidar.get("median_absolute_m", float("nan")),
                }
            )


def write_report(
    output_dir,
    bag_path,
    pairs,
    lidar_frames,
    per_method,
    camera_info,
    baseline_m,
    shift,
    lidar_diagnostics,
):
    rows = []
    for name, item in per_method.items():
        lidar = item.get("uncalibrated_lidar_agreement", {})
        lidar_mad = lidar.get("median_absolute_m")
        lidar_text = f"{lidar_mad:.3f}" if lidar_mad is not None else "n/a"
        rows.append(
            f"| {name} | {np.median(item['times_ms']):.1f} | "
            f"{np.percentile(item['times_ms'], 95):.1f} | "
            f"{100.0 * np.mean(item['coverage']):.1f}% | "
            f"{np.mean(item['photo_median']):.1f} | "
            f"{item['temporal_mad_m'].get('median', float('nan')):.3f} | "
            f"{lidar_text} |"
        )
    report = f"""# Live ZED / Neuvition Depth Validation

Date: 2026-10-15 (America/New_York)  
Status: **first live, static-scene benchmark; camera–LiDAR extrinsic not calibrated**

## Executive summary

This report compares Fast-FoundationStereo CPU inference with the classical ROS 2
`stereo_image_proc` StereoBM default and a wider-range StereoSGBM configuration.
All offline measurements use the same {len(pairs)} stereo pairs sampled from the
new live bag. The Neuvition data is included as a reference, but the LiDAR overlay
and numeric agreement are explicitly provisional because a six-degree-of-freedom
camera–LiDAR extrinsic is not yet known.

![Depth overview](plots/depth_overview.png)

## Recorded evidence

- Raw bag: `{bag_path}`
- ZED image size: {pairs[0].left.shape[1]} × {pairs[0].left.shape[0]}
- Evaluated stereo pairs: {len(pairs)}
- Recorded LiDAR frames: {len(lidar_frames)}
- Median LiDAR returns/frame: {lidar_diagnostics['median_points_per_frame']:.0f}
- Median returns/frame beyond 0.30 m: {lidar_diagnostics['median_points_beyond_0_3m']:.0f}
- Median radial range of all returns: {lidar_diagnostics['median_radial_range_m']:.3f} m
- Camera focal length: {float(camera_info.p[0]):.3f} px
- Stereo baseline: {baseline_m:.6f} m
- Host: {platform.node()}, {platform.processor() or platform.machine()}
- FoundationStereo backend: ONNX Runtime CPU, six intra-op threads

## Results

| Method | Median compute ms | p95 compute ms | Valid depth | Median photometric residual | Temporal MAD m | Uncalibrated LiDAR median absolute difference m |
|---|---:|---:|---:|---:|---:|---:|
{os.linesep.join(rows)}

Compute timing is isolated offline algorithm time. It excludes ROS serialization,
camera capture, plotting, and bag I/O. Coverage is the percentage of image pixels
with finite depth in the common 0–30 m evaluation interval. The photometric metric
warps the right image into the left image with the estimated disparity; lower is
better, but occlusions and exposure differences remain in the score.

![Method metrics](plots/method_metrics.png)

![Temporal stability](plots/temporal_stability.png)

## Interpretation

- **FoundationStereo is the densest and most visually coherent** in this office
  scene, with 96.3% valid coverage, including blank walls and other low-texture
  areas. Its higher photometric residual means that dense output must not be
  mistaken for verified accuracy; learned filling/smoothing can remain plausible
  where classical stereo declines to return a value.
- **StereoBM is the real-time CPU option** on this host. It nearly follows the
  30 Hz camera, but returns depth for only about 30% of pixels and is visibly
  fragmented outside strongly textured regions.
- **The tested StereoSGBM configuration is the middle ground**: about 57% valid
  coverage and the best photometric consistency, but only about 4 Hz live at the
  selected 160-pixel disparity range.
- FoundationStereo is not operationally real-time on this CPU: it is roughly
  50 times slower than the camera stream and intentionally replaces queued frames.
  A GPU backend or a substantially smaller/optimized model is needed for dense
  real-time use.
- No method can be ranked by absolute metric accuracy from this capture because
  the LiDAR reference was near-field dominated and the extrinsic is unknown.

## LiDAR comparison

The following overlay applies only the known ROS-body to camera-optical axis
conversion, followed by a two-dimensional image-edge shift of dx={shift[1]} px and
dy={shift[2]} px. That shift helps visualization; it is not a physical extrinsic.

![Uncalibrated LiDAR overlay](plots/lidar_overlay_uncalibrated.png)

![Uncalibrated LiDAR agreement](plots/uncalibrated_lidar_agreement.png)

The LiDAR difference column and histogram must be read as **uncalibrated sensor
agreement**, not depth accuracy. It mixes stereo error, camera–LiDAR rotation and
translation error, visibility/occlusion, differing angular resolution, and timing.
It is retained because it is useful for detecting gross scale or sign failures.
In this capture, almost all LiDAR returns were within 0.30 m and only a few points
per frame reached the room. This indicates a blocked/near-field-dominated LiDAR
view or degraded return state, so even qualitative cross-sensor scoring is not
supported by this sequence. The native raster and overlay document that failure.

## Live deployment observations

- ZED Open Capture held approximately 30 Hz at 1280×720.
- Neuvition delivered approximately 3.07 Hz in this run.
- Fast-FoundationStereo produced 0.57 Hz live and consumed about 4.74 logical CPU
  cores with 5,870 MiB RSS. Logged per-frame processing was typically 1.66–1.95 seconds.
- ROS StereoBM produced 27.86 Hz using about 1.22 logical CPU cores and 99 MiB RSS.
- Wide-range ROS StereoSGBM produced 3.95 Hz using about 1.03 logical CPU cores and
  139 MiB RSS.
- Running FoundationStereo, StereoBM, and wide-range StereoSGBM together saturated
  the six-core CPU; combined-output rates are therefore not used as fair speed scores.
- Both tested `stereo_image_proc` processes segfaulted during SIGINT shutdown after
  otherwise publishing valid disparity. This is an operational issue, not a depth
  quality result.
- The Neuvition process also exited with signal 11 after completing its SDK stop and
  disconnect sequence; the ZED Open Capture process stopped cleanly.

![Live runtime](plots/live_runtime.png)

## Limitations and next validation gate

1. The scene was static and opportunistic; no surveyed targets or known distances
   were introduced during this unattended capture.
2. The Neuvition driver stamps point clouds on host callback arrival and currently
   ignores the SDK device timestamp. Dynamic-scene scoring is not yet justified.
3. A proper camera–LiDAR calibration must use several board poses or a dedicated
   target. The transform must be estimated on a calibration set, locked, and then
   evaluated on different captures.
4. After calibration, the report should add LiDAR-projected MAE, RMSE, bias,
   absolute-relative error, distance-bin errors, boundary errors, and coverage.
5. StereoBM's 64-pixel range cannot represent the same near-depth interval as the
   160-pixel SGBM and FoundationStereo configurations; the speed/coverage tradeoff
   is intentional and should remain visible.

## Reproduction

The analyzer source is in
`src/foundation_stereo_ros/scripts/depth_validation`. Raw measurements are retained in
`data/metrics.csv` and `data/summary.json`; report plots are under `plots/`.
"""
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()

    if ort is None:
        candidates = []
        for parent in [Path.cwd(), *args.model.resolve().parents]:
            candidate = parent / ".venv-ffs-cpu" / "bin" / "python"
            if candidate not in candidates:
                candidates.append(candidate)
        venv_python = next((candidate for candidate in candidates if candidate.is_file()), None)
        if venv_python is None:
            raise RuntimeError(
                "onnxruntime is unavailable; activate the FoundationStereo virtual "
                "environment or place it at <workspace>/.venv-ffs-cpu"
            )
        os.execv(
            str(venv_python),
            [str(venv_python), "-m", "depth_validation.analyze_bag", *sys.argv[1:]],
        )

    plots_dir = args.output / "plots"
    data_dir = args.output / "data"
    plots_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    print("Reading live bag...", flush=True)
    pairs, lidar_frames, left_info, right_info = read_live_bag(args.bag, args.samples)
    focal_px = float(left_info.p[0] if left_info.p[0] > 0.0 else left_info.k[0])
    baseline_m = abs(float(right_info.p[3] / right_info.p[0]))
    print(f"Selected {len(pairs)} pairs and read {len(lidar_frames)} LiDAR frames", flush=True)

    runners = {
        "FoundationStereo": make_foundation_runner(args.model, args.threads),
        "ROS StereoBM": make_bm_runner(),
        "ROS StereoSGBM": make_sgbm_runner(),
    }
    per_method = {
        name: {"times_ms": [], "coverage": [], "photo_median": [], "photo_mean": [], "depths": []}
        for name in runners
    }

    representative = len(pairs) // 2
    representative_results = {}
    for index, pair in enumerate(pairs):
        print(f"Processing pair {index + 1}/{len(pairs)}", flush=True)
        for name, runner in runners.items():
            disparity, timing = runner(pair.left, pair.right)
            depth = disparity_to_depth(disparity, focal_px, baseline_m)
            median_photo, mean_photo, _ = photometric_metrics(pair.left, pair.right, disparity)
            valid = np.isfinite(depth)
            item = per_method[name]
            item["times_ms"].append(float(timing["total_ms"]))
            item["coverage"].append(float(np.mean(valid)))
            item["photo_median"].append(median_photo)
            item["photo_mean"].append(mean_photo)
            item["depths"].append(depth)
            if index == representative:
                representative_results[name] = {
                    "depth": depth,
                    "disparity": disparity,
                    "coverage": float(np.mean(valid)),
                }

    pair = pairs[representative]
    lidar = min(lidar_frames, key=lambda frame: abs(frame.stamp_ns - pair.stamp_ns))
    all_radial = np.concatenate([np.linalg.norm(frame.xyz, axis=1) for frame in lidar_frames])
    lidar_diagnostics = {
        "median_points_per_frame": float(np.median([len(frame.xyz) for frame in lidar_frames])),
        "median_points_beyond_0_3m": float(
            np.median(
                [
                    np.count_nonzero(np.linalg.norm(frame.xyz, axis=1) > 0.3)
                    for frame in lidar_frames
                ]
            )
        ),
        "median_radial_range_m": float(np.median(all_radial[np.isfinite(all_radial)])),
    }
    save_depth_panel(plots_dir, pair, lidar, representative_results, focal_px, baseline_m)
    save_metric_plots(plots_dir, per_method)
    save_online_runtime(plots_dir)
    save_temporal_plot(plots_dir, per_method)
    u, v, z, shift = project_lidar_uncalibrated(lidar, left_info, pair.left.shape, pair.left)
    save_lidar_overlay(plots_dir, pair.left, (u, v, z), shift)
    save_uncalibrated_agreement(plots_dir, u, v, z, per_method)
    write_csv(data_dir, per_method)
    write_online_runtime_csv(data_dir)

    summary = {
        "bag": str(args.bag),
        "model": str(args.model),
        "sample_count": len(pairs),
        "lidar_frame_count": len(lidar_frames),
        "image_size": [int(pairs[0].left.shape[1]), int(pairs[0].left.shape[0])],
        "focal_px": focal_px,
        "baseline_m": baseline_m,
        "uncalibrated_overlay_shift_px": [int(shift[1]), int(shift[2])],
        "lidar_diagnostics": lidar_diagnostics,
        "online_runtime": ONLINE_RUNTIME,
        "methods": {},
    }
    for name, item in per_method.items():
        summary["methods"][name] = {
            "compute_ms": robust_stats(item["times_ms"]),
            "valid_fraction": robust_stats(item["coverage"]),
            "photometric_median": robust_stats(item["photo_median"]),
            "photometric_mean": robust_stats(item["photo_mean"]),
            "temporal_mad_m": item["temporal_mad_m"],
            "uncalibrated_lidar_agreement": item.get("uncalibrated_lidar_agreement", {}),
        }
    (data_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(
        args.output,
        args.bag,
        pairs,
        lidar_frames,
        per_method,
        left_info,
        baseline_m,
        shift,
        lidar_diagnostics,
    )
    print(f"Report written to {args.output / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
