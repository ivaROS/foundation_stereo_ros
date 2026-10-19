#!/usr/bin/env python3

"""Analyze sequential ZED, Neuvition, and isolated stereo-estimator bags."""

import argparse
import csv
import json
import math
import warnings
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


LEFT_IMAGE = "/zed2i/left/image_rect_color"
RIGHT_IMAGE = "/zed2i/right/image_rect_color"
LEFT_INFO = "/zed2i/left/camera_info"
RIGHT_INFO = "/zed2i/right/camera_info"
LIDAR_DEPTH = "/neuvition_depth"

METHODS = {
    "ROS StereoBM": ("ros_bm", "/zed2i/disparity"),
    "ROS StereoSGBM": ("ros_sgbm", "/zed2i/disparity"),
    "FoundationStereo": ("foundation_3threads", "/foundation_stereo/disparity"),
}

RESOURCE_FILES = {
    "ROS StereoBM": "resources_ros_bm.csv",
    "ROS StereoSGBM": "resources_ros_sgbm.csv",
    "FoundationStereo": "resources_foundation_3threads.csv",
}

RUNTIME = {
    "ROS StereoBM": {
        "output_frames": 143,
        "input_pairs": 154,
        "quality_replay_rate": 0.5,
        "peak_rss_mib": 106468 / 1024,
        "cpu_seconds": 9.14 + 0.58,
        "compute_median_ms": 12.679,
        "compute_p95_ms": 13.064,
        "note": "Pace-limited quality replay; not a maximum-throughput test",
    },
    "ROS StereoSGBM": {
        "output_frames": 133,
        "input_pairs": 154,
        "quality_replay_rate": 0.25,
        "peak_rss_mib": 108808 / 1024,
        "cpu_seconds": 27.55 + 0.37,
        "compute_median_ms": 198.724,
        "compute_p95_ms": 209.439,
        "note": "Pace-limited quality replay; 10 synchronized pairs not emitted",
    },
    "FoundationStereo": {
        "output_frames": 8,
        "input_pairs": 154,
        "quality_replay_rate": 1.0,
        "peak_rss_mib": 6023956 / 1024,
        "cpu_seconds": 37.58 + 2.12,
        "first_frame_seconds": 1.978,
        "compute_median_ms": 1978.0,
        "compute_p95_ms": 1978.0,
        "threads": 3,
        "note": "Real-time replay; latest-frame replacement is intentional",
    },
}


def open_reader(path):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(path), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    types = {item.name: get_message(item.type) for item in reader.get_all_topics_and_types()}
    return reader, types


def stamp_ns(message):
    header = message.header
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


def image_array(message):
    encoding = message.encoding.lower()
    height, width, step = int(message.height), int(message.width), int(message.step)
    if encoding in ("32fc1", "32fc"):
        return np.frombuffer(message.data, np.float32).reshape(height, step // 4)[:, :width].copy()
    raw = np.frombuffer(message.data, np.uint8).reshape(height, step)
    if encoding == "bgr8":
        return raw[:, : width * 3].reshape(height, width, 3).copy()
    if encoding == "rgb8":
        rgb = raw[:, : width * 3].reshape(height, width, 3)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if encoding in ("mono8", "8uc1"):
        gray = raw[:, :width].copy()
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    raise ValueError(f"Unsupported image encoding {message.encoding}")


def read_disparity_stamps(path, topic):
    reader, types = open_reader(path)
    stamps = []
    while reader.has_next():
        name, data, _ = reader.read_next()
        if name == topic:
            stamps.append(stamp_ns(deserialize_message(data, types[name])))
    return stamps


def read_disparities(path, topic, wanted):
    reader, types = open_reader(path)
    result = {}
    while reader.has_next():
        name, data, _ = reader.read_next()
        if name != topic:
            continue
        message = deserialize_message(data, types[name])
        stamp = stamp_ns(message)
        if stamp in wanted:
            result[stamp] = {
                "disparity": image_array(message.image),
                "f": float(message.f),
                "t": float(message.t),
                "min": float(message.min_disparity),
                "max": float(message.max_disparity),
            }
    return result


def read_stereo(path, wanted):
    reader, types = open_reader(path)
    left, right = {}, {}
    left_info = right_info = None
    all_left, all_right = set(), set()
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic not in {LEFT_IMAGE, RIGHT_IMAGE, LEFT_INFO, RIGHT_INFO}:
            continue
        message = deserialize_message(data, types[topic])
        if topic == LEFT_INFO:
            left_info = message
        elif topic == RIGHT_INFO:
            right_info = message
        elif topic == LEFT_IMAGE:
            key = stamp_ns(message)
            all_left.add(key)
            if key in wanted:
                left[key] = image_array(message)
        else:
            key = stamp_ns(message)
            all_right.add(key)
            if key in wanted:
                right[key] = image_array(message)
    return left, right, left_info, right_info, all_left & all_right


def read_lidar_depths(path):
    reader, types = open_reader(path)
    frames = []
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic == LIDAR_DEPTH:
            frames.append(image_array(deserialize_message(data, types[topic])))
    return frames


def disparity_to_depth(item):
    disparity = item["disparity"]
    valid = np.isfinite(disparity) & (disparity > max(0.0, item["min"]))
    depth = np.full(disparity.shape, np.nan, np.float32)
    depth[valid] = item["f"] * item["t"] / disparity[valid]
    depth[(depth < 0.3) | (depth > 30.0)] = np.nan
    return depth


def photometric(left_bgr, right_bgr, disparity):
    left = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    right = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    height, width = left.shape
    x, y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    source_x = x - disparity.astype(np.float32)
    warped = cv2.remap(right, source_x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    valid = np.isfinite(disparity) & (disparity > 0) & (source_x >= 0)
    residual = np.full(left.shape, np.nan, np.float32)
    residual[valid] = np.abs(left[valid] - warped[valid])
    return residual


def stats(values):
    values = np.asarray(values, np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"count": 0}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
    }


def detailed_depth_stats(values):
    values = np.asarray(values, np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"count": 0}
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "p01": float(np.percentile(values, 1)),
        "p05": float(np.percentile(values, 5)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def spatial_depth_metrics(depths):
    stack = np.stack(depths)
    frame_count, height, width = stack.shape
    regions = []

    def add_region(region, row, column, values, pixel_total):
        finite = values[np.isfinite(values)]
        entry = {
            "region": region,
            "row": row,
            "column": column,
            "valid_percent": 100.0 * finite.size / pixel_total,
            **detailed_depth_stats(finite),
        }
        regions.append(entry)

    add_region("full", -1, -1, stack, stack.size)
    y0, y1 = height // 4, 3 * height // 4
    x0, x1 = width // 4, 3 * width // 4
    center = stack[:, y0:y1, x0:x1]
    add_region("center_50pct", -1, -1, center, center.size)

    y_edges = np.linspace(0, height, 4, dtype=int)
    x_edges = np.linspace(0, width, 4, dtype=int)
    for row in range(3):
        for column in range(3):
            cell = stack[:, y_edges[row]:y_edges[row + 1], x_edges[column]:x_edges[column + 1]]
            add_region(f"grid_r{row}_c{column}", row, column, cell, cell.size)

    valid = stack[np.isfinite(stack)]
    bands = []
    for label, low, high in [
        ("0.3–1 m", 0.3, 1.0),
        ("1–2 m", 1.0, 2.0),
        ("2–5 m", 2.0, 5.0),
        ("5–10 m", 5.0, 10.0),
        ("10–30 m", 10.0, 30.000001),
    ]:
        count = int(np.count_nonzero((valid >= low) & (valid < high)))
        bands.append({
            "band": label,
            "low_m": low,
            "high_m": high,
            "count": count,
            "percent_of_valid": 100.0 * count / valid.size if valid.size else 0.0,
            "percent_of_all_pixels": 100.0 * count / stack.size,
        })
    return {"shape": [frame_count, height, width], "regions": regions, "bands": bands}


def detect_near_regions(depth, threshold_m=2.0):
    mask = (np.isfinite(depth) & (depth >= 0.3) & (depth < threshold_m)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, component_stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    minimum_area = max(100, int(0.002 * depth.shape[0] * depth.shape[1]))
    regions = []
    for label in range(1, count):
        x, y, width, height, area = component_stats[label]
        if area < minimum_area:
            continue
        values = depth[(labels == label) & np.isfinite(depth)]
        if not values.size:
            continue
        regions.append({
            "x": int(x), "y": int(y), "width": int(width), "height": int(height),
            "area_px": int(area), "median_depth_m": float(np.median(values)),
            "p05_depth_m": float(np.percentile(values, 5)),
        })
    regions.sort(key=lambda item: item["area_px"], reverse=True)
    return mask.astype(bool), regions


def choose_evenly(values, count):
    values = sorted(values)
    if len(values) <= count:
        return values
    indices = np.linspace(0, len(values) - 1, count).round().astype(int)
    return [values[index] for index in indices]


def evaluate_method(outputs, left, right, selected):
    depths, coverage, photo_median, photo_mean = [], [], [], []
    residuals = []
    for key in selected:
        output = outputs[key]
        depth = disparity_to_depth(output)
        residual = photometric(left[key], right[key], output["disparity"])
        valid_residual = residual[np.isfinite(residual)]
        depths.append(depth)
        residuals.append(residual)
        coverage.append(float(np.mean(np.isfinite(depth))))
        photo_median.append(float(np.median(valid_residual)))
        photo_mean.append(float(np.mean(valid_residual)))
    stack = np.stack(depths)
    valid_count = np.sum(np.isfinite(stack), axis=0)
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", category=RuntimeWarning)
        center = np.nanmedian(stack, axis=0)
        temporal_mad = np.nanmedian(np.abs(stack - center), axis=0)
    stable = valid_count >= max(3, math.ceil(0.7 * len(depths)))
    temporal_values = temporal_mad[stable & np.isfinite(temporal_mad)]
    return {
        "depths": depths,
        "residuals": residuals,
        "coverage": coverage,
        "photo_median": photo_median,
        "photo_mean": photo_mean,
        "temporal_mad": stats(temporal_values),
    }


def evaluate_lidar(frames):
    stack = np.stack(frames)
    valid = np.isfinite(stack) & (stack > 0.3) & (stack < 30.0)
    cleaned = np.where(valid, stack, np.nan)
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", category=RuntimeWarning)
        center = np.nanmedian(cleaned, axis=0)
        mad = np.nanmedian(np.abs(cleaned - center), axis=0)
    persistent = np.sum(valid, axis=0) >= math.ceil(0.7 * len(frames))
    return {
        "frames": len(frames),
        "coverage": stats(np.mean(valid, axis=(1, 2))),
        "depth": stats(cleaned.ravel()),
        "temporal_mad": stats(mad[persistent & np.isfinite(mad)]),
        "representative": cleaned[len(frames) // 2],
        "all_valid": cleaned[np.isfinite(cleaned)],
    }


def save_overview(path, left_image, results, index, lidar):
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), constrained_layout=True)
    axes[0, 0].imshow(cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("ZED rectified left image")
    for axis, (name, result) in zip([axes[0, 1], axes[0, 2], axes[1, 0]], results.items()):
        image = axis.imshow(result["depths"][index], cmap="turbo", vmin=0.3, vmax=12)
        axis.set_title(f"{name} depth")
        fig.colorbar(image, ax=axis, label="m")
    image = axes[1, 1].imshow(lidar["representative"], cmap="turbo", vmin=0.3, vmax=12)
    axes[1, 1].set_title("Neuvition native depth (separate capture)")
    fig.colorbar(image, ax=axes[1, 1], label="m")
    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.02,
        0.95,
        "Camera and LiDAR are side-by-side.\n"
        "Captures were sequential and the rig was stationary.\n\n"
        "No camera–LiDAR extrinsic is known.\n"
        "The LiDAR panel is qualitative only and is not\n"
        "pixel-aligned with the camera panels.",
        va="top",
        fontsize=13,
    )
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_metrics(path, results):
    names = list(results)
    coverage = [100 * np.mean(results[name]["coverage"]) for name in names]
    photo = [np.mean(results[name]["photo_median"]) for name in names]
    temporal = [results[name]["temporal_mad"].get("median", np.nan) for name in names]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    colors = ["#4c78a8", "#f58518", "#54a24b"]
    axes[0].bar(names, coverage, color=colors)
    axes[0].set_ylabel("Valid pixels, 0.3–30 m (%)")
    axes[0].set_ylim(0, 100)
    axes[1].bar(names, photo, color=colors)
    axes[1].set_ylabel("Median grayscale residual (lower better)")
    axes[2].bar(names, temporal, color=colors)
    axes[2].set_ylabel("Temporal depth MAD (m, lower better)")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", rotation=20)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def read_resource_traces(data_dir):
    traces = {}
    for name, filename in RESOURCE_FILES.items():
        with (data_dir / filename).open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        elapsed = np.asarray([float(row["elapsed_s"]) for row in rows])
        cpu = np.asarray([float(row["cpu_percent"]) for row in rows])
        rss = np.asarray([float(row["rss_mib"]) for row in rows])
        active = cpu > 5.0
        traces[name] = {
            "elapsed_s": elapsed,
            "cpu_percent": cpu,
            "rss_mib": rss,
            "summary": {
                "sample_count": len(rows),
                "interval_s": float(np.median(np.diff(elapsed))),
                "active_cpu_median_percent": float(np.median(cpu[active])),
                "active_cpu_mean_percent": float(np.mean(cpu[active])),
                "cpu_peak_percent": float(np.max(cpu)),
                "rss_median_mib": float(np.median(rss)),
                "rss_peak_mib": float(np.max(rss)),
            },
        }
    return traces


def save_resource_traces(path, resources):
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True, sharex=True)
    for name, item in resources.items():
        axes[0].plot(item["elapsed_s"], item["cpu_percent"], linewidth=1.8, label=name)
        axes[1].plot(item["elapsed_s"], item["rss_mib"], linewidth=1.8, label=name)
    axes[0].axhline(100, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    axes[0].set_ylabel("Process CPU (%)")
    axes[0].set_title("Isolated process resource samples (100% = one logical core)")
    axes[1].set_ylabel("Resident memory (MiB)")
    axes[1].set_xlabel("Elapsed measurement time (s)")
    axes[1].set_yscale("log")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_runtime(path):
    names = list(RUNTIME)
    completeness = [100 * RUNTIME[name]["output_frames"] / RUNTIME[name]["input_pairs"] for name in names]
    memory = [RUNTIME[name]["peak_rss_mib"] for name in names]
    compute_ms = [RUNTIME[name]["compute_median_ms"] for name in names]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    colors = ["#4c78a8", "#f58518", "#54a24b"]
    axes[0].bar(names, completeness, color=colors)
    axes[0].set_ylabel("Emitted / synchronized input (%)")
    axes[0].set_ylim(0, 105)
    axes[1].bar(names, compute_ms, color=colors)
    axes[1].set_ylabel("Isolated compute latency (ms)")
    axes[1].set_yscale("log")
    axes[2].bar(names, memory, color=colors)
    axes[2].set_ylabel("Peak RSS (MiB)")
    axes[2].set_yscale("log")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", rotation=20)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_ranges(path, results, lidar):
    fig, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    bins = np.linspace(0.3, 12, 80)
    for name, result in results.items():
        values = np.concatenate([depth[np.isfinite(depth)] for depth in result["depths"]])
        axis.hist(values, bins=bins, density=True, histtype="step", linewidth=2, label=name)
    axis.hist(lidar["all_valid"], bins=bins, density=True, histtype="step", linewidth=2,
              linestyle="--", label="Neuvition (native FOV)")
    axis.set_xlabel("Depth (m)")
    axis.set_ylabel("Normalized sample density")
    axis.set_title("Depth distributions — LiDAR is not pixel-aligned")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_depth_statistics(path, spatial):
    names = list(spatial)
    x = np.arange(len(names))
    width = 0.23
    full = {name: spatial[name]["regions"][0] for name in names}
    fig, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    axis.bar(x - width, [full[name]["median"] for name in names], width, label="Median")
    axis.bar(x, [full[name]["mean"] for name in names], width, label="Mean")
    axis.bar(x + width, [full[name]["p95"] for name in names], width, label="p95")
    axis.set_xticks(x, names, rotation=18)
    axis.set_ylabel("Depth (m)")
    axis.set_title("Valid-pixel depth distribution summary")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_spatial_heatmaps(path, spatial):
    names = list(spatial)
    fig, axes = plt.subplots(2, len(names), figsize=(4.3 * len(names), 7.5), constrained_layout=True)
    for column, name in enumerate(names):
        cells = [item for item in spatial[name]["regions"] if item["row"] >= 0]
        median = np.full((3, 3), np.nan)
        coverage = np.full((3, 3), np.nan)
        for item in cells:
            median[item["row"], item["column"]] = item.get("median", np.nan)
            coverage[item["row"], item["column"]] = item["valid_percent"]
        image = axes[0, column].imshow(median, cmap="turbo", vmin=0.3, vmax=8)
        axes[0, column].set_title(f"{name}\nmedian depth (m)")
        axes[1, column].imshow(coverage, cmap="viridis", vmin=0, vmax=100)
        axes[1, column].set_title("valid coverage (%)")
        for row in range(3):
            for cell_column in range(3):
                axes[0, column].text(cell_column, row, f"{median[row, cell_column]:.2f}",
                                     ha="center", va="center", color="white", fontsize=10)
                axes[1, column].text(cell_column, row, f"{coverage[row, cell_column]:.1f}",
                                     ha="center", va="center", color="white", fontsize=10)
        for row_axis in axes[:, column]:
            row_axis.set_xticks([])
            row_axis.set_yticks([])
    fig.colorbar(image, ax=axes[0, :], label="m", shrink=0.8)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_near_regions(path, left_image, representative_depths):
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
    detections = {}
    for axis, (name, depth) in zip(axes.flat, representative_depths.items()):
        mask, regions = detect_near_regions(depth)
        detections[name] = regions
        if name == "Neuvition":
            axis.imshow(depth, cmap="turbo", vmin=0.3, vmax=8)
        else:
            axis.imshow(cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB))
        overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
        overlay[mask] = (1.0, 0.1, 0.1, 0.28)
        axis.imshow(overlay)
        for region in regions[:8]:
            rectangle = plt.Rectangle(
                (region["x"], region["y"]), region["width"], region["height"],
                fill=False, edgecolor="yellow", linewidth=1.5,
            )
            axis.add_patch(rectangle)
            axis.text(region["x"], region["y"], f"{region['median_depth_m']:.2f} m",
                      color="yellow", fontsize=8, va="bottom")
        axis.set_title(f"{name}: {len(regions)} connected near regions (<2 m)")
        axis.set_xticks([])
        axis.set_yticks([])
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return detections


def write_spatial_csvs(data_dir, spatial, detections):
    region_rows = []
    band_rows = []
    detection_rows = []
    for name, item in spatial.items():
        for region in item["regions"]:
            region_rows.append({"source": name, **region})
        for band in item["bands"]:
            band_rows.append({"source": name, **band})
    for name, regions in detections.items():
        for index, region in enumerate(regions, start=1):
            detection_rows.append({"source": name, "region_id": index, **region})
    for filename, rows in [
        ("depth_region_statistics.csv", region_rows),
        ("depth_band_statistics.csv", band_rows),
        ("near_region_detections.csv", detection_rows),
    ]:
        with (data_dir / filename).open("w", newline="", encoding="utf-8") as stream:
            if rows:
                writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)


def write_report(output, args, selected, total_pairs, results, lidar, resources, spatial, detections):
    table = []
    for name, result in results.items():
        runtime = RUNTIME[name]
        resource = resources[name]["summary"]
        table.append(
            f"| {name} | {100*np.mean(result['coverage']):.1f}% | "
            f"{np.mean(result['photo_median']):.1f} | "
            f"{result['temporal_mad'].get('median', float('nan')):.3f} | "
            f"{runtime['output_frames']}/{runtime['input_pairs']} | "
            f"{resource['active_cpu_median_percent']:.1f}% | "
            f"{resource['cpu_peak_percent']:.1f}% | "
            f"{resource['rss_peak_mib']:.0f} |"
        )
    depth_rows = []
    area_rows = []
    band_rows = []
    for name, item in spatial.items():
        full = item["regions"][0]
        center = item["regions"][1]
        depth_rows.append(
            f"| {name} | {full['min']:.3f} | {full['p01']:.3f} | "
            f"{full['mean']:.3f} | {full['median']:.3f} | {full['p95']:.3f} | "
            f"{full['p99']:.3f} | {full['max']:.3f} |"
        )
        area_rows.append(
            f"| {name} | {full['valid_percent']:.1f}% | {center['valid_percent']:.1f}% | "
            f"{center['median']:.3f} | {len(detections[name])} |"
        )
        band_rows.append(
            f"| {name} | " + " | ".join(f"{band['percent_of_valid']:.1f}%" for band in item["bands"]) + " |"
        )
    report = f"""# Sequential ZED / Neuvition depth validation

Date: 2026-10-15 (America/New_York)  
Test mode: **one depth estimator at a time**  
Rig geometry: **ZED camera beside the Neuvition LiDAR**

## Executive summary

This run uses fresh, stationary-scene captures. The ZED and Neuvition drivers were
run separately. ROS StereoBM, ROS StereoSGBM, and FoundationStereo were then run
in isolated passes; no two depth estimators ran together. Quality scores use the
same {len(selected)} timestamp-matched ZED pairs common to every estimator.

The Neuvition capture is a qualitative reference only. Its native depth view and
depth distribution are shown, but no pixelwise camera–LiDAR accuracy metric is
reported because the six-degree-of-freedom extrinsic is unknown and the captures
are sequential.

![Representative depth comparison](plots/depth_overview.png)

## Comparable stereo results

| Method | Valid depth | Photometric residual | Temporal MAD (m) | Outputs / input pairs | Active CPU median | CPU peak | Peak RSS (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

Coverage includes finite depth from 0.3 to 30 m. Photometric residual is the
median absolute grayscale error after warping the right image to the left with
the reported disparity; lower is better. Temporal MAD measures static-scene
per-pixel repeatability and is only computed where at least 70% of selected
frames are valid. These are self-consistency metrics, not ground-truth accuracy.

![Quality metrics](plots/quality_metrics.png)

The practical tradeoff is clear in this scene: StereoBM is the fastest but
sparsest result; SGBM roughly doubles valid coverage and has the best photometric
consistency, at substantially higher compute cost; FoundationStereo is by far the
densest and remains temporally stable, but its worse photometric score shows why
dense learned output should not automatically be treated as verified geometry.

## Speed and resource behavior

![Runtime evidence](plots/runtime_metrics.png)

- ROS StereoBM emitted 143/154 synchronized pairs during a 0.5× quality replay.
- ROS StereoSGBM emitted 133/154 pairs during a 0.25× replay. Its quality results
  are still computed only on frames common to every method.
- FoundationStereo used a normal 1× replay and three ONNX CPU threads. It emitted
  8/154 disparities, its first frame took 1.978 s, and its latest-frame policy
  intentionally replaced queued inputs. Peak RSS was about 5.75 GiB.
- Classical matcher-only latency on 20 cached 1280×720 pairs was 12.679 ms median
  for BM (13.064 ms p95) and 198.724 ms for SGBM (209.439 ms p95). It excludes
  grayscale conversion, ROS transport, and serialization. Foundation's 1.978 s
  figure is an end-to-end node frame, so the latency bars show deployment scale
  rather than a perfectly identical code boundary.
- Both installed `stereo_image_proc` algorithms segfaulted after SIGINT shutdown,
  after their output bags had finalized successfully. This is a lifecycle defect,
  not a quality result.

## CPU and memory traces

Each estimator was replayed separately and sampled every 0.5 seconds. CPU is the
process CPU percentage, where 100% represents one fully occupied logical core.
The table reports the median over active samples (CPU >5%), plus the observed
peak. RSS is resident memory, not virtual address space.

![CPU and memory time series](plots/resource_traces.png)

- StereoBM: {resources['ROS StereoBM']['summary']['active_cpu_median_percent']:.1f}% median active CPU,
  {resources['ROS StereoBM']['summary']['cpu_peak_percent']:.1f}% peak CPU, and
  {resources['ROS StereoBM']['summary']['rss_peak_mib']:.1f} MiB peak RSS.
- StereoSGBM: {resources['ROS StereoSGBM']['summary']['active_cpu_median_percent']:.1f}% median active CPU,
  {resources['ROS StereoSGBM']['summary']['cpu_peak_percent']:.1f}% peak CPU, and
  {resources['ROS StereoSGBM']['summary']['rss_peak_mib']:.1f} MiB peak RSS.
- FoundationStereo (three threads): {resources['FoundationStereo']['summary']['active_cpu_median_percent']:.1f}% median active CPU,
  {resources['FoundationStereo']['summary']['cpu_peak_percent']:.1f}% peak CPU, and
  {resources['FoundationStereo']['summary']['rss_peak_mib']:.1f} MiB peak RSS.

The unaggregated samples are retained as `data/resources_*.csv`.

## LiDAR observation

The LiDAR bag contains {lidar['frames']} native depth frames. Median valid coverage
from 0.3–30 m is {100*lidar['coverage'].get('median', float('nan')):.1f}%, median
valid depth is {lidar['depth'].get('median', float('nan')):.3f} m, and median
static-scene temporal MAD is {lidar['temporal_mad'].get('median', float('nan')):.3f} m.

![Depth distributions](plots/depth_distributions.png)

The distribution plot is a scene-level plausibility check only: the sensor fields
of view, sampling density, viewpoints, and occlusions differ. It must not be read
as pointwise error or calibration evidence.

## Depth values and image areas

The following statistics use valid depth pixels in the 0.3–30 m evaluation range,
aggregated over the six common stereo frames or all 49 LiDAR frames. Raw minimum
and maximum are included as requested, but p1 and p99 are better robust endpoints
because a single mismatched pixel can dominate min/max.

| Source | Min m | p1 m | Mean m | Median m | p95 m | p99 m | Max m |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(depth_rows)}

![Depth statistic summary](plots/depth_statistics.png)

| Source | Full valid | Center 50% valid | Center median m | Connected near regions |
|---|---:|---:|---:|---:|
{chr(10).join(area_rows)}

The 3×3 maps below show how coverage and median depth vary spatially. Neuvition's
grid is in its own native image coordinates and is not registered to the ZED grid.

![Spatial depth statistics](plots/spatial_depth_heatmaps.png)

Distance composition is reported as a percentage of each source's valid pixels:

| Source | 0.3–1 m | 1–2 m | 2–5 m | 5–10 m | 10–30 m |
|---|---:|---:|---:|---:|---:|
{chr(10).join(band_rows)}

## Depth-only near-region detection

![Near-region detections](plots/near_region_detections.png)

Red masks and yellow boxes are connected regions with depth below 2 m after
small morphological cleanup; labels show regional median depth. They can support
proximity or obstacle proposals, but they are not semantic object detections.
Fragmentation and missing depth directly affect the number of detected regions.

## Conclusion

- **Best dense result:** FoundationStereo, at 97.0% valid depth and good static
  repeatability. On this CPU it is an offline/low-rate option: roughly 1.98 s per
  frame, about 2.5 active logical cores, and about 5.7 GiB RSS.
- **Best classical quality/speed compromise:** ROS StereoSGBM. It reached about
  60% coverage and the lowest photometric residual, with a matcher-only rate of
  about 5 frames/s and only about 143 MiB peak RSS.
- **Best low-cost option:** ROS StereoBM. Its matcher is very fast and used less
  than 100 MiB in the sampled run, but its 33% coverage leaves large low-texture
  regions without depth.
- **LiDAR verdict:** the Neuvition native depth is stable and scene-level ranges
  are broadly plausible, but this experiment cannot establish stereo accuracy
  against LiDAR. No method should be declared metrically most accurate until the
  side-by-side sensor extrinsic is calibrated and evaluated on separate data.

## Evidence and limitations

- ZED bag: `{args.stereo_bag}` ({total_pairs} synchronized pairs found)
- Neuvition bag: `{args.lidar_bag}`
- Recorded estimator outputs: `{args.outputs}`
- Common evaluated timestamps: {len(selected)}
- Hardware remained stationary, but the captures were sequential rather than
  hardware-synchronized.
- A calibrated target or measured camera–LiDAR extrinsic is required before MAE,
  RMSE, bias, relative error, or distance-binned LiDAR accuracy is scientifically
  defensible.
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stereo-bag", required=True, type=Path)
    parser.add_argument("--lidar-bag", required=True, type=Path)
    parser.add_argument("--outputs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    plots = args.output / "plots"
    data = args.output / "data"
    plots.mkdir(exist_ok=True)
    data.mkdir(exist_ok=True)

    stamp_sets = {}
    for name, (directory, topic) in METHODS.items():
        stamp_sets[name] = set(read_disparity_stamps(args.outputs / directory, topic))
        print(f"{name}: {len(stamp_sets[name])} disparity messages", flush=True)
    common = set.intersection(*stamp_sets.values())
    selected = choose_evenly(common, args.samples)
    if len(selected) < 3:
        raise RuntimeError(f"Only {len(selected)} common output timestamps")

    left, right, left_info, right_info, raw_pairs = read_stereo(args.stereo_bag, set(selected))
    selected = [key for key in selected if key in left and key in right]
    if len(selected) < 3 or left_info is None or right_info is None:
        raise RuntimeError("Insufficient matched images or missing calibration")

    results = {}
    for name, (directory, topic) in METHODS.items():
        print(f"Evaluating {name}", flush=True)
        outputs = read_disparities(args.outputs / directory, topic, set(selected))
        results[name] = evaluate_method(outputs, left, right, selected)

    lidar_frames = read_lidar_depths(args.lidar_bag)
    lidar = evaluate_lidar(lidar_frames)
    resources = read_resource_traces(data)
    spatial = {name: spatial_depth_metrics(result["depths"]) for name, result in results.items()}
    lidar_clean = [
        np.where(np.isfinite(frame) & (frame >= 0.3) & (frame <= 30.0), frame, np.nan)
        for frame in lidar_frames
    ]
    spatial["Neuvition"] = spatial_depth_metrics(lidar_clean)
    middle = len(selected) // 2
    save_overview(plots / "depth_overview.png", left[selected[middle]], results, middle, lidar)
    save_metrics(plots / "quality_metrics.png", results)
    save_runtime(plots / "runtime_metrics.png")
    save_resource_traces(plots / "resource_traces.png", resources)
    save_ranges(plots / "depth_distributions.png", results, lidar)
    save_depth_statistics(plots / "depth_statistics.png", spatial)
    save_spatial_heatmaps(plots / "spatial_depth_heatmaps.png", spatial)
    representative_depths = {
        name: result["depths"][middle] for name, result in results.items()
    }
    representative_depths["Neuvition"] = lidar["representative"]
    detections = save_near_regions(
        plots / "near_region_detections.png", left[selected[middle]], representative_depths
    )
    write_spatial_csvs(data, spatial, detections)

    rows = []
    summary = {"selected_timestamps_ns": selected, "raw_stereo_pairs": len(raw_pairs), "methods": {}, "lidar": {k: v for k, v in lidar.items() if k not in {"representative", "all_valid"}}, "spatial_depth": spatial, "near_region_detections": detections}
    for name, result in results.items():
        entry = {
            "valid_fraction": stats(result["coverage"]),
            "photometric_median": stats(result["photo_median"]),
            "photometric_mean": stats(result["photo_mean"]),
            "temporal_mad_m": result["temporal_mad"],
            "runtime": RUNTIME[name],
            "resources": resources[name]["summary"],
        }
        summary["methods"][name] = entry
        rows.append({
            "method": name,
            "evaluated_frames": len(selected),
            "valid_percent": 100 * entry["valid_fraction"]["mean"],
            "photometric_median": entry["photometric_median"]["mean"],
            "temporal_mad_m": entry["temporal_mad_m"].get("median", float("nan")),
            "emitted_frames": RUNTIME[name]["output_frames"],
            "input_pairs": RUNTIME[name]["input_pairs"],
            "peak_rss_mib": RUNTIME[name]["peak_rss_mib"],
            "active_cpu_median_percent": resources[name]["summary"]["active_cpu_median_percent"],
            "sampled_cpu_peak_percent": resources[name]["summary"]["cpu_peak_percent"],
            "sampled_rss_peak_mib": resources[name]["summary"]["rss_peak_mib"],
        })
    with (data / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (data / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(
        args.output, args, selected, len(raw_pairs), results, lidar, resources, spatial, detections
    )
    print(f"Wrote {args.output / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
