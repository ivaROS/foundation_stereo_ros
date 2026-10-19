#!/usr/bin/env python3

"""Compare a FoundationStereo ONNX export with a recorded reference output."""

import argparse
import json
import math
import time
import warnings
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort

from depth_validation.analyze_sequential import (
    detailed_depth_stats,
    disparity_to_depth as recorded_disparity_to_depth,
    evaluate_method,
    read_disparities,
    read_stereo,
    stats,
)
from depth_validation.benchmark_foundation_cpu import (
    normalize_nchw,
)


def resize_uniform_and_pad(image, target_height, target_width):
    source_height, source_width = image.shape[:2]
    scale = min(target_width / source_width, target_height / source_height)
    scaled_width = max(1, int(round(source_width * scale)))
    scaled_height = max(1, int(round(source_height * scale)))
    resized = cv2.resize(
        image, (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR
    )
    padded = cv2.copyMakeBorder(
        resized,
        0,
        target_height - scaled_height,
        0,
        target_width - scaled_width,
        cv2.BORDER_REPLICATE,
    )
    return padded, scaled_height, scaled_width


def restore_disparity(raw, input_height, input_width, model_height, model_width,
                      scaled_height, scaled_width):
    disparity = np.maximum(raw.reshape(model_height, model_width), 0.0).astype(
        np.float32, copy=False
    )
    cropped = disparity[:scaled_height, :scaled_width]
    restored = cv2.resize(
        cropped, (input_width, input_height), interpolation=cv2.INTER_NEAREST
    )
    return np.ascontiguousarray(
        restored * (float(input_width) / scaled_width), dtype=np.float32
    )


def make_runner(model_path, threads):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    session = ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
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
        right_model, _, _ = resize_uniform_and_pad(
            right_rgb, model_height, model_width
        )
        raw = session.run(
            ["disparity"],
            {
                "left_image": normalize_nchw(left_model),
                "right_image": normalize_nchw(right_model),
            },
        )[0]
        disparity = restore_disparity(
            raw,
            input_height,
            input_width,
            model_height,
            model_width,
            scaled_height,
            scaled_width,
        )
        columns = np.arange(input_width, dtype=np.float32)[None, :]
        disparity[(columns - disparity) < 0.0] = np.inf
        return disparity, time.perf_counter() - started

    return run, [int(model_height), int(model_width)]


def temporal_depth_mad(depths):
    stack = np.stack(depths)
    valid_count = np.sum(np.isfinite(stack), axis=0)
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", category=RuntimeWarning)
        center = np.nanmedian(stack, axis=0)
        mad = np.nanmedian(np.abs(stack - center), axis=0)
    stable = valid_count >= max(3, math.ceil(0.7 * len(depths)))
    return stats(mad[stable & np.isfinite(mad)])


def summarize_evaluation(evaluation):
    return {
        "valid_fraction": stats(evaluation["coverage"]),
        "photometric_median": stats(evaluation["photo_median"]),
        "photometric_mean": stats(evaluation["photo_mean"]),
        "temporal_mad_m": evaluation["temporal_mad"],
        "depth_m": detailed_depth_stats(np.stack(evaluation["depths"])),
    }


def mask_depth_range(outputs, minimum_depth, maximum_depth):
    masked = {}
    for stamp, item in outputs.items():
        disparity = item["disparity"].copy()
        with np.errstate(divide="ignore", invalid="ignore"):
            depth = float(item["f"] * item["t"]) / disparity
        outside = (
            ~np.isfinite(depth)
            | (depth < minimum_depth)
            | (depth > maximum_depth)
        )
        disparity[outside] = np.nan
        masked[stamp] = {**item, "disparity": disparity}
    return masked


def save_plot(path, left, reference_depth, candidate_depth,
              reference_label, candidate_label, minimum_depth, maximum_depth):
    difference = np.abs(candidate_depth - reference_depth)
    figure, axes = plt.subplots(1, 4, figsize=(18, 4.5), constrained_layout=True)
    axes[0].imshow(cv2.cvtColor(left, cv2.COLOR_BGR2RGB))
    axes[0].set_title("ZED left image")
    for axis, data, title in (
        (axes[1], reference_depth, reference_label),
        (axes[2], candidate_depth, candidate_label),
    ):
        image = axis.imshow(
            data, cmap="turbo", vmin=minimum_depth, vmax=maximum_depth
        )
        axis.set_title(title)
        figure.colorbar(image, ax=axis, label="depth (m)", fraction=0.046)
    image = axes[3].imshow(difference, cmap="magma", vmin=0.0, vmax=1.0)
    axes[3].set_title("Absolute depth difference")
    figure.colorbar(image, ax=axes[3], label="difference (m)", fraction=0.046)
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zed-bag", required=True, type=Path)
    parser.add_argument("--reference-bag", required=True, type=Path)
    parser.add_argument("--reference-summary", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--minimum-depth", type=float, default=0.3)
    parser.add_argument("--maximum-depth", type=float, default=20.0)
    parser.add_argument(
        "--reference-topic", default="/foundation_stereo/disparity"
    )
    parser.add_argument(
        "--reference-label", default="320×736, 4 iterations"
    )
    parser.add_argument("--candidate-label")
    args = parser.parse_args()
    if args.minimum_depth <= 0.0 or args.maximum_depth <= args.minimum_depth:
        parser.error("depth range must satisfy 0 < minimum-depth < maximum-depth")

    args.output.mkdir(parents=True, exist_ok=True)
    baseline_summary = json.loads(args.reference_summary.read_text(encoding="utf-8"))
    selected = [int(value) for value in baseline_summary["selected_timestamps_ns"]]
    wanted = set(selected)
    left, right, left_info, right_info, _ = read_stereo(args.zed_bag, wanted)
    missing = wanted - (set(left) & set(right))
    if missing:
        raise RuntimeError(f"missing {len(missing)} selected stereo pairs")
    if left_info is None or right_info is None:
        raise RuntimeError("camera calibration is missing")

    focal = float(left_info.p[0] if left_info.p[0] > 0.0 else left_info.k[0])
    baseline = abs(float(right_info.p[3] / right_info.p[0]))
    reference_outputs = read_disparities(
        args.reference_bag, args.reference_topic, wanted
    )
    if set(reference_outputs) != wanted:
        raise RuntimeError("reference disparity bag is missing selected timestamps")

    runner, model_shape = make_runner(args.model, args.threads)
    candidate_outputs = {}
    latencies = []
    for index, key in enumerate(selected, start=1):
        disparity, elapsed = runner(left[key], right[key])
        latencies.append(elapsed)
        candidate_outputs[key] = {
            "disparity": disparity,
            "f": focal,
            "t": baseline,
            "min": 0.0,
            "max": 192.0,
        }
        print(f"frame {index}/{len(selected)}: {elapsed:.3f} s", flush=True)

    reference_outputs = mask_depth_range(
        reference_outputs, args.minimum_depth, args.maximum_depth
    )
    candidate_outputs = mask_depth_range(
        candidate_outputs, args.minimum_depth, args.maximum_depth
    )
    reference = evaluate_method(reference_outputs, left, right, selected)
    candidate = evaluate_method(candidate_outputs, left, right, selected)
    absolute_depth_difference = []
    relative_depth_difference = []
    signed_depth_difference = []
    for reference_depth, candidate_depth in zip(
        reference["depths"], candidate["depths"]
    ):
        overlap = np.isfinite(reference_depth) & np.isfinite(candidate_depth)
        signed = candidate_depth[overlap] - reference_depth[overlap]
        signed_depth_difference.append(signed)
        absolute_depth_difference.append(np.abs(signed))
        relative_depth_difference.append(np.abs(signed) / reference_depth[overlap])

    result = {
        "candidate_model": str(args.model.resolve()),
        "candidate_model_shape": model_shape,
        "threads": args.threads,
        "frames": len(selected),
        "evaluation_depth_range_m": [
            args.minimum_depth,
            args.maximum_depth,
        ],
        "candidate_latency_seconds": stats(latencies),
        "reference": summarize_evaluation(reference),
        "candidate": summarize_evaluation(candidate),
        "candidate_vs_reference": {
            "absolute_depth_difference_m": detailed_depth_stats(
                np.concatenate(absolute_depth_difference)
            ),
            "signed_depth_difference_m": detailed_depth_stats(
                np.concatenate(signed_depth_difference)
            ),
            "absolute_relative_depth_difference": detailed_depth_stats(
                np.concatenate(relative_depth_difference)
            ),
        },
    }
    (args.output / "quality_comparison.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    representative = len(selected) // 2
    save_plot(
        args.output / "quality_comparison.png",
        left[selected[representative]],
        reference["depths"][representative],
        candidate["depths"][representative],
        args.reference_label,
        args.candidate_label or f"{model_shape[0]}×{model_shape[1]}",
        args.minimum_depth,
        args.maximum_depth,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
