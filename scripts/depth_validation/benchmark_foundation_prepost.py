#!/usr/bin/env python3

"""Benchmark FoundationStereo preprocessing and post-processing without inference."""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from depth_validation.benchmark_foundation_cpu import read_pair


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


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


def normalize_nchw(image):
    normalized = (image.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
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


def summarize(values):
    values = np.asarray(values, dtype=np.float64) * 1000.0
    return {
        "mean_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "minimum_ms": float(np.min(values)),
        "maximum_ms": float(np.max(values)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True, type=Path)
    parser.add_argument("--height", required=True, type=int)
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--pair-index", type=int, default=15)
    parser.add_argument("--focal-px", type=float, default=557.665631247)
    parser.add_argument("--baseline-m", type=float, default=0.119985001)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    _, left_rgb, right_rgb = read_pair(args.bag, args.pair_index)
    input_height, input_width = left_rgb.shape[:2]
    scale = min(args.width / input_width, args.height / input_height)
    scaled_width = max(1, int(round(input_width * scale)))
    scaled_height = max(1, int(round(input_height * scale)))
    raw_disparity = np.linspace(
        1.0, 63.0, args.height * args.width, dtype=np.float32
    ).reshape(args.height, args.width)

    preprocess = []
    postprocess = []
    for index in range(args.warmup + args.runs):
        started = time.perf_counter()
        left_model, _, _ = resize_uniform_and_pad(
            left_rgb, args.height, args.width
        )
        right_model, _, _ = resize_uniform_and_pad(
            right_rgb, args.height, args.width
        )
        left_tensor = normalize_nchw(left_model)
        right_tensor = normalize_nchw(right_model)
        prepared = time.perf_counter()

        cropped = raw_disparity[:scaled_height, :scaled_width]
        disparity = nearest_resize_cpp(cropped, input_height, input_width)
        disparity = np.ascontiguousarray(
            disparity * (float(input_width) / scaled_width), dtype=np.float32
        )
        columns = np.arange(input_width, dtype=np.float32)[None, :]
        disparity[(columns - disparity) < 0.0] = np.inf
        with np.errstate(divide="ignore", invalid="ignore"):
            depth = np.float32(args.focal_px * args.baseline_m) / disparity
        depth[(depth < 0.3) | (depth > 10.0)] = np.nan
        finished = time.perf_counter()
        if index >= args.warmup:
            preprocess.append(prepared - started)
            postprocess.append(finished - prepared)
        if left_tensor.shape != right_tensor.shape or depth.shape != (720, 1280):
            raise RuntimeError("unexpected benchmark output shape")

    cv2.ocl.setUseOpenCL(True)
    result = {
        "input_shape": list(left_rgb.shape),
        "model_shape": [args.height, args.width],
        "runs": args.runs,
        "cpu": {
            "preprocess_stereo_pair": summarize(preprocess),
            "postprocess_disparity_and_depth": summarize(postprocess),
            "combined_median_ms": float(
                1000.0 * (np.median(preprocess) + np.median(postprocess))
            ),
        },
        "opencv_opencl": {
            "compiled": "OpenCL:                        YES"
            in cv2.getBuildInformation(),
            "runtime_available": bool(cv2.ocl.haveOpenCL()),
            "enabled": bool(cv2.ocl.useOpenCL()),
        },
        "opencv_cuda_devices": int(cv2.cuda.getCudaEnabledDeviceCount()),
    }
    rendered = json.dumps(result, indent=2)
    print(rendered, flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
