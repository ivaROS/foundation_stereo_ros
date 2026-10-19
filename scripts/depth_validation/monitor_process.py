#!/usr/bin/env python3

"""Sample one process's CPU and resident memory for benchmark evidence."""

import argparse
import csv
import time
from pathlib import Path

import psutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--duration", type=float, default=20.0)
    args = parser.parse_args()

    process = psutil.Process(args.pid)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    process.cpu_percent(interval=None)
    rows = []
    while time.monotonic() - started < args.duration:
        time.sleep(args.interval)
        try:
            memory = process.memory_info()
            rows.append(
                {
                    "method": args.label,
                    "elapsed_s": time.monotonic() - started,
                    "cpu_percent": process.cpu_percent(interval=None),
                    "rss_mib": memory.rss / (1024 * 1024),
                    "vms_mib": memory.vms / (1024 * 1024),
                    "threads": process.num_threads(),
                }
            )
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            break
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["method", "elapsed_s", "cpu_percent", "rss_mib", "vms_mib", "threads"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} samples to {args.output}")


if __name__ == "__main__":
    main()
