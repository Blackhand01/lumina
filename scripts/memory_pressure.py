#!/usr/bin/env python3
"""Inject controlled memory pressure for Lumina regime tests."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.telemetry import snapshot


def allocate_pressure(target_mb: int, chunk_mb: int, touch_stride: int) -> list[bytearray]:
    chunks: list[bytearray] = []
    bytes_per_chunk = chunk_mb * 1024 * 1024
    allocated = 0
    while allocated < target_mb:
        chunk = bytearray(bytes_per_chunk)
        for offset in range(0, len(chunk), touch_stride):
            chunk[offset] = 1
        chunks.append(chunk)
        allocated += chunk_mb
        print(f"allocated_mb={allocated}", flush=True)
    return chunks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mb", type=int, default=512, help="Memory pressure target in MB.")
    parser.add_argument("--seconds", type=float, default=30.0, help="How long to hold memory.")
    parser.add_argument("--chunk-mb", type=int, default=64)
    parser.add_argument("--touch-stride", type=int, default=4096)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.mb <= 0:
        parser.error("--mb must be positive")
    if args.seconds < 0:
        parser.error("--seconds must be non-negative")
    if args.chunk_mb <= 0:
        parser.error("--chunk-mb must be positive")

    start = snapshot(os.getpid())
    plan = {
        "target_mb": args.mb,
        "seconds": args.seconds,
        "chunk_mb": args.chunk_mb,
        "start_pressure": start.pressure,
        "start_swap_mb": round(start.swap_used_mb, 2),
        "start_process_rss_mb": round(start.process_rss_mb, 2),
    }
    print(json.dumps(plan, sort_keys=True))
    if args.dry_run:
        return 0

    chunks = allocate_pressure(args.mb, args.chunk_mb, args.touch_stride)
    deadline = time.monotonic() + args.seconds
    while time.monotonic() < deadline:
        time.sleep(min(1.0, max(deadline - time.monotonic(), 0.0)))

    end = snapshot(os.getpid())
    print(
        json.dumps(
            {
                "held_chunks": len(chunks),
                "end_pressure": end.pressure,
                "end_swap_mb": round(end.swap_used_mb, 2),
                "end_process_rss_mb": round(end.process_rss_mb, 2),
                "swap_delta_mb": round(end.swap_used_mb - start.swap_used_mb, 2),
                "pageout_delta": round(end.pageouts - start.pageouts, 2),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

