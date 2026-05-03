"""Memory-soak helper used to force resource contention during experiments."""

from __future__ import annotations

import time


def allocate_touched_memory(target_mib: int, *, chunk_mib: int = 64, touch_stride: int = 4096) -> list[bytearray]:
    """Allocate and touch memory so the OS commits physical pages."""

    if target_mib <= 0:
        raise ValueError("target_mib must be positive")
    if chunk_mib <= 0:
        raise ValueError("chunk_mib must be positive")

    chunks: list[bytearray] = []
    allocated = 0
    while allocated < target_mib:
        chunk = bytearray(chunk_mib * 1024 * 1024)
        for offset in range(0, len(chunk), touch_stride):
            chunk[offset] = 1
        chunks.append(chunk)
        allocated += chunk_mib
    return chunks


def hold(chunks: list[bytearray], seconds: float) -> None:
    """Keep allocated memory alive for a fixed duration."""

    deadline = time.monotonic() + max(seconds, 0.0)
    while time.monotonic() < deadline:
        time.sleep(min(1.0, deadline - time.monotonic()))
    if not chunks:
        return
