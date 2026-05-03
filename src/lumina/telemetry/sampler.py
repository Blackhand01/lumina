"""Periodic telemetry sampling."""

from __future__ import annotations

import threading

from .macos import TelemetrySnapshot, snapshot


class TelemetrySampler:
    """Collect telemetry snapshots in a background thread."""

    def __init__(self, pid: int, interval_seconds: float = 0.25) -> None:
        self.pid = pid
        self.interval_seconds = interval_seconds
        self.samples: list[TelemetrySnapshot] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> "TelemetrySampler":
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.append(snapshot(self.pid))
            self._stop.wait(self.interval_seconds)

    def delta(self, field: str) -> float:
        if len(self.samples) < 2:
            return 0.0
        return float(getattr(self.samples[-1], field)) - float(getattr(self.samples[0], field))
