"""macOS memory telemetry for Apple Silicon experiments."""

from __future__ import annotations

import re
import subprocess
import threading
import time
from dataclasses import asdict, dataclass


PRESSURE_RANK = {"unknown": 0, "green": 1, "yellow": 2, "red": 3}


def run_text(command: list[str], timeout: int = 20) -> str:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return (completed.stdout or "") + (completed.stderr or "")


def parse_decimal(value: str) -> float:
    return float(value.strip().replace(",", "."))


def parse_size_to_mb(value: str, unit: str) -> float:
    number = parse_decimal(value)
    unit = unit.upper()
    if unit.startswith("G"):
        return number * 1024
    if unit.startswith("K"):
        return number / 1024
    return number


def parse_swapusage(text: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for key in ("total", "used", "free"):
        match = re.search(rf"{key}\s*=\s*([0-9.,]+)([KMG])", text)
        if match:
            values[f"swap_{key}_mb"] = parse_size_to_mb(match.group(1), match.group(2))
    return values


def parse_vm_stat(text: str) -> dict[str, float]:
    page_size_match = re.search(r"page size of\s+([0-9]+)\s+bytes", text)
    page_size = int(page_size_match.group(1)) if page_size_match else 16384
    pages: dict[str, float] = {}

    for line in text.splitlines():
        match = re.match(r'([^:"]+|"[^"]+"):\s+([0-9.]+)\.', line.strip())
        if not match:
            continue
        key = match.group(1).strip().strip('"').lower().replace(" ", "_").replace("-", "_")
        pages[key] = parse_decimal(match.group(2))

    result: dict[str, float] = {"vm_page_size_bytes": float(page_size)}
    for key, value in pages.items():
        result[f"vm_{key}_pages"] = value
        result[f"vm_{key}_mb"] = value * page_size / (1024 * 1024)
    return result


def pressure_from_free_percent(free_percent: float | None) -> str:
    if free_percent is None:
        return "unknown"
    if free_percent >= 20:
        return "green"
    if free_percent >= 10:
        return "yellow"
    return "red"


def memory_pressure() -> tuple[str, float | None]:
    text = run_text(["memory_pressure"], timeout=30)
    match = re.search(r"System-wide memory free percentage:\s*([0-9.,]+)%", text)
    if not match:
        return "unknown", None
    free_percent = parse_decimal(match.group(1))
    return pressure_from_free_percent(free_percent), free_percent


def rss_mb(pid: int) -> float:
    text = run_text(["ps", "-o", "rss=", "-p", str(pid)], timeout=5).strip()
    if not text:
        return 0.0
    return parse_decimal(text.splitlines()[0].strip()) / 1024


def hw_memsize_mb() -> float:
    text = run_text(["sysctl", "-n", "hw.memsize"], timeout=5).strip()
    if not text:
        return 0.0
    return parse_decimal(text) / (1024 * 1024)


@dataclass
class MemorySnapshot:
    timestamp_monotonic: float
    process_rss_mb: float
    pressure: str
    pressure_free_percent: float | None
    swap_total_mb: float
    swap_used_mb: float
    swap_free_mb: float
    unified_total_mb: float
    unified_available_mb: float
    unified_used_mb: float
    compressor_mb: float
    compressed_mb: float
    wired_mb: float
    active_mb: float
    inactive_mb: float
    speculative_mb: float
    free_mb: float
    pageins: float
    pageouts: float
    swapins: float
    swapouts: float

    def to_record(self, prefix: str) -> dict[str, float | str | None]:
        return {f"{prefix}_{key}": value for key, value in asdict(self).items()}


def snapshot(pid: int) -> MemorySnapshot:
    vm = parse_vm_stat(run_text(["vm_stat"], timeout=20))
    swap = parse_swapusage(run_text(["sysctl", "vm.swapusage"], timeout=10))
    pressure, free_percent = memory_pressure()
    total_mb = hw_memsize_mb()

    free_mb = vm.get("vm_pages_free_mb", 0.0)
    speculative_mb = vm.get("vm_pages_speculative_mb", 0.0)
    available_mb = free_mb + speculative_mb
    used_mb = max(total_mb - available_mb, 0.0) if total_mb else 0.0

    return MemorySnapshot(
        timestamp_monotonic=time.monotonic(),
        process_rss_mb=rss_mb(pid),
        pressure=pressure,
        pressure_free_percent=free_percent,
        swap_total_mb=swap.get("swap_total_mb", 0.0),
        swap_used_mb=swap.get("swap_used_mb", 0.0),
        swap_free_mb=swap.get("swap_free_mb", 0.0),
        unified_total_mb=total_mb,
        unified_available_mb=available_mb,
        unified_used_mb=used_mb,
        compressor_mb=vm.get("vm_pages_occupied_by_compressor_mb", 0.0),
        compressed_mb=vm.get("vm_pages_stored_in_compressor_mb", 0.0),
        wired_mb=vm.get("vm_pages_wired_down_mb", 0.0),
        active_mb=vm.get("vm_pages_active_mb", 0.0),
        inactive_mb=vm.get("vm_pages_inactive_mb", 0.0),
        speculative_mb=speculative_mb,
        free_mb=free_mb,
        pageins=vm.get("vm_pageins_pages", 0.0),
        pageouts=vm.get("vm_pageouts_pages", 0.0),
        swapins=vm.get("vm_swapins_pages", 0.0),
        swapouts=vm.get("vm_swapouts_pages", 0.0),
    )


def worst_pressure(left: str, right: str) -> str:
    return left if PRESSURE_RANK.get(left, 0) >= PRESSURE_RANK.get(right, 0) else right


class MemorySampler:
    def __init__(self, pid: int, interval_seconds: float = 0.25) -> None:
        self.pid = pid
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self.samples: list[MemorySnapshot] = []
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def __enter__(self) -> "MemorySampler":
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def _sample(self) -> None:
        while not self._stop.is_set():
            self.samples.append(snapshot(self.pid))
            self._stop.wait(self.interval_seconds)

    def peak(self, field: str) -> float:
        values = [float(getattr(sample, field)) for sample in self.samples]
        return max(values) if values else float(getattr(snapshot(self.pid), field))

    @property
    def peak_rss_mb(self) -> float:
        return self.peak("process_rss_mb")

    @property
    def peak_unified_used_mb(self) -> float:
        return self.peak("unified_used_mb")

    @property
    def peak_swap_used_mb(self) -> float:
        return self.peak("swap_used_mb")

    @property
    def peak_compressor_mb(self) -> float:
        return self.peak("compressor_mb")
