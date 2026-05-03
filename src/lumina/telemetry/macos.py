"""macOS telemetry helpers for Apple Silicon experiments."""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import asdict, dataclass


def _run_text(command: list[str], timeout: int = 10) -> str:
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return (completed.stdout or "") + (completed.stderr or "")


def _decimal(value: str) -> float:
    return float(value.strip().replace(",", "."))


def _size_mib(value: str, unit: str) -> float:
    number = _decimal(value)
    unit = unit.upper()
    if unit.startswith("G"):
        return number * 1024
    if unit.startswith("K"):
        return number / 1024
    return number


def _parse_swapusage(text: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for key in ("total", "used", "free"):
        match = re.search(rf"{key}\s*=\s*([0-9.,]+)([KMG])", text)
        if match:
            values[f"swap_{key}_mib"] = _size_mib(match.group(1), match.group(2))
    return values


def _parse_vm_stat(text: str) -> dict[str, float]:
    page_size_match = re.search(r"page size of\s+([0-9]+)\s+bytes", text)
    page_size = int(page_size_match.group(1)) if page_size_match else 16384
    values: dict[str, float] = {"page_size_bytes": float(page_size)}
    for line in text.splitlines():
        match = re.match(r'([^:"]+|"[^"]+"):\s+([0-9.]+)\.', line.strip())
        if not match:
            continue
        key = match.group(1).strip().strip('"').lower().replace(" ", "_").replace("-", "_")
        pages = _decimal(match.group(2))
        values[f"{key}_pages"] = pages
        values[f"{key}_mib"] = pages * page_size / (1024 * 1024)
    return values


def _memory_pressure() -> tuple[str, float | None]:
    text = _run_text(["memory_pressure"], timeout=30)
    match = re.search(r"System-wide memory free percentage:\s*([0-9.,]+)%", text)
    if not match:
        return "unknown", None
    free_percent = _decimal(match.group(1))
    if free_percent >= 20:
        return "green", free_percent
    if free_percent >= 10:
        return "yellow", free_percent
    return "red", free_percent


def _thermal_state() -> tuple[str, float | None, float | None, float | None]:
    text = _run_text(["pmset", "-g", "therm"], timeout=5).lower()
    if not text:
        return "unknown", None, None, None

    thermal = "unknown"
    level: float | None = None
    match = re.search(r"thermal warning level:\s*([0-9]+)", text)
    if match:
        level = _decimal(match.group(1))
        if level <= 0:
            thermal = "nominal"
        elif level == 1:
            thermal = "fair"
        elif level == 2:
            thermal = "serious"
        else:
            thermal = "critical"
    elif "no thermal warning level has been recorded" in text:
        thermal = "nominal"

    cpu_limit = _speed_limit(text, "cpu")
    gpu_limit = _speed_limit(text, "gpu")
    limits = [value for value in (cpu_limit, gpu_limit) if value is not None]
    if limits:
        minimum = min(limits)
        if minimum <= 40:
            thermal = "critical"
        elif minimum <= 70 and thermal not in {"critical"}:
            thermal = "serious"
        elif minimum < 100 and thermal in {"unknown", "nominal"}:
            thermal = "fair"
    return thermal, level, cpu_limit, gpu_limit


def _speed_limit(text: str, device: str) -> float | None:
    for pattern in (
        rf"{device}[_ ]speed[_ ]limit\s*[:=]\s*([0-9.]+)",
        rf"{device}[_ ]power[_ ]limit\s*[:=]\s*([0-9.]+)",
    ):
        match = re.search(pattern, text)
        if match:
            return _decimal(match.group(1))
    return None


def _rss_mib(pid: int) -> float:
    text = _run_text(["ps", "-o", "rss=", "-p", str(pid)], timeout=5).strip()
    return _decimal(text.splitlines()[0]) / 1024 if text else 0.0


def _physical_memory_mib() -> float:
    text = _run_text(["sysctl", "-n", "hw.memsize"], timeout=5).strip()
    return _decimal(text) / (1024 * 1024) if text else 0.0


@dataclass(frozen=True)
class TelemetrySnapshot:
    timestamp_monotonic: float
    process_rss_mib: float
    memory_pressure: str
    memory_free_percent: float | None
    physical_memory_mib: float
    unified_available_mib: float
    unified_used_mib: float
    swap_used_mib: float
    compressed_mib: float
    compressor_mib: float
    pageouts: float
    swapouts: float
    thermal_state: str
    thermal_level: float | None
    cpu_speed_limit: float | None
    gpu_speed_limit: float | None

    def to_record(self) -> dict[str, float | str | None]:
        return asdict(self)


def snapshot(pid: int) -> TelemetrySnapshot:
    vm = _parse_vm_stat(_run_text(["vm_stat"], timeout=20))
    swap = _parse_swapusage(_run_text(["sysctl", "vm.swapusage"], timeout=10))
    pressure, free_percent = _memory_pressure()
    thermal, thermal_level, cpu_limit, gpu_limit = _thermal_state()
    total_mib = _physical_memory_mib()
    free_mib = vm.get("pages_free_mib", 0.0)
    speculative_mib = vm.get("pages_speculative_mib", 0.0)
    available_mib = free_mib + speculative_mib

    return TelemetrySnapshot(
        timestamp_monotonic=time.monotonic(),
        process_rss_mib=_rss_mib(pid),
        memory_pressure=pressure,
        memory_free_percent=free_percent,
        physical_memory_mib=total_mib,
        unified_available_mib=available_mib,
        unified_used_mib=max(total_mib - available_mib, 0.0) if total_mib else 0.0,
        swap_used_mib=swap.get("swap_used_mib", 0.0),
        compressed_mib=vm.get("pages_stored_in_compressor_mib", 0.0),
        compressor_mib=vm.get("pages_occupied_by_compressor_mib", 0.0),
        pageouts=vm.get("pageouts_pages", 0.0),
        swapouts=vm.get("swapouts_pages", 0.0),
        thermal_state=thermal,
        thermal_level=thermal_level,
        cpu_speed_limit=cpu_limit,
        gpu_speed_limit=gpu_limit,
    )
