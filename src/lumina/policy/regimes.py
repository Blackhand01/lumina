"""Operating-regime state and classification for Lumina policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


OperatingRegimeName = Literal["under_saturated", "saturation", "contention", "unknown"]

PRESSURE_RANK = {"unknown": 0, "green": 1, "yellow": 2, "red": 3}


@dataclass(frozen=True)
class PolicyState:
    """Observable state used by the Checkpoint 4 controller."""

    prompt_tokens: int
    generated_tokens: int = 0
    estimated_full_kv_mb: float = 0.0
    budget_uma_mb: float | None = None
    memory_pressure: str = "unknown"
    pressure_injection: str = "none"
    swap_delta_mb: float = 0.0
    pageout_delta: float = 0.0
    swapout_delta: float = 0.0
    compressed_memory_delta_mb: float = 0.0
    recent_latency_ms_per_token: float = 1.0
    latency_p95_ms_per_token: float = 1.0
    recent_latency_trend: float = 0.0
    task_profile: str = "generic"
    target_regime: OperatingRegimeName = "unknown"

    @property
    def p95_ratio(self) -> float:
        if self.recent_latency_ms_per_token <= 0:
            return 1.0
        return self.latency_p95_ms_per_token / self.recent_latency_ms_per_token


def pressure_rank(pressure: str) -> int:
    return PRESSURE_RANK.get(pressure, 0)


def classify_operating_regime(state: PolicyState) -> OperatingRegimeName:
    """Classify hardware regime from OS/runtime proxy metrics."""

    injected = state.pressure_injection not in {"", "none", "budget_only"}
    pressure = pressure_rank(state.memory_pressure)
    has_swap = state.swap_delta_mb > 32 or state.swapout_delta > 0
    has_pageout = state.pageout_delta > 0
    p95_spike = state.p95_ratio >= 1.5

    if injected and (pressure >= pressure_rank("yellow") or has_swap or has_pageout or p95_spike):
        return "contention"
    if injected and state.target_regime == "contention":
        return "contention"
    if pressure >= pressure_rank("yellow") or has_swap or has_pageout:
        return "saturation"

    budget = state.budget_uma_mb
    if budget is not None and state.estimated_full_kv_mb > 0:
        if budget <= state.estimated_full_kv_mb * 0.60:
            return "saturation"

    return "under_saturated"

