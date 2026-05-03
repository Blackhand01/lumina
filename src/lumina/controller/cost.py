"""Runtime cost model for feasibility-constrained policy selection."""

from __future__ import annotations

from dataclasses import dataclass


PRESSURE_RANK = {"unknown": 0, "green": 1, "yellow": 2, "red": 3}
THERMAL_RANK = {"unknown": 0, "nominal": 1, "fair": 2, "serious": 3, "critical": 4}


@dataclass(frozen=True)
class SystemState:
    memory_pressure: str = "unknown"
    thermal_state: str = "unknown"
    swap_delta_mib: float = 0.0
    pageout_delta: float = 0.0
    latency_p95_ratio: float = 1.0
    available_memory_mib: float | None = None


@dataclass(frozen=True)
class RuntimeCostWeights:
    memory: float = 1.0
    swap: float = 4.0
    thermal: float = 2.0
    latency_tail: float = 1.5


def runtime_cost(
    *,
    estimated_cache_mib: float,
    state: SystemState,
    weights: RuntimeCostWeights = RuntimeCostWeights(),
) -> float:
    """Estimate normalized system cost for one candidate policy."""

    memory_ratio = _memory_ratio(estimated_cache_mib, state.available_memory_mib)
    swap_penalty = max(state.swap_delta_mib, 0.0) / 256.0 + max(state.pageout_delta, 0.0) / 1000.0
    pressure_penalty = max(PRESSURE_RANK.get(state.memory_pressure, 0) - PRESSURE_RANK["green"], 0) * 0.5
    thermal_penalty = max(THERMAL_RANK.get(state.thermal_state, 0) - THERMAL_RANK["nominal"], 0) * 0.5
    latency_penalty = max(state.latency_p95_ratio - 1.0, 0.0)

    return (
        weights.memory * memory_ratio
        + weights.swap * (swap_penalty + pressure_penalty)
        + weights.thermal * thermal_penalty
        + weights.latency_tail * latency_penalty
    )


def _memory_ratio(estimated_cache_mib: float, available_memory_mib: float | None) -> float:
    if available_memory_mib is None or available_memory_mib <= 0:
        return 0.0
    ratio = estimated_cache_mib / available_memory_mib
    if ratio <= 1.0:
        return ratio
    return 1.0 + (ratio - 1.0) ** 2
