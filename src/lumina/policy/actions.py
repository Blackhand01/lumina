"""Hybrid KV-cache action definitions for Checkpoint 4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class HybridKVAction:
    """A joint retention/precision action for KV-cache policy search."""

    name: str
    retention: float
    kv_bits: int
    description: str

    def __post_init__(self) -> None:
        if not 0 < self.retention <= 1:
            raise ValueError(f"retention must be in (0, 1], got {self.retention}")
        if self.kv_bits <= 0:
            raise ValueError(f"kv_bits must be positive, got {self.kv_bits}")

    @property
    def uses_retention(self) -> bool:
        return self.retention < 1.0

    @property
    def uses_quantization(self) -> bool:
        return self.kv_bits < 16


DEFAULT_ACTION_GRID: tuple[HybridKVAction, ...] = (
    HybridKVAction("full_16", 1.00, 16, "Full KV cache control."),
    HybridKVAction("full_8", 1.00, 8, "Static MLX q8 KV cache baseline."),
    HybridKVAction("r50_b16", 0.50, 16, "Retention-only rotating cache baseline."),
    HybridKVAction("r75_b8", 0.75, 8, "Conservative hybrid retention plus q8."),
    HybridKVAction("r50_b8", 0.50, 8, "Balanced hybrid retention plus q8."),
    HybridKVAction("r50_b4", 0.50, 4, "Aggressive retention plus 4-bit KV."),
    HybridKVAction("r30_b4", 0.30, 4, "Stress memory budget action."),
)


@dataclass(frozen=True)
class BackendSupport:
    """Backend feasibility declaration for action filtering."""

    quantized_kv_bits: tuple[int, ...] = (8,)
    supports_rotating_cache: bool = True
    supports_quantized_rotating_cache: bool = False
    allow_simulated: bool = False


@dataclass(frozen=True)
class BackendFeasibility:
    feasible: bool
    status: str
    reason: str


def backend_feasibility(action: HybridKVAction, support: BackendSupport) -> BackendFeasibility:
    """Return whether an action is executable by the current backend."""

    if action.retention < 1.0 and not support.supports_rotating_cache:
        return BackendFeasibility(False, "infeasible", "retention requires rotating cache support")

    if action.kv_bits == 16:
        return BackendFeasibility(True, "real", "fp16/bf16 KV path available")

    if action.kv_bits not in support.quantized_kv_bits:
        if support.allow_simulated:
            return BackendFeasibility(True, "simulated", f"kv_bits={action.kv_bits} is simulated")
        return BackendFeasibility(False, "infeasible", f"kv_bits={action.kv_bits} not supported by backend")

    if action.retention < 1.0 and not support.supports_quantized_rotating_cache:
        if support.allow_simulated:
            return BackendFeasibility(True, "simulated", "quantized rotating cache is simulated")
        return BackendFeasibility(False, "infeasible", "RotatingKVCache quantization NYI")

    return BackendFeasibility(True, "real", "backend path available")


def actions_by_name(actions: Iterable[HybridKVAction] = DEFAULT_ACTION_GRID) -> dict[str, HybridKVAction]:
    return {action.name: action for action in actions}

