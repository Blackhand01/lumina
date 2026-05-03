"""Map theoretical KV-cache actions onto backend-feasible actions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from lumina.kv import KVPolicyAction


class FeasibilityStatus(StrEnum):
    REAL = "real"
    BACKEND_INFEASIBLE = "backend_infeasible"
    SIMULATED = "simulated"


@dataclass(frozen=True)
class BackendCapabilities:
    """Declared cache capabilities for an inference backend."""

    name: str
    version: str = "unknown"
    full_precision_bits: int = 16
    quantized_kv_bits: tuple[int, ...] = ()
    supports_retention: bool = False
    supports_quantized_retention: bool = False
    allow_simulated: bool = False


@dataclass(frozen=True)
class FeasibilityResult:
    action: KVPolicyAction
    status: FeasibilityStatus
    reason: str

    @property
    def feasible(self) -> bool:
        return self.status in {FeasibilityStatus.REAL, FeasibilityStatus.SIMULATED}


def classify_action(action: KVPolicyAction, backend: BackendCapabilities) -> FeasibilityResult:
    """Classify whether a KV action is executable by a backend."""

    if action.uses_retention and not backend.supports_retention:
        return _maybe_simulated(action, backend, "retention is not exposed by backend")

    if action.kv_bits == backend.full_precision_bits:
        return FeasibilityResult(action, FeasibilityStatus.REAL, "full-precision KV path available")

    if action.kv_bits not in backend.quantized_kv_bits:
        return _maybe_simulated(action, backend, f"{action.kv_bits}-bit KV is not exposed by backend")

    if action.uses_retention and not backend.supports_quantized_retention:
        return _maybe_simulated(action, backend, "retention and quantized KV are not composable in backend")

    return FeasibilityResult(action, FeasibilityStatus.REAL, "backend path available")


def feasible_matrix(
    actions: tuple[KVPolicyAction, ...],
    backend: BackendCapabilities,
) -> tuple[FeasibilityResult, ...]:
    """Classify a batch of actions against one backend declaration."""

    return tuple(classify_action(action, backend) for action in actions)


def _maybe_simulated(action: KVPolicyAction, backend: BackendCapabilities, reason: str) -> FeasibilityResult:
    if backend.allow_simulated:
        return FeasibilityResult(action, FeasibilityStatus.SIMULATED, reason)
    return FeasibilityResult(action, FeasibilityStatus.BACKEND_INFEASIBLE, reason)
