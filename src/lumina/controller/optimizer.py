"""Policy selection over backend-feasible candidates."""

from __future__ import annotations

from dataclasses import dataclass

from lumina.backend import FeasibilityStatus
from lumina.kv import KVPolicyAction


@dataclass(frozen=True)
class PolicyCandidate:
    action: KVPolicyAction
    feasibility: FeasibilityStatus
    quality_score: float
    system_cost: float
    memory_mib: float
    reason: str = ""

    @property
    def admissible(self) -> bool:
        return self.feasibility == FeasibilityStatus.REAL


def select_policy(candidates: tuple[PolicyCandidate, ...], *, max_cost: float) -> PolicyCandidate | None:
    """Select the highest-quality real policy under a runtime cost threshold."""

    admissible = [candidate for candidate in candidates if candidate.admissible and candidate.system_cost <= max_cost]
    if not admissible:
        return None
    return max(admissible, key=lambda candidate: (candidate.quality_score, -candidate.system_cost, -candidate.memory_mib))
