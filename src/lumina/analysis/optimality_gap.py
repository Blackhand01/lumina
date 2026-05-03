"""Backend-induced optimality gap computation."""

from __future__ import annotations

from dataclasses import dataclass

from lumina.controller import PolicyCandidate


@dataclass(frozen=True)
class OptimalityGap:
    theoretical_best: PolicyCandidate
    feasible_best: PolicyCandidate

    @property
    def value(self) -> float:
        return max(self.theoretical_best.quality_score - self.feasible_best.quality_score, 0.0)


def compute_optimality_gap(
    theoretical_candidates: tuple[PolicyCandidate, ...],
    feasible_candidates: tuple[PolicyCandidate, ...],
) -> OptimalityGap | None:
    """Compute the score gap between theoretical and backend-feasible optima."""

    if not theoretical_candidates or not feasible_candidates:
        return None
    theoretical_best = max(theoretical_candidates, key=lambda candidate: candidate.quality_score)
    real_candidates = [candidate for candidate in feasible_candidates if candidate.admissible]
    if not real_candidates:
        return None
    feasible_best = max(real_candidates, key=lambda candidate: candidate.quality_score)
    return OptimalityGap(theoretical_best=theoretical_best, feasible_best=feasible_best)
