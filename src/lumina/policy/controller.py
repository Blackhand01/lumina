"""Regime-aware hybrid KV-cache controller."""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass
from typing import Iterable

from .actions import BackendSupport, HybridKVAction, backend_feasibility
from .actions import DEFAULT_ACTION_GRID
from .cost import CostBreakdown, CostWeights, ModelKVProfile, score_action
from .regimes import PolicyState, classify_operating_regime


@dataclass(frozen=True)
class PolicyDecision:
    selected_action: HybridKVAction | None
    selected_cost: CostBreakdown | None
    operating_regime_observed: str
    state: PolicyState
    costs: tuple[CostBreakdown, ...]
    timestamp: str

    @property
    def selected_action_name(self) -> str | None:
        return None if self.selected_action is None else self.selected_action.name

    def to_record(self) -> dict[str, object]:
        selected_cost = None if self.selected_cost is None else self.selected_cost.to_record()
        return {
            "timestamp": self.timestamp,
            "selected_action": self.selected_action_name,
            "selected_cost": selected_cost,
            "operating_regime_target": self.state.target_regime,
            "operating_regime_observed": self.operating_regime_observed,
            "state": asdict(self.state),
            "costs": [cost.to_record() for cost in self.costs],
        }


class HybridPolicyController:
    """Evaluate a grid of hybrid KV actions and choose the minimum-cost action."""

    def __init__(
        self,
        *,
        model_profile: ModelKVProfile,
        actions: Iterable[HybridKVAction] = DEFAULT_ACTION_GRID,
        weights: CostWeights | None = None,
        backend_support: BackendSupport | None = None,
    ) -> None:
        self.model_profile = model_profile
        self.actions = tuple(actions)
        self.weights = weights or CostWeights()
        self.backend_support = backend_support or BackendSupport()

    def decide(self, state: PolicyState) -> PolicyDecision:
        if state.estimated_full_kv_mb <= 0:
            state = PolicyState(
                **(asdict(state) | {"estimated_full_kv_mb": self.model_profile.full_kv_cache_mb(state.prompt_tokens)})
            )
        observed_regime = classify_operating_regime(state)
        costs = tuple(
            score_action(
                action,
                state,
                self.weights,
                backend_feasibility(action, self.backend_support),
            )
            for action in self.actions
        )
        feasible_costs = [cost for cost in costs if cost.feasible]
        selected_cost = min(feasible_costs, key=lambda cost: cost.total) if feasible_costs else None
        selected_action = None
        if selected_cost is not None:
            selected_action = next(action for action in self.actions if action.name == selected_cost.action_name)
        return PolicyDecision(
            selected_action=selected_action,
            selected_cost=selected_cost,
            operating_regime_observed=observed_regime,
            state=state,
            costs=costs,
            timestamp=dt.datetime.now(dt.timezone.utc).isoformat(),
        )

