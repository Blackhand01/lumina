"""Regime-aware hybrid KV-cache controller."""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass
from typing import Iterable

from .actions import BackendSupport, HybridKVAction, backend_feasibility, make_dynamic_retention_action
from .actions import DEFAULT_ACTION_GRID
from .cost import CostBreakdown, CostWeights, ModelKVProfile, precision_memory_multiplier, score_action
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
        enable_dynamic_retention: bool = False,
        dynamic_kv_bits: Iterable[int] = (8, 16),
        dynamic_min_retention: float = 0.30,
        dynamic_budget_margin: float = 0.999,
        replace_static_retention: bool = False,
    ) -> None:
        self.model_profile = model_profile
        base_actions = tuple(actions)
        if replace_static_retention:
            base_actions = tuple(action for action in base_actions if action.retention >= 1.0)
        self.actions = base_actions
        self.weights = weights or CostWeights()
        self.backend_support = backend_support or BackendSupport()
        self.enable_dynamic_retention = enable_dynamic_retention
        self.dynamic_kv_bits = tuple(int(bits) for bits in dynamic_kv_bits)
        self.dynamic_min_retention = dynamic_min_retention
        self.dynamic_budget_margin = dynamic_budget_margin
        self.replace_static_retention = replace_static_retention

    def _dynamic_actions(self, state: PolicyState) -> tuple[HybridKVAction, ...]:
        if not self.enable_dynamic_retention:
            return ()
        if state.budget_uma_mb is None or state.budget_uma_mb <= 0:
            return ()
        if state.estimated_full_kv_mb <= 0:
            return ()

        actions: list[HybridKVAction] = []
        seen: set[str] = set()
        margin = min(max(self.dynamic_budget_margin, 0.0), 1.0)
        for kv_bits in self.dynamic_kv_bits:
            precision_ratio = kv_bits / 16
            denominator = state.estimated_full_kv_mb * precision_ratio * precision_memory_multiplier(kv_bits)
            if denominator <= 0:
                continue
            retention = (state.budget_uma_mb * margin) / denominator
            if retention >= 1.0:
                continue
            retention = max(retention, self.dynamic_min_retention)
            action = make_dynamic_retention_action(retention=retention, kv_bits=kv_bits)
            if action.name not in seen:
                actions.append(action)
                seen.add(action.name)
        return tuple(actions)

    def candidate_actions(self, state: PolicyState) -> tuple[HybridKVAction, ...]:
        actions = self.actions + self._dynamic_actions(state)
        by_name: dict[str, HybridKVAction] = {}
        for action in actions:
            by_name.setdefault(action.name, action)
        return tuple(by_name.values())

    def decide(self, state: PolicyState) -> PolicyDecision:
        if state.estimated_full_kv_mb <= 0:
            state = PolicyState(
                **(asdict(state) | {"estimated_full_kv_mb": self.model_profile.full_kv_cache_mb(state.prompt_tokens)})
            )
        observed_regime = classify_operating_regime(state)
        actions = self.candidate_actions(state)
        costs = tuple(
            score_action(
                action,
                state,
                self.weights,
                backend_feasibility(action, self.backend_support),
            )
            for action in actions
        )
        feasible_costs = [cost for cost in costs if cost.feasible]
        selected_cost = min(feasible_costs, key=lambda cost: cost.total) if feasible_costs else None
        selected_action = None
        if selected_cost is not None:
            selected_action = next(action for action in actions if action.name == selected_cost.action_name)
        return PolicyDecision(
            selected_action=selected_action,
            selected_cost=selected_cost,
            operating_regime_observed=observed_regime,
            state=state,
            costs=costs,
            timestamp=dt.datetime.now(dt.timezone.utc).isoformat(),
        )
