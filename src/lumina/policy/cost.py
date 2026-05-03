"""Cost model for regime-aware hybrid KV-cache policy selection."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .actions import BackendFeasibility, HybridKVAction
from .regimes import PolicyState, pressure_rank


MB = 1024 * 1024


@dataclass(frozen=True)
class ModelKVProfile:
    """Model metadata required to estimate KV cache size."""

    num_hidden_layers: int
    num_key_value_heads: int
    head_dim: int
    dtype_bits: int = 16

    @classmethod
    def from_config(cls, config_path: Path) -> "ModelKVProfile":
        data = json.loads(config_path.read_text(encoding="utf-8"))
        hidden_size = int(data["hidden_size"])
        attention_heads = int(data["num_attention_heads"])
        head_dim = int(data.get("head_dim") or hidden_size // attention_heads)
        return cls(
            num_hidden_layers=int(data["num_hidden_layers"]),
            num_key_value_heads=int(data.get("num_key_value_heads", attention_heads)),
            head_dim=head_dim,
            dtype_bits=16,
        )

    def full_kv_cache_mb(self, tokens: int) -> float:
        bytes_per_token = (
            self.num_hidden_layers
            * self.num_key_value_heads
            * self.head_dim
            * 2
            * (self.dtype_bits / 8)
        )
        return tokens * bytes_per_token / MB


@dataclass(frozen=True)
class CostWeights:
    alpha_quality: float = 4.0
    beta_latency: float = 1.0
    gamma_memory: float = 2.0
    delta_swap_risk: float = 3.0
    epsilon_bandwidth_pressure: float = 3.0


WEIGHT_PROFILES: dict[str, CostWeights] = {
    "quality_first": CostWeights(alpha_quality=6.0, beta_latency=1.0, gamma_memory=1.5, delta_swap_risk=2.0, epsilon_bandwidth_pressure=2.0),
    "conservative": CostWeights(alpha_quality=6.0, beta_latency=1.0, gamma_memory=1.5, delta_swap_risk=2.0, epsilon_bandwidth_pressure=2.0),
    "balanced": CostWeights(),
    "memory_first": CostWeights(alpha_quality=2.0, beta_latency=1.0, gamma_memory=4.0, delta_swap_risk=5.0, epsilon_bandwidth_pressure=4.0),
    "stress": CostWeights(alpha_quality=1.0, beta_latency=1.0, gamma_memory=6.0, delta_swap_risk=6.0, epsilon_bandwidth_pressure=5.0),
}


@dataclass(frozen=True)
class CostBreakdown:
    action_name: str
    feasible: bool
    backend_status: str
    reason: str
    estimated_kv_cache_mb: float
    memory_budget_mb: float | None
    quality_loss: float
    latency: float
    memory_usage: float
    swap_risk: float
    bandwidth_pressure: float
    total: float

    def to_record(self) -> dict[str, float | str | bool | None]:
        record = asdict(self)
        record["estimated_kv_cache_mb"] = round(self.estimated_kv_cache_mb, 3)
        record["quality_loss"] = round(self.quality_loss, 6)
        record["latency"] = round(self.latency, 6)
        record["memory_usage"] = round(self.memory_usage, 6)
        record["swap_risk"] = round(self.swap_risk, 6)
        record["bandwidth_pressure"] = round(self.bandwidth_pressure, 6)
        record["total"] = round(self.total, 6) if self.total != float("inf") else "inf"
        return record


def estimate_action_kv_mb(action: HybridKVAction, full_kv_mb: float) -> float:
    """Estimate action KV memory from full FP16 KV size."""

    precision_ratio = action.kv_bits / 16
    metadata_multiplier = precision_memory_multiplier(action.kv_bits)
    metadata_overhead_mb = 0.0
    return full_kv_mb * action.retention * precision_ratio * metadata_multiplier + metadata_overhead_mb


def precision_memory_multiplier(kv_bits: int) -> float:
    """Return empirical cache overhead multiplier for a KV precision."""

    if kv_bits < 16:
        # Checkpoint 3 observed q8 at about 53.1% of full KV, not exact 50%.
        return 1.062
    return 1.0


def quality_loss_prior(action: HybridKVAction, state: PolicyState) -> float:
    if action.retention >= 1.0 and action.kv_bits >= 8:
        return 0.0

    retention_loss = 0.0
    if action.retention < 1.0:
        r = action.retention
        if r >= 0.75:
            retention_loss = 0.12 * ((1.0 - r) / 0.25)
        elif r >= 0.50:
            retention_loss = 0.12 + (0.35 - 0.12) * ((0.75 - r) / 0.25)
        else:
            retention_loss = 0.35 + (0.65 - 0.35) * ((0.50 - r) / 0.20)
        retention_loss = min(max(retention_loss, 0.0), 1.0)

    precision_loss = 0.0
    if action.kv_bits < 8:
        precision_loss = 0.10 if action.kv_bits == 4 else 0.25

    retrieval_multiplier = 1.25 if state.task_profile in {"needle", "document_qa", "retrieval"} else 1.0
    return min((retention_loss + precision_loss) * retrieval_multiplier, 1.0)


def latency_prior(action: HybridKVAction) -> float:
    latency = 1.0
    if action.retention < 1.0:
        latency -= (1.0 - action.retention) * 0.10
    if action.kv_bits < 16:
        latency += 0.02
    if action.kv_bits < 8:
        latency += 0.08
    return max(latency, 0.70)


def normalized_memory_usage(estimated_mb: float, full_kv_mb: float, budget_mb: float | None) -> float:
    denominator = budget_mb if budget_mb and budget_mb > 0 else full_kv_mb
    if denominator <= 0:
        return 0.0
    return estimated_mb / denominator


def swap_risk(state: PolicyState, estimated_mb: float) -> float:
    pressure_component = max(pressure_rank(state.memory_pressure) - pressure_rank("green"), 0) * 0.40
    swap_component = min(max(state.swap_delta_mb, 0.0) / 256.0, 2.0)
    pageout_component = min(max(state.pageout_delta, 0.0) / 1000.0, 2.0)
    budget_component = 0.0
    if state.budget_uma_mb and state.budget_uma_mb > 0:
        budget_ratio = estimated_mb / state.budget_uma_mb
        budget_component = max(budget_ratio - 0.75, 0.0) / 0.25
    return pressure_component + swap_component + pageout_component + budget_component


def bandwidth_pressure(state: PolicyState, estimated_mb: float) -> float:
    p95_component = max(state.p95_ratio - 1.0, 0.0)
    trend_component = max(state.recent_latency_trend, 0.0)
    compressed_component = min(max(state.compressed_memory_delta_mb, 0.0) / 512.0, 2.0)
    swapout_component = min(max(state.swapout_delta, 0.0) / 1000.0, 2.0)
    memory_ratio = normalized_memory_usage(estimated_mb, state.estimated_full_kv_mb, state.budget_uma_mb)
    return (p95_component + trend_component + compressed_component + swapout_component) * max(memory_ratio, 0.1)


def score_action(
    action: HybridKVAction,
    state: PolicyState,
    weights: CostWeights,
    feasibility: BackendFeasibility,
) -> CostBreakdown:
    full_kv_mb = state.estimated_full_kv_mb
    estimated_mb = estimate_action_kv_mb(action, full_kv_mb)
    memory_budget = state.budget_uma_mb

    if not feasibility.feasible:
        return CostBreakdown(
            action.name,
            False,
            feasibility.status,
            feasibility.reason,
            estimated_mb,
            memory_budget,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            float("inf"),
        )

    if memory_budget is not None and estimated_mb > memory_budget:
        return CostBreakdown(
            action.name,
            False,
            feasibility.status,
            f"estimated KV {estimated_mb:.2f} MB exceeds budget {memory_budget:.2f} MB",
            estimated_mb,
            memory_budget,
            0.0,
            0.0,
            normalized_memory_usage(estimated_mb, full_kv_mb, memory_budget),
            0.0,
            0.0,
            float("inf"),
        )

    quality = quality_loss_prior(action, state)
    latency = latency_prior(action)
    memory = normalized_memory_usage(estimated_mb, full_kv_mb, memory_budget)
    swap = swap_risk(state, estimated_mb)
    bandwidth = bandwidth_pressure(state, estimated_mb)
    total = (
        weights.alpha_quality * quality
        + weights.beta_latency * latency
        + weights.gamma_memory * memory
        + weights.delta_swap_risk * swap
        + weights.epsilon_bandwidth_pressure * bandwidth
    )
    return CostBreakdown(
        action.name,
        True,
        feasibility.status,
        feasibility.reason,
        estimated_mb,
        memory_budget,
        quality,
        latency,
        memory,
        swap,
        bandwidth,
        total,
    )
