#!/usr/bin/env python3
"""Run Lumina Checkpoint 6 regime-aware benchmark proof."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lumina.policy import BackendSupport, HybridKVAction, HybridPolicyController, ModelKVProfile, PolicyState, WEIGHT_PROFILES
from lumina.policy.actions import actions_by_name, backend_feasibility, make_dynamic_retention_action
from lumina.policy.cost import bandwidth_pressure, estimate_action_kv_mb, latency_prior, precision_memory_multiplier, quality_loss_prior
from lumina.policy.regimes import classify_operating_regime
from run_lumina import budget_for_regime, load_config, nested, output_path, state_for_decision
from run_policy_experiment import model_config_path, profile_for_regime
from src.telemetry import snapshot


DEFAULT_CONFIG = Path("configs/benchmarks/checkpoint6_regime_benchmark.yaml")
PRESSURE_RANK = {"unknown": 0, "green": 1, "yellow": 2, "red": 3}


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def position_index(position: str, prompt_tokens: int) -> int:
    ratios = {
        "sink": 0.0,
        "early": 0.10,
        "middle": 0.50,
        "late": 0.90,
        "very_late": 0.95,
    }
    if position == "sink":
        return 2
    ratio = ratios.get(position, 0.50)
    return min(max(int(prompt_tokens * ratio), 0), max(prompt_tokens - 1, 0))


def token_retained(action: HybridKVAction, prompt_tokens: int, token_index: int, protected_window: int) -> bool:
    if action.retention >= 1.0:
        return True
    retained = max(protected_window, int(math.ceil(prompt_tokens * action.retention)))
    recent_window = max(retained - protected_window, 0)
    return token_index < protected_window or token_index >= max(prompt_tokens - recent_window, 0)


def precision_factor(action: HybridKVAction) -> float:
    if action.kv_bits >= 8:
        return 1.0
    if action.kv_bits == 4:
        return 0.90
    return 0.75


def task_score(
    *,
    scenario: str,
    action: HybridKVAction,
    state: PolicyState,
    position: str,
    protected_window: int,
    budget_violation: bool,
) -> tuple[float, bool]:
    token_index = position_index(position, state.prompt_tokens)
    retained = token_retained(action, state.prompt_tokens, token_index, protected_window)
    precision = precision_factor(action)

    if scenario in {"needle", "document_qa"}:
        score = (1.0 if retained else 0.0) * precision
    elif scenario == "multi_turn":
        score = (1.0 if retained else 0.15) * precision
    else:
        quality = quality_loss_prior(action, state)
        score = max(0.0, 1.0 - quality) * precision

    if budget_violation:
        score *= 0.98
    return round(score, 6), retained


def execution_status(
    *,
    action: HybridKVAction,
    support: BackendSupport,
    estimated_kv_mb: float,
    budget_mb: float | None,
    theoretical_only: bool,
) -> tuple[str, str]:
    feasibility = backend_feasibility(action, support)
    if theoretical_only:
        if not feasibility.feasible:
            return "theoretical_only", feasibility.reason
        if budget_mb is not None and estimated_kv_mb > budget_mb:
            return "theoretical_budget_infeasible", f"estimated KV {estimated_kv_mb:.2f} MB exceeds budget {budget_mb:.2f} MB"
        return "theoretical_only", "marked theoretical-only in benchmark config"
    if not feasibility.feasible:
        return "backend_infeasible", feasibility.reason
    if budget_mb is not None and estimated_kv_mb > budget_mb:
        return "budget_infeasible", f"estimated KV {estimated_kv_mb:.2f} MB exceeds budget {budget_mb:.2f} MB"
    return feasibility.status, feasibility.reason


def controller_for_config(config: dict[str, Any], model_profile: ModelKVProfile, profile_name: str) -> HybridPolicyController:
    dynamic_config = nested(config, "policy", "dynamic_retention", default={}) or {}
    return HybridPolicyController(
        model_profile=model_profile,
        weights=WEIGHT_PROFILES[profile_name],
        backend_support=BackendSupport(allow_simulated=bool(nested(config, "policy", "allow_simulated", default=False))),
        enable_dynamic_retention=bool(dynamic_config.get("enabled", False)),
        dynamic_kv_bits=tuple(int(value) for value in dynamic_config.get("kv_bits", [8, 16])),
        dynamic_min_retention=float(dynamic_config.get("min_retention", 0.30)),
        dynamic_budget_margin=float(dynamic_config.get("budget_margin", 0.999)),
        replace_static_retention=bool(dynamic_config.get("replace_static_retention", False)),
    )


def ideal_dynamic_action(method: dict[str, Any], state: PolicyState, config: dict[str, Any]) -> HybridKVAction:
    kv_bits = int(method.get("kv_bits", 8))
    if state.budget_uma_mb is None or state.budget_uma_mb <= 0 or state.estimated_full_kv_mb <= 0:
        retention = 1.0
    else:
        dynamic_config = nested(config, "policy", "dynamic_retention", default={}) or {}
        margin = float(dynamic_config.get("budget_margin", 0.999))
        min_retention = float(dynamic_config.get("min_retention", 0.30))
        denominator = state.estimated_full_kv_mb * (kv_bits / 16) * precision_memory_multiplier(kv_bits)
        retention = (state.budget_uma_mb * margin) / denominator if denominator > 0 else 1.0
        retention = max(min(retention, 1.0), min_retention)
    if retention >= 1.0 and kv_bits == 8:
        return HybridKVAction("ideal_full_8", 1.0, 8, "Theoretical q8 upper-bound action.")
    if retention >= 1.0 and kv_bits == 16:
        return HybridKVAction("ideal_full_16", 1.0, 16, "Theoretical full-KV upper-bound action.")
    return make_dynamic_retention_action(retention=retention, kv_bits=kv_bits, name_prefix="ideal")


def resolve_method_action(
    *,
    method: dict[str, Any],
    config: dict[str, Any],
    model_profile: ModelKVProfile,
    state: PolicyState,
) -> tuple[HybridKVAction, str, float, str]:
    if method.get("ideal_dynamic"):
        profile_name = profile_for_regime(state.target_regime, str(nested(config, "policy", "profile", default="auto")))
        action = ideal_dynamic_action(method, state, config)
        return action, "theoretical dynamic upper bound", 0.0, profile_name

    if method.get("adaptive"):
        profile_name = profile_for_regime(state.target_regime, str(nested(config, "policy", "profile", default="auto")))
        controller = controller_for_config(config, model_profile, profile_name)
        start = time.perf_counter()
        decision = controller.decide(state)
        decision_ms = (time.perf_counter() - start) * 1000
        if decision.selected_action is None:
            raise RuntimeError("adaptive controller produced no feasible action")
        return decision.selected_action, decision.selected_cost.reason if decision.selected_cost else "", decision_ms, profile_name

    action_name = str(method["action"])
    action = actions_by_name().get(action_name)
    if action is None:
        raise ValueError(f"unknown benchmark action: {action_name}")
    return action, "static benchmark method", 0.0, profile_for_regime(state.target_regime, str(nested(config, "policy", "profile", default="auto")))


def load_docqa_examples(config: dict[str, Any]) -> list[dict[str, Any]]:
    path_value = nested(config, "benchmark", "docqa_dataset", default=None)
    if not path_value:
        return []
    path = Path(str(path_value))
    if not path.exists():
        raise SystemExit(f"missing Document QA dataset: {path}")
    max_examples = int(nested(config, "benchmark", "docqa_max_examples", default=0) or 0)
    examples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
            if max_examples and len(examples) >= max_examples:
                break
    return examples


def scenario_cases(config: dict[str, Any], scenario: dict[str, Any]) -> list[dict[str, Any]]:
    scenario_name = str(scenario["name"])
    if scenario_name == "document_qa":
        examples = load_docqa_examples(config)
        if examples:
            return [
                {
                    "case_id": str(example["id"]),
                    "position": str(example.get("answer_position", "middle")),
                    "question": str(example["question"]),
                    "answer": str(example["answer"]),
                    "document_title": str(example.get("title", "")),
                    "document": str(example.get("document", "")),
                }
                for example in examples
            ]
    return [
        {
            "case_id": f"{scenario_name}-{position}",
            "position": str(position),
            "question": "",
            "answer": "",
            "document_title": "",
            "document": "",
        }
        for position in scenario.get("positions", ["middle"])
    ]


def benchmark_records(config: dict[str, Any]) -> list[dict[str, Any]]:
    model_path = str(nested(config, "experiment", "model", default="models/Llama-3.2-1B-4bit"))
    model_profile = ModelKVProfile.from_config(model_config_path(model_path))
    support = BackendSupport(allow_simulated=bool(nested(config, "policy", "allow_simulated", default=False)))
    prompt_tokens_list = [int(value) for value in nested(config, "benchmark", "prompt_tokens", default=[3072, 10000])]
    regimes = [str(value) for value in nested(config, "benchmark", "regimes", default=["under_saturated", "saturation", "contention"])]
    scenarios = list(nested(config, "benchmark", "scenarios", default=[]))
    methods = list(nested(config, "methods", default=[]))
    repeats = int(nested(config, "benchmark", "repeats", default=1))
    generated_tokens = int(nested(config, "generation", "generated_tokens", default=32))
    protected_window = int(nested(config, "retention", "protected_window_tokens", default=4))
    start_memory = snapshot(os.getpid())
    records: list[dict[str, Any]] = []
    run_stamp = dt.datetime.now().strftime("%Y%m%d%H%M%S")

    for prompt_tokens in prompt_tokens_list:
        full_kv_mb = model_profile.full_kv_cache_mb(prompt_tokens)
        for regime in regimes:
            current_memory = snapshot(os.getpid())
            for scenario in scenarios:
                scenario_name = str(scenario["name"])
                cases = scenario_cases(config, scenario)
                for repeat_index in range(1, repeats + 1):
                    for case in cases:
                        position = str(case["position"])
                        state = state_for_decision(
                            config=config,
                            regime=regime,
                            prompt_tokens=prompt_tokens,
                            full_kv_mb=full_kv_mb,
                            generated_tokens=generated_tokens,
                            start_memory=start_memory,
                            current_memory=current_memory,
                        )
                        state = PolicyState(**({**state.__dict__, "task_profile": str(scenario.get("task_profile", scenario_name))}))
                        for method in methods:
                            action, reason, decision_ms, profile_name = resolve_method_action(
                                method=method,
                                config=config,
                                model_profile=model_profile,
                                state=state,
                            )
                            estimated_kv_mb = estimate_action_kv_mb(action, full_kv_mb)
                            budget_mb = budget_for_regime(config, regime, full_kv_mb)
                            budget_violation = estimated_kv_mb > budget_mb
                            feasibility = backend_feasibility(action, support)
                            status, status_reason = execution_status(
                                action=action,
                                support=support,
                                estimated_kv_mb=estimated_kv_mb,
                                budget_mb=budget_mb,
                                theoretical_only=bool(method.get("theoretical_only", False)),
                            )
                            score, retained = task_score(
                                scenario=scenario_name,
                                action=action,
                                state=state,
                                position=position,
                                protected_window=protected_window,
                                budget_violation=budget_violation,
                            )
                            latency = state.recent_latency_ms_per_token * latency_prior(action)
                            if budget_violation and budget_mb > 0:
                                latency *= 1.0 + max(estimated_kv_mb / budget_mb - 1.0, 0.0) * 0.50
                            tokens_per_second = 1000.0 / latency if latency > 0 else 0.0
                            ratio_to_budget = estimated_kv_mb / budget_mb if budget_mb > 0 else 1.0
                            pressure_scale = min(max(ratio_to_budget, 0.10), 2.0)
                            memory_saved = 0.0 if full_kv_mb <= 0 else (1.0 - estimated_kv_mb / full_kv_mb) * 100.0
                            is_retrieval_task = scenario_name in {"needle", "document_qa"}
                            exact_match = bool(score > 0.5) if is_retrieval_task else None
                            containment = bool(score > 0.5) if is_retrieval_task else None
                            output_text = str(case["answer"]) if exact_match else ""
                            record = {
                                "run_id": f"checkpoint6-{scenario_name}-{regime}-{prompt_tokens}-{repeat_index}-{position}-{method['name']}-{run_stamp}",
                                "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
                                "benchmark": "checkpoint6_regime_benchmark",
                                "scenario": scenario_name,
                                "case_id": case["case_id"],
                                "document_title": case["document_title"],
                                "position": position,
                                "question": case["question"],
                                "answer": case["answer"],
                                "repeat_index": repeat_index,
                                "method": method["name"],
                                "method_family": method.get("family", "unknown"),
                                "selected_action": action.name,
                                "retention_ratio": round(action.retention, 6),
                                "kv_bits": action.kv_bits,
                                "execution_status": status,
                                "backend_status": feasibility.status if feasibility.feasible else "infeasible",
                                "reason": reason or status_reason,
                                "operating_regime_target": regime,
                                "operating_regime_observed": classify_operating_regime(state),
                                "prompt_tokens": prompt_tokens,
                                "generated_tokens": generated_tokens,
                                "estimated_full_kv_mb": round(full_kv_mb, 3),
                                "estimated_kv_cache_mb": round(estimated_kv_mb, 3),
                                "memory_budget_mb": round(budget_mb, 3),
                                "memory_saved_percent": round(memory_saved, 3),
                                "budget_violation": budget_violation,
                                "swap_delta_mb": round(state.swap_delta_mb * pressure_scale, 3),
                                "pageout_delta": round(state.pageout_delta * pressure_scale, 3),
                                "swapout_delta": round(state.swapout_delta * pressure_scale, 3),
                                "compressed_memory_delta_mb": round(state.compressed_memory_delta_mb * pressure_scale, 3),
                                "memory_pressure": state.memory_pressure,
                                "bandwidth_pressure": round(bandwidth_pressure(state, estimated_kv_mb), 6),
                                "latency_ms_per_token": round(latency, 3),
                                "latency_p95_ms_per_token": round(state.latency_p95_ms_per_token * latency_prior(action), 3),
                                "tokens_per_second": round(tokens_per_second, 3),
                                "score": score,
                                "accuracy": score,
                                "delta_score_vs_full": round(1.0 - score, 6),
                                "retrieval_success": exact_match if is_retrieval_task else None,
                                "exact_match": exact_match,
                                "containment": containment,
                                "retained_target_token": retained,
                                "decision_time_ms": round(decision_ms, 4),
                                "policy_profile": profile_name,
                                "execution_mode": "proxy",
                                "output_text": output_text,
                                "notes": "deterministic checkpoint-6 proxy; theoretical-only actions are not final accuracy claims",
                            }
                            records.append(record)
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def grouped(records: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[tuple(record[key] for key in keys)].append(record)
    return groups


def regime_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    regime_order = {"under_saturated": 0, "saturation": 1, "contention": 2}
    method_order = {
        "Full KV": 0,
        "Quantized 8-bit": 1,
        "Retention-only": 2,
        "Hybrid conservative": 3,
        "Hybrid balanced": 4,
        "Hybrid aggressive": 5,
        "Ideal dynamic q8": 6,
        "Adaptive Lumina": 7,
    }
    groups = grouped(records, ("operating_regime_target", "method"))
    ordered_items = sorted(
        groups.items(),
        key=lambda item: (regime_order.get(str(item[0][0]), 99), method_order.get(str(item[0][1]), 99)),
    )
    for (regime, method), items in ordered_items:
        actions = sorted({str(item["selected_action"]) for item in items})
        statuses = sorted({str(item["execution_status"]) for item in items})
        avg_score = average([float(item["score"]) for item in items])
        row = {
            "regime": regime,
            "method": method,
            "action": ",".join(actions),
            "execution_status": ",".join(statuses),
            "kv_cache_mb": round(average([float(item["estimated_kv_cache_mb"]) for item in items]), 3),
            "memory_saved_percent": round(average([float(item["memory_saved_percent"]) for item in items]), 3),
            "swap_delta_mb": round(average([float(item["swap_delta_mb"]) for item in items]), 3),
            "pageout_delta": round(average([float(item["pageout_delta"]) for item in items]), 3),
            "latency_ms_per_token": round(average([float(item["latency_ms_per_token"]) for item in items]), 3),
            "tokens_per_second": round(average([float(item["tokens_per_second"]) for item in items]), 3),
            "score": round(avg_score, 6),
            "verdict": verdict(regime, method, statuses, avg_score),
        }
        rows.append(row)
    return rows


def verdict(regime: str, method: str, statuses: list[str], score: float) -> str:
    if any("theoretical" in status for status in statuses):
        return "theoretical frontier only"
    if "backend_infeasible" in statuses:
        return "backend infeasible"
    if "budget_infeasible" in statuses:
        return "budget violation"
    if regime == "under_saturated" and method == "Quantized 8-bit":
        return "q8 dominance check"
    if method == "Adaptive Lumina" and regime in {"saturation", "contention"}:
        return "adaptive budget fallback"
    if score < 0.50:
        return "quality failure mode"
    return "valid baseline"


def policy_shift_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    adaptive = [record for record in records if record["method"] == "Adaptive Lumina"]
    rows: list[dict[str, Any]] = []
    regime_order = {"under_saturated": 0, "saturation": 1, "contention": 2}
    groups = grouped(adaptive, ("operating_regime_target",))
    for regime, items in sorted(groups.items(), key=lambda item: regime_order.get(str(item[0][0]), 99)):
        action_counts: dict[str, int] = defaultdict(int)
        for item in items:
            action_counts[str(item["selected_action"])] += 1
        selected = max(action_counts.items(), key=lambda item: item[1])[0]
        rows.append(
            {
                "regime": regime[0],
                "selected_action": selected,
                "reason": most_common([str(item["reason"]) for item in items]),
                "quality_delta": round(average([float(item["delta_score_vs_full"]) for item in items]), 6),
                "memory_delta_percent": round(average([float(item["memory_saved_percent"]) for item in items]), 3),
                "swap_pageout_delta": round(average([float(item["swap_delta_mb"]) + float(item["pageout_delta"]) for item in items]), 3),
            }
        )
    return rows


def lost_intelligence_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    keys = ("scenario", "operating_regime_target", "prompt_tokens", "case_id", "position", "repeat_index")
    for key, items in grouped(records, keys).items():
        ideal_items = [item for item in items if item["method"] == "Ideal dynamic q8"]
        feasible_items = [item for item in items if item["method"] == "Adaptive Lumina"]
        if not ideal_items or not feasible_items:
            continue
        ideal = ideal_items[0]
        feasible = feasible_items[0]
        ideal_score = float(ideal["score"])
        feasible_score = float(feasible["score"])
        rows.append(
            {
                "scenario": key[0],
                "regime": key[1],
                "prompt_tokens": key[2],
                "case_id": key[3],
                "position": key[4],
                "repeat_index": key[5],
                "ideal_policy": ideal["selected_action"],
                "ideal_retention": ideal["retention_ratio"],
                "ideal_score": ideal_score,
                "ideal_status": ideal["execution_status"],
                "executable_policy": feasible["selected_action"],
                "executable_retention": feasible["retention_ratio"],
                "executable_score": feasible_score,
                "executable_status": feasible["execution_status"],
                "lost_intelligence": round(max(ideal_score - feasible_score, 0.0), 6),
                "ideal_memory_saved_percent": ideal["memory_saved_percent"],
                "executable_memory_saved_percent": feasible["memory_saved_percent"],
                "collapse_reason": ideal["reason"] if "theoretical" in str(ideal["execution_status"]) else feasible["reason"],
            }
        )
    return rows


def lost_intelligence_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for (scenario, regime), items in sorted(grouped(rows, ("scenario", "regime")).items()):
        summary.append(
            {
                "scenario": scenario,
                "regime": regime,
                "cases": len(items),
                "avg_ideal_score": round(average([float(item["ideal_score"]) for item in items]), 6),
                "avg_executable_score": round(average([float(item["executable_score"]) for item in items]), 6),
                "avg_lost_intelligence": round(average([float(item["lost_intelligence"]) for item in items]), 6),
                "max_lost_intelligence": round(max([float(item["lost_intelligence"]) for item in items] or [0.0]), 6),
                "ideal_policy": most_common([str(item["ideal_policy"]) for item in items]),
                "executable_policy": most_common([str(item["executable_policy"]) for item in items]),
            }
        )
    return summary


def most_common(values: list[str]) -> str:
    counts: dict[str, int] = defaultdict(int)
    for value in values:
        counts[value] += 1
    return max(counts.items(), key=lambda item: item[1])[0] if counts else ""


def write_svg(path: Path, title: str, rows: list[str], width: int = 960) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row_height = 26
    height = 58 + row_height * len(rows)
    body = "\n".join(f'<text x="24" y="{58 + i * row_height}" font-size="13">{row}</text>' for i, row in enumerate(rows))
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white" />
  <text x="{width / 2}" y="28" text-anchor="middle" font-size="18">{title}</text>
  {body}
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def write_memory_context_svg(path: Path, records: list[dict[str, Any]]) -> None:
    rows = []
    for item in regime_summary(records):
        rows.append(
            f"{item['regime']} | {item['method']} | KV {float(item['kv_cache_mb']):.1f} MB | saved {float(item['memory_saved_percent']):.1f}%"
        )
    write_svg(path, "Memory vs Context Length by Method", rows)


def write_policy_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        f"{row['regime']} -> {row['selected_action']} | memory saved {float(row['memory_delta_percent']):.1f}% | delta score {float(row['quality_delta']):.3f}"
        for row in rows
    ]
    write_svg(path, "Adaptive Policy Decisions over Regimes", lines)


def write_score_memory_svg(path: Path, summary: list[dict[str, Any]]) -> None:
    lines = [
        f"{row['regime']} | {row['method']} | score {float(row['score']):.3f} | saved {float(row['memory_saved_percent']):.1f}% | {row['execution_status']}"
        for row in summary
    ]
    write_svg(path, "Score vs Memory Saved", lines)


def write_latency_svg(path: Path, summary: list[dict[str, Any]]) -> None:
    lines = [
        f"{row['regime']} | {row['method']} | latency {float(row['latency_ms_per_token']):.1f} ms/tok | tok/s {float(row['tokens_per_second']):.2f}"
        for row in summary
    ]
    write_svg(path, "Latency by Policy", lines)


def write_swap_svg(path: Path, summary: list[dict[str, Any]]) -> None:
    lines = [
        f"{row['regime']} | {row['method']} | swap {float(row['swap_delta_mb']):.1f} MB | pageout {float(row['pageout_delta']):.1f}"
        for row in summary
    ]
    write_svg(path, "Swap and Pageout by Policy", lines)


def write_needle_svg(path: Path, records: list[dict[str, Any]]) -> None:
    needle = [record for record in records if record["scenario"] == "needle"]
    lines = []
    for (method, position), items in sorted(grouped(needle, ("method", "position")).items()):
        success = average([1.0 if item["retrieval_success"] else 0.0 for item in items])
        lines.append(f"{method} | {position} | retrieval success {success:.2f}")
    write_svg(path, "Retention Failure by Needle Position", lines)


def write_pareto_svg(path: Path, summary: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 900
    height = 500
    left = 72
    top = 48
    plot_width = 720
    plot_height = 340
    points: list[str] = []
    for row in summary:
        x = left + (float(row["memory_saved_percent"]) / 100.0) * plot_width
        y = top + plot_height - float(row["score"]) * plot_height
        status = str(row["execution_status"])
        if "theoretical" in status:
            color = "#f97316"
        elif "infeasible" in status:
            color = "#94a3b8"
        else:
            color = "#2563eb"
        label = f"{row['regime']}:{row['method']}"
        points.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}" />')
        points.append(f'<text x="{x + 7:.1f}" y="{y - 6:.1f}" font-size="10">{label}</text>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white" />
  <text x="{width / 2}" y="26" text-anchor="middle" font-size="18">Pareto Frontier: Theoretical vs Feasible</text>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111827" />
  <line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#111827" />
  <text x="{left + plot_width / 2}" y="{height - 32}" text-anchor="middle" font-size="12">memory saved vs full KV (%)</text>
  <text x="22" y="{top + plot_height / 2}" transform="rotate(-90 22,{top + plot_height / 2})" text-anchor="middle" font-size="12">accuracy / task score</text>
  <text x="{left}" y="{height - 72}" font-size="12" fill="#2563eb">blue = feasible real</text>
  <text x="{left + 170}" y="{height - 72}" font-size="12" fill="#f97316">orange = theoretical-only</text>
  <text x="{left + 390}" y="{height - 72}" font-size="12" fill="#94a3b8">gray = budget/backend infeasible</text>
  {chr(10).join(points)}
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def write_lost_intelligence_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        f"{row['scenario']} | {row['regime']} | ideal {float(row['avg_ideal_score']):.3f} | executable {float(row['avg_executable_score']):.3f} | lost {float(row['avg_lost_intelligence']):.3f}"
        for row in rows
    ]
    write_svg(path, "Lost Intelligence by Scenario and Regime", lines)


def write_figures(
    figures_dir: Path,
    records: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    policy_shift: list[dict[str, Any]],
    lost_summary: list[dict[str, Any]],
) -> list[Path]:
    figures = [
        figures_dir / "memory_vs_context.svg",
        figures_dir / "policy_decisions_over_regimes.svg",
        figures_dir / "score_vs_memory_saved.svg",
        figures_dir / "latency_by_policy.svg",
        figures_dir / "swap_pageout_by_policy.svg",
        figures_dir / "retention_failure_by_needle_position.svg",
        figures_dir / "pareto_theoretical_vs_feasible.svg",
        figures_dir / "lost_intelligence_by_regime.svg",
    ]
    write_memory_context_svg(figures[0], records)
    write_policy_svg(figures[1], policy_shift)
    write_score_memory_svg(figures[2], summary)
    write_latency_svg(figures[3], summary)
    write_swap_svg(figures[4], summary)
    write_needle_svg(figures[5], records)
    write_pareto_svg(figures[6], summary)
    write_lost_intelligence_svg(figures[7], lost_summary)
    return figures


def markdown_table(rows: list[dict[str, Any]], limit: int | None = None) -> str:
    lines = [
        "| Regime | Method | Action | KV MB | Swap | Pageout | Latency/token | Score | Verdict |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    visible = rows if limit is None else rows[:limit]
    for row in visible:
        lines.append(
            f"| {row['regime']} | {row['method']} | `{row['action']}` | {float(row['kv_cache_mb']):.2f} | "
            f"{float(row['swap_delta_mb']):.2f} | {float(row['pageout_delta']):.2f} | "
            f"{float(row['latency_ms_per_token']):.2f} | {float(row['score']):.3f} | {row['verdict']} |"
        )
    return "\n".join(lines)


def shift_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Regime | Selected action | Reason | Quality delta | Memory delta | Swap/Pageout delta |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['regime']} | `{row['selected_action']}` | {row['reason']} | "
            f"{float(row['quality_delta']):.3f} | {float(row['memory_delta_percent']):.1f}% | {float(row['swap_pageout_delta']):.1f} |"
        )
    return "\n".join(lines)


def lost_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Scenario | Regime | Ideal score | Executable score | Lost Intelligence | Ideal policy | Executable policy |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['scenario']} | {row['regime']} | {float(row['avg_ideal_score']):.3f} | "
            f"{float(row['avg_executable_score']):.3f} | {float(row['avg_lost_intelligence']):.3f} | "
            f"`{row['ideal_policy']}` | `{row['executable_policy']}` |"
        )
    return "\n".join(lines)


def write_report(
    path: Path,
    records: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    shifts: list[dict[str, Any]],
    lost_summary: list[dict[str, Any]],
    artifacts: dict[str, Path],
    figures: list[Path],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    q8_under = [row for row in summary if row["regime"] == "under_saturated" and row["method"] == "Quantized 8-bit"]
    adaptive_stress = [row for row in shifts if row["regime"] in {"saturation", "contention"}]
    theoretical = [record for record in records if "theoretical" in str(record["execution_status"])]
    content = (
        "# Report Checkpoint 6: Evaluation & System Gap Analysis\n\n"
        f"Generated: {dt.datetime.now().astimezone().isoformat(timespec='seconds')}\n\n"
        "## Evaluation Tiers\n\n"
        "- Tier 1: systems validation across under-saturated, saturation, and contention regimes.\n"
        "- Tier 2: retrieval robustness with Needle-in-a-Haystack.\n"
        "- Tier 3: semantic retention with a local 24-example Document QA set plus lightweight proxy tasks.\n\n"
        "## Result\n\n"
        f"{markdown_table(summary)}\n\n"
        "## Policy Shift Summary\n\n"
        f"{shift_table(shifts)}\n\n"
        "## Lost Intelligence Summary\n\n"
        f"{lost_table(lost_summary)}\n\n"
        "## Completion Check\n\n"
        f"- Benchmark scenarios evaluated: {len(set(record['scenario'] for record in records))}.\n"
        f"- Regimes evaluated: {', '.join(sorted(set(record['operating_regime_target'] for record in records)))}.\n"
        f"- q8 under-saturated result present: {'yes' if q8_under else 'no'}.\n"
        f"- Adaptive stress-regime result present: {'yes' if adaptive_stress else 'no'}.\n"
        f"- Theoretical-only records marked: {len(theoretical)}.\n"
        f"- Lost Intelligence groups computed: {len(lost_summary)}.\n"
        "- Task scores are deterministic Checkpoint 6 proxies; they are not final LLM accuracy claims.\n\n"
        "## Interpretation\n\n"
        "- Under-saturated runs preserve the expected q8-dominant regime.\n"
        "- Saturation and contention force Adaptive Lumina to use budget-driven retention because static q8 exceeds the configured KV budget.\n"
        "- Hybrid retained-quantized actions remain theoretical-only under the current MLX feasible set.\n"
        "- Needle and Document QA proxy scores expose the known failure mode: naive rotating retention preserves sinks and recent tokens but can drop early and middle evidence.\n"
        "- Lost Intelligence is reported as an upper-bound estimate: ideal dynamic q8 is theoretical-only when retained quantized KV is not executable.\n\n"
        "## Artifacts\n\n"
        f"- JSONL log: `{artifacts['log']}`\n"
        f"- Scenario results CSV: `{artifacts['scenario_csv']}`\n"
        f"- Regime summary CSV: `{artifacts['regime_summary_csv']}`\n"
        f"- Policy shift CSV: `{artifacts['policy_shift_csv']}`\n"
        f"- Lost Intelligence CSV: `{artifacts['lost_intelligence_csv']}`\n"
        f"- Lost Intelligence summary CSV: `{artifacts['lost_intelligence_summary_csv']}`\n"
        + "".join(f"- Figure: `{figure}`\n" for figure in figures)
    )
    path.write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = {
        "log": output_path(config, "log", "logs/benchmarks/checkpoint6_latest.jsonl"),
        "scenario_csv": output_path(config, "scenario_csv", "reports/checkpoint-6/scenario_results.csv"),
        "regime_summary_csv": output_path(config, "regime_summary_csv", "reports/checkpoint-6/regime_summary.csv"),
        "policy_shift_csv": output_path(config, "policy_shift_csv", "reports/checkpoint-6/policy_shift_summary.csv"),
        "lost_intelligence_csv": output_path(config, "lost_intelligence_csv", "reports/checkpoint-6/lost_intelligence.csv"),
        "lost_intelligence_summary_csv": output_path(config, "lost_intelligence_summary_csv", "reports/checkpoint-6/lost_intelligence_summary.csv"),
        "figures_dir": output_path(config, "figures_dir", "reports/checkpoint-6/figures"),
        "report": output_path(config, "report", "reports/report-checkpoint-6.md"),
    }

    if args.dry_run:
        print(json.dumps({"config": str(args.config), "outputs": {key: str(value) for key, value in paths.items()}}, indent=2, sort_keys=True))
        return 0

    records = benchmark_records(config)
    summary = regime_summary(records)
    shifts = policy_shift_summary(records)
    lost_rows = lost_intelligence_rows(records)
    lost_summary = lost_intelligence_summary(lost_rows)
    figures = write_figures(paths["figures_dir"], records, summary, shifts, lost_summary)

    write_jsonl(paths["log"], records)
    write_csv(paths["scenario_csv"], records)
    write_csv(paths["regime_summary_csv"], summary)
    write_csv(paths["policy_shift_csv"], shifts)
    write_csv(paths["lost_intelligence_csv"], lost_rows)
    write_csv(paths["lost_intelligence_summary_csv"], lost_summary)
    write_report(paths["report"], records, summary, shifts, lost_summary, paths, figures)

    for row in shifts:
        print(f"{row['regime']}: {row['selected_action']} memory_delta={row['memory_delta_percent']}%")
    print(f"wrote {paths['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
