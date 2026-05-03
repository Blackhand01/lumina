#!/usr/bin/env python3
"""Run Lumina Checkpoint 5 as an end-to-end Adaptive Memory OS experiment."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lumina.policy import BackendSupport, HybridKVAction, HybridPolicyController, ModelKVProfile, PolicyState, WEIGHT_PROFILES
from src.telemetry import MemorySnapshot, snapshot

from run_policy_experiment import cache_policy_for_action, model_config_path, profile_for_regime


DEFAULT_CONFIG = Path("configs/experiments/checkpoint5_runtime_os.yaml")


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"missing config: {path}")
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML is required for YAML experiment configs") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"config must be a mapping: {path}")
    return data


def nested(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def output_path(config: dict[str, Any], key: str, default: str) -> Path:
    return Path(str(nested(config, "outputs", key, default=default)))


def budget_for_regime(config: dict[str, Any], regime: str, full_kv_mb: float) -> float:
    ratios = nested(config, "memory", "budget_ratio", default={}) or {}
    ratio = float(ratios.get(regime, 1.0))
    budget = full_kv_mb * ratio
    if regime == "under_saturated":
        headroom = float(nested(config, "memory", "min_under_saturated_headroom_mb", default=256))
        budget = max(budget, full_kv_mb + headroom)
    return budget


def state_for_decision(
    *,
    config: dict[str, Any],
    regime: str,
    prompt_tokens: int,
    full_kv_mb: float,
    generated_tokens: int,
    start_memory: MemorySnapshot,
    current_memory: MemorySnapshot,
) -> PolicyState:
    priors = nested(config, "regime_priors", regime, default={}) or {}
    task_profile = str(nested(config, "scenario", "task_profile", default="generic"))
    pressure = str(priors.get("memory_pressure", current_memory.pressure))
    return PolicyState(
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        estimated_full_kv_mb=full_kv_mb,
        budget_uma_mb=budget_for_regime(config, regime, full_kv_mb),
        memory_pressure=pressure,
        pressure_injection=str(priors.get("pressure_injection", "none")),
        swap_delta_mb=float(priors.get("swap_delta_mb", current_memory.swap_used_mb - start_memory.swap_used_mb)),
        pageout_delta=float(priors.get("pageout_delta", current_memory.pageouts - start_memory.pageouts)),
        swapout_delta=float(priors.get("swapout_delta", current_memory.swapouts - start_memory.swapouts)),
        compressed_memory_delta_mb=float(
            priors.get("compressed_memory_delta_mb", current_memory.compressed_mb - start_memory.compressed_mb)
        ),
        recent_latency_ms_per_token=float(priors.get("latency_ms_per_token", 1.0)),
        latency_p95_ms_per_token=float(priors.get("latency_p95_ms_per_token", priors.get("latency_ms_per_token", 1.0))),
        recent_latency_trend=float(priors.get("recent_latency_trend", 0.0)),
        task_profile=task_profile,
        target_regime=regime,  # type: ignore[arg-type]
    )


def run_mlx_action(record: dict[str, Any], model: Any, tokenizer: Any, max_tokens: int) -> dict[str, Any]:
    from compare_compression import build_prompt, run_policy
    from lumina.policy.actions import actions_by_name

    action_name = str(record["selected_action"])
    action = actions_by_name().get(action_name)
    if action is None:
        action = HybridKVAction(
            action_name,
            float(record["retention_ratio"]),
            int(record["kv_bits"]),
            "Runtime selected dynamic action.",
        )
    policy = cache_policy_for_action(action, int(record["prompt_tokens"]))
    _, prompt_token_ids = build_prompt(tokenizer, int(record["prompt_tokens"]))
    actual = run_policy(
        model=model,
        tokenizer=tokenizer,
        policy=policy,
        prompt_label=f"{record['operating_regime_target']}_{record['prompt_tokens']}",
        prompt_tokens=prompt_token_ids,
        max_tokens=max_tokens,
        reference_text=None,
        reference_tokens=None,
    )
    return record | {
        "execution_mode": "mlx",
        "generated_tokens": actual["generated_tokens"],
        "cache_policy": actual["cache_policy"],
        "measured_kv_cache_mb": actual["kv_cache_mb"],
        "latency_ms_per_token": actual["latency_ms_per_token"],
        "tokens_per_second": actual["tokens_per_second"],
        "total_time_sec": actual["total_time_sec"],
        "peak_memory_mb": actual["peak_process_rss_mb"],
        "peak_process_rss_mb": actual["peak_process_rss_mb"],
        "peak_unified_used_mb": actual["peak_unified_used_mb"],
        "swap_mb": actual["end_swap_mb"],
        "swap_delta_mb": actual["swap_delta_mb"],
        "memory_pressure": actual["memory_pressure"],
        "memory_pressure_start": actual["memory_pressure_start"],
        "memory_pressure_end": actual["memory_pressure_end"],
        "compressed_memory_mb": actual["compressed_memory_mb"],
        "output_text": actual["output_text"],
    }


def decision_to_record(
    *,
    run_id: str,
    config: dict[str, Any],
    prompt_tokens: int,
    step_index: int,
    profile_name: str,
    decision_time_ms: float,
    decision: Any,
    start_memory: MemorySnapshot,
    current_memory: MemorySnapshot,
) -> dict[str, Any]:
    selected = decision.selected_action
    selected_cost = decision.selected_cost
    dynamic_enabled = bool(nested(config, "policy", "dynamic_retention", "enabled", default=False))
    estimated_kv_mb = None if selected_cost is None else selected_cost.estimated_kv_cache_mb
    budget_mb = decision.state.budget_uma_mb
    budget_violation = bool(estimated_kv_mb is not None and budget_mb is not None and estimated_kv_mb > budget_mb)
    max_overhead = float(nested(config, "policy", "max_decision_overhead_fraction", default=0.05))
    generation_proxy_ms = max(decision.state.recent_latency_ms_per_token, 1.0) * max(decision.state.generated_tokens, 1)
    overhead_fraction = decision_time_ms / max(decision_time_ms + generation_proxy_ms, 1.0)
    return {
        "run_id": run_id,
        "timestamp": decision.timestamp,
        "experiment": nested(config, "experiment", "name", default="checkpoint5_runtime_os"),
        "scenario": nested(config, "scenario", "name", default="unknown"),
        "model": nested(config, "experiment", "model", default="models/Llama-3.2-1B-4bit"),
        "backend": nested(config, "experiment", "backend", default="mlx"),
        "quantization": nested(config, "experiment", "quantization", default="4bit_model"),
        "benchmark": "checkpoint5_runtime_os",
        "step_index": step_index,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": decision.state.generated_tokens,
        "cache_policy": None if selected is None else selected.name,
        "selected_action": None if selected is None else selected.name,
        "retention_ratio": None if selected is None else selected.retention,
        "kv_bits": None if selected is None else selected.kv_bits,
        "dynamic_retention_enabled": dynamic_enabled,
        "is_dynamic_action": bool(selected is not None and selected.name.startswith("dynamic_")),
        "policy_profile": profile_name,
        "operating_regime_target": decision.state.target_regime,
        "operating_regime_observed": decision.operating_regime_observed,
        "pressure_injection": decision.state.pressure_injection,
        "estimated_full_kv_mb": round(decision.state.estimated_full_kv_mb, 3),
        "estimated_kv_cache_mb": None if estimated_kv_mb is None else round(estimated_kv_mb, 3),
        "memory_budget_mb": None if budget_mb is None else round(budget_mb, 3),
        "budget_violation": budget_violation,
        "backend_status": None if selected_cost is None else selected_cost.backend_status,
        "decision_reason": None if selected_cost is None else selected_cost.reason,
        "score": None if selected_cost is None or math.isinf(selected_cost.total) else round(selected_cost.total, 6),
        "policy_costs": [cost.to_record() for cost in decision.costs],
        "decision_time_ms": round(decision_time_ms, 4),
        "decision_overhead_fraction": round(overhead_fraction, 6),
        "decision_overhead_ok": overhead_fraction <= max_overhead,
        "memory_pressure": decision.state.memory_pressure,
        "swap_delta_mb": decision.state.swap_delta_mb,
        "pageout_delta": decision.state.pageout_delta,
        "swapout_delta": decision.state.swapout_delta,
        "compressed_memory_delta_mb": decision.state.compressed_memory_delta_mb,
        "latency_ms_per_token": decision.state.recent_latency_ms_per_token,
        "latency_p95_ms_per_token": decision.state.latency_p95_ms_per_token,
        "tokens_per_second": 0.0,
        "peak_memory_mb": current_memory.process_rss_mb,
        "swap_mb": current_memory.swap_used_mb,
        "compressed_memory_mb": current_memory.compressed_mb,
        "execution_mode": "controller",
        "notes": "runtime decision loop; use --run-mlx to execute selected real actions",
        **start_memory.to_record("telemetry_start"),
        **current_memory.to_record("telemetry_current"),
    }


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


def runtime_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "run_id": record["run_id"],
            "step_index": record["step_index"],
            "target_regime": record["operating_regime_target"],
            "observed_regime": record["operating_regime_observed"],
            "prompt_tokens": record["prompt_tokens"],
            "selected_action": record["selected_action"],
            "retention_ratio": record["retention_ratio"],
            "kv_bits": record["kv_bits"],
            "backend_status": record["backend_status"],
            "estimated_kv_cache_mb": record["estimated_kv_cache_mb"],
            "memory_budget_mb": record["memory_budget_mb"],
            "budget_violation": record["budget_violation"],
            "decision_time_ms": record["decision_time_ms"],
            "decision_overhead_fraction": record["decision_overhead_fraction"],
            "score": record["score"],
            "execution_mode": record["execution_mode"],
        }
        for record in records
    ]


def budget_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "run_id": record["run_id"],
            "target_regime": record["operating_regime_target"],
            "prompt_tokens": record["prompt_tokens"],
            "selected_action": record["selected_action"],
            "retention_ratio": record["retention_ratio"],
            "kv_bits": record["kv_bits"],
            "estimated_kv_cache_mb": record["estimated_kv_cache_mb"],
            "memory_budget_mb": record["memory_budget_mb"],
            "budget_violation": record["budget_violation"],
            "backend_status": record["backend_status"],
            "reason": record["decision_reason"],
        }
        for record in records
    ]


def cost_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        for cost in record["policy_costs"]:
            rows.append(
                {
                    "run_id": record["run_id"],
                    "target_regime": record["operating_regime_target"],
                    "prompt_tokens": record["prompt_tokens"],
                    **cost,
                }
            )
    return rows


def write_action_svg(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 900
    left = 210
    top = 54
    row_height = 38
    height = top + row_height * len(records) + 44
    colors = {"full_16": "#64748b", "full_8": "#2563eb", "r50_b16": "#f97316"}
    rows = []
    for index, record in enumerate(records):
        y = top + index * row_height
        full = float(record["estimated_full_kv_mb"])
        kv = float(record["estimated_kv_cache_mb"] or 0.0)
        bar_width = 430 * min(kv / max(full, 1.0), 1.0)
        action = str(record["selected_action"])
        color = "#16a34a" if action.startswith("dynamic_") else colors.get(action, "#111827")
        label = f"{record['operating_regime_target']} / {record['prompt_tokens']}"
        rows.append(f'<text x="18" y="{y + 22}" font-size="13">{label}</text>')
        rows.append(f'<rect x="{left}" y="{y + 7}" width="430" height="17" fill="#e5e7eb" />')
        rows.append(f'<rect x="{left}" y="{y + 7}" width="{bar_width:.1f}" height="17" fill="{color}" />')
        rows.append(
            f'<text x="{left + 442}" y="{y + 21}" font-size="13">{action} '
            f'({kv:.1f} MB / {float(record["memory_budget_mb"]):.1f} MB)</text>'
        )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white" />
  <text x="{width / 2}" y="28" text-anchor="middle" font-size="18">Checkpoint 5 Runtime Policy Decisions</text>
  <text x="{left}" y="46" font-size="12">selected KV cache relative to full KV</text>
  {chr(10).join(rows)}
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Regime | Prompt tokens | Selected action | Backend | Est. KV MB | Budget MB | Decision ms | Budget violation |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['target_regime']} | {row['prompt_tokens']} | `{row['selected_action']}` | "
            f"{row['backend_status']} | {float(row['estimated_kv_cache_mb']):.2f} | "
            f"{float(row['memory_budget_mb']):.2f} | {float(row['decision_time_ms']):.3f} | "
            f"{row.get('budget_violation', False)} |"
        )
    return "\n".join(lines)


def write_report(
    path: Path,
    records: list[dict[str, Any]],
    runtime_csv: Path,
    budget_csv: Path,
    cost_csv: Path,
    action_svg: Path,
    log_path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = runtime_rows(records)
    selected = sorted({str(record["selected_action"]) for record in records})
    execution_modes = sorted({str(record["execution_mode"]) for record in records})
    violations = [record for record in records if record["budget_violation"]]
    overhead_failures = [record for record in records if not record["decision_overhead_ok"]]
    dynamic_enabled = any(bool(record.get("dynamic_retention_enabled")) for record in records)
    dynamic_selected = [record for record in records if record.get("is_dynamic_action")]
    content = (
        "# Report Checkpoint 5: System Integration & Runtime OS\n\n"
        f"Generated: {dt.datetime.now().astimezone().isoformat(timespec='seconds')}\n\n"
        "## Result\n\n"
        f"{markdown_table(rows)}\n\n"
        "## Completion Check\n\n"
        f"- End-to-end runtime decisions from config: yes.\n"
        f"- Distinct selected actions: {len(selected)} (`{', '.join(selected)}`).\n"
        f"- Execution modes: {', '.join(execution_modes)}.\n"
        f"- Dynamic budget-driven retention enabled: {'yes' if dynamic_enabled else 'no'}.\n"
        f"- Dynamic actions selected: {len(dynamic_selected)}.\n"
        f"- Budget violations: {len(violations)}.\n"
        f"- Decision overhead failures: {len(overhead_failures)}.\n"
        "- Every decision logs cost, feasibility, backend status, and telemetry snapshot: yes.\n\n"
        "## Interpretation\n\n"
        "- Checkpoint 5 integrates controller, model KV profile, memory budget, telemetry, and report generation.\n"
        "- The runtime currently performs controller-level decisions by default; real MLX execution is available with `--run-mlx` for feasible actions.\n"
        "- Saturation and contention regimes are represented through explicit budget and pressure priors unless an external pressure process is used.\n"
        "- Budget-driven retention computes the smallest required token drop for each candidate precision instead of using only coarse retention steps.\n"
        "- Dynamic `r<1,b=8` remains backend-infeasible in standard MLX when retained quantized KV is not available; dynamic `r<1,b=16` is the real fallback.\n\n"
        "## Artifacts\n\n"
        f"- JSONL log: `{log_path}`\n"
        f"- Runtime decisions CSV: `{runtime_csv}`\n"
        f"- Budget enforcement CSV: `{budget_csv}`\n"
        f"- Cost terms CSV: `{cost_csv}`\n"
        f"- Runtime action plot: `{action_svg}`\n"
    )
    path.write_text(content, encoding="utf-8")


def run_runtime(config: dict[str, Any], *, run_mlx: bool) -> list[dict[str, Any]]:
    model_path = str(nested(config, "experiment", "model", default="models/Llama-3.2-1B-4bit"))
    model_profile = ModelKVProfile.from_config(model_config_path(model_path))
    support = BackendSupport(allow_simulated=bool(nested(config, "policy", "allow_simulated", default=False)))
    dynamic_config = nested(config, "policy", "dynamic_retention", default={}) or {}
    prompt_tokens_list = [int(value) for value in nested(config, "scenario", "prompt_tokens", default=[3072, 10000])]
    regimes = [str(value) for value in nested(config, "scenario", "regimes", default=["under_saturated", "saturation", "contention"])]
    profile = str(nested(config, "policy", "profile", default="auto"))
    max_tokens = int(nested(config, "generation", "max_tokens", default=1))
    run_stamp = dt.datetime.now().strftime("%Y%m%d%H%M%S")
    start_memory = snapshot(os.getpid())
    records: list[dict[str, Any]] = []

    mlx_model = None
    mlx_tokenizer = None
    if run_mlx:
        from mlx_lm import load

        mlx_model, mlx_tokenizer = load(model_path)

    step_index = 0
    for prompt_tokens in prompt_tokens_list:
        full_kv_mb = model_profile.full_kv_cache_mb(prompt_tokens)
        for regime in regimes:
            step_index += 1
            current_memory = snapshot(os.getpid())
            profile_name = profile_for_regime(regime, profile)
            controller = HybridPolicyController(
                model_profile=model_profile,
                weights=WEIGHT_PROFILES[profile_name],
                backend_support=support,
                enable_dynamic_retention=bool(dynamic_config.get("enabled", False)),
                dynamic_kv_bits=tuple(int(value) for value in dynamic_config.get("kv_bits", [8, 16])),
                dynamic_min_retention=float(dynamic_config.get("min_retention", 0.30)),
                dynamic_budget_margin=float(dynamic_config.get("budget_margin", 0.999)),
                replace_static_retention=bool(dynamic_config.get("replace_static_retention", False)),
            )
            state = state_for_decision(
                config=config,
                regime=regime,
                prompt_tokens=prompt_tokens,
                full_kv_mb=full_kv_mb,
                generated_tokens=max_tokens,
                start_memory=start_memory,
                current_memory=current_memory,
            )
            decision_start = time.perf_counter()
            decision = controller.decide(state)
            decision_time_ms = (time.perf_counter() - decision_start) * 1000
            run_id = f"checkpoint5-{regime}-{prompt_tokens}-{run_stamp}"
            record = decision_to_record(
                run_id=run_id,
                config=config,
                prompt_tokens=prompt_tokens,
                step_index=step_index,
                profile_name=profile_name,
                decision_time_ms=decision_time_ms,
                decision=decision,
                start_memory=start_memory,
                current_memory=current_memory,
            )
            if run_mlx and record["backend_status"] == "real" and mlx_model is not None and mlx_tokenizer is not None:
                record = run_mlx_action(record, mlx_model, mlx_tokenizer, max_tokens)
            records.append(record)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-mlx", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.run_mlx:
        config.setdefault("generation", {})["run_mlx"] = True
    run_mlx = bool(nested(config, "generation", "run_mlx", default=False))

    paths = {
        "log": output_path(config, "log", "logs/adaptive_os/runtime_os_latest.jsonl"),
        "runtime_csv": output_path(config, "runtime_csv", "reports/checkpoint-5/runtime_decisions.csv"),
        "budget_csv": output_path(config, "budget_csv", "reports/checkpoint-5/budget_enforcement.csv"),
        "cost_csv": output_path(config, "cost_csv", "reports/checkpoint-5/runtime_cost_terms.csv"),
        "action_svg": output_path(config, "action_svg", "reports/checkpoint-5/runtime_policy_actions.svg"),
        "report": output_path(config, "report", "reports/report-checkpoint-5.md"),
    }

    if args.dry_run:
        print(json.dumps({"config": str(args.config), "run_mlx": run_mlx, "outputs": {k: str(v) for k, v in paths.items()}}, indent=2, sort_keys=True))
        return 0

    records = run_runtime(config, run_mlx=run_mlx)
    write_jsonl(paths["log"], records)
    write_csv(paths["runtime_csv"], runtime_rows(records))
    write_csv(paths["budget_csv"], budget_rows(records))
    write_csv(paths["cost_csv"], cost_rows(records))
    write_action_svg(paths["action_svg"], records)
    write_report(paths["report"], records, paths["runtime_csv"], paths["budget_csv"], paths["cost_csv"], paths["action_svg"], paths["log"])

    for record in records:
        print(
            f"{record['operating_regime_target']} tokens={record['prompt_tokens']}: "
            f"{record['selected_action']} ({record['backend_status']}) budget_violation={record['budget_violation']}"
        )
    print(f"wrote {paths['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
