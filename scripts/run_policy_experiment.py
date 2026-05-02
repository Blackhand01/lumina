#!/usr/bin/env python3
"""Run the Checkpoint 4 regime-aware policy experiment."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lumina.cache import EvictionCachePolicy, FullCachePolicy, QuantizedCachePolicy
from lumina.policy import BackendSupport, HybridPolicyController, ModelKVProfile, PolicyState, WEIGHT_PROFILES
from lumina.policy.actions import DEFAULT_ACTION_GRID, HybridKVAction
from lumina.policy.cost import estimate_action_kv_mb, quality_loss_prior


def parse_prompt_tokens(value: str) -> list[int]:
    tokens = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not tokens:
        raise argparse.ArgumentTypeError("expected at least one prompt length")
    if any(token <= 0 for token in tokens):
        raise argparse.ArgumentTypeError("prompt lengths must be positive")
    return tokens


def model_config_path(model: str) -> Path:
    path = Path(model)
    if path.is_dir():
        return path / "config.json"
    return path


def profile_for_regime(regime: str, profile_name: str) -> str:
    if profile_name != "auto":
        return profile_name
    if regime == "under_saturated":
        return "balanced"
    if regime == "saturation":
        return "memory_first"
    if regime == "contention":
        return "stress"
    return "balanced"


def synthetic_state(
    *,
    regime: str,
    prompt_tokens: int,
    full_kv_mb: float,
    allow_simulated: bool,
) -> PolicyState:
    if regime == "under_saturated":
        return PolicyState(
            prompt_tokens=prompt_tokens,
            estimated_full_kv_mb=full_kv_mb,
            budget_uma_mb=max(full_kv_mb * 2.0, full_kv_mb + 256),
            memory_pressure="green",
            pressure_injection="none",
            recent_latency_ms_per_token=50.0,
            latency_p95_ms_per_token=52.0,
            task_profile="document_qa",
            target_regime="under_saturated",
        )
    if regime == "saturation":
        return PolicyState(
            prompt_tokens=prompt_tokens,
            estimated_full_kv_mb=full_kv_mb,
            budget_uma_mb=full_kv_mb * 0.515,
            memory_pressure="yellow",
            pressure_injection="budget_only",
            swap_delta_mb=48.0,
            pageout_delta=25.0,
            compressed_memory_delta_mb=128.0,
            recent_latency_ms_per_token=55.0,
            latency_p95_ms_per_token=72.0,
            recent_latency_trend=0.12,
            task_profile="document_qa",
            target_regime="saturation",
        )
    if regime == "contention":
        return PolicyState(
            prompt_tokens=prompt_tokens,
            estimated_full_kv_mb=full_kv_mb,
            budget_uma_mb=full_kv_mb * (0.22 if allow_simulated else 0.515),
            memory_pressure="yellow",
            pressure_injection="memory_process",
            swap_delta_mb=96.0,
            pageout_delta=80.0,
            swapout_delta=12.0,
            compressed_memory_delta_mb=320.0,
            recent_latency_ms_per_token=58.0,
            latency_p95_ms_per_token=110.0,
            recent_latency_trend=0.35,
            task_profile="document_qa",
            target_regime="contention",
        )
    raise ValueError(f"unknown regime: {regime}")


def decision_record(
    *,
    run_id: str,
    model: str,
    prompt_tokens: int,
    profile_name: str,
    decision: Any,
) -> dict[str, Any]:
    selected = decision.selected_action
    selected_cost = decision.selected_cost
    return {
        "run_id": run_id,
        "timestamp": decision.timestamp,
        "model": model,
        "backend": "mlx",
        "quantization": "4bit_model",
        "benchmark": "checkpoint4_policy_controller",
        "prompt_tokens": prompt_tokens,
        "generated_tokens": 0,
        "cache_policy": None if selected is None else selected.name,
        "selected_action": None if selected is None else selected.name,
        "retention_ratio": None if selected is None else selected.retention,
        "kv_bits": None if selected is None else selected.kv_bits,
        "policy_profile": profile_name,
        "operating_regime_target": decision.state.target_regime,
        "operating_regime_observed": decision.operating_regime_observed,
        "pressure_injection": decision.state.pressure_injection,
        "estimated_full_kv_mb": round(decision.state.estimated_full_kv_mb, 3),
        "estimated_kv_cache_mb": None if selected_cost is None else round(selected_cost.estimated_kv_cache_mb, 3),
        "memory_budget_mb": None if selected_cost is None else round(selected_cost.memory_budget_mb or 0.0, 3),
        "backend_status": None if selected_cost is None else selected_cost.backend_status,
        "decision_reason": None if selected_cost is None else selected_cost.reason,
        "score": None if selected_cost is None or math.isinf(selected_cost.total) else round(selected_cost.total, 6),
        "policy_costs": [cost.to_record() for cost in decision.costs],
        "memory_pressure": decision.state.memory_pressure,
        "swap_delta_mb": decision.state.swap_delta_mb,
        "pageout_delta": decision.state.pageout_delta,
        "swapout_delta": decision.state.swapout_delta,
        "compressed_memory_delta_mb": decision.state.compressed_memory_delta_mb,
        "latency_ms_per_token": decision.state.recent_latency_ms_per_token,
        "latency_p95_ms_per_token": decision.state.latency_p95_ms_per_token,
        "tokens_per_second": 0.0,
        "peak_memory_mb": 0.0,
        "swap_mb": 0.0,
        "compressed_memory_mb": 0.0,
        "notes": "controller-only policy decision; use --run-mlx for real generation",
    }


def cache_policy_for_action(action: HybridKVAction, prompt_tokens: int) -> Any:
    if action.name == "full_16":
        return FullCachePolicy()
    if action.name == "full_8":
        return QuantizedCachePolicy(bits=8)
    if action.retention < 1.0 and action.kv_bits == 16:
        return EvictionCachePolicy(max_kv_size=max(1, int(prompt_tokens * action.retention)), name=action.name)
    raise ValueError(f"action {action.name} has no real MLX cache path")


def run_mlx_for_records(records: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    from compare_compression import build_prompt, run_policy
    from mlx_lm import load

    model, tokenizer = load(args.model)
    by_prompt: dict[int, list[int]] = {}
    output_records: list[dict[str, Any]] = []
    action_by_name = {action.name: action for action in DEFAULT_ACTION_GRID}

    for record in records:
        if record["backend_status"] != "real":
            output_records.append(record)
            continue
        action = action_by_name[str(record["selected_action"])]
        prompt_tokens = int(record["prompt_tokens"])
        if prompt_tokens not in by_prompt:
            _, token_ids = build_prompt(tokenizer, prompt_tokens)
            by_prompt[prompt_tokens] = token_ids
        policy = cache_policy_for_action(action, prompt_tokens)
        actual = run_policy(
            model=model,
            tokenizer=tokenizer,
            policy=policy,
            prompt_label=f"{record['operating_regime_target']}_{prompt_tokens}",
            prompt_tokens=by_prompt[prompt_tokens],
            max_tokens=args.max_tokens,
            reference_text=None,
            reference_tokens=None,
        )
        merged = record | {
            "benchmark": "checkpoint4_policy_mlx",
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
            "notes": "real MLX generation for selected feasible action",
        }
        output_records.append(merged)
    return output_records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not rows:
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summary_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        rows.append(
            {
                "run_id": record["run_id"],
                "target_regime": record["operating_regime_target"],
                "observed_regime": record["operating_regime_observed"],
                "prompt_tokens": record["prompt_tokens"],
                "budget_mb": record["memory_budget_mb"],
                "full_kv_mb": record["estimated_full_kv_mb"],
                "selected_action": record["selected_action"],
                "backend_status": record["backend_status"],
                "estimated_kv_mb": record["estimated_kv_cache_mb"],
                "score": record["score"],
                "reason": record["decision_reason"],
            }
        )
    return rows


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


def pareto_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    action_lookup = {action.name: action for action in DEFAULT_ACTION_GRID}
    for record in records:
        regime = str(record["operating_regime_target"])
        prompt_tokens = int(record["prompt_tokens"])
        key = (regime, prompt_tokens)
        if key in seen:
            continue
        seen.add(key)
        state = PolicyState(
            prompt_tokens=prompt_tokens,
            estimated_full_kv_mb=float(record["estimated_full_kv_mb"]),
            budget_uma_mb=float(record["memory_budget_mb"]),
            memory_pressure=str(record["memory_pressure"]),
            pressure_injection=str(record["pressure_injection"]),
            swap_delta_mb=float(record["swap_delta_mb"]),
            pageout_delta=float(record["pageout_delta"]),
            swapout_delta=float(record["swapout_delta"]),
            compressed_memory_delta_mb=float(record["compressed_memory_delta_mb"]),
            recent_latency_ms_per_token=float(record["latency_ms_per_token"]),
            latency_p95_ms_per_token=float(record["latency_p95_ms_per_token"]),
            task_profile="document_qa",
            target_regime=regime,  # type: ignore[arg-type]
        )
        costs_by_action = {str(cost["action_name"]): cost for cost in record["policy_costs"]}
        for action_name, cost in costs_by_action.items():
            action = action_lookup[action_name]
            estimated_mb = estimate_action_kv_mb(action, state.estimated_full_kv_mb)
            memory_saved_percent = 100.0 * (1.0 - estimated_mb / max(state.estimated_full_kv_mb, 1.0))
            theoretical_quality_loss = quality_loss_prior(action, state)
            backend_status = str(cost["backend_status"])
            budget_feasible = estimated_mb <= float(record["memory_budget_mb"])
            backend_feasible = backend_status == "real"
            feasible_set = "mlx_feasible" if backend_feasible and budget_feasible else "theoretical_only"
            rows.append(
                {
                    "target_regime": regime,
                    "prompt_tokens": prompt_tokens,
                    "action_name": action_name,
                    "retention": action.retention,
                    "kv_bits": action.kv_bits,
                    "estimated_kv_cache_mb": round(estimated_mb, 3),
                    "memory_saved_percent": round(memory_saved_percent, 3),
                    "quality_loss_prior": round(theoretical_quality_loss, 6),
                    "backend_status": backend_status,
                    "budget_feasible": budget_feasible,
                    "feasible_set": feasible_set,
                    "collapse_reason": "" if feasible_set == "mlx_feasible" else str(cost["reason"]),
                }
            )
    return rows


def markdown_summary(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Regime | Prompt tokens | Budget MB | Selected action | Backend | Est. KV MB | Score | Reason |",
        "| --- | ---: | ---: | --- | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['target_regime']} | {row['prompt_tokens']} | {float(row['budget_mb']):.2f} | "
            f"`{row['selected_action']}` | {row['backend_status']} | {float(row['estimated_kv_mb']):.2f} | "
            f"{float(row['score']):.3f} | {row['reason']} |"
        )
    return "\n".join(lines)


def markdown_pareto_table(rows: list[dict[str, Any]]) -> str:
    selected = [
        row for row in rows
        if row["prompt_tokens"] == 10000 and row["target_regime"] == "saturation"
    ]
    if not selected:
        selected = rows[:]
    lines = [
        "| Action | r | b | Memory saved | Quality prior | Feasible set | Collapse reason |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in selected:
        lines.append(
            f"| `{row['action_name']}` | {float(row['retention']):.2f} | {int(row['kv_bits'])} | "
            f"{float(row['memory_saved_percent']):.1f}% | {float(row['quality_loss_prior']):.3f} | "
            f"{row['feasible_set']} | {row['collapse_reason']} |"
        )
    return "\n".join(lines)


def write_report(
    path: Path,
    records: list[dict[str, Any]],
    summary_csv: Path,
    cost_csv: Path,
    pareto_csv: Path,
    frontier_svg: Path,
    pareto_svg: Path,
    log_path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = summary_rows(records)
    selected = {str(row["selected_action"]) for row in rows if row["selected_action"]}
    under = [row for row in rows if row["target_regime"] == "under_saturated"]
    stress = [row for row in rows if row["target_regime"] in {"saturation", "contention"}]
    q8_under = any(row["selected_action"] == "full_8" for row in under)
    stress_shift = any(row["selected_action"] != "full_8" for row in stress)
    simulated = any(row["backend_status"] == "simulated" for row in rows)
    pareto = pareto_rows(records)
    theoretical_only = [row for row in pareto if row["feasible_set"] == "theoretical_only"]
    mlx_feasible = [row for row in pareto if row["feasible_set"] == "mlx_feasible"]
    content = (
        "# Report Checkpoint 4: Regime-Aware Hybrid Adaptive Policy Engine\n\n"
        f"Generato: {dt.datetime.now().astimezone().isoformat(timespec='seconds')}\n\n"
        "## Risultato\n\n"
        f"{markdown_summary(rows)}\n\n"
        "## Completion Check\n\n"
        f"- Azioni distinte selezionate: {len(selected)} (`{', '.join(sorted(selected))}`).\n"
        f"- q8 domina in under-saturated: {'yes' if q8_under else 'no'}.\n"
        f"- Policy shift sotto stress: {'yes' if stress_shift else 'no'}.\n"
        f"- Azioni simulate presenti: {'yes' if simulated else 'no'}.\n\n"
        "## Interpretazione\n\n"
        "- `full_8` e' atteso come scelta dominante quando Memory Pressure e swap non sono un vincolo.\n"
        "- Le azioni `r < 1, b < 16` restano infeasible nel backend MLX corrente se la quantizzazione di `RotatingKVCache` non e' disponibile.\n"
        "- Retention-only `r50_b16` e' il fallback reale per stressare budget stretti senza simulare 4-bit KV.\n"
        "- Questo report valida il controller e il logging; le affermazioni finali richiedono run MLX e benchmark retrieval.\n\n"
        "## Backend-Induced Policy Collapse\n\n"
        "Il risultato principale non e' solo che la policy cambia da `full_8` a `r50_b16`. "
        "Il risultato e' che lo spazio teorico delle azioni `A = {(r,b)}` viene ristretto dal backend MLX a un feasible set piu' piccolo `F subset A`.\n\n"
        "In pratica, Lumina vede azioni ibride teoricamente interessanti come `(0.75,8)` e `(0.5,8)`, "
        "ma il backend corrente non puo' eseguirle come punti reali perche' retention e quantizzazione KV non sono componibili nel percorso cache disponibile.\n\n"
        f"- Punti MLX-feasible nel plot Pareto: {len(mlx_feasible)}.\n"
        f"- Punti theoretical-only nel plot Pareto: {len(theoretical_only)}.\n"
        "- Nome del fenomeno: **Backend-Induced Policy Collapse**.\n\n"
        "## Pareto Feasible Set\n\n"
        "Tabella per regime `saturation` e 10000 prompt token, utile come lettura paper-style della frontiera:\n\n"
        f"{markdown_pareto_table(pareto)}\n\n"
        "Claim aggiornato:\n\n"
        "> Lumina is a constraint-aware controller that exposes the gap between optimal memory allocation policies and backend-feasible execution.\n\n"
        "## Artefatti\n\n"
        f"- Log JSONL: `{log_path}`\n"
        f"- Sommario regime/action: `{summary_csv}`\n"
        f"- Costi per azione: `{cost_csv}`\n"
        f"- Pareto feasible set CSV: `{pareto_csv}`\n"
        f"- Grafico regime/action: `{frontier_svg}`\n"
        f"- Grafico Pareto feasible set: `{pareto_svg}`\n"
    )
    path.write_text(content, encoding="utf-8")


def write_regime_svg(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 860
    row_height = 42
    top = 58
    left = 190
    height = top + row_height * len(records) + 34
    action_colors = {
        "full_16": "#64748b",
        "full_8": "#2563eb",
        "r50_b16": "#f97316",
        "r75_b8": "#16a34a",
        "r50_b8": "#0d9488",
        "r50_b4": "#dc2626",
        "r30_b4": "#7c2d12",
    }
    rows = []
    for index, record in enumerate(records):
        y = top + index * row_height
        action = str(record["selected_action"])
        color = action_colors.get(action, "#111827")
        budget = float(record["memory_budget_mb"])
        kv = float(record["estimated_kv_cache_mb"])
        full = float(record["estimated_full_kv_mb"])
        bar_width = 420 * min(kv / max(full, 1.0), 1.0)
        rows.append(f'<text x="16" y="{y + 24}" font-size="13">{record["operating_regime_target"]} / {record["prompt_tokens"]}</text>')
        rows.append(f'<rect x="{left}" y="{y + 8}" width="420" height="18" fill="#e5e7eb" />')
        rows.append(f'<rect x="{left}" y="{y + 8}" width="{bar_width:.1f}" height="18" fill="{color}" />')
        rows.append(f'<text x="{left + 432}" y="{y + 23}" font-size="13">{action} ({kv:.1f} MB, budget {budget:.1f})</text>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white" />
  <text x="{width / 2}" y="28" text-anchor="middle" font-size="18">Checkpoint 4 Policy Action by Regime</text>
  <text x="{left}" y="48" font-size="12">estimated KV cache relative to full KV</text>
  {chr(10).join(rows)}
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def write_pareto_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = [
        row for row in rows
        if row["prompt_tokens"] == 10000 and row["target_regime"] == "saturation"
    ]
    if not selected:
        selected = rows[:]
    width = 900
    height = 520
    left = 86
    top = 54
    plot_width = width - left - 52
    plot_height = height - top - 86
    max_x = max(float(row["memory_saved_percent"]) for row in selected) + 5
    max_y = max(float(row["quality_loss_prior"]) for row in selected) + 0.10

    def x_pos(value: float) -> float:
        return left + (value / max(max_x, 1.0)) * plot_width

    def y_pos(value: float) -> float:
        return top + plot_height - (value / max(max_y, 0.1)) * plot_height

    points = []
    for row in selected:
        x = x_pos(float(row["memory_saved_percent"]))
        y = y_pos(float(row["quality_loss_prior"]))
        feasible = row["feasible_set"] == "mlx_feasible"
        fill = "#2563eb" if feasible else "white"
        stroke = "#2563eb" if feasible else "#dc2626"
        radius = 6 if feasible else 7
        dash = "" if feasible else ' stroke-dasharray="4 3"'
        points.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="2"{dash} />')
        points.append(f'<text x="{x + 9:.1f}" y="{y - 8:.1f}" font-size="12">{row["action_name"]}</text>')

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white" />
  <text x="{width / 2}" y="28" text-anchor="middle" font-size="18">Pareto Frontier vs MLX Feasible Set</text>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111827" />
  <line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#111827" />
  <text x="{width / 2}" y="{height - 24}" text-anchor="middle" font-size="13">Memory saved vs full KV (%)</text>
  <text x="24" y="{top + plot_height / 2}" transform="rotate(-90 24,{top + plot_height / 2})" text-anchor="middle" font-size="13">Quality loss prior</text>
  <text x="{left + 8}" y="{top + 18}" font-size="12" fill="#2563eb">filled = MLX feasible</text>
  <text x="{left + 8}" y="{top + 36}" font-size="12" fill="#dc2626">dashed = theoretical only</text>
  {chr(10).join(points)}
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def run_controller(args: argparse.Namespace) -> list[dict[str, Any]]:
    config_path = model_config_path(args.model)
    if not config_path.exists():
        raise SystemExit(f"missing model config: {config_path}")
    model_profile = ModelKVProfile.from_config(config_path)
    support = BackendSupport(allow_simulated=args.allow_simulated)
    records: list[dict[str, Any]] = []
    run_stamp = dt.datetime.now().strftime("%Y%m%d%H%M%S")

    for prompt_tokens in args.prompt_tokens:
        full_kv_mb = model_profile.full_kv_cache_mb(prompt_tokens)
        for regime in args.regimes:
            profile_name = profile_for_regime(regime, args.profile)
            controller = HybridPolicyController(
                model_profile=model_profile,
                weights=WEIGHT_PROFILES[profile_name],
                backend_support=support,
            )
            state = synthetic_state(
                regime=regime,
                prompt_tokens=prompt_tokens,
                full_kv_mb=full_kv_mb,
                allow_simulated=args.allow_simulated,
            )
            decision = controller.decide(state)
            run_id = f"checkpoint4-{regime}-{prompt_tokens}-{run_stamp}"
            records.append(
                decision_record(
                    run_id=run_id,
                    model=args.model,
                    prompt_tokens=prompt_tokens,
                    profile_name=profile_name,
                    decision=decision,
                )
            )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/Llama-3.2-1B-4bit")
    parser.add_argument("--prompt-tokens", type=parse_prompt_tokens, default=parse_prompt_tokens("3072,10000"))
    parser.add_argument("--regimes", nargs="+", default=["under_saturated", "saturation", "contention"])
    parser.add_argument("--profile", choices=["auto", *WEIGHT_PROFILES.keys()], default="auto")
    parser.add_argument("--allow-simulated", action="store_true")
    parser.add_argument("--run-mlx", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--log", type=Path, default=Path("logs/policy/policy_latest.jsonl"))
    parser.add_argument("--summary-csv", type=Path, default=Path("reports/checkpoint-4/regime_action_summary.csv"))
    parser.add_argument("--cost-csv", type=Path, default=Path("reports/checkpoint-4/policy_cost_terms.csv"))
    parser.add_argument("--pareto-csv", type=Path, default=Path("reports/checkpoint-4/pareto_feasible_set.csv"))
    parser.add_argument("--frontier-svg", type=Path, default=Path("reports/checkpoint-4/retention_precision_by_regime.svg"))
    parser.add_argument("--pareto-svg", type=Path, default=Path("reports/checkpoint-4/pareto_feasible_set.svg"))
    parser.add_argument("--report", type=Path, default=Path("reports/report-checkpoint-4.md"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")

    if args.dry_run:
        print(
            json.dumps(
                {
                    "model": args.model,
                    "prompt_tokens": args.prompt_tokens,
                    "regimes": args.regimes,
                    "profile": args.profile,
                    "allow_simulated": args.allow_simulated,
                    "run_mlx": args.run_mlx,
                    "log": str(args.log),
                    "summary_csv": str(args.summary_csv),
                    "cost_csv": str(args.cost_csv),
                    "pareto_csv": str(args.pareto_csv),
                    "frontier_svg": str(args.frontier_svg),
                    "pareto_svg": str(args.pareto_svg),
                    "report": str(args.report),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    records = run_controller(args)
    if args.run_mlx:
        records = run_mlx_for_records(records, args)

    write_jsonl(args.log, records)
    write_csv(args.summary_csv, summary_rows(records))
    write_csv(args.cost_csv, cost_rows(records))
    pareto = pareto_rows(records)
    write_csv(args.pareto_csv, pareto)
    write_regime_svg(args.frontier_svg, records)
    write_pareto_svg(args.pareto_svg, pareto)
    write_report(
        args.report,
        records,
        args.summary_csv,
        args.cost_csv,
        args.pareto_csv,
        args.frontier_svg,
        args.pareto_svg,
        args.log,
    )

    for record in records:
        print(
            f"{record['operating_regime_target']} tokens={record['prompt_tokens']}: "
            f"{record['selected_action']} ({record['backend_status']}) score={record['score']}"
        )
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
