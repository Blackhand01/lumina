#!/usr/bin/env python3
"""Run a minimal Needle-in-a-Haystack benchmark for Lumina retention analysis."""

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

from lumina.policy import BackendSupport, HybridKVAction
from lumina.policy.actions import actions_by_name, backend_feasibility
from lumina.policy.cost import estimate_action_kv_mb
from run_lumina import load_config, nested, output_path
from run_policy_experiment import cache_policy_for_action, model_config_path
from src.telemetry import snapshot, worst_pressure


DEFAULT_CONFIG = Path("configs/benchmarks/needle_minimal.yaml")
MB = 1024 * 1024


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def make_action(name: str) -> HybridKVAction:
    predefined = actions_by_name()
    if name in predefined:
        return predefined[name]
    if name.startswith("r") and "_b" in name:
        left, right = name.split("_b", 1)
        retention = int(left[1:]) / 100
        return HybridKVAction(name, retention, int(right), "Needle benchmark action.")
    raise ValueError(f"unknown needle action: {name}")


def position_ratio(position: str) -> float:
    return {
        "sink": 0.0,
        "early": 0.10,
        "middle": 0.50,
        "late": 0.85,
        "very_late": 0.95,
    }.get(position, 0.50)


def position_index(position: str, prompt_tokens: int, protected_window: int) -> int:
    if position == "sink":
        return min(2, max(prompt_tokens - 1, 0))
    return min(max(int(prompt_tokens * position_ratio(position)), protected_window), max(prompt_tokens - 1, 0))


def retained_by_rotating_cache(action: HybridKVAction, prompt_tokens: int, token_index: int, protected_window: int) -> bool:
    if action.retention >= 1.0:
        return True
    retained_tokens = max(protected_window, int(math.ceil(prompt_tokens * action.retention)))
    recent_window = max(retained_tokens - protected_window, 0)
    recent_start = max(prompt_tokens - recent_window, 0)
    return token_index < protected_window or token_index >= recent_start


def build_needle_prompt(
    tokenizer: Any,
    *,
    prompt_tokens: int,
    needle_position: str,
    answer: str,
) -> tuple[str, list[int], int]:
    header = (
        "You are performing a memory retrieval benchmark. Read the context and answer the final question "
        "with only the secret code.\n\n"
    )
    needle = f"\nIMPORTANT FACT: the secret code is {answer}.\n"
    question = "\nQuestion: What is the secret code? Answer with only the code.\nAnswer:"
    filler = (
        "This filler sentence discusses local inference, memory pressure, latency, throughput, and cache policy. "
        "It contains no secret code and should not be used as the answer.\n"
    )

    target_needle_tokens = position_index(needle_position, prompt_tokens, 4)
    before = header
    while len(tokenizer.encode(before)) < target_needle_tokens:
        before += filler
    after = ""
    while len(tokenizer.encode(before + needle + after + question)) < prompt_tokens:
        after += filler
    prompt = before + needle + after + question
    token_ids = tokenizer.encode(prompt)
    needle_index = len(tokenizer.encode(before))
    return prompt, token_ids, needle_index


def decode_generated(tokenizer: Any, tokens: list[int]) -> str:
    if hasattr(tokenizer, "decode"):
        return tokenizer.decode(tokens)
    return " ".join(str(token) for token in tokens)


def run_mlx_trial(
    *,
    model: Any,
    tokenizer: Any,
    action: HybridKVAction,
    prompt_tokens: int,
    position: str,
    answer: str,
    max_tokens: int,
) -> dict[str, Any]:
    import mlx.core as mx
    from mlx_lm.generate import generate_step

    _, token_ids, needle_index = build_needle_prompt(
        tokenizer,
        prompt_tokens=prompt_tokens,
        needle_position=position,
        answer=answer,
    )
    policy = cache_policy_for_action(action, len(token_ids))
    prompt_cache = policy.make_cache(model)
    start_memory = snapshot(os.getpid())
    start = time.perf_counter()
    generated: list[int] = []
    for token, _ in generate_step(
        mx.array(token_ids, dtype=mx.int32),
        model,
        max_tokens=max_tokens,
        prompt_cache=prompt_cache,
        **policy.generation_kwargs(),
    ):
        generated.append(int(token))
    elapsed = time.perf_counter() - start
    end_memory = snapshot(os.getpid())
    output = decode_generated(tokenizer, generated)
    success = answer.lower() in output.lower()
    count = len(generated)
    return {
        "needle_token_index": needle_index,
        "retrieval_success": success,
        "output_text": output.strip(),
        "generated_tokens": count,
        "latency_ms_per_token": round(elapsed * 1000 / count, 3) if count else 0.0,
        "tokens_per_second": round(count / elapsed, 3) if elapsed else 0.0,
        "kv_cache_mb": round(policy.cache_nbytes(prompt_cache) / MB, 3),
        "memory_pressure": worst_pressure(start_memory.pressure, end_memory.pressure),
        "swap_delta_mb": round(end_memory.swap_used_mb - start_memory.swap_used_mb, 3),
        "execution_mode": "mlx",
    }


def proxy_trial(
    *,
    action: HybridKVAction,
    prompt_tokens: int,
    position: str,
    protected_window: int,
) -> dict[str, Any]:
    needle_index = position_index(position, prompt_tokens, protected_window)
    retained = retained_by_rotating_cache(action, prompt_tokens, needle_index, protected_window)
    success = retained and action.kv_bits >= 8
    return {
        "needle_token_index": needle_index,
        "retrieval_success": success,
        "output_text": "",
        "generated_tokens": 0,
        "latency_ms_per_token": 0.0,
        "tokens_per_second": 0.0,
        "kv_cache_mb": None,
        "memory_pressure": "unknown",
        "swap_delta_mb": 0.0,
        "execution_mode": "proxy",
    }


def generate_records(config: dict[str, Any], *, run_mlx: bool) -> list[dict[str, Any]]:
    model_path = str(nested(config, "experiment", "model", default="models/Llama-3.2-1B-4bit"))
    model_profile = None
    config_path = model_config_path(model_path)
    if config_path.exists():
        from lumina.policy import ModelKVProfile

        model_profile = ModelKVProfile.from_config(config_path)

    actions = [make_action(str(name)) for name in nested(config, "actions", default=["full_16", "full_8", "r50_b16"])]
    prompt_tokens_list = [int(value) for value in nested(config, "benchmark", "prompt_tokens", default=[3072])]
    positions = [str(value) for value in nested(config, "benchmark", "positions", default=["early", "middle", "late"])]
    repeats = int(nested(config, "benchmark", "repeats", default=3))
    max_tokens = int(nested(config, "benchmark", "max_tokens", default=24))
    protected_window = int(nested(config, "retention", "protected_window_tokens", default=4))
    support = BackendSupport()
    stamp = dt.datetime.now().strftime("%Y%m%d%H%M%S")
    records: list[dict[str, Any]] = []

    model = None
    tokenizer = None
    if run_mlx:
        from mlx_lm import load

        model, tokenizer = load(model_path)

    for prompt_tokens in prompt_tokens_list:
        full_kv_mb = model_profile.full_kv_cache_mb(prompt_tokens) if model_profile is not None else 0.0
        for repeat in range(1, repeats + 1):
            for position in positions:
                answer = f"LUMINA-{prompt_tokens}-{position}-{repeat}"
                for action in actions:
                    feasibility = backend_feasibility(action, support)
                    can_run = feasibility.feasible and feasibility.status == "real" and not (
                        action.retention < 1.0 and action.kv_bits < 16
                    )
                    if run_mlx and can_run and model is not None and tokenizer is not None:
                        result = run_mlx_trial(
                            model=model,
                            tokenizer=tokenizer,
                            action=action,
                            prompt_tokens=prompt_tokens,
                            position=position,
                            answer=answer,
                            max_tokens=max_tokens,
                        )
                    else:
                        result = proxy_trial(
                            action=action,
                            prompt_tokens=prompt_tokens,
                            position=position,
                            protected_window=protected_window,
                        )
                    estimated_kv_mb = estimate_action_kv_mb(action, full_kv_mb) if full_kv_mb else 0.0
                    records.append(
                        {
                            "run_id": f"needle-{prompt_tokens}-{position}-{repeat}-{action.name}-{stamp}",
                            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
                            "benchmark": "needle_minimal",
                            "model": model_path,
                            "prompt_tokens": prompt_tokens,
                            "position": position,
                            "repeat_index": repeat,
                            "answer": answer,
                            "selected_action": action.name,
                            "retention_ratio": round(action.retention, 6),
                            "kv_bits": action.kv_bits,
                            "protected_window_tokens": protected_window,
                            "needle_token_index": result["needle_token_index"],
                            "retained_target_token": retained_by_rotating_cache(
                                action,
                                prompt_tokens,
                                int(result["needle_token_index"]),
                                protected_window,
                            ),
                            "retrieval_success": bool(result["retrieval_success"]),
                            "execution_mode": result["execution_mode"],
                            "backend_status": feasibility.status if feasibility.feasible else "infeasible",
                            "backend_reason": feasibility.reason,
                            "estimated_full_kv_mb": round(full_kv_mb, 3),
                            "estimated_kv_cache_mb": round(estimated_kv_mb, 3),
                            "memory_saved_percent": round(0.0 if full_kv_mb <= 0 else (1.0 - estimated_kv_mb / full_kv_mb) * 100.0, 3),
                            "latency_ms_per_token": result["latency_ms_per_token"],
                            "tokens_per_second": result["tokens_per_second"],
                            "measured_kv_cache_mb": result["kv_cache_mb"],
                            "memory_pressure": result["memory_pressure"],
                            "swap_delta_mb": result["swap_delta_mb"],
                            "output_text": result["output_text"],
                            "notes": "minimal needle benchmark; proxy mode measures retention-induced reachability",
                        }
                    )
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


def summary_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = grouped(records, ("selected_action", "retention_ratio", "kv_bits", "position", "execution_mode"))
    for (action, retention, kv_bits, position, mode), items in sorted(groups.items(), key=lambda item: (float(item[0][1]), str(item[0][3]))):
        success_rate = average([1.0 if item["retrieval_success"] else 0.0 for item in items])
        rows.append(
            {
                "selected_action": action,
                "retention_ratio": retention,
                "kv_bits": kv_bits,
                "position": position,
                "execution_mode": mode,
                "runs": len(items),
                "retrieval_success_rate": round(success_rate, 6),
                "avg_memory_saved_percent": round(average([float(item["memory_saved_percent"]) for item in items]), 3),
                "avg_estimated_kv_cache_mb": round(average([float(item["estimated_kv_cache_mb"]) for item in items]), 3),
            }
        )
    return rows


def write_curve_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 860
    height = 430
    left = 70
    top = 42
    plot_width = 690
    plot_height = 300
    points: list[str] = []
    colors = {"sink": "#0f172a", "early": "#dc2626", "middle": "#f97316", "late": "#2563eb", "very_late": "#16a34a"}
    for row in rows:
        x = left + float(row["retention_ratio"]) * plot_width
        y = top + plot_height - float(row["retrieval_success_rate"]) * plot_height
        color = colors.get(str(row["position"]), "#111827")
        points.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}" />')
        points.append(f'<text x="{x + 6:.1f}" y="{y - 6:.1f}" font-size="10">{row["position"]}</text>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white" />
  <text x="{width / 2}" y="24" text-anchor="middle" font-size="18">Needle Retrieval vs Retention</text>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111827" />
  <line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#111827" />
  <text x="{left + plot_width / 2}" y="{height - 32}" text-anchor="middle" font-size="12">retention ratio r</text>
  <text x="20" y="{top + plot_height / 2}" transform="rotate(-90 20,{top + plot_height / 2})" text-anchor="middle" font-size="12">retrieval success</text>
  {chr(10).join(points)}
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def write_failure_map_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 900
    row_height = 25
    height = 56 + row_height * len(rows)
    lines = []
    for index, row in enumerate(rows):
        y = 56 + index * row_height
        success = float(row["retrieval_success_rate"])
        color = "#16a34a" if success >= 0.99 else "#dc2626" if success <= 0.01 else "#f97316"
        label = (
            f"{row['selected_action']} | r={float(row['retention_ratio']):.2f} | "
            f"{row['position']} | success={success:.2f} | mode={row['execution_mode']}"
        )
        lines.append(f'<rect x="18" y="{y - 14}" width="12" height="12" fill="{color}" />')
        lines.append(f'<text x="40" y="{y - 4}" font-size="12">{label}</text>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white" />
  <text x="{width / 2}" y="26" text-anchor="middle" font-size="18">Needle Failure Map</text>
  {chr(10).join(lines)}
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def markdown_table(rows: list[dict[str, Any]], limit: int = 24) -> str:
    lines = [
        "| Action | r | b | Position | Mode | Runs | Success | Memory Saved |",
        "| --- | ---: | ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows[:limit]:
        lines.append(
            f"| `{row['selected_action']}` | {float(row['retention_ratio']):.2f} | {int(row['kv_bits'])} | "
            f"{row['position']} | {row['execution_mode']} | {row['runs']} | "
            f"{float(row['retrieval_success_rate']):.2f} | {float(row['avg_memory_saved_percent']):.1f}% |"
        )
    return "\n".join(lines)


def write_report(path: Path, records: list[dict[str, Any]], rows: list[dict[str, Any]], artifacts: dict[str, Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    failures = [record for record in records if not record["retrieval_success"]]
    content = (
        "# Minimal Needle Benchmark\n\n"
        f"Generated: {dt.datetime.now().astimezone().isoformat(timespec='seconds')}\n\n"
        "## Result\n\n"
        f"{markdown_table(rows)}\n\n"
        "## Completion Check\n\n"
        f"- Total trials: {len(records)}.\n"
        f"- Unique prompt lengths: {len(set(record['prompt_tokens'] for record in records))}.\n"
        f"- Unique needle positions: {len(set(record['position'] for record in records))}.\n"
        f"- Failed retrieval trials: {len(failures)}.\n"
        "- Logged fields include retention, kv_bits, needle position, protected window, retained target token, and success/failure.\n\n"
        "## Interpretation\n\n"
        "- Proxy mode measures whether the needle survives the cache policy's retention geometry.\n"
        "- The protected window preserves sink-position needles, while early and middle needles fail under rotating retention.\n"
        "- MLX execution can be enabled with `--run-mlx` for backend-feasible actions when a slower semantic run is needed.\n\n"
        "## Artifacts\n\n"
        f"- JSONL log: `{artifacts['log']}`\n"
        f"- Results CSV: `{artifacts['results_csv']}`\n"
        f"- Summary CSV: `{artifacts['summary_csv']}`\n"
        f"- Retrieval curve: `{artifacts['curve_svg']}`\n"
        f"- Failure map: `{artifacts['failure_map_svg']}`\n"
    )
    path.write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-mlx", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = {
        "log": output_path(config, "log", "logs/benchmarks/needle_minimal_latest.jsonl"),
        "results_csv": output_path(config, "results_csv", "reports/checkpoint-6/needle_minimal_results.csv"),
        "summary_csv": output_path(config, "summary_csv", "reports/checkpoint-6/needle_retrieval_summary.csv"),
        "curve_svg": output_path(config, "curve_svg", "reports/checkpoint-6/figures/retrieval_vs_retention.svg"),
        "failure_map_svg": output_path(config, "failure_map_svg", "reports/checkpoint-6/figures/needle_failure_map.svg"),
        "report": output_path(config, "report", "reports/checkpoint-6/report_needle_minimal.md"),
    }

    if args.dry_run:
        print(json.dumps({"config": str(args.config), "run_mlx": args.run_mlx, "outputs": {key: str(value) for key, value in paths.items()}}, indent=2, sort_keys=True))
        return 0

    records = generate_records(config, run_mlx=args.run_mlx)
    rows = summary_rows(records)
    write_jsonl(paths["log"], records)
    write_csv(paths["results_csv"], records)
    write_csv(paths["summary_csv"], rows)
    write_curve_svg(paths["curve_svg"], rows)
    write_failure_map_svg(paths["failure_map_svg"], rows)
    write_report(paths["report"], records, rows, paths)

    print(f"trials={len(records)} failures={sum(1 for record in records if not record['retrieval_success'])}")
    print(f"wrote {paths['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
