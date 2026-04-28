#!/usr/bin/env python3
"""Compare Lumina KV cache compression policies against full cache."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import difflib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lumina.cache import CachePolicy, policy_from_name
from src.telemetry import MemorySampler, snapshot, worst_pressure


PROMPT_BASE = (
    "This is a controlled local inference benchmark on Apple Silicon. "
    "The model must keep track of operational details including memory pressure, "
    "swap usage, compressed memory, prompt length, token throughput, latency, "
    "repeatability, and structured logging. "
)
PROMPT_LABELS = ["short", "medium", "long"]


def tokenizer_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text))


def build_prompt(tokenizer: Any, target_tokens: int) -> tuple[str, list[int]]:
    header = (
        "You are evaluating a local LLM baseline. Summarize the operational "
        "tradeoffs and keep the repeated measurement facts consistent.\n\n"
    )
    prompt = header
    iteration = 1
    while tokenizer_count(tokenizer, prompt) < target_tokens:
        prompt += f"{iteration}. {PROMPT_BASE}"
        iteration += 1
    token_ids = tokenizer.encode(prompt)
    if len(token_ids) > target_tokens + 32 and hasattr(tokenizer, "decode"):
        decoded = tokenizer.decode(token_ids[:target_tokens])
        if decoded:
            prompt = decoded
            token_ids = tokenizer.encode(prompt)
    return prompt, token_ids


def parse_prompt_targets(value: str) -> list[int]:
    targets = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(targets) != 3:
        raise argparse.ArgumentTypeError("expected exactly three comma-separated prompt token targets")
    return targets


def decode(tokenizer: Any, token_ids: list[int]) -> str:
    if hasattr(tokenizer, "decode"):
        return tokenizer.decode(token_ids)
    return " ".join(str(token_id) for token_id in token_ids)


def text_error(reference: str, candidate: str) -> float:
    if not reference and not candidate:
        return 0.0
    ratio = difflib.SequenceMatcher(None, reference, candidate).ratio()
    return max(0.0, min(1.0, 1.0 - ratio))


def token_error(reference: list[int], candidate: list[int]) -> float:
    if not reference and not candidate:
        return 0.0
    length = max(len(reference), len(candidate), 1)
    matches = sum(1 for left, right in zip(reference, candidate) if left == right)
    return 1.0 - (matches / length)


def run_policy(
    *,
    model: Any,
    tokenizer: Any,
    policy: CachePolicy,
    prompt_label: str,
    prompt_tokens: list[int],
    max_tokens: int,
    reference_text: str | None,
    reference_tokens: list[int] | None,
) -> dict[str, Any]:
    import mlx.core as mx
    from mlx_lm.generate import generate_step

    prompt_cache = policy.make_cache(model)
    start_memory = snapshot(os.getpid())
    start = time.perf_counter()
    generated_tokens: list[int] = []
    with MemorySampler(os.getpid()) as sampler:
        for token, _ in generate_step(
            mx.array(prompt_tokens, dtype=mx.int32),
            model,
            max_tokens=max_tokens,
            prompt_cache=prompt_cache,
            **policy.generation_kwargs(),
        ):
            generated_tokens.append(int(token))
    elapsed = time.perf_counter() - start
    end_memory = snapshot(os.getpid())

    output_text = decode(tokenizer, generated_tokens)
    cache_bytes = policy.cache_nbytes(prompt_cache)
    generated_count = len(generated_tokens)
    latency = elapsed * 1000 / generated_count if generated_count else 0.0
    throughput = generated_count / elapsed if elapsed else 0.0
    pressure = worst_pressure(start_memory.pressure, end_memory.pressure)

    return {
        "run_id": f"compression-{prompt_label}-{policy.name}-{dt.datetime.now().strftime('%Y%m%d%H%M%S')}",
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "benchmark": "compression",
        "prompt_label": prompt_label,
        "prompt_tokens": len(prompt_tokens),
        "generated_tokens": generated_count,
        "method": policy.name,
        "method_description": policy.description,
        "cache_policy": policy.name,
        "kv_cache_mb": round(cache_bytes / (1024 * 1024), 3),
        "latency_ms_per_token": round(latency, 3),
        "tokens_per_second": round(throughput, 3),
        "total_time_sec": round(elapsed, 3),
        "peak_process_rss_mb": round(sampler.peak_rss_mb, 2),
        "peak_unified_used_mb": round(sampler.peak_unified_used_mb, 2),
        "start_swap_mb": round(start_memory.swap_used_mb, 2),
        "end_swap_mb": round(end_memory.swap_used_mb, 2),
        "swap_delta_mb": round(end_memory.swap_used_mb - start_memory.swap_used_mb, 2),
        "memory_pressure": pressure,
        "memory_pressure_start": start_memory.pressure,
        "memory_pressure_end": end_memory.pressure,
        "compressed_memory_mb": round(end_memory.compressed_mb, 2),
        "output_text": output_text,
        "generated_token_ids": generated_tokens,
        "quality_text_error": None if reference_text is None else round(text_error(reference_text, output_text), 4),
        "quality_token_error": None if reference_tokens is None else round(token_error(reference_tokens, generated_tokens), 4),
    }


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    full_by_prompt = {
        record["prompt_label"]: record
        for record in records
        if record["method"] == "full"
    }
    rows: list[dict[str, Any]] = []
    methods = sorted({record["method"] for record in records}, key=lambda value: (value != "full", value))
    for method in methods:
        items = [record for record in records if record["method"] == method]
        savings: list[float] = []
        latency_overheads: list[float] = []
        for item in items:
            full = full_by_prompt[item["prompt_label"]]
            full_cache = float(full["kv_cache_mb"])
            full_latency = float(full["latency_ms_per_token"])
            savings.append(0.0 if full_cache == 0 else (1 - float(item["kv_cache_mb"]) / full_cache) * 100)
            latency_overheads.append(0.0 if full_latency == 0 else (float(item["latency_ms_per_token"]) / full_latency - 1) * 100)
        quality_errors = [
            float(item["quality_text_error"])
            for item in items
            if item.get("quality_text_error") is not None
        ]
        row = {
            "method": method,
            "runs": len(items),
            "avg_kv_cache_mb": average([float(item["kv_cache_mb"]) for item in items]),
            "avg_ram_savings_percent": average(savings),
            "avg_quality_error": average(quality_errors),
            "avg_latency_ms_per_token": average([float(item["latency_ms_per_token"]) for item in items]),
            "avg_latency_overhead_percent": average(latency_overheads),
            "avg_tokens_per_second": average([float(item["tokens_per_second"]) for item in items]),
            "worst_memory_pressure": worst_record_pressure(items),
        }
        row["verdict"] = verdict(row)
        rows.append(row)
    return rows


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def worst_record_pressure(records: list[dict[str, Any]]) -> str:
    pressure = "unknown"
    for record in records:
        pressure = worst_pressure(pressure, str(record.get("memory_pressure", "unknown")))
    return pressure


def verdict(row: dict[str, Any]) -> str:
    if row["method"] == "full":
        return "Baseline"
    savings = float(row["avg_ram_savings_percent"])
    error = float(row["avg_quality_error"])
    overhead = float(row["avg_latency_overhead_percent"])
    if savings >= 40 and error <= 0.10 and overhead <= 15:
        return "Ottimo"
    if savings >= 30 and error <= 0.20 and overhead <= 35:
        return "Promettente"
    if overhead > 50:
        return "Troppo lento"
    if error > 0.35:
        return "Qualita' debole"
    return "Da rivedere"


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def write_tradeoff_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 760
    height = 430
    left = 72
    top = 48
    plot_width = width - left - 42
    plot_height = height - top - 68
    max_x = max([float(row["avg_ram_savings_percent"]) for row in rows] + [1.0])
    max_y = max([float(row["avg_quality_error"]) for row in rows] + [0.01])
    points = []
    for row in rows:
        x = left + (float(row["avg_ram_savings_percent"]) / max(max_x, 1.0)) * plot_width
        y = top + plot_height - (float(row["avg_quality_error"]) / max(max_y, 0.01)) * plot_height
        points.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#2563eb" />')
        points.append(f'<text x="{x + 8:.1f}" y="{y - 8:.1f}" font-size="12">{row["method"]}</text>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white" />
  <text x="{width / 2}" y="26" text-anchor="middle" font-size="18">Compression Trade-off</text>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111827" />
  <line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#111827" />
  <text x="{width / 2}" y="{height - 16}" text-anchor="middle" font-size="12">RAM saved (%)</text>
  <text x="20" y="{top + plot_height / 2}" transform="rotate(-90 20,{top + plot_height / 2})" text-anchor="middle" font-size="12">quality error</text>
  {chr(10).join(points)}
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Metodo | Risparmio RAM | Errore qualita' | Latenza ms/tok | Overhead latenza | Verdict |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {float(row['avg_ram_savings_percent']):.1f}% | "
            f"{float(row['avg_quality_error']):.3f} | "
            f"{float(row['avg_latency_ms_per_token']):.2f} | "
            f"{float(row['avg_latency_overhead_percent']):.1f}% | {row['verdict']} |"
        )
    return "\n".join(lines)


def write_report(path: Path, rows: list[dict[str, Any]], records: list[dict[str, Any]], log_path: Path, csv_path: Path, svg_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    winner = choose_winner(rows)
    content = (
        "# Report Checkpoint 3: Compression Engine\n\n"
        f"Generato: {dt.datetime.now().astimezone().isoformat(timespec='seconds')}\n\n"
        "## Tabella della Verita'\n\n"
        f"{markdown_table(rows)}\n\n"
        "## Vincitore\n\n"
        f"{winner}\n\n"
        "## Note Tecniche\n\n"
        "- `full` e' il controllo con KV cache piena.\n"
        "- `quantized_8bit` usa il supporto MLX per KV cache quantizzata.\n"
        "- `pca_35` usa proiettori PCA offline dai dump del Checkpoint 2 e ricostruisce la cache full quando il modello legge K/V.\n"
        "- `eviction` usa una rotating cache: conserva l'inizio e la finestra piu' recente.\n"
        "- L'errore qualita' e' misurato come drift testuale rispetto alla full cache sullo stesso prompt.\n\n"
        "## Artefatti\n\n"
        f"- Log JSONL: `{log_path}`\n"
        f"- Sommario CSV: `{csv_path}`\n"
        f"- Grafico trade-off: `{svg_path}`\n"
    )
    path.write_text(content, encoding="utf-8")


def choose_winner(rows: list[dict[str, Any]]) -> str:
    candidates = [row for row in rows if row["method"] != "full"]
    if not candidates:
        return "Nessun metodo compresso valutato."
    candidates.sort(
        key=lambda row: (
            float(row["avg_quality_error"]) > 0.20,
            float(row["avg_latency_overhead_percent"]) > 35,
            -float(row["avg_ram_savings_percent"]),
            float(row["avg_quality_error"]),
        )
    )
    best = candidates[0]
    return (
        f"Metodo candidato: `{best['method']}`. "
        f"Risparmio medio {float(best['avg_ram_savings_percent']):.1f}%, "
        f"errore qualita' {float(best['avg_quality_error']):.3f}, "
        f"latenza {float(best['avg_latency_ms_per_token']):.2f} ms/token."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/Llama-3.2-1B-4bit")
    parser.add_argument("--prompt-tokens", type=parse_prompt_targets, default="384,1536,3072")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--policies", default="full,quantized_8bit,pca_35,eviction")
    parser.add_argument("--samples-dir", type=Path, default=Path("data/cache_samples"))
    parser.add_argument("--pca-rank", type=int, default=35)
    parser.add_argument("--eviction-size", type=int, default=1024)
    parser.add_argument("--log", type=Path, default=Path("logs/compression/compression_latest.jsonl"))
    parser.add_argument("--summary-csv", type=Path, default=Path("reports/compression_summary.csv"))
    parser.add_argument("--tradeoff-svg", type=Path, default=Path("reports/compression_tradeoff.svg"))
    parser.add_argument("--report", type=Path, default=Path("reports/report-checkpoint-3.md"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    policy_names = [name.strip() for name in args.policies.split(",") if name.strip()]
    if args.dry_run:
        print(json.dumps(vars(args) | {"policy_names": policy_names}, indent=2, default=str))
        return 0

    from mlx_lm import load

    model, tokenizer = load(args.model)
    policies = [
        policy_from_name(
            name,
            samples_dir=args.samples_dir,
            pca_rank=args.pca_rank,
            eviction_size=args.eviction_size,
        )
        for name in policy_names
    ]

    records: list[dict[str, Any]] = []
    references: dict[str, tuple[str, list[int]]] = {}
    for prompt_label, target_tokens in zip(PROMPT_LABELS, args.prompt_tokens):
        _, prompt_tokens = build_prompt(tokenizer, target_tokens)
        for policy in policies:
            reference = references.get(prompt_label)
            record = run_policy(
                model=model,
                tokenizer=tokenizer,
                policy=policy,
                prompt_label=prompt_label,
                prompt_tokens=prompt_tokens,
                max_tokens=args.max_tokens,
                reference_text=None if reference is None else reference[0],
                reference_tokens=None if reference is None else reference[1],
            )
            if policy.name == "full":
                references[prompt_label] = (
                    str(record["output_text"]),
                    list(record["generated_token_ids"]),
                )
                record["quality_text_error"] = 0.0
                record["quality_token_error"] = 0.0
            records.append(record)
            print(
                f"{prompt_label} {policy.name}: "
                f"{record['kv_cache_mb']} MB, {record['latency_ms_per_token']} ms/tok, "
                f"quality_error={record['quality_text_error']}"
            )

    rows = summarize(records)
    write_jsonl(args.log, records)
    write_csv(args.summary_csv, rows)
    write_tradeoff_svg(args.tradeoff_svg, rows)
    write_report(args.report, rows, records, args.log, args.summary_csv, args.tradeoff_svg)
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
