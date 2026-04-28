#!/usr/bin/env python3
"""Run a repeatable MLX baseline and write structured JSONL logs."""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata as metadata
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


def tokenizer_decode(tokenizer: Any, token_ids: list[int]) -> str:
    if hasattr(tokenizer, "decode"):
        return tokenizer.decode(token_ids)
    return ""


def build_prompt(tokenizer: Any, target_tokens: int) -> tuple[str, int]:
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
    if len(token_ids) > target_tokens + 32:
        decoded = tokenizer_decode(tokenizer, token_ids[:target_tokens])
        if decoded:
            prompt = decoded

    return prompt, tokenizer_count(tokenizer, prompt)


def parse_prompt_targets(value: str) -> list[int]:
    targets = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(targets) != 3:
        raise argparse.ArgumentTypeError("expected exactly three comma-separated prompt token targets")
    if any(target <= 0 for target in targets):
        raise argparse.ArgumentTypeError("prompt token targets must be positive")
    return targets


def import_mlx_lm() -> tuple[Any, Any]:
    try:
        from mlx_lm import generate, load
    except ImportError as exc:
        raise SystemExit(
            "mlx-lm is not installed. Run `make checkpoint1-deps` first."
        ) from exc
    return load, generate


def generate_text(generate: Any, model: Any, tokenizer: Any, prompt: str, max_tokens: int) -> str:
    result = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=False,
    )
    return str(result)


def make_record(
    *,
    model_name: str,
    quantization: str,
    prompt_label: str,
    repeat_index: int,
    target_prompt_tokens: int,
    prompt_tokens: int,
    generated_tokens: int,
    elapsed_seconds: float,
    peak_memory_mb: float,
    start_memory: Any,
    end_memory: Any,
    peak_unified_used_mb: float,
    peak_swap_used_mb: float,
    peak_compressor_mb: float,
) -> dict[str, Any]:
    tokens_per_second = generated_tokens / elapsed_seconds if elapsed_seconds > 0 else 0.0
    latency_ms_per_token = (
        elapsed_seconds * 1000 / generated_tokens if generated_tokens > 0 else 0.0
    )
    pressure = worst_pressure(start_memory.pressure, end_memory.pressure)
    swap_delta = end_memory.swap_used_mb - start_memory.swap_used_mb
    unified_delta = end_memory.unified_used_mb - start_memory.unified_used_mb
    compressed_delta = end_memory.compressed_mb - start_memory.compressed_mb
    pageout_delta = end_memory.pageouts - start_memory.pageouts
    swapout_delta = end_memory.swapouts - start_memory.swapouts

    record = {
        "run_id": f"baseline-{prompt_label}-{repeat_index}-{uuid.uuid4().hex[:8]}",
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model": model_name,
        "backend": "mlx",
        "quantization": quantization,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "cache_policy": "full_kv_cache",
        "peak_memory_mb": round(peak_memory_mb, 2),
        "swap_mb": round(end_memory.swap_used_mb, 2),
        "memory_pressure": pressure,
        "compressed_memory_mb": round(end_memory.compressed_mb, 2),
        "latency_ms_per_token": round(latency_ms_per_token, 3),
        "tokens_per_second": round(tokens_per_second, 3),
        "benchmark": "baseline",
        "score": round(tokens_per_second, 3),
        "notes": (
            f"prompt_label={prompt_label}; repeat={repeat_index}; "
            f"total_time_sec={elapsed_seconds:.3f}; swap_delta_mb={swap_delta:.2f}; "
            f"unified_delta_mb={unified_delta:.2f}; "
            "thermal_throttling=unknown"
        ),
        "prompt_label": prompt_label,
        "repeat_index": repeat_index,
        "target_prompt_tokens": target_prompt_tokens,
        "total_time_sec": round(elapsed_seconds, 3),
        "start_swap_mb": round(start_memory.swap_used_mb, 2),
        "end_swap_mb": round(end_memory.swap_used_mb, 2),
        "swap_delta_mb": round(swap_delta, 2),
        "memory_pressure_start": start_memory.pressure,
        "memory_pressure_end": end_memory.pressure,
        "memory_pressure_free_percent_start": start_memory.pressure_free_percent,
        "memory_pressure_free_percent_end": end_memory.pressure_free_percent,
        "compressed_memory_start_mb": round(start_memory.compressed_mb, 2),
        "compressed_memory_end_mb": round(end_memory.compressed_mb, 2),
        "thermal_throttling": "unknown",
        "telemetry_source": "vm_stat+memory_pressure+sysctl",
        "start_unified_used_mb": round(start_memory.unified_used_mb, 2),
        "end_unified_used_mb": round(end_memory.unified_used_mb, 2),
        "unified_delta_mb": round(unified_delta, 2),
        "peak_unified_used_mb": round(peak_unified_used_mb, 2),
        "start_unified_available_mb": round(start_memory.unified_available_mb, 2),
        "end_unified_available_mb": round(end_memory.unified_available_mb, 2),
        "start_process_rss_mb": round(start_memory.process_rss_mb, 2),
        "end_process_rss_mb": round(end_memory.process_rss_mb, 2),
        "peak_process_rss_mb": round(peak_memory_mb, 2),
        "peak_swap_mb": round(peak_swap_used_mb, 2),
        "start_compressor_mb": round(start_memory.compressor_mb, 2),
        "end_compressor_mb": round(end_memory.compressor_mb, 2),
        "peak_compressor_mb": round(peak_compressor_mb, 2),
        "compressed_memory_delta_mb": round(compressed_delta, 2),
        "pageout_delta": round(pageout_delta, 2),
        "swapout_delta": round(swapout_delta, 2),
    }
    record.update(start_memory.to_record("telemetry_start"))
    record.update(end_memory.to_record("telemetry_end"))
    return record


def run_baseline(args: argparse.Namespace) -> None:
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    load, generate = import_mlx_lm()
    print(f"Loading model: {args.model}", flush=True)
    model, tokenizer = load(args.model)

    try:
        import mlx.core as mx

        mx.random.seed(args.seed)
    except Exception:
        pass

    mlx_version = metadata.version("mlx")
    mlx_lm_version = metadata.version("mlx-lm")
    print(f"mlx={mlx_version} mlx-lm={mlx_lm_version}", flush=True)

    with output_path.open("a", encoding="utf-8") as handle:
        for label, target_tokens in zip(PROMPT_LABELS, args.prompt_tokens):
            prompt, prompt_tokens = build_prompt(tokenizer, target_tokens)
            for repeat_index in range(1, args.runs + 1):
                start_memory = snapshot(os.getpid())

                start = time.perf_counter()
                with MemorySampler(os.getpid()) as sampler:
                    generated = generate_text(
                        generate,
                        model,
                        tokenizer,
                        prompt,
                        args.max_tokens,
                    )
                elapsed = time.perf_counter() - start

                generated_tokens = tokenizer_count(tokenizer, generated)
                end_memory = snapshot(os.getpid())

                record = make_record(
                    model_name=args.model,
                    quantization=args.quantization,
                    prompt_label=label,
                    repeat_index=repeat_index,
                    target_prompt_tokens=target_tokens,
                    prompt_tokens=prompt_tokens,
                    generated_tokens=generated_tokens,
                    elapsed_seconds=elapsed,
                    peak_memory_mb=sampler.peak_rss_mb,
                    start_memory=start_memory,
                    end_memory=end_memory,
                    peak_unified_used_mb=sampler.peak_unified_used_mb,
                    peak_swap_used_mb=sampler.peak_swap_used_mb,
                    peak_compressor_mb=sampler.peak_compressor_mb,
                )

                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                print(
                    f"{label} run {repeat_index}/{args.runs}: "
                    f"{record['tokens_per_second']} tok/s, "
                    f"{record['peak_memory_mb']} MB peak RSS, "
                    f"pressure={record['memory_pressure']}",
                    flush=True,
                )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--quantization", default="4bit")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--prompt-tokens", type=parse_prompt_targets, default="384,1536,3072")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.runs <= 0:
        parser.error("--runs must be positive")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")

    if args.dry_run:
        print(
            json.dumps(
                {
                    "model": args.model,
                    "quantization": args.quantization,
                    "runs": args.runs,
                    "max_tokens": args.max_tokens,
                    "prompt_tokens": args.prompt_tokens,
                    "output": args.output,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    run_baseline(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
