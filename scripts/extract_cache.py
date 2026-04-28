#!/usr/bin/env python3
"""Extract MLX KV cache tensors for offline memory observatory analysis."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.telemetry import snapshot


PROMPT_BASE = (
    "This is a controlled local inference benchmark on Apple Silicon. "
    "The model must keep track of operational details including memory pressure, "
    "swap usage, compressed memory, prompt length, token throughput, latency, "
    "repeatability, and structured logging. "
)
PROMPT_LABELS = ["short", "medium", "long"]


def parse_prompt_targets(value: str) -> list[int]:
    targets = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not targets:
        raise argparse.ArgumentTypeError("expected at least one prompt token target")
    if any(target <= 0 for target in targets):
        raise argparse.ArgumentTypeError("prompt token targets must be positive")
    return targets


def tokenizer_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text))


def build_prompt(tokenizer: Any, target_tokens: int) -> tuple[str, list[int]]:
    header = (
        "You are evaluating a local LLM KV cache. Preserve measurement facts and "
        "produce stable internal activations for offline analysis.\n\n"
    )
    prompt = header
    iteration = 1
    while tokenizer_count(tokenizer, prompt) < target_tokens:
        prompt += f"{iteration}. {PROMPT_BASE}"
        iteration += 1
    token_ids = tokenizer.encode(prompt)
    return prompt, token_ids


def as_numpy(array: Any) -> np.ndarray:
    return np.array(array, copy=True)


def cache_state(cache_item: Any) -> tuple[Any, Any]:
    state = getattr(cache_item, "state", None)
    if state is None:
        return None, None
    keys, values = state
    return keys, values


def layer_summary(layer_index: int, kind: str, array: np.ndarray) -> dict[str, Any]:
    values = array.astype(np.float32, copy=False)
    return {
        "layer": layer_index,
        "kind": kind,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "nbytes": int(array.nbytes),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def write_cache_sample(
    *,
    model: Any,
    tokenizer: Any,
    model_name: str,
    output_root: Path,
    prompt_label: str,
    target_tokens: int,
    max_cache_mb: float,
) -> Path:
    import mlx.core as mx

    prompt, token_ids = build_prompt(tokenizer, target_tokens)
    tokens = mx.array(token_ids, dtype=mx.int32)[None, :]
    cache = model.make_cache()

    start_memory = snapshot(os.getpid())
    start = time.perf_counter()
    logits = model(tokens, cache=cache)
    mx.eval(logits)
    elapsed = time.perf_counter() - start
    end_memory = snapshot(os.getpid())

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    sample_dir = output_root / f"{prompt_label}_{len(token_ids)}tok_{timestamp}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    total_cache_bytes = 0
    for layer_index, cache_item in enumerate(cache):
        keys, values = cache_state(cache_item)
        if keys is None or values is None:
            continue
        key_array = as_numpy(keys)
        value_array = as_numpy(values)
        total_cache_bytes += key_array.nbytes + value_array.nbytes
        if total_cache_bytes / (1024 * 1024) > max_cache_mb:
            raise SystemExit(
                f"cache dump exceeds --max-cache-mb={max_cache_mb}; "
                f"current={total_cache_bytes / (1024 * 1024):.1f} MB"
            )
        np.save(sample_dir / f"layer_{layer_index:02d}_key.npy", key_array)
        np.save(sample_dir / f"layer_{layer_index:02d}_value.npy", value_array)
        summaries.append(layer_summary(layer_index, "key", key_array))
        summaries.append(layer_summary(layer_index, "value", value_array))

    metadata = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model": model_name,
        "backend": "mlx",
        "prompt_label": prompt_label,
        "target_prompt_tokens": target_tokens,
        "prompt_tokens": len(token_ids),
        "prompt_preview": prompt[:240],
        "layers": len(cache),
        "cache_total_mb": round(total_cache_bytes / (1024 * 1024), 3),
        "forward_time_sec": round(elapsed, 3),
        "telemetry_source": "vm_stat+memory_pressure+sysctl",
        "telemetry_start": start_memory.to_record("start"),
        "telemetry_end": end_memory.to_record("end"),
        "tensor_files": sorted(path.name for path in sample_dir.glob("*.npy")),
        "summaries": summaries,
    }
    (sample_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return sample_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-tokens", type=parse_prompt_targets, default="384,1536,3072")
    parser.add_argument("--output-root", type=Path, default=Path("data/cache_samples"))
    parser.add_argument("--max-cache-mb", type=float, default=2048.0)
    parser.add_argument("--labels", default="short,medium,long")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    labels = [item.strip() for item in args.labels.split(",") if item.strip()]
    if len(labels) < len(args.prompt_tokens):
        labels += [f"prompt_{index}" for index in range(len(labels), len(args.prompt_tokens))]

    if args.dry_run:
        print(
            json.dumps(
                {
                    "model": args.model,
                    "prompt_tokens": args.prompt_tokens,
                    "output_root": str(args.output_root),
                    "labels": labels[: len(args.prompt_tokens)],
                    "max_cache_mb": args.max_cache_mb,
                },
                indent=2,
            )
        )
        return 0

    from mlx_lm import load

    args.output_root.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load(args.model)
    for label, target_tokens in zip(labels, args.prompt_tokens):
        sample_dir = write_cache_sample(
            model=model,
            tokenizer=tokenizer,
            model_name=args.model,
            output_root=args.output_root,
            prompt_label=label,
            target_tokens=target_tokens,
            max_cache_mb=args.max_cache_mb,
        )
        print(f"wrote {sample_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
