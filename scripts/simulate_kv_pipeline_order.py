#!/usr/bin/env python3
"""Simulate KV compression order: retention->quantization vs quantization->retention.

The script compares two pipelines against an original FP16 KV cache:

Pipeline A (r+b):
  1. Retain 70% of FP16 tokens, protecting the first N attention-sink tokens.
  2. Quantize the retained cache to 8-bit.
  3. Dequantize for reconstruction and fill evicted tokens with zeros.

Pipeline B (b+r):
  1. Quantize the full FP16 cache to 8-bit.
  2. Dequantize the full cache.
  3. Retain 70% of quantized/dequantized tokens, protecting the first N tokens.
  4. Fill evicted tokens with zeros.

This is a simulation, not an MLX kernel path.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Quantized:
    values: np.ndarray
    scale: np.ndarray
    zero_point: np.ndarray


def quantize_uint8(x: np.ndarray, axis: tuple[int, ...] | None = None) -> Quantized:
    """Affine uint8 quantization with optional per-channel axes kept."""

    if axis is None:
        x_min = np.min(x, keepdims=True)
        x_max = np.max(x, keepdims=True)
    else:
        x_min = np.min(x, axis=axis, keepdims=True)
        x_max = np.max(x, axis=axis, keepdims=True)

    scale = (x_max - x_min) / 255.0
    scale = np.where(scale == 0, np.ones_like(scale), scale)
    zero_point = np.round(-x_min / scale)
    q = np.round(x / scale + zero_point)
    q = np.clip(q, 0, 255).astype(np.uint8)
    return Quantized(values=q, scale=scale.astype(np.float32), zero_point=zero_point.astype(np.float32))


def dequantize_uint8(q: Quantized) -> np.ndarray:
    return (q.values.astype(np.float32) - q.zero_point) * q.scale


def quant_dequant_uint8(x: np.ndarray, axis: tuple[int, ...] | None = None) -> np.ndarray:
    return dequantize_uint8(quantize_uint8(x, axis=axis))


def token_energy(kv: np.ndarray) -> np.ndarray:
    """Compute per-token energy from a cache shaped [2, layers, seq, heads, dim]."""

    return np.mean(kv.astype(np.float32) ** 2, axis=(0, 1, 3, 4))


def retention_mask(
    kv_for_scoring: np.ndarray,
    retention: float,
    protected_tokens: int,
) -> np.ndarray:
    seq_len = kv_for_scoring.shape[2]
    keep_count = max(protected_tokens, int(round(seq_len * retention)))
    keep_count = min(keep_count, seq_len)

    mask = np.zeros(seq_len, dtype=bool)
    mask[: min(protected_tokens, seq_len)] = True

    remaining = np.arange(protected_tokens, seq_len)
    slots = keep_count - int(mask.sum())
    if slots > 0 and remaining.size > 0:
        energy = token_energy(kv_for_scoring)
        ranked_remaining = remaining[np.argsort(energy[remaining])[::-1]]
        mask[ranked_remaining[:slots]] = True
    return mask


def apply_mask(kv: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(kv, dtype=np.float32)
    out[:, :, mask, :, :] = kv[:, :, mask, :, :].astype(np.float32)
    return out


def mse(reference: np.ndarray, candidate: np.ndarray) -> float:
    diff = reference.astype(np.float32) - candidate.astype(np.float32)
    return float(np.mean(diff * diff))


def make_synthetic_kv(
    *,
    rng: np.random.Generator,
    layers: int,
    seq_len: int,
    heads: int,
    dim: int,
    protected_tokens: int,
) -> np.ndarray:
    kv = rng.normal(0.0, 0.55, size=(2, layers, seq_len, heads, dim)).astype(np.float32)

    # Attention sinks / first tokens: larger magnitude and shared low-frequency structure.
    sink_len = min(protected_tokens, seq_len)
    if sink_len:
        kv[:, :, :sink_len, :, :] += rng.normal(0.0, 1.25, size=(2, layers, sink_len, heads, dim))

    # Add a few high-energy retrieval-like tokens outside the sink window.
    if seq_len > protected_tokens + 8:
        needle_count = max(1, seq_len // 32)
        positions = rng.choice(np.arange(protected_tokens, seq_len), size=needle_count, replace=False)
        kv[:, :, positions, :, :] += rng.normal(0.0, 1.0, size=(2, layers, needle_count, heads, dim))

    return kv.astype(np.float16)


def pipeline_a(
    kv_fp16: np.ndarray,
    retention: float,
    protected_tokens: int,
    quant_axis: tuple[int, ...] | None,
) -> tuple[np.ndarray, np.ndarray]:
    mask = retention_mask(kv_fp16, retention, protected_tokens)
    retained = apply_mask(kv_fp16, mask)
    # Quantize only retained non-zero cache. This represents r+b.
    reconstructed = quant_dequant_uint8(retained, axis=quant_axis)
    reconstructed[:, :, ~mask, :, :] = 0.0
    return reconstructed, mask


def pipeline_b(
    kv_fp16: np.ndarray,
    retention: float,
    protected_tokens: int,
    quant_axis: tuple[int, ...] | None,
) -> tuple[np.ndarray, np.ndarray]:
    quantized_full = quant_dequant_uint8(kv_fp16.astype(np.float32), axis=quant_axis)
    mask = retention_mask(quantized_full, retention, protected_tokens)
    reconstructed = apply_mask(quantized_full, mask)
    return reconstructed, mask


def fixed_mask_commutativity_check(
    kv_fp16: np.ndarray,
    retention: float,
    protected_tokens: int,
    quant_axis: tuple[int, ...] | None,
) -> tuple[float, float]:
    mask = retention_mask(kv_fp16, retention, protected_tokens)

    retained_first = apply_mask(kv_fp16, mask)
    a = quant_dequant_uint8(retained_first, axis=quant_axis)
    a[:, :, ~mask, :, :] = 0.0

    q_full = quant_dequant_uint8(kv_fp16.astype(np.float32), axis=quant_axis)
    b = apply_mask(q_full, mask)
    return mse(kv_fp16, a), mse(kv_fp16, b)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--retention", type=float, default=0.70)
    parser.add_argument("--protected-tokens", type=int, default=4)
    parser.add_argument(
        "--per-channel",
        action="store_true",
        help="Use per-(K/V, layer, head, dim) quantization across sequence tokens.",
    )
    args = parser.parse_args()

    if not 0 < args.retention <= 1:
        parser.error("--retention must be in (0, 1]")

    rng = np.random.default_rng(args.seed)
    kv = make_synthetic_kv(
        rng=rng,
        layers=args.layers,
        seq_len=args.seq_len,
        heads=args.heads,
        dim=args.dim,
        protected_tokens=args.protected_tokens,
    )

    # Per-channel quantization keeps K/V, layer, head and dim axes, reducing
    # quantization error in a way closer to practical grouped quantization.
    quant_axis = (2,) if args.per_channel else None

    a, mask_a = pipeline_a(kv, args.retention, args.protected_tokens, quant_axis)
    b, mask_b = pipeline_b(kv, args.retention, args.protected_tokens, quant_axis)
    fixed_a, fixed_b = fixed_mask_commutativity_check(kv, args.retention, args.protected_tokens, quant_axis)

    print("KV pipeline order simulation")
    print(f"shape={kv.shape}, retention={args.retention}, protected_tokens={args.protected_tokens}")
    print(f"quantization={'per-channel-over-sequence' if args.per_channel else 'global'}")
    print()
    print("Adaptive retention masks:")
    print(f"  Pipeline A r+b MSE: {mse(kv, a):.8f}")
    print(f"  Pipeline B b+r MSE: {mse(kv, b):.8f}")
    print(f"  mask agreement: {np.mean(mask_a == mask_b) * 100:.2f}%")
    print(f"  protected window kept A: {bool(mask_a[:args.protected_tokens].all())}")
    print(f"  protected window kept B: {bool(mask_b[:args.protected_tokens].all())}")
    print()
    print("Fixed-mask commutativity check:")
    print(f"  R then Q MSE: {fixed_a:.8f}")
    print(f"  Q then R MSE: {fixed_b:.8f}")
    print("  If the mask and quantizer calibration are identical, the two operators are nearly equivalent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

