#!/usr/bin/env python3
"""Run offline PCA diagnostics over extracted KV cache tensors."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def tensor_files(sample_dir: Path) -> list[Path]:
    return sorted(sample_dir.glob("layer_*_*.npy"))


def parse_tensor_name(path: Path) -> tuple[int, str]:
    match = re.match(r"layer_(\d+)_(key|value)\.npy", path.name)
    if not match:
        raise ValueError(f"unexpected tensor file name: {path}")
    return int(match.group(1)), match.group(2)


def pca_matrix(array: np.ndarray, max_samples: int) -> np.ndarray:
    # Cache layout: [batch, kv_heads, tokens, head_dim]. Use token/head rows
    # and head_dim columns so PCA measures redundancy within the projected head.
    if array.ndim != 4:
        raise ValueError(f"expected 4D KV tensor, got shape={array.shape}")
    _, heads, tokens, dim = array.shape
    matrix = array.astype(np.float32, copy=False).transpose(0, 2, 1, 3).reshape(-1, dim)
    if len(matrix) > max_samples:
        indices = np.linspace(0, len(matrix) - 1, max_samples).astype(np.int64)
        matrix = matrix[indices]
    return matrix


def pca_stats(matrix: np.ndarray) -> dict[str, Any]:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    if len(matrix) <= 1:
        explained = np.zeros_like(singular_values)
    else:
        explained = (singular_values**2) / (len(matrix) - 1)
    total = float(explained.sum())
    ratio = explained / total if total > 0 else np.zeros_like(explained)
    cumulative = np.cumsum(ratio)

    def components_for(threshold: float) -> int:
        hits = np.where(cumulative >= threshold)[0]
        return int(hits[0] + 1) if len(hits) else int(len(cumulative))

    return {
        "explained_variance_ratio": ratio.tolist(),
        "components_90": components_for(0.90),
        "components_95": components_for(0.95),
        "components_99": components_for(0.99),
        "top1_variance": float(ratio[0]) if len(ratio) else 0.0,
        "top4_variance": float(cumulative[min(3, len(cumulative) - 1)]) if len(cumulative) else 0.0,
    }


def token_energy(array: np.ndarray) -> np.ndarray:
    # Mean L2 energy per token across batch/head/dim. Used as a light proxy for
    # token informativeness in the cache.
    values = array.astype(np.float32, copy=False)
    return np.sqrt((values * values).mean(axis=(0, 1, 3)))


def analyze_sample(sample_dir: Path, max_samples: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metadata_path = sample_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    prompt_label = str(metadata.get("prompt_label", sample_dir.name))
    prompt_tokens = int(metadata.get("prompt_tokens", 0))

    rows: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []
    for path in tensor_files(sample_dir):
        layer, kind = parse_tensor_name(path)
        array = np.load(path, mmap_mode="r")
        matrix = pca_matrix(array, max_samples=max_samples)
        stats = pca_stats(matrix)
        shape = list(array.shape)
        rows.append(
            {
                "sample_dir": str(sample_dir),
                "prompt_label": prompt_label,
                "prompt_tokens": prompt_tokens,
                "layer": layer,
                "kind": kind,
                "shape": "x".join(str(item) for item in shape),
                "dtype": str(array.dtype),
                "nbytes": int(array.nbytes),
                "n_features": int(matrix.shape[1]),
                "n_samples": int(matrix.shape[0]),
                "components_90": stats["components_90"],
                "components_95": stats["components_95"],
                "components_99": stats["components_99"],
                "top1_variance": stats["top1_variance"],
                "top4_variance": stats["top4_variance"],
                "explained_variance_ratio": json.dumps(stats["explained_variance_ratio"]),
            }
        )
        energies = token_energy(array)
        if len(energies):
            top_indices = np.argsort(energies)[-10:][::-1]
            for rank, token_index in enumerate(top_indices, start=1):
                token_rows.append(
                    {
                        "sample_dir": str(sample_dir),
                        "prompt_label": prompt_label,
                        "prompt_tokens": prompt_tokens,
                        "layer": layer,
                        "kind": kind,
                        "rank": rank,
                        "token_index": int(token_index),
                        "energy": float(energies[token_index]),
                    }
                )
    return rows, token_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_svg_bars(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[f"{row['prompt_label']}:{row['kind']}"].append(row)

    width = 920
    height = 480
    left = 64
    top = 44
    plot_width = width - left - 36
    plot_height = height - top - 72
    series = sorted(grouped.items())
    max_layer = max((int(row["layer"]) for row in rows), default=0)
    max_value = max((float(row["components_95"]) for row in rows), default=1.0)
    colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706", "#0891b2"]
    polylines: list[str] = []
    legend: list[str] = []
    for index, (name, items) in enumerate(series):
        points = []
        item_by_layer = {int(item["layer"]): item for item in items}
        for layer in range(max_layer + 1):
            item = item_by_layer.get(layer)
            if not item:
                continue
            x = left + (layer / max(max_layer, 1)) * plot_width
            y = top + plot_height - (float(item["components_95"]) / max(max_value, 1.0)) * plot_height
            points.append(f"{x:.1f},{y:.1f}")
        color = colors[index % len(colors)]
        polylines.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2" />')
        legend.append(f'<text x="{left + (index % 3) * 250}" y="{height - 44 + (index // 3) * 16}" font-size="12" fill="{color}">{name}</text>')

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white" />
  <text x="{width / 2}" y="26" text-anchor="middle" font-size="18" font-family="Arial">{title}</text>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111827" />
  <line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#111827" />
  <text x="{width / 2}" y="{height - 12}" text-anchor="middle" font-size="12">layer</text>
  <text x="20" y="{top + plot_height / 2}" transform="rotate(-90 20,{top + plot_height / 2})" text-anchor="middle" font-size="12">components for 95% variance</text>
  {chr(10).join(polylines)}
  {chr(10).join(legend)}
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def write_report(path: Path, rows: list[dict[str, Any]], token_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("# Report Checkpoint 2: Memory Observatory\n\nNessun dump analizzato.\n", encoding="utf-8")
        return

    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_kind[str(row["kind"])].append(row)

    def avg_components(kind: str, field: str) -> float:
        items = by_kind.get(kind, [])
        return float(np.mean([float(item[field]) for item in items])) if items else 0.0

    most_compressible = sorted(rows, key=lambda row: (float(row["components_95"]), str(row["kind"])))[:8]
    least_compressible = sorted(rows, key=lambda row: (float(row["components_95"]), str(row["kind"])), reverse=True)[:8]

    def table(items: list[dict[str, Any]]) -> str:
        lines = [
            "| Prompt | Layer | K/V | Comp 90 | Comp 95 | Comp 99 | Top1 var |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for item in items:
            lines.append(
                f"| {item['prompt_label']} | {item['layer']} | {item['kind']} | "
                f"{item['components_90']} | {item['components_95']} | {item['components_99']} | "
                f"{float(item['top1_variance']):.3f} |"
            )
        return "\n".join(lines)

    content = (
        "# Report Checkpoint 2: Memory Observatory\n\n"
        f"Generato: {dt.datetime.now().astimezone().isoformat(timespec='seconds')}\n\n"
        "## Sintesi\n\n"
        f"- Tensori analizzati: {len(rows)}\n"
        f"- Campioni cache: {len({row['sample_dir'] for row in rows})}\n"
        f"- Componenti medie per 95% varianza, key: {avg_components('key', 'components_95'):.2f}\n"
        f"- Componenti medie per 95% varianza, value: {avg_components('value', 'components_95'):.2f}\n\n"
        "## Layer Piu' Comprimibili\n\n"
        f"{table(most_compressible)}\n\n"
        "## Layer Meno Comprimibili\n\n"
        f"{table(least_compressible)}\n\n"
        "## Ipotesi Operative\n\n"
        "- I layer con meno componenti per preservare il 95% della varianza sono candidati iniziali per compressione KV.\n"
        "- Key e value vanno trattati separatamente: se le componenti medie divergono, usare policy diverse.\n"
        "- I token ad alta energia in `reports/figures/token_energy_top.csv` sono candidati da preservare con priorita'.\n\n"
        "## Limiti\n\n"
        "- PCA offline diagnostica: non misura ancora impatto su accuracy o perplexity.\n"
        "- Le conclusioni dipendono dai prompt estratti; servono short, medium e long per una strategia robusta.\n"
        "- Le curve indicano ridondanza lineare, non garantiscono compressione runtime senza perdita.\n\n"
        "## Artefatti\n\n"
        "- Sommario PCA: `reports/figures/cache_pca_summary.csv`\n"
        "- Token energy: `reports/figures/token_energy_top.csv`\n"
        "- Grafico PCA: `reports/figures/pca_components_95_by_layer.svg`\n"
    )
    path.write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=Path("data/cache_samples"))
    parser.add_argument("--figures", type=Path, default=Path("reports/figures"))
    parser.add_argument("--report", type=Path, default=Path("reports/memory_observatory.md"))
    parser.add_argument("--max-samples", type=int, default=50000)
    args = parser.parse_args()

    sample_dirs = [path for path in sorted(args.samples.iterdir()) if path.is_dir()] if args.samples.exists() else []
    rows: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []
    for sample_dir in sample_dirs:
        sample_rows, sample_token_rows = analyze_sample(sample_dir, args.max_samples)
        rows.extend(sample_rows)
        token_rows.extend(sample_token_rows)

    write_csv(args.figures / "cache_pca_summary.csv", rows)
    write_csv(args.figures / "token_energy_top.csv", token_rows)
    write_svg_bars(args.figures / "pca_components_95_by_layer.svg", rows, "PCA Components by Layer")
    write_report(args.report, rows, token_rows)
    print(f"analyzed {len(rows)} tensors")
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
