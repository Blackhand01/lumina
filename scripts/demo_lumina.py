#!/usr/bin/env python3
"""Print a compact Lumina demo summary from Checkpoint 6 artifacts."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"missing artifact: {path}. Run scripts/run_benchmarks.py first.")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=Path("reports/checkpoint-6/regime_summary.csv"))
    parser.add_argument("--policy-shift", type=Path, default=Path("reports/checkpoint-6/policy_shift_summary.csv"))
    args = parser.parse_args()

    summary = read_rows(args.summary)
    shifts = read_rows(args.policy_shift)

    print("Lumina-1 Checkpoint 6 Demo")
    print("")
    print("Policy shifts:")
    for row in shifts:
        print(
            f"- {row['regime']}: {row['selected_action']} "
            f"(memory saved {row['memory_delta_percent']}%, quality delta {row['quality_delta']})"
        )

    print("")
    print("Baseline comparison:")
    for row in summary:
        if row["method"] in {"Full KV", "Quantized 8-bit", "Adaptive Lumina"}:
            print(
                f"- {row['regime']} / {row['method']}: action={row['action']}, "
                f"KV={row['kv_cache_mb']} MB, score={row['score']}, verdict={row['verdict']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
