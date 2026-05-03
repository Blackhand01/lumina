# Report Checkpoint 5: System Integration & Runtime OS

Generated: 2026-05-02T22:45:24+02:00

## Result

| Regime | Prompt tokens | Selected action | Backend | Est. KV MB | Budget MB | Decision ms | Budget violation |
| --- | ---: | --- | --- | ---: | ---: | ---: | --- |
| under_saturated | 3072 | `full_8` | real | 50.98 | 352.00 | 0.529 | False |

## Completion Check

- End-to-end runtime decisions from config: yes.
- Distinct selected actions: 1 (`full_8`).
- Execution modes: mlx.
- Dynamic budget-driven retention enabled: yes.
- Dynamic actions selected: 0.
- Budget violations: 0.
- Decision overhead failures: 0.
- Every decision logs cost, feasibility, backend status, and telemetry snapshot: yes.

## Interpretation

- Checkpoint 5 integrates controller, model KV profile, memory budget, telemetry, and report generation.
- The runtime currently performs controller-level decisions by default; real MLX execution is available with `--run-mlx` for feasible actions.
- Saturation and contention regimes are represented through explicit budget and pressure priors unless an external pressure process is used.
- Budget-driven retention computes the smallest required token drop for each candidate precision instead of using only coarse retention steps.
- Dynamic `r<1,b=8` remains backend-infeasible in standard MLX when retained quantized KV is not available; dynamic `r<1,b=16` is the real fallback.

## Artifacts

- JSONL log: `logs/adaptive_os/runtime_os_mlx_smoke.jsonl`
- Runtime decisions CSV: `reports/checkpoint-5/runtime_decisions_mlx_smoke.csv`
- Budget enforcement CSV: `reports/checkpoint-5/budget_enforcement_mlx_smoke.csv`
- Cost terms CSV: `reports/checkpoint-5/runtime_cost_terms_mlx_smoke.csv`
- Runtime action plot: `reports/checkpoint-5/runtime_policy_actions_mlx_smoke.svg`
