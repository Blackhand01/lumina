# Report Checkpoint 5: System Integration & Runtime OS

Generated: 2026-05-02T22:45:17+02:00

## Result

| Regime | Prompt tokens | Selected action | Backend | Est. KV MB | Budget MB | Decision ms | Budget violation |
| --- | ---: | --- | --- | ---: | ---: | ---: | --- |
| under_saturated | 3072 | `full_8` | real | 50.98 | 352.00 | 0.208 | False |
| saturation | 3072 | `dynamic_r514_b16` | real | 49.39 | 49.44 | 0.106 | False |
| contention | 3072 | `dynamic_r514_b16` | real | 49.39 | 49.44 | 0.132 | False |
| under_saturated | 10000 | `full_8` | real | 165.94 | 625.00 | 0.071 | False |
| saturation | 10000 | `dynamic_r514_b16` | real | 160.78 | 160.94 | 0.092 | False |
| contention | 10000 | `dynamic_r514_b16` | real | 160.78 | 160.94 | 0.150 | False |

## Completion Check

- End-to-end runtime decisions from config: yes.
- Distinct selected actions: 2 (`dynamic_r514_b16, full_8`).
- Execution modes: controller.
- Dynamic budget-driven retention enabled: yes.
- Dynamic actions selected: 4.
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

- JSONL log: `logs/adaptive_os/runtime_os_latest.jsonl`
- Runtime decisions CSV: `reports/checkpoint-5/runtime_decisions.csv`
- Budget enforcement CSV: `reports/checkpoint-5/budget_enforcement.csv`
- Cost terms CSV: `reports/checkpoint-5/runtime_cost_terms.csv`
- Runtime action plot: `reports/checkpoint-5/runtime_policy_actions.svg`
