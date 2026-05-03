# Report Checkpoint 6: Evaluation & System Gap Analysis

Generated: 2026-05-03T13:45:16+02:00

## Evaluation Tiers

- Tier 1: systems validation across under-saturated, saturation, and contention regimes.
- Tier 2: retrieval robustness with Needle-in-a-Haystack.
- Tier 3: semantic retention with a local 24-example Document QA set plus lightweight proxy tasks.

## Result

| Regime | Method | Action | KV MB | Swap | Pageout | Latency/token | Score | Verdict |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| under_saturated | Full KV | `full_16` | 204.25 | 0.00 | 0.00 | 50.00 | 1.000 | valid baseline |
| under_saturated | Quantized 8-bit | `full_8` | 108.46 | 0.00 | 0.00 | 51.00 | 1.000 | q8 dominance check |
| under_saturated | Retention-only | `r50_b16` | 102.12 | 0.00 | 0.00 | 47.50 | 0.373 | quality failure mode |
| under_saturated | Hybrid conservative | `r75_b8` | 81.34 | 0.00 | 0.00 | 49.75 | 0.688 | theoretical frontier only |
| under_saturated | Hybrid balanced | `r50_b8` | 54.23 | 0.00 | 0.00 | 48.50 | 0.373 | theoretical frontier only |
| under_saturated | Hybrid aggressive | `r30_b4` | 16.27 | 0.00 | 0.00 | 51.50 | 0.325 | theoretical frontier only |
| under_saturated | Ideal dynamic q8 | `ideal_full_8` | 108.46 | 0.00 | 0.00 | 51.00 | 1.000 | theoretical frontier only |
| under_saturated | Adaptive Lumina | `full_8` | 108.46 | 0.00 | 0.00 | 51.00 | 1.000 | valid baseline |
| saturation | Full KV | `full_16` | 204.25 | 93.20 | 48.54 | 80.90 | 0.980 | budget violation |
| saturation | Quantized 8-bit | `full_8` | 108.46 | 49.49 | 25.78 | 56.97 | 0.980 | budget violation |
| saturation | Retention-only | `r50_b16` | 102.12 | 46.60 | 24.27 | 52.25 | 0.373 | quality failure mode |
| saturation | Hybrid conservative | `r75_b8` | 81.34 | 37.12 | 19.33 | 54.73 | 0.688 | theoretical frontier only |
| saturation | Hybrid balanced | `r50_b8` | 54.23 | 24.75 | 12.89 | 53.35 | 0.373 | theoretical frontier only |
| saturation | Hybrid aggressive | `r30_b4` | 16.27 | 7.42 | 3.87 | 56.65 | 0.325 | theoretical frontier only |
| saturation | Ideal dynamic q8 | `ideal_r969_b8` | 105.08 | 47.95 | 24.98 | 55.93 | 1.000 | theoretical frontier only |
| saturation | Adaptive Lumina | `dynamic_r514_b16` | 105.08 | 47.95 | 24.98 | 52.33 | 0.682 | adaptive budget fallback |
| contention | Full KV | `full_16` | 204.25 | 186.41 | 155.34 | 85.31 | 0.980 | budget violation |
| contention | Quantized 8-bit | `full_8` | 108.46 | 98.98 | 82.48 | 60.08 | 0.980 | budget violation |
| contention | Retention-only | `r50_b16` | 102.12 | 93.20 | 77.67 | 55.10 | 0.373 | quality failure mode |
| contention | Hybrid conservative | `r75_b8` | 81.34 | 74.24 | 61.86 | 57.71 | 0.688 | theoretical frontier only |
| contention | Hybrid balanced | `r50_b8` | 54.23 | 49.49 | 41.24 | 56.26 | 0.373 | theoretical frontier only |
| contention | Hybrid aggressive | `r30_b4` | 16.27 | 14.85 | 12.37 | 59.74 | 0.325 | theoretical frontier only |
| contention | Ideal dynamic q8 | `ideal_r969_b8` | 105.08 | 95.90 | 79.92 | 58.98 | 1.000 | theoretical frontier only |
| contention | Adaptive Lumina | `dynamic_r514_b16` | 105.08 | 95.90 | 79.92 | 55.18 | 0.682 | adaptive budget fallback |

## Policy Shift Summary

| Regime | Selected action | Reason | Quality delta | Memory delta | Swap/Pageout delta |
| --- | --- | --- | ---: | ---: | ---: |
| under_saturated | `full_8` | backend path available | 0.000 | 46.9% | 0.0 |
| saturation | `dynamic_r514_b16` | fp16/bf16 KV path available | 0.318 | 48.6% | 72.9 |
| contention | `dynamic_r514_b16` | fp16/bf16 KV path available | 0.318 | 48.6% | 175.8 |

## Lost Intelligence Summary

| Scenario | Regime | Ideal score | Executable score | Lost Intelligence | Ideal policy | Executable policy |
| --- | --- | ---: | ---: | ---: | --- | --- |
| document_qa | contention | 1.000 | 0.667 | 0.333 | `ideal_r969_b8` | `dynamic_r514_b16` |
| document_qa | saturation | 1.000 | 0.667 | 0.333 | `ideal_r969_b8` | `dynamic_r514_b16` |
| document_qa | under_saturated | 1.000 | 1.000 | 0.000 | `ideal_full_8` | `full_8` |
| long_context | contention | 0.985 | 0.663 | 0.322 | `ideal_r969_b8` | `dynamic_r514_b16` |
| long_context | saturation | 0.985 | 0.663 | 0.322 | `ideal_r969_b8` | `dynamic_r514_b16` |
| long_context | under_saturated | 1.000 | 1.000 | 0.000 | `ideal_full_8` | `full_8` |
| multi_turn | contention | 1.000 | 0.717 | 0.283 | `ideal_r969_b8` | `dynamic_r514_b16` |
| multi_turn | saturation | 1.000 | 0.717 | 0.283 | `ideal_r969_b8` | `dynamic_r514_b16` |
| multi_turn | under_saturated | 1.000 | 1.000 | 0.000 | `ideal_full_8` | `full_8` |
| needle | contention | 1.000 | 0.750 | 0.250 | `ideal_r969_b8` | `dynamic_r514_b16` |
| needle | saturation | 1.000 | 0.750 | 0.250 | `ideal_r969_b8` | `dynamic_r514_b16` |
| needle | under_saturated | 1.000 | 1.000 | 0.000 | `ideal_full_8` | `full_8` |

## Completion Check

- Benchmark scenarios evaluated: 4.
- Regimes evaluated: contention, saturation, under_saturated.
- q8 under-saturated result present: yes.
- Adaptive stress-regime result present: yes.
- Theoretical-only records marked: 768.
- Lost Intelligence groups computed: 12.
- Task scores are deterministic Checkpoint 6 proxies; they are not final LLM accuracy claims.

## Interpretation

- Under-saturated runs preserve the expected q8-dominant regime.
- Saturation and contention force Adaptive Lumina to use budget-driven retention because static q8 exceeds the configured KV budget.
- Hybrid retained-quantized actions remain theoretical-only under the current MLX feasible set.
- Needle and Document QA proxy scores expose the known failure mode: naive rotating retention preserves sinks and recent tokens but can drop early and middle evidence.
- Lost Intelligence is reported as an upper-bound estimate: ideal dynamic q8 is theoretical-only when retained quantized KV is not executable.

## Artifacts

- JSONL log: `logs/benchmarks/checkpoint6_latest.jsonl`
- Scenario results CSV: `reports/checkpoint-6/scenario_results.csv`
- Regime summary CSV: `reports/checkpoint-6/regime_summary.csv`
- Policy shift CSV: `reports/checkpoint-6/policy_shift_summary.csv`
- Lost Intelligence CSV: `reports/checkpoint-6/lost_intelligence.csv`
- Lost Intelligence summary CSV: `reports/checkpoint-6/lost_intelligence_summary.csv`
- Figure: `reports/checkpoint-6/figures/memory_vs_context.svg`
- Figure: `reports/checkpoint-6/figures/policy_decisions_over_regimes.svg`
- Figure: `reports/checkpoint-6/figures/score_vs_memory_saved.svg`
- Figure: `reports/checkpoint-6/figures/latency_by_policy.svg`
- Figure: `reports/checkpoint-6/figures/swap_pageout_by_policy.svg`
- Figure: `reports/checkpoint-6/figures/retention_failure_by_needle_position.svg`
- Figure: `reports/checkpoint-6/figures/pareto_theoretical_vs_feasible.svg`
- Figure: `reports/checkpoint-6/figures/lost_intelligence_by_regime.svg`
