# Minimal Needle Benchmark

Generated: 2026-05-03T13:23:58+02:00

## Result

| Action | r | b | Position | Mode | Runs | Success | Memory Saved |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: |
| `r30_b16` | 0.30 | 16 | early | proxy | 6 | 0.00 | 70.0% |
| `r30_b4` | 0.30 | 4 | early | proxy | 6 | 0.00 | 92.0% |
| `r30_b16` | 0.30 | 16 | late | proxy | 6 | 1.00 | 70.0% |
| `r30_b4` | 0.30 | 4 | late | proxy | 6 | 0.00 | 92.0% |
| `r30_b16` | 0.30 | 16 | middle | proxy | 6 | 0.00 | 70.0% |
| `r30_b4` | 0.30 | 4 | middle | proxy | 6 | 0.00 | 92.0% |
| `r30_b16` | 0.30 | 16 | sink | proxy | 6 | 1.00 | 70.0% |
| `r30_b4` | 0.30 | 4 | sink | proxy | 6 | 0.00 | 92.0% |
| `r30_b16` | 0.30 | 16 | very_late | proxy | 6 | 1.00 | 70.0% |
| `r30_b4` | 0.30 | 4 | very_late | proxy | 6 | 0.00 | 92.0% |
| `r50_b16` | 0.50 | 16 | early | proxy | 6 | 0.00 | 50.0% |
| `r50_b8` | 0.50 | 8 | early | proxy | 6 | 0.00 | 73.5% |
| `r50_b16` | 0.50 | 16 | late | proxy | 6 | 1.00 | 50.0% |
| `r50_b8` | 0.50 | 8 | late | proxy | 6 | 1.00 | 73.5% |
| `r50_b16` | 0.50 | 16 | middle | proxy | 6 | 0.00 | 50.0% |
| `r50_b8` | 0.50 | 8 | middle | proxy | 6 | 0.00 | 73.5% |
| `r50_b16` | 0.50 | 16 | sink | proxy | 6 | 1.00 | 50.0% |
| `r50_b8` | 0.50 | 8 | sink | proxy | 6 | 1.00 | 73.5% |
| `r50_b16` | 0.50 | 16 | very_late | proxy | 6 | 1.00 | 50.0% |
| `r50_b8` | 0.50 | 8 | very_late | proxy | 6 | 1.00 | 73.5% |
| `r75_b16` | 0.75 | 16 | early | proxy | 6 | 0.00 | 25.0% |
| `r75_b8` | 0.75 | 8 | early | proxy | 6 | 0.00 | 60.2% |
| `r75_b16` | 0.75 | 16 | late | proxy | 6 | 1.00 | 25.0% |
| `r75_b8` | 0.75 | 8 | late | proxy | 6 | 1.00 | 60.2% |

## Completion Check

- Total trials: 240.
- Unique prompt lengths: 2.
- Unique needle positions: 5.
- Failed retrieval trials: 78.
- Logged fields include retention, kv_bits, needle position, protected window, retained target token, and success/failure.

## Interpretation

- Proxy mode measures whether the needle survives the cache policy's retention geometry.
- The protected window preserves sink-position needles, while early and middle needles fail under rotating retention.
- MLX execution can be enabled with `--run-mlx` for backend-feasible actions when a slower semantic run is needed.

## Artifacts

- JSONL log: `logs/benchmarks/needle_minimal_latest.jsonl`
- Results CSV: `reports/checkpoint-6/needle_minimal_results.csv`
- Summary CSV: `reports/checkpoint-6/needle_retrieval_summary.csv`
- Retrieval curve: `reports/checkpoint-6/figures/retrieval_vs_retention.svg`
- Failure map: `reports/checkpoint-6/figures/needle_failure_map.svg`
