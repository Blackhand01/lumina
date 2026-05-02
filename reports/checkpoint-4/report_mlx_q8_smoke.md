# Report Checkpoint 4: Regime-Aware Hybrid Adaptive Policy Engine

Generato: 2026-05-02T21:15:24+02:00

## Risultato

| Regime | Prompt tokens | Budget MB | Selected action | Backend | Est. KV MB | Score | Reason |
| --- | ---: | ---: | --- | --- | ---: | ---: | --- |
| under_saturated | 3072 | 352.00 | `full_8` | real | 50.98 | 1.327 | backend path available |

## Completion Check

- Azioni distinte selezionate: 1 (`full_8`).
- q8 domina in under-saturated: yes.
- Policy shift sotto stress: no.
- Azioni simulate presenti: no.

## Interpretazione

- `full_8` e' atteso come scelta dominante quando Memory Pressure e swap non sono un vincolo.
- Le azioni `r < 1, b < 16` restano infeasible nel backend MLX corrente se la quantizzazione di `RotatingKVCache` non e' disponibile.
- Retention-only `r50_b16` e' il fallback reale per stressare budget stretti senza simulare 4-bit KV.
- Questo report valida il controller e il logging; le affermazioni finali richiedono run MLX e benchmark retrieval.

## Artefatti

- Log JSONL: `logs/policy/policy_mlx_q8_smoke.jsonl`
- Sommario regime/action: `reports/checkpoint-4/regime_action_summary_mlx_q8_smoke.csv`
- Costi per azione: `reports/checkpoint-4/policy_cost_terms_mlx_q8_smoke.csv`
- Grafico regime/action: `reports/checkpoint-4/retention_precision_by_regime_mlx_q8_smoke.svg`
