# Report Checkpoint 4: Regime-Aware Hybrid Adaptive Policy Engine

Generato: 2026-05-02T21:41:14+02:00

## Risultato

| Regime | Prompt tokens | Budget MB | Selected action | Backend | Est. KV MB | Score | Reason |
| --- | ---: | ---: | --- | --- | ---: | ---: | --- |
| under_saturated | 3072 | 352.00 | `full_8` | real | 50.98 | 1.327 | backend path available |
| saturation | 3072 | 49.44 | `r50_b16` | real | 48.00 | 15.826 | fp16/bf16 KV path available |
| contention | 3072 | 49.44 | `r50_b16` | real | 48.00 | 26.787 | fp16/bf16 KV path available |
| under_saturated | 10000 | 625.00 | `full_8` | real | 165.94 | 1.583 | backend path available |
| saturation | 10000 | 160.94 | `r50_b16` | real | 156.25 | 15.826 | fp16/bf16 KV path available |
| contention | 10000 | 160.94 | `r50_b16` | real | 156.25 | 26.787 | fp16/bf16 KV path available |

## Completion Check

- Azioni distinte selezionate: 2 (`full_8, r50_b16`).
- q8 domina in under-saturated: yes.
- Policy shift sotto stress: yes.
- Azioni simulate presenti: no.

## Interpretazione

- `full_8` e' atteso come scelta dominante quando Memory Pressure e swap non sono un vincolo.
- Le azioni `r < 1, b < 16` restano infeasible nel backend MLX corrente se la quantizzazione di `RotatingKVCache` non e' disponibile.
- Retention-only `r50_b16` e' il fallback reale per stressare budget stretti senza simulare 4-bit KV.
- Questo report valida il controller e il logging; le affermazioni finali richiedono run MLX e benchmark retrieval.

## Backend-Induced Policy Collapse

Il risultato principale non e' solo che la policy cambia da `full_8` a `r50_b16`. Il risultato e' che lo spazio teorico delle azioni `A = {(r,b)}` viene ristretto dal backend MLX a un feasible set piu' piccolo `F subset A`.

In pratica, Lumina vede azioni ibride teoricamente interessanti come `(0.75,8)` e `(0.5,8)`, ma il backend corrente non puo' eseguirle come punti reali perche' retention e quantizzazione KV non sono componibili nel percorso cache disponibile.

- Punti MLX-feasible nel plot Pareto: 10.
- Punti theoretical-only nel plot Pareto: 32.
- Nome del fenomeno: **Backend-Induced Policy Collapse**.

## Pareto Feasible Set

Tabella per regime `saturation` e 10000 prompt token, utile come lettura paper-style della frontiera:

| Action | r | b | Memory saved | Quality prior | Feasible set | Collapse reason |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `full_16` | 1.00 | 16 | 0.0% | 0.000 | theoretical_only | estimated KV 312.50 MB exceeds budget 160.94 MB |
| `full_8` | 1.00 | 8 | 46.9% | 0.000 | theoretical_only | estimated KV 165.94 MB exceeds budget 160.94 MB |
| `r50_b16` | 0.50 | 16 | 50.0% | 0.438 | mlx_feasible |  |
| `r75_b8` | 0.75 | 8 | 60.2% | 0.150 | theoretical_only | RotatingKVCache quantization NYI |
| `r50_b8` | 0.50 | 8 | 73.5% | 0.438 | theoretical_only | RotatingKVCache quantization NYI |
| `r50_b4` | 0.50 | 4 | 86.7% | 0.562 | theoretical_only | kv_bits=4 not supported by backend |
| `r30_b4` | 0.30 | 4 | 92.0% | 0.938 | theoretical_only | kv_bits=4 not supported by backend |

Claim aggiornato:

> Lumina is a constraint-aware controller that exposes the gap between optimal memory allocation policies and backend-feasible execution.

## Artefatti

- Log JSONL: `logs/policy/policy_20260502_214114.jsonl`
- Sommario regime/action: `reports/checkpoint-4/regime_action_summary.csv`
- Costi per azione: `reports/checkpoint-4/policy_cost_terms.csv`
- Pareto feasible set CSV: `reports/checkpoint-4/pareto_feasible_set.csv`
- Grafico regime/action: `reports/checkpoint-4/retention_precision_by_regime.svg`
- Grafico Pareto feasible set: `reports/checkpoint-4/pareto_feasible_set.svg`
