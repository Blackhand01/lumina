# Report Checkpoint 3: Compression Engine

Generato: 2026-04-28T11:54:47+02:00

## Tabella della Verita'

| Metodo | Risparmio RAM | Errore qualita' | Latenza ms/tok | Overhead latenza | Verdict |
| --- | --- | --- | --- | --- | --- |
| full | 0.0% | 0.000 | 50.39 | 0.0% | Baseline |
| eviction | 27.5% | 0.474 | 50.29 | -3.0% | Qualita' debole |
| pca_35 | 48.5% | 0.891 | 64.00 | 21.0% | Qualita' debole |
| quantized_8bit | 46.9% | 0.000 | 50.55 | -1.8% | Ottimo |

## Vincitore

Metodo candidato: `quantized_8bit`. Risparmio medio 46.9%, errore qualita' 0.000, latenza 50.55 ms/token.

## Note Tecniche

- `full` e' il controllo con KV cache piena.
- `quantized_8bit` usa il supporto MLX per KV cache quantizzata.
- `pca_35` usa proiettori PCA offline dai dump del Checkpoint 2 e ricostruisce la cache full quando il modello legge K/V.
- `eviction` usa una rotating cache: conserva l'inizio e la finestra piu' recente.
- L'errore qualita' e' misurato come drift testuale rispetto alla full cache sullo stesso prompt.

## Artefatti

- Log JSONL: `logs/compression/compression_20260428_115401.jsonl`
- Sommario CSV: `reports/compression_summary.csv`
- Grafico trade-off: `reports/compression_tradeoff.svg`
