# Report Checkpoint 1: Baseline Engine

Data run: 2026-04-27  
Report generato: 2026-04-27T23:13:07+02:00  
Modello: `models/Llama-3.2-1B-4bit`  
Backend: MLX  
Hardware: MacBook Air M1, 8 GB RAM

## Obiettivo

Stabilire una baseline ripetibile di inferenza locale misurando throughput, latenza e memoria al crescere della lunghezza del prompt. Questa baseline usa full KV cache ed e' il riferimento per i checkpoint successivi.

## Risultati Sintetici

- Run analizzati: 9
- File log: logs/baseline/run_llama32_1b.jsonl
- Limite pratico iniziale: long (~3097 token prompt) con full KV cache.
- Peggior Memory Pressure osservata: green
- Picco RSS massimo registrato: 293.67 MB
- Swap massimo registrato nei run: 0.00 MB
- Swap preflight sistema: `vm.swapusage: total = 2048,00M  used = 968,00M  free = 1080,00M  (encrypted)`

| Lunghezza contesto | Run | Prompt tokens | Generated tokens | Avg tokens/sec | Delta performance | Avg ms/token | Peak RSS max MB | Swap delta max MB | Memory Pressure |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| short | 3 | 409 | 129 | 41.826 | Baseline (100%) | 24.426 | 293.7 | 0.0 | green |
| medium | 3 | 1561 | 129 | 26.229 | -37.3% | 38.176 | 98.3 | 0.0 | green |
| long | 3 | 3097 | 129 | 17.237 | -58.8% | 58.023 | 88.0 | 0.0 | green |

## Analisi Tecnica

Il throughput cala da 41.83 tokens/sec a 17.24 tokens/sec passando da 409 a 3097 prompt token, con una variazione complessiva di -58.8%. La latenza media per token cresce in modo coerente con l'aumento del contesto.

La Memory Pressure registrata nei run e' rimasta verde e lo `swap_delta_mb` nei record baseline e' rimasto a 0.0 MB. Il preflight mostrava comunque swap di sistema gia' attivo prima dei run; per benchmark futuri conviene continuare a registrare sia preflight sia metriche per-run.

Il thermal throttling e' marcato come `unknown` nei log perche' non viene misurato automaticamente dallo script. Durante run lunghi va osservato con `mactop` e annotato manualmente se compare throttling.

## Grafici

- Memoria vs lunghezza prompt: `reports/baseline_memory_vs_prompt.svg`
- Tokens/sec vs lunghezza prompt: `reports/baseline_tokens_per_sec_vs_prompt.svg`

## Deliverable

- Runner baseline: `scripts/run_baseline.py`
- Log strutturati: `logs/baseline/run_llama32_1b.jsonl`
- Tabella CSV: `reports/baseline_summary.csv`
- Report: `reports/baseline_report.md`
- Grafici SVG: `reports/baseline_memory_vs_prompt.svg`, `reports/baseline_tokens_per_sec_vs_prompt.svg`

## Conclusioni

Il Checkpoint 1 e' considerato completato: esistono tre lunghezze prompt, tre ripetizioni per lunghezza, log JSONL strutturati, report riassuntivo e grafici iniziali. La baseline full KV cache e' pronta come riferimento per il Checkpoint 2: Memory Observatory.
