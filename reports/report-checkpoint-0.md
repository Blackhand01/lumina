# Report Checkpoint 0: System Setup

Data esecuzione: 2026-04-27 22:35:32 CEST  
Repository: `/Users/stefanoroybisignano/Desktop/lumina`

## Sintesi

Il Checkpoint 0 e' stato completato per la parte di setup: la struttura base del repository esiste, e' stato creato un ambiente Python locale, MLX e' installato e importabile, lo schema log JSON e' disponibile in `configs/log_schema.json`, e questo report documenta lo stato iniziale della macchina.

La macchina e' utilizzabile per preparazione e test minimi. Per benchmark affidabili del Checkpoint 1 e' consigliato chiudere applicazioni pesanti o riavviare prima della misura, perche' al momento del controllo lo swap era gia' attivo a circa 1.0 GiB.

## Sistema

- Modello macchina: MacBookAir10,1
- CPU/SoC: Apple M1
- Architettura: arm64
- RAM fisica: 8 GiB
- macOS: 26.4.1, build 25E253
- Homebrew: 5.1.8
- `mactop`: v2.1.2

## Stato disco

Output sintetico da `df -h`:

- Volume root `/`: 228 GiB totali, 12 GiB usati, 44 GiB disponibili, 22% capacita'
- Volume dati `/System/Volumes/Data`: 228 GiB totali, 161 GiB usati, 44 GiB disponibili, 79% capacita'
- Volume VM `/System/Volumes/VM`: 2.0 GiB usati, 44 GiB disponibili

Cache e directory rilevanti:

- Cache HuggingFace: 4.9 GiB in `~/.cache/huggingface/hub`
- Download: nessun elemento rilevante per spazio; il file piu' grande rilevato e' circa 6.4 MiB

Valutazione: lo spazio e' sufficiente per il setup e per esperimenti piccoli. Prima di scaricare modelli grandi, considerare la pulizia della cache HuggingFace o verificare nuovamente `df -h`.

## Stato memoria

Controlli eseguiti:

```text
PhysMem: 7463M used (1252M wired, 2724M compressor), 170M unused.
vm.swapusage: total = 2048.00M  used = 1032.00M  free = 1016.00M  (encrypted)
System-wide memory free percentage: 47%
```

Valutazione: la memoria non mostra blocchi critici, ma lo swap e la memoria compressa sono gia' significativi a riposo. Per benchmark, registrare sempre swap e memoria compressa prima e dopo ogni run.

## Ambiente Python

Ambiente creato:

- Percorso: `.venv`
- Python: 3.11.7
- Eseguibile: `/Users/stefanoroybisignano/Desktop/lumina/.venv/bin/python`
- MLX: 0.31.2

Nota: il comando globale `python3` punta a un ambiente PlatformIO (`/Users/stefanoroybisignano/.platformio/penv/bin/python3`). Per questo progetto usare sempre l'ambiente locale:

```bash
source .venv/bin/activate
python -c "import mlx.core as mx; print(mx.sum(mx.array([1, 2, 3])).item())"
```

Test minimo eseguito con successo:

```text
mlx 0.31.2
mlx_sum 6
```

## Struttura repository

Cartelle base presenti:

```text
checkpoints/
configs/
data/
docs/
docs/blueprints/
logs/
notebooks/
reports/
scripts/
src/
```

Deliverable creati o verificati:

- `reports/system_report.md`
- `configs/log_schema.json`
- `.venv/` locale con MLX installato
- struttura repository base

## Schema log

Il formato minimo per gli esperimenti e' stato definito in `configs/log_schema.json`. I campi obbligatori includono:

- `run_id`
- `timestamp`
- `model`
- `backend`
- `quantization`
- `prompt_tokens`
- `generated_tokens`
- `cache_policy`
- `peak_memory_mb`
- `swap_mb`
- `memory_pressure`
- `compressed_memory_mb`
- `latency_ms_per_token`
- `tokens_per_second`
- `benchmark`
- `score`
- `notes`

## Criteri di completamento

- Mac in stato documentato: completato
- Spazio disco verificato: completato
- Swap misurato: completato, con nota operativa per benchmark
- MLX importabile: completato
- Schema log disponibile: completato
- Cartelle base del progetto: completato
- Test Python minimo: completato

## Raccomandazioni per Checkpoint 1

1. Attivare sempre `.venv` prima di eseguire script o notebook.
2. Registrare `df -h`, `sysctl vm.swapusage` e memoria compressa prima di ogni benchmark significativo.
3. Chiudere app pesanti o riavviare se lo swap resta intorno a 1 GiB prima del test.
4. Usare `mactop` durante esperimenti lunghi per osservare CPU/GPU, memoria e swap.
5. Salvare ogni risultato futuro secondo `configs/log_schema.json`.
