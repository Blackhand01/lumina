# Report Checkpoint 2: Memory Observatory

Generato: 2026-04-27T23:33:06+02:00

## Sintesi

- Tensori analizzati: 96
- Campioni cache: 3
- Componenti medie per 95% varianza, key: 44.23
- Componenti medie per 95% varianza, value: 52.85

## Layer Piu' Comprimibili

| Prompt | Layer | K/V | Comp 90 | Comp 95 | Comp 99 | Top1 var |
| --- | --- | --- | --- | --- | --- | --- |
| short | 3 | key | 28 | 35 | 54 | 0.228 |
| medium | 3 | key | 31 | 37 | 53 | 0.218 |
| long | 3 | key | 33 | 39 | 53 | 0.216 |
| medium | 9 | key | 31 | 39 | 55 | 0.191 |
| short | 4 | key | 31 | 39 | 56 | 0.144 |
| short | 9 | key | 29 | 39 | 56 | 0.202 |
| long | 9 | key | 31 | 40 | 54 | 0.185 |
| medium | 4 | key | 34 | 41 | 55 | 0.137 |

## Layer Meno Comprimibili

| Prompt | Layer | K/V | Comp 90 | Comp 95 | Comp 99 | Top1 var |
| --- | --- | --- | --- | --- | --- | --- |
| short | 1 | value | 48 | 55 | 62 | 0.057 |
| short | 2 | value | 47 | 55 | 62 | 0.067 |
| short | 3 | value | 48 | 55 | 62 | 0.065 |
| short | 11 | value | 48 | 55 | 62 | 0.061 |
| long | 1 | value | 47 | 54 | 62 | 0.061 |
| long | 2 | value | 46 | 54 | 62 | 0.073 |
| long | 14 | value | 47 | 54 | 62 | 0.046 |
| long | 15 | value | 46 | 54 | 62 | 0.105 |

## Ipotesi Operative

- I layer con meno componenti per preservare il 95% della varianza sono candidati iniziali per compressione KV.
- Key e value vanno trattati separatamente: se le componenti medie divergono, usare policy diverse.
- I token ad alta energia in `reports/figures/token_energy_top.csv` sono candidati da preservare con priorita'.

## Limiti

- PCA offline diagnostica: non misura ancora impatto su accuracy o perplexity.
- Le conclusioni dipendono dai prompt estratti; servono short, medium e long per una strategia robusta.
- Le curve indicano ridondanza lineare, non garantiscono compressione runtime senza perdita.

## Artefatti

- Sommario PCA: `reports/figures/cache_pca_summary.csv`
- Token energy: `reports/figures/token_energy_top.csv`
- Grafico PCA: `reports/figures/pca_components_95_by_layer.svg`
