# Orion Voice — Experiment Program

## Single Metric
**Cleartext verification accuracy (%)** — then FHE accuracy + FHE-Clear agreement.

## Fixed Infrastructure (do not modify)
- `speaker_verify/features.py` — feature extraction logic
- `speaker_verify/dataset.py` — dataset loading and pair generation
- `speaker_verify/fhe_inference.py` — FHE pipeline
- `configs/fhe_config.yml` — CKKS parameters
- `demo/server.py`, `demo/client.py` — client-server flow
- `tests/` — all test files

## Experiment Surface (iterate on)

| Parameter | Current | Candidates |
|-----------|---------|------------|
| n_mfcc | 20 | 13, 20, 26 |
| embedding | mean+std | mean+std, mean+std+delta |
| model_width | 128→64 | 64→32, 128→64, 256→128 |
| activation_degree | 7 | 5, 7, 9 |
| learning_rate | 1e-3 | 1e-4, 5e-4, 1e-3, 5e-3 |
| epochs | 200 | 100, 200, 500 |
| n_pairs | 2000 | 1000, 2000, 5000 |

## Experiment Loop

1. Modify hyperparameters in `demo/train_model.py` args
2. Train: `python demo/train_model.py --n-pairs N --epochs E --lr LR`
3. Record cleartext accuracy
4. If improved: run `python demo/fhe_demo.py --num-samples 10`
5. Record FHE accuracy + FHE-Clear agreement
6. Log to `results.tsv`
7. Keep best, discard rest

## Results Format (`results.tsv`)

```
experiment	n_mfcc	width	degree	lr	epochs	n_pairs	clear_acc	fhe_acc	fhe_clear_agree	fhe_time_avg	status
baseline	20	128-64	7	1e-3	200	2000	0.82	0.80	9/10	16.1	ok
```
