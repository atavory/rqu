# Exact Unlearning via Quantized Sufficient Statistics

Code for the paper "Exact Unlearning via Quantized Sufficient Statistics" (NeurIPS 2026 submission).

## Overview

QSS achieves O(1) exact machine unlearning by separating a model into a frozen
*schema* (encoder + RQ codebook, trained on a small retained set D°) and mutable
*content* (sum-decomposable accumulators over all data). Unlearning a point is a
constant-time subtraction from the accumulators.

Pipeline: frozen encoder → L2-normalize → Faiss RQ encoding → accumulator tables → prediction.

## Setup

```bash
pip install numpy scipy faiss-cpu scikit-learn lightgbm
```

## Code Structure

```
algs/
  qss.py          # Core QSS: RQ codebook training, encoding, accumulator
                  #   build/predict, boosted QSS (g + α·QSS), evaluation
  lgbm.py         # LightGBM leaf-contribution feature extraction for tabular data

system/
  t10_unified_runner.py          # Main experiment runner (all datasets, all modes)
  sisa_timing_runner.py          # SISA baseline timing
  rebuild_timing_lgbm.py         # Codebook rebuild timing (tabular)
  deletion_timing.py             # Accumulator deletion timing
  fixed_rho_lgbm_runner.py       # Fixed-ρ experiments (tabular)
  download_largescale_datasets.py # Dataset download utilities
```

## Key Design

- **Accumulators are sum-decomposable**: S0[level, centroid, class] counts,
  n[level, centroid] counts, N_co[(k,m)][c_k, c_m] co-occurrence counts.
  Unlearning subtracts one point's contributions — O(ℓ) per deletion.
- **Boosted QSS**: base model g (LP or MLP on D°) + α · QSS on residuals.
  α is cross-validated; α=0 recovers g exactly, so QSS ≥ g-only by construction.
- **Codebook**: Faiss ResidualQuantizer, beam=1 (greedy assignment), trained on
  D° (QSS-E mode) or all data (QSS-L mode).

## Dependencies

```
numpy scipy faiss-cpu scikit-learn lightgbm
```
