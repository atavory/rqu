# Exact Unlearning via Quantized Sufficient Statistics

Code for the paper "Exact Unlearning via Quantized Sufficient Statistics" (NeurIPS 2026 submission).

## Overview

QSS replaces a neural prediction head with hash-table lookups over Faiss Residual Quantization codes.
Unlearning is exact (bitwise identical to retrain-from-scratch) and O(1) in dataset size:
subtract the deleted point's contributions from per-cell accumulators, then re-derive predictions.

```
input → frozen encoder f(x) → LDA preconditioning → L2-normalize
      → Residual Quantization → code tuple (c₁,...,cₗ)
      → lookup sufficient statistics tables → prediction
```

## Files

| File | Contents |
|------|----------|
| `qss.py` | Core QSS algorithms: RVQ encoding, label-residual sufficient statistics, QSS model (train / predict / unlearn), boosted QSS, SISA baseline, evaluation metrics |
| `lgbm.py` | LightGBM-based QSS for tabular datasets: leaf embeddings, SISA-LGBM |
| `lda.py` | LDA-K preconditioning (supervised feature projection on non-deletable data) |
| `evaluation.py` | Verification and attack code: unlearning correctness, retention curves, LiRA MIA, codebook fragility |

## Quick start

```python
import numpy as np
from qss import l2_normalize, qss_train, qss_model_predict, qss_unlearn, encode_rq

# Train
embeddings = l2_normalize(your_embeddings)  # (N, d) float32
model = qss_train(embeddings, labels, K=num_classes, b=256, ell=8)

# Predict
scores = qss_model_predict(model, l2_normalize(test_embeddings))
predictions = scores.argmax(axis=1)

# Unlearn (exact, O(1) in N)
del_codes = encode_rq(model.rq, l2_normalize(points_to_delete), b=256, ell=8)
model_after = qss_unlearn(model, del_codes, labels_to_delete)
```

## Requirements

```
numpy
scipy
faiss-cpu
torch
scikit-learn
lightgbm
```

## License

See LICENSE file.
