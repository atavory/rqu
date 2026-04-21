#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

"""LDA-K preconditioning for QSS.

Fits Linear Discriminant Analysis on the non-deletable portion (D^o) to
produce supervised feature projections. This is NOT centroid projection —
LDA uses the between/within-class scatter matrices via sklearn's SVD solver.

Binary (K=2): full-rank orthonormal rotation with the LDA discriminant
direction as the first axis, preserving all d dimensions.

Multi-class (K>2): sklearn LDA transform giving min(K-1, d) components.

Output is always L2-normalized before feeding into RQ.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import null_space
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from qss import l2_normalize


def fit_lda(
    train_emb: np.ndarray,
    train_labels: np.ndarray,
    n_classes: int,
) -> dict:
    """Fit LDA preconditioning on non-deletable data.

    Args:
        train_emb: (N, d) float32 embeddings from D^o.
        train_labels: (N,) int64 labels from D^o.
        n_classes: number of classes K.

    Returns dict with fitted parameters for transform_lda().
    """
    lda = LinearDiscriminantAnalysis(solver="svd")
    lda.fit(train_emb.astype(np.float64), train_labels)

    if n_classes == 2:
        w = lda.scalings_[:, 0].copy()
        w = w / np.linalg.norm(w)
        ns = null_space(w.reshape(1, -1))
        W = np.column_stack([w, ns])
        return {"type": "binary", "W": W, "mean": lda.xbar_, "lda": lda}
    else:
        return {"type": "multiclass", "lda": lda}


def transform_lda(
    emb: np.ndarray,
    lda_params: dict,
) -> np.ndarray:
    """Transform embeddings using fitted LDA, then L2-normalize.

    Args:
        emb: (N, d) float32 embeddings.
        lda_params: output of fit_lda().

    Returns: (N, d') float32 L2-normalized transformed embeddings.
    """
    if lda_params["type"] == "binary":
        W = lda_params["W"]
        mean = lda_params["mean"]
        out = ((emb.astype(np.float64) - mean) @ W).astype(np.float32)
    else:
        out = lda_params["lda"].transform(
            emb.astype(np.float64)
        ).astype(np.float32)
    return l2_normalize(out)
