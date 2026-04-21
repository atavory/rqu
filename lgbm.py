#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

"""LightGBM-based QSS algorithms for tabular datasets.

Core functions:
- Train LightGBM with given hyperparameters
- Extract per-tree leaf value embeddings (raw and z-scored)
- CV scoring for Optuna
- SISA with LightGBM shards

No Meta-specific imports — suitable for open-source release.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import lightgbm as lgb
import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)


# ── Dataset loading (from fi_trunk_tail .pt files) ──


def load_trunk_tail_pt(path: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load a fi_trunk_tail .pt dataset file.

    Returns (X, y, metadata) where X is float32 features and y is labels.
    Concatenates train/val/test splits and continuous/categorical features.
    """
    import torch

    d = torch.load(path, weights_only=False)
    metadata = d.get("metadata", {})

    parts = []
    labels = []
    for split in ["train", "val", "test"]:
        cont_key = f"{split}_continuous"
        cat_key = f"{split}_categorical"
        label_key = f"{split}_labels"

        cont = d[cont_key].numpy() if cont_key in d else np.empty((0, 0))
        cat = d[cat_key].numpy() if cat_key in d else np.empty((0, 0))

        if cont.shape[1] > 0 and cat.shape[1] > 0:
            parts.append(np.hstack([cont, cat.astype(np.float32)]))
        elif cont.shape[1] > 0:
            parts.append(cont)
        elif cat.shape[1] > 0:
            parts.append(cat.astype(np.float32))
        else:
            raise ValueError(f"No features in {split}")

        labels.append(d[label_key].numpy())

    X = np.concatenate(parts, axis=0).astype(np.float32)
    y = np.concatenate(labels, axis=0)

    return X, y, metadata


# ── LightGBM training ──


def train_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    task_type: str,
    K: int,
    n_estimators: int = 100,
    num_leaves: int = 31,
    learning_rate: float = 0.1,
    min_child_samples: int = 20,
    subsample: float = 1.0,
    colsample_bytree: float = 1.0,
    reg_alpha: float = 0.0,
    reg_lambda: float = 0.0,
) -> lgb.LGBMClassifier | lgb.LGBMRegressor:
    """Train a LightGBM model with given hyperparameters."""
    params = {
        "n_estimators": n_estimators,
        "num_leaves": num_leaves,
        "learning_rate": learning_rate,
        "min_child_samples": min_child_samples,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "reg_alpha": reg_alpha,
        "reg_lambda": reg_lambda,
        "verbose": -1,
        "n_jobs": -1,
    }

    if task_type == "classification":
        if K == 2:
            params["objective"] = "binary"
        else:
            params["objective"] = "multiclass"
            params["num_class"] = K
        model = lgb.LGBMClassifier(**params)
    else:
        params["objective"] = "regression"
        model = lgb.LGBMRegressor(**params)

    model.fit(X_train, y_train)
    return model


def lgbm_predict_proba(
    model: lgb.LGBMClassifier | lgb.LGBMRegressor,
    X: np.ndarray,
    task_type: str,
    K: int,
) -> np.ndarray:
    """Get predictions from LightGBM model.

    For classification: returns (N, K) probability matrix.
    For regression: returns (N,) predictions.
    """
    if task_type == "classification":
        proba = model.predict_proba(X)
        if K == 2 and proba.shape[1] == 1:
            proba = np.column_stack([1 - proba, proba])
        # Pad to K columns if classes were missing from training data
        if proba.shape[1] < K:
            full = np.zeros((proba.shape[0], K), dtype=np.float64)
            full[:, model.classes_.astype(int)] = proba
            return full
        return proba.astype(np.float64)
    else:
        return model.predict(X).astype(np.float64)


# ── Embedding extraction ──


def extract_leaf_values(
    model: lgb.LGBMClassifier | lgb.LGBMRegressor,
    X: np.ndarray,
) -> np.ndarray:
    """Extract per-tree leaf values for each data point.

    Returns (N, T) array where T = n_estimators (or n_estimators * K for
    multiclass). Each entry is the leaf output value for that tree.
    """
    booster = model.booster_
    leaf_indices = model.predict(X, pred_leaf=True)  # (N, T) or (N, T*K)

    n_samples = len(X)
    n_trees = leaf_indices.shape[1]

    # Vectorized: build lookup table per tree
    tree_values = np.empty((n_samples, n_trees), dtype=np.float64)
    for tree_idx in range(n_trees):
        unique_leaves = np.unique(leaf_indices[:, tree_idx])
        leaf_to_val = {
            int(leaf): booster.get_leaf_output(tree_idx, int(leaf))
            for leaf in unique_leaves
        }
        tree_values[:, tree_idx] = np.vectorize(leaf_to_val.get)(
            leaf_indices[:, tree_idx]
        )

    return tree_values.astype(np.float32)


def extract_embeddings(
    model: lgb.LGBMClassifier | lgb.LGBMRegressor,
    X: np.ndarray,
) -> dict[str, np.ndarray]:
    """Extract both embedding types from a trained LightGBM model.

    Returns dict with:
        "contrib_raw": per-tree leaf values, natural scale
        "contrib_zscore": per-tree leaf values, z-scored per tree
    """
    raw = extract_leaf_values(model, X)

    # Z-score per tree
    mean = raw.mean(axis=0, keepdims=True)
    std = raw.std(axis=0, keepdims=True) + 1e-8
    zscore = ((raw - mean) / std).astype(np.float32)

    return {
        "contrib_raw": raw,
        "contrib_zscore": zscore,
    }


def train_shallow_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    task_type: str,
    K: int,
    max_depth: int,
) -> lgb.LGBMClassifier | lgb.LGBMRegressor:
    """Train a shallow LightGBM for one-hot embedding extraction.

    Uses 20 trees with num_leaves = 2**max_depth (fully balanced).
    """
    params = {
        "n_estimators": 20,
        "num_leaves": 2**max_depth,
        "max_depth": max_depth,
        "learning_rate": 0.1,
        "verbose": -1,
        "n_jobs": -1,
    }

    if task_type == "classification":
        if K == 2:
            params["objective"] = "binary"
        else:
            params["objective"] = "multiclass"
            params["num_class"] = K
        model = lgb.LGBMClassifier(**params)
    else:
        params["objective"] = "regression"
        model = lgb.LGBMRegressor(**params)

    model.fit(X_train, y_train)
    return model


def extract_onehot_embeddings(
    shallow_model: lgb.LGBMClassifier | lgb.LGBMRegressor,
    X: np.ndarray,
) -> np.ndarray:
    """Extract one-hot encoded leaf indices from a shallow LightGBM.

    Returns (N, T * num_leaves_per_tree) sparse-ish float32 array.
    Each tree contributes a one-hot block of size num_leaves_per_tree.
    """
    leaf_indices = shallow_model.predict(X, pred_leaf=True)  # (N, T)
    n_samples, n_trees = leaf_indices.shape

    # Build one-hot per tree
    blocks = []
    for t in range(n_trees):
        leaves_t = leaf_indices[:, t]
        n_leaves = int(leaves_t.max()) + 1
        oh = np.zeros((n_samples, n_leaves), dtype=np.float32)
        oh[np.arange(n_samples), leaves_t] = 1.0
        blocks.append(oh)

    return np.hstack(blocks)


def extract_onehot_with_fit(
    shallow_model: lgb.LGBMClassifier | lgb.LGBMRegressor,
    X_fit: np.ndarray,
    X_transform: np.ndarray,
) -> np.ndarray:
    """Extract one-hot embeddings, using X_fit to determine the leaf count per tree.

    Ensures X_transform gets the same dimensionality as X_fit.
    """
    fit_leaves = shallow_model.predict(X_fit, pred_leaf=True)
    transform_leaves = shallow_model.predict(X_transform, pred_leaf=True)
    n_trees = fit_leaves.shape[1]
    n_samples = len(X_transform)

    blocks = []
    for t in range(n_trees):
        n_leaves = int(fit_leaves[:, t].max()) + 1
        oh = np.zeros((n_samples, n_leaves), dtype=np.float32)
        valid = transform_leaves[:, t] < n_leaves
        oh[np.arange(n_samples)[valid], transform_leaves[:, t][valid]] = 1.0
        blocks.append(oh)

    return np.hstack(blocks)


def extract_embeddings_with_stats(
    model: lgb.LGBMClassifier | lgb.LGBMRegressor,
    X_fit: np.ndarray,
    X_transform: np.ndarray,
) -> dict[str, np.ndarray]:
    """Extract embeddings, computing z-score stats from X_fit but applying to X_transform.

    Use this when z-score statistics should come from training data only.
    """
    raw_fit = extract_leaf_values(model, X_fit)
    raw_transform = extract_leaf_values(model, X_transform)

    mean = raw_fit.mean(axis=0, keepdims=True)
    std = raw_fit.std(axis=0, keepdims=True) + 1e-8
    zscore = ((raw_transform - mean) / std).astype(np.float32)

    return {
        "contrib_raw": raw_transform,
        "contrib_zscore": zscore,
    }


# ── CV scoring for Optuna ──


def lgbm_cv_score(
    X_train: np.ndarray,
    y_train: np.ndarray,
    task_type: str,
    K: int,
    n_estimators: int = 100,
    num_leaves: int = 31,
    learning_rate: float = 0.1,
    min_child_samples: int = 20,
    subsample: float = 1.0,
    colsample_bytree: float = 1.0,
    reg_alpha: float = 0.0,
    reg_lambda: float = 0.0,
) -> float:
    """Train LightGBM with 5-fold CV, return mean score."""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        model = train_lgbm(
            X_tr, y_tr, task_type, K,
            n_estimators=n_estimators,
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
        )

        if task_type == "classification":
            pred = model.predict(X_val)
            score = float(np.mean(pred == y_val))
        else:
            pred = model.predict(X_val)
            score = float(pearsonr(y_val, pred)[0])

        scores.append(score)

    return float(np.mean(scores))


# ── SISA with LightGBM ──


def eval_sisa_lgbm(
    X_all: np.ndarray,
    y_all: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task_type: str,
    K: int,
    n_shards: int = 5,
    seed: int = 0,
    **lgbm_params: Any,
) -> dict[str, Any]:
    """Evaluate SISA baseline with LightGBM.

    Returns dict with sisa_acc/sisa_r, avg_sisa_del_s, shard times.
    """
    N = len(y_all)
    rng = np.random.RandomState(seed + 2000)
    perm = rng.permutation(N)
    shard_size = N // n_shards

    shard_preds = []
    shard_times = []

    for s in range(n_shards):
        shard_idx = perm[s * shard_size : (s + 1) * shard_size]
        X_shard, y_shard = X_all[shard_idx], y_all[shard_idx]

        t0 = time.time()
        model = train_lgbm(X_shard, y_shard, task_type, K, **lgbm_params)
        shard_time = time.time() - t0
        shard_times.append(shard_time)

        if task_type == "classification":
            shard_preds.append(lgbm_predict_proba(model, X_test, task_type, K))
        else:
            shard_preds.append(model.predict(X_test))

    # Ensemble: average probabilities (classification) or predictions (regression)
    if task_type == "classification":
        avg_proba = np.mean(shard_preds, axis=0)
        ensemble_pred = avg_proba.argmax(axis=1)
        metric = float(np.mean(ensemble_pred == y_test))
        metric_name = "sisa_acc"
    else:
        avg_pred = np.mean(shard_preds, axis=0)
        metric = float(pearsonr(y_test, avg_pred)[0])
        metric_name = "sisa_r"

    return {
        metric_name: metric,
        "avg_sisa_del_s": float(np.mean(shard_times)),
        "shard_times_s": [float(t) for t in shard_times],
    }
