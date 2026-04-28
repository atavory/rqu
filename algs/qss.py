#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Core QSS (Quantized Sufficient Statistics) algorithms.

This module contains the Meta-independent core of the QSS system:
- RVQ encoding/decoding (Faiss ResidualQuantizer)
- Label-residual sufficient statistics (build, predict, evaluate)
- Centroid projection (preconditioning)
- Linear probe training (PyTorch)
- SISA baseline evaluation
- Memory and latency measurement

No Meta-specific imports (ManifoldClient, MediaLog, etc.) — suitable for
open-source release.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import faiss
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ── L2 normalization ──


def l2_normalize(X: np.ndarray) -> np.ndarray:
    """L2-normalize rows of X."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return (X / norms).astype(np.float32)


# ── RVQ encoding ──


def train_rvq(
    embeddings: np.ndarray,
    d: int,
    b: int,
    ell: int,
) -> faiss.ResidualQuantizer:
    """Train a Faiss ResidualQuantizer on embeddings (no labels).

    Args:
        embeddings: (N, d) float32 array, should be L2-normalized.
        d: embedding dimension.
        b: codebook size (must be power of 2).
        ell: number of RQ levels.

    Returns:
        Trained ResidualQuantizer with max_beam_size=1 (greedy nearest-centroid).
    """
    nbits = int(np.log2(b))
    rq = faiss.ResidualQuantizer(d, ell, nbits)
    rq.train_type = faiss.ResidualQuantizer.Train_default
    rq.max_beam_size = 1  # greedy nearest-centroid (Voronoi cells)
    t0 = time.time()
    rq.train(np.ascontiguousarray(embeddings, dtype=np.float32))
    logger.info(f"RVQ training: d={d}, b={b}, ell={ell} in {time.time() - t0:.1f}s")
    return rq


def unpack_codes(codes_packed: np.ndarray, nbits: int, ell: int) -> np.ndarray:
    """Unpack bit-packed RQ codes (LSB-first) into per-level centroid indices."""
    if nbits == 8:
        return codes_packed[:, :ell].astype(np.int32)
    n = codes_packed.shape[0]
    result = np.zeros((n, ell), dtype=np.int32)
    mask = (1 << nbits) - 1
    for i in range(n):
        val = int.from_bytes(codes_packed[i].tobytes(), byteorder="little")
        for level in range(ell):
            result[i, level] = (val >> (level * nbits)) & mask
    return result


def encode_rq(
    rq: faiss.ResidualQuantizer,
    emb: np.ndarray,
    b: int,
    ell: int,
) -> np.ndarray:
    """Encode embeddings with a trained RQ. Returns (N, ell) int32 codes."""
    nbits = int(np.log2(b))
    cp = rq.compute_codes(np.ascontiguousarray(emb, dtype=np.float32))
    if nbits == 8:
        return cp[:, :ell].astype(np.int32)
    # Vectorized bit unpacking — avoids O(N*ell) Python loop
    mask = (1 << nbits) - 1
    total_bits = nbits * ell
    total_bytes = (total_bits + 7) // 8
    raw = cp[:, :total_bytes]
    # Convert each row's bytes to a big integer via uint64 chunks
    n = raw.shape[0]
    result = np.zeros((n, ell), dtype=np.int32)
    # Process in uint64 chunks for speed
    raw_u8 = raw.astype(np.uint8)
    for level in range(ell):
        bit_offset = level * nbits
        byte_start = bit_offset // 8
        bit_start = bit_offset % 8
        # Read up to 8 bytes starting at byte_start (enough for any nbits <= 32)
        bytes_needed = (bit_start + nbits + 7) // 8
        acc = np.zeros(n, dtype=np.uint64)
        for j in range(bytes_needed):
            if byte_start + j < raw_u8.shape[1]:
                acc |= raw_u8[:, byte_start + j].astype(np.uint64) << (8 * j)
        result[:, level] = ((acc >> bit_start) & mask).astype(np.int32)
    return result


# ── Label-residual sufficient statistics ──


def precompute_label_stats(
    codes: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    ell: int,
    b: int,
) -> tuple:
    """Precompute class counts, centroid counts, and joint counts.

    These are independent of smoothing parameters and only need to be
    computed once per (b, ell) config.
    """
    class_counts = np.zeros((ell, b, num_classes), dtype=np.float64)
    centroid_counts = np.zeros((ell, b), dtype=np.int64)
    for level in range(ell):
        np.add.at(class_counts[level], (codes[:, level], labels), 1)
        np.add.at(centroid_counts[level], codes[:, level], 1)
    joints = {}
    for level in range(ell):
        for k in range(level):
            joint = np.zeros((b, b), dtype=np.int64)
            np.add.at(joint, (codes[:, k], codes[:, level]), 1)
            joints[(k, level)] = joint
    return class_counts, centroid_counts, joints


def build_label_residual_stats(
    codes: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    ell: int,
    b: int,
    alphas: list | None = None,
    cached_stats: tuple | None = None,
) -> list:
    """Build label-residual sufficient statistics.

    Each level fits the residual of the prediction from all previous levels.
    Uses joint centroid counts to avoid materializing (n, num_classes) arrays.

    Args:
        alphas: per-level smoothing pseudocounts, list of ell floats.
                Level 0 shrinks toward uniform 1/C; levels > 0 shrink toward 0.
        cached_stats: from precompute_label_stats. If None, computes on the fly.

    Returns list of ell arrays, each (b, num_classes).
    """
    if cached_stats is not None:
        class_counts, centroid_counts, joints = cached_stats
    else:
        class_counts, centroid_counts, joints = precompute_label_stats(
            codes,
            labels,
            num_classes,
            ell,
            b,
        )

    level_predictions = []
    for level in range(ell):
        centroid_sum = class_counts[level].copy()
        for k in range(level):
            joint = joints[(k, level)]
            centroid_sum -= joint.T.astype(np.float64) @ level_predictions[k]
        mask = centroid_counts[level] > 0
        alpha_m = alphas[level] if alphas is not None else 0.0
        pred = np.zeros((b, num_classes), dtype=np.float64)
        if level == 0:
            pred[mask] = (centroid_sum[mask] + alpha_m) / (
                centroid_counts[level][mask, np.newaxis] + alpha_m * num_classes
            )
        else:
            pred[mask] = centroid_sum[mask] / (
                centroid_counts[level][mask, np.newaxis] + alpha_m
            )
        level_predictions.append(pred)
    return level_predictions


def predict_label_residual(
    codes: np.ndarray,
    level_predictions: list,
    num_classes: int,
) -> np.ndarray:
    """Predict by summing level predictions and taking argmax."""
    n = codes.shape[0]
    combined = np.zeros((n, num_classes), dtype=np.float64)
    for level in range(len(level_predictions)):
        combined += level_predictions[level][codes[:, level]]
    return np.argmax(combined, axis=1)


def get_combined_scores(
    codes: np.ndarray,
    level_predictions: list,
    num_classes: int,
) -> np.ndarray:
    """Sum level predictions to get combined scores (n, num_classes)."""
    n = codes.shape[0]
    combined = np.zeros((n, num_classes), dtype=np.float64)
    for level in range(len(level_predictions)):
        combined += level_predictions[level][codes[:, level]]
    return combined


def make_alphas(alpha: float, alpha_growth: float, ell: int) -> list:
    """Create per-level alpha list from base alpha and growth rate."""
    return [alpha * (alpha_growth**m) for m in range(ell)]


# ── QSS prediction (one-hot labels) ──


def qss_predict(
    codes_priv: np.ndarray,
    y_priv: np.ndarray,
    codes_test: np.ndarray,
    K: int,
    b: int,
    ell: int,
) -> np.ndarray:
    """QSS prediction with one-hot label accumulation.

    Builds label-residual sufficient statistics from private data codes/labels,
    then predicts on test codes. Returns (n_test, K) score matrix.
    """
    S0 = np.zeros((ell, b, K), dtype=np.float64)
    n = np.zeros((ell, b), dtype=np.int64)
    N_co: dict[tuple[int, int], np.ndarray] = {}
    for level in range(ell):
        np.add.at(S0[level], (codes_priv[:, level], y_priv), 1)
        np.add.at(n[level], codes_priv[:, level], 1)
    for m in range(ell):
        for k in range(m):
            joint = np.zeros((b, b), dtype=np.int64)
            np.add.at(joint, (codes_priv[:, k], codes_priv[:, m]), 1)
            N_co[(k, m)] = joint
    mu = []
    for m in range(ell):
        S_m = S0[m].copy()
        for k in range(m):
            S_m -= N_co[(k, m)].T.astype(np.float64) @ mu[k]
        pred = np.zeros((b, K), dtype=np.float64)
        occ = n[m] > 0
        pred[occ] = S_m[occ] / n[m][occ, np.newaxis]
        mu.append(pred)
    nt = codes_test.shape[0]
    scores = np.zeros((nt, K), dtype=np.float64)
    for level in range(ell):
        scores += mu[level][codes_test[:, level]]
    return scores


def qss_predict_real(
    codes_priv: np.ndarray,
    residuals: np.ndarray,
    codes_test: np.ndarray,
    K: int,
    b: int,
    ell: int,
) -> np.ndarray:
    """Like qss_predict but accumulates real-valued vectors instead of one-hot."""
    S0 = np.zeros((ell, b, K), dtype=np.float64)
    n = np.zeros((ell, b), dtype=np.int64)
    N_co: dict[tuple[int, int], np.ndarray] = {}
    for level in range(ell):
        np.add.at(S0[level], codes_priv[:, level], residuals)
        np.add.at(n[level], codes_priv[:, level], 1)
    for m in range(ell):
        for k in range(m):
            joint = np.zeros((b, b), dtype=np.int64)
            np.add.at(joint, (codes_priv[:, k], codes_priv[:, m]), 1)
            N_co[(k, m)] = joint
    mu = []
    for m in range(ell):
        S_m = S0[m].copy()
        for k in range(m):
            S_m -= N_co[(k, m)].T.astype(np.float64) @ mu[k]
        pred = np.zeros((b, K), dtype=np.float64)
        occ = n[m] > 0
        pred[occ] = S_m[occ] / n[m][occ, np.newaxis]
        mu.append(pred)
    nt = codes_test.shape[0]
    scores = np.zeros((nt, K), dtype=np.float64)
    for level in range(ell):
        scores += mu[level][codes_test[:, level]]
    return scores


# ── Centroid projection (preconditioning) ──


def compute_centroid_features(
    X_pub: np.ndarray,
    y_pub: np.ndarray,
    X_target: np.ndarray,
    K: int,
    task_type: str,
) -> tuple[np.ndarray, int]:
    """Compute centroid projection on public data, apply to target.

    Binary (K=2): w = (mu+ - mu-), normalized. Append X @ w * sqrt(D).
    Multi-class (K>2): K centroids, normalized. Append X @ centroids.T.
    For regression: discretize into 10 quantile bins, treat as multi-class.

    Returns (F_target, n_proj_dims).
    """
    D = X_pub.shape[1]

    if task_type != "classification":
        n_bins = 10
        bin_edges = np.percentile(y_pub, np.linspace(0, 100, n_bins + 1)[1:-1])
        y_pub = np.digitize(y_pub, bin_edges).astype(np.int64)
        K = len(np.unique(y_pub))

    classes = np.unique(y_pub)

    if len(classes) < 2:
        return X_target, 0

    if len(classes) == 2:
        mask_pos = y_pub == classes[1]
        mask_neg = y_pub == classes[0]
        mu_pos = X_pub[mask_pos].mean(axis=0)
        mu_neg = X_pub[mask_neg].mean(axis=0)
        w = mu_pos - mu_neg
        w_norm = np.linalg.norm(w)
        if w_norm > 1e-10:
            w = w / w_norm
        proj = (X_target @ w.reshape(-1, 1)) * np.sqrt(D)
        F = np.concatenate([X_target, proj.astype(np.float32)], axis=1)
        return F.astype(np.float32), 1
    else:
        centroids = []
        for c in classes:
            mask = y_pub == c
            if mask.sum() > 0:
                mu_c = X_pub[mask].mean(axis=0)
                c_norm = np.linalg.norm(mu_c)
                if c_norm > 1e-10:
                    mu_c = mu_c / c_norm
                centroids.append(mu_c)
            else:
                centroids.append(np.zeros(D, dtype=np.float64))
        C = np.stack(centroids, axis=1).astype(np.float32)  # (D, K)
        proj = X_target @ C  # (N, K)
        F = np.concatenate([X_target, proj], axis=1)
        return F.astype(np.float32), len(centroids)


# ── Linear probe (PyTorch) ──


def train_linear_probe(
    X_np: np.ndarray,
    y_np: np.ndarray,
    task_type: str,
    K: int,
    lr: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    device: str = "cpu",
) -> nn.Module:
    """Train a PyTorch linear probe. Returns the trained model."""
    D = X_np.shape[1]
    X_t = torch.tensor(X_np, dtype=torch.float32, device=device)

    if task_type == "classification":
        y_t = torch.tensor(y_np, dtype=torch.long, device=device)
        model = nn.Linear(D, K).to(device)
        criterion: nn.Module = nn.CrossEntropyLoss()
    else:
        y_t = torch.tensor(y_np, dtype=torch.float32, device=device).unsqueeze(1)
        model = nn.Linear(D, 1).to(device)
        criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    n = len(X_t)

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            out = model(X_t[idx])
            loss = criterion(out, y_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def train_mlp(
    X_np: np.ndarray,
    y_np: np.ndarray,
    task_type: str,
    K: int,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 20,
    batch_size: int = 256,
    hidden_dim: int = 128,
    patience: int = 3,
    device: str = "cpu",
) -> nn.Module:
    """Train a 1-hidden-layer MLP with early stopping. Returns trained model."""
    D = X_np.shape[1]
    X_t = torch.tensor(X_np, dtype=torch.float32, device=device)

    out_dim = K if task_type == "classification" else 1
    if task_type == "classification":
        y_t = torch.tensor(y_np, dtype=torch.long, device=device)
        criterion: nn.Module = nn.CrossEntropyLoss()
    else:
        y_t = torch.tensor(y_np, dtype=torch.float32, device=device).unsqueeze(1)
        criterion = nn.MSELoss()

    model = nn.Sequential(
        nn.Linear(D, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    n = len(X_t)

    # 90/10 split for early stopping (seeded for reproducibility)
    n_val = max(n // 10, 1)
    gen = torch.Generator().manual_seed(42)
    perm_es = torch.randperm(n, generator=gen)
    val_idx_es = perm_es[:n_val]
    train_idx_es = perm_es[n_val:]
    X_train, y_train = X_t[train_idx_es], y_t[train_idx_es]
    X_val_es, y_val_es = X_t[val_idx_es], y_t[val_idx_es]

    best_loss = float("inf")
    best_state = None
    wait = 0

    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(len(X_train), device=device)
        for i in range(0, len(X_train), batch_size):
            idx = perm[i : i + batch_size]
            out = model(X_train[idx])
            loss = criterion(out, y_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Early stopping check
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_es), y_val_es).item()
        model.train()

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def train_final_mlp_model(
    X: np.ndarray,
    y: np.ndarray,
    task_type: str,
    K: int,
    device: str = "cpu",
    needs_scaler: bool = False,
    mlp_params: dict[str, Any] | None = None,
) -> tuple[nn.Module, StandardScaler | None]:
    """Train MLP g model. Uses Optuna params from mlp_params if provided."""
    scaler = None
    X_s = X.copy()
    use_scaler = needs_scaler
    if mlp_params is not None:
        use_scaler = mlp_params.get("use_scaler", needs_scaler)
    if use_scaler:
        scaler = StandardScaler().fit(X)
        X_s = scaler.transform(X)

    if mlp_params is not None:
        model = train_mlp(
            X_s, y, task_type, K,
            lr=mlp_params["best_lr"],
            weight_decay=mlp_params["best_weight_decay"],
            epochs=mlp_params["best_epochs"],
            batch_size=mlp_params["best_batch_size"],
            hidden_dim=mlp_params["best_hidden_dim"],
            patience=mlp_params["best_patience"],
            device=device,
        )
    else:
        model = train_mlp(X_s, y, task_type, K, device=device)
    return model, scaler


def train_final_model(
    X: np.ndarray,
    y: np.ndarray,
    params: dict[str, Any],
    task_type: str,
    K: int,
    device: str = "cpu",
) -> tuple[nn.Module, StandardScaler | None]:
    """Train a final model with best Optuna params."""
    use_scaler = params.get("use_scaler", False)
    scaler = None
    X_s = X.copy()
    if use_scaler:
        scaler = StandardScaler().fit(X)
        X_s = scaler.transform(X)
    model = train_linear_probe(
        X_s,
        y,
        task_type,
        K,
        lr=params["best_lr"],
        weight_decay=params["best_weight_decay"],
        epochs=params["best_epochs"],
        batch_size=params["best_batch_size"],
        device=device,
    )
    return model, scaler


def predict_with_model(
    model: nn.Module,
    X: np.ndarray,
    scaler: StandardScaler | None = None,
    task_type: str = "classification",
    device: str = "cpu",
) -> np.ndarray:
    """Get predictions from a trained linear probe."""
    if scaler is not None:
        X = scaler.transform(X)
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        out = model(X_t)
        if task_type == "classification":
            return torch.softmax(out, dim=1).cpu().numpy()
        else:
            return out.squeeze().cpu().numpy()


# ── QSS evaluation wrappers ──


def run_qss_on_features(
    F_pub: np.ndarray,
    y_pub: np.ndarray,
    F_priv: np.ndarray,
    y_priv: np.ndarray,
    F_test: np.ndarray,
    y_test: np.ndarray,
    b: int,
    ell: int,
    task_type: str,
    K: int,
    F_rq_train: np.ndarray | None = None,
) -> float:
    """Run QSS evaluation on feature arrays.

    Args:
        F_rq_train: Data to train RQ on. Defaults to F_pub (QSS-E).
            Pass concat(F_pub, F_priv) for QSS-L.
    """
    F_pub_n = l2_normalize(F_pub)
    F_priv_n = l2_normalize(F_priv)
    F_test_n = l2_normalize(F_test)

    if F_rq_train is not None:
        F_rq_train_n = l2_normalize(F_rq_train)
    else:
        F_rq_train_n = F_pub_n

    # Subsample RQ training data if too large — RQ only needs a representative
    # codebook, not all data. Training cost scales linearly with N.
    MAX_RQ_TRAIN = 50_000
    if F_rq_train_n.shape[0] > MAX_RQ_TRAIN:
        rng = np.random.RandomState(42)
        idx = rng.choice(F_rq_train_n.shape[0], MAX_RQ_TRAIN, replace=False)
        F_rq_train_n = F_rq_train_n[idx]

    D = F_rq_train_n.shape[1]
    nbits = int(np.log2(b))

    rq = faiss.ResidualQuantizer(D, ell, nbits)
    rq.train_type = faiss.ResidualQuantizer.Train_default
    rq.max_beam_size = 1
    rq.train(np.ascontiguousarray(F_rq_train_n, dtype=np.float32))

    priv_codes = encode_rq(rq, F_priv_n, b, ell)
    test_codes = encode_rq(rq, F_test_n, b, ell)

    if task_type == "classification":
        scores = qss_predict(priv_codes, y_priv, test_codes, K, b, ell)
        return float((scores.argmax(axis=1) == y_test).mean())
    else:
        bin_edges = np.percentile(y_priv, np.arange(1, 100))
        y_binned_priv = np.digitize(y_priv.ravel(), bin_edges).astype(np.int64)
        K_reg = max(int(y_binned_priv.max()) + 1, 2)
        scores = qss_predict(priv_codes, y_binned_priv, test_codes, K_reg, b, ell)
        bin_centers = np.concatenate(
            [
                [bin_edges[0] - 1],
                (bin_edges[:-1] + bin_edges[1:]) / 2,
                [bin_edges[-1] + 1],
            ]
        )[:K_reg]
        probs = scores - scores.max(axis=1, keepdims=True)
        probs = np.exp(probs) / np.exp(probs).sum(axis=1, keepdims=True)
        y_pred = probs @ bin_centers
        return float(pearsonr(y_test, y_pred)[0])


def run_qss_boosted(
    F_rq_train: np.ndarray,
    F_priv: np.ndarray,
    y_priv: np.ndarray,
    F_test: np.ndarray,
    y_test: np.ndarray,
    g_proba_priv: np.ndarray,
    g_proba_test: np.ndarray,
    b: int,
    ell: int,
    task_type: str,
    K: int,
    alpha: float = 1.0,
) -> float:
    """Run boosted QSS: y_hat = g(x) + alpha * qss_residual(x)."""
    F_rq_n = l2_normalize(F_rq_train)
    F_priv_n = l2_normalize(F_priv)
    F_test_n = l2_normalize(F_test)

    # Subsample RQ training data if too large
    MAX_RQ_TRAIN = 50_000
    F_rq_train_sub = F_rq_n
    if F_rq_n.shape[0] > MAX_RQ_TRAIN:
        rng = np.random.RandomState(42)
        idx = rng.choice(F_rq_n.shape[0], MAX_RQ_TRAIN, replace=False)
        F_rq_train_sub = F_rq_n[idx]

    D = F_rq_n.shape[1]
    nbits = int(np.log2(b))

    rq = faiss.ResidualQuantizer(D, ell, nbits)
    rq.train_type = faiss.ResidualQuantizer.Train_default
    rq.max_beam_size = 1
    rq.train(np.ascontiguousarray(F_rq_train_sub, dtype=np.float32))

    priv_codes = encode_rq(rq, F_priv_n, b, ell)
    test_codes = encode_rq(rq, F_test_n, b, ell)

    if task_type == "classification":
        y_oh = np.zeros((len(y_priv), K), dtype=np.float64)
        y_oh[np.arange(len(y_priv)), y_priv] = 1.0
        residuals = y_oh - g_proba_priv
        residual_scores = qss_predict_real(
            priv_codes,
            residuals,
            test_codes,
            K,
            b,
            ell,
        )
        final_scores = g_proba_test + alpha * residual_scores
        return float((final_scores.argmax(axis=1) == y_test).mean())
    else:
        residuals = (y_priv - g_proba_priv).reshape(-1, 1)
        residual_scores = qss_predict_real(
            priv_codes,
            residuals,
            test_codes,
            1,
            b,
            ell,
        )
        final_pred = g_proba_test + alpha * residual_scores.ravel()
        return float(pearsonr(y_test, final_pred)[0])


# ── SISA baseline ──


def eval_sisa_shards(
    X_train_per_shard: list[np.ndarray],
    y_train_per_shard: list[np.ndarray],
    X_test: np.ndarray,
    y_test: np.ndarray,
    params: dict[str, Any],
    task_type: str,
    K: int,
    device: str = "cpu",
) -> tuple[float, list[dict[str, float]]]:
    """Evaluate SISA: train per shard with same params, average predictions."""
    shard_preds = []
    shard_times = []

    for s, (X_train, y_train) in enumerate(zip(X_train_per_shard, y_train_per_shard)):
        t0 = time.time()
        model, scaler = train_final_model(
            X_train, y_train, params, task_type, K, device
        )
        preds = predict_with_model(model, X_test, scaler, task_type, device)
        train_s = time.time() - t0
        shard_times.append({"shard": float(s), "train_s": train_s})
        shard_preds.append(preds)

    if task_type == "classification":
        avg_proba = np.mean(shard_preds, axis=0)
        score = float((avg_proba.argmax(axis=1) == y_test).mean())
    else:
        avg_pred = np.mean(shard_preds, axis=0)
        score = float(pearsonr(y_test, avg_pred)[0])

    return score, shard_times


# ── Memory + latency measurement ──


def measure_qss_memory_and_latency(
    F_pub: np.ndarray,
    y_pub: np.ndarray,
    F_priv: np.ndarray,
    y_priv: np.ndarray,
    F_test: np.ndarray,
    y_test: np.ndarray,
    b: int,
    ell: int,
    task_type: str,
    K: int,
) -> dict[str, float]:
    """Measure QSS data structure memory and per-query prediction latency."""
    F_pub_n = l2_normalize(F_pub)
    F_priv_n = l2_normalize(F_priv)
    F_test_n = l2_normalize(F_test)

    D = F_pub_n.shape[1]
    nbits = int(np.log2(b))

    # Subsample RQ training data if too large
    MAX_RQ_TRAIN = 50_000
    F_train = F_pub_n
    if F_pub_n.shape[0] > MAX_RQ_TRAIN:
        rng = np.random.RandomState(42)
        idx = rng.choice(F_pub_n.shape[0], MAX_RQ_TRAIN, replace=False)
        F_train = F_pub_n[idx]

    rq = faiss.ResidualQuantizer(D, ell, nbits)
    rq.train_type = faiss.ResidualQuantizer.Train_default
    rq.max_beam_size = 1
    rq.train(np.ascontiguousarray(F_train, dtype=np.float32))

    priv_codes = encode_rq(rq, F_priv_n, b, ell)

    n_counts = np.zeros((ell, b), dtype=np.int64)
    if task_type == "classification":
        y_acc = y_priv
        K_acc = K
    else:
        bin_edges = np.percentile(y_priv, np.arange(1, 100))
        y_acc = np.digitize(y_priv.ravel(), bin_edges).astype(np.int64)
        K_acc = max(int(y_acc.max()) + 1, 2)

    S0 = np.zeros((ell, b, K_acc), dtype=np.float64)
    N_co: dict[tuple[int, int], np.ndarray] = {}
    for level in range(ell):
        np.add.at(S0[level], (priv_codes[:, level], y_acc), 1)
        np.add.at(n_counts[level], priv_codes[:, level], 1)
    for m in range(ell):
        for k in range(m):
            joint = np.zeros((b, b), dtype=np.int64)
            np.add.at(joint, (priv_codes[:, k], priv_codes[:, m]), 1)
            N_co[(k, m)] = joint

    mu = []
    for m in range(ell):
        S_m = S0[m].copy()
        for k in range(m):
            S_m -= N_co[(k, m)].T.astype(np.float64) @ mu[k]
        pred = np.zeros((b, K_acc), dtype=np.float64)
        occ = n_counts[m] > 0
        pred[occ] = S_m[occ] / n_counts[m][occ, np.newaxis]
        mu.append(pred)

    # Memory
    mem_S0 = S0.nbytes
    mem_n = n_counts.nbytes
    mem_Nco = sum(v.nbytes for v in N_co.values())
    mem_mu = sum(m.nbytes for m in mu)
    mem_rq = ell * b * D * 4
    total_mem_bytes = mem_S0 + mem_n + mem_Nco + mem_mu + mem_rq

    # Latency
    test_codes = encode_rq(rq, F_test_n, b, ell)
    n_test = min(len(test_codes), 1000)

    # Warm up
    for i in range(min(10, n_test)):
        code_i = test_codes[i : i + 1]
        scores_i = np.zeros((1, K_acc), dtype=np.float64)
        for level in range(ell):
            scores_i += mu[level][code_i[:, level]]

    # Time single-query prediction
    t0 = time.time()
    for i in range(n_test):
        code_i = test_codes[i : i + 1]
        scores_i = np.zeros((1, K_acc), dtype=np.float64)
        for level in range(ell):
            scores_i += mu[level][code_i[:, level]]
    predict_latency_us = (time.time() - t0) / n_test * 1e6

    # Time encode (single query)
    n_encode = min(100, len(F_test_n))
    t0 = time.time()
    for i in range(n_encode):
        _ = rq.compute_codes(
            np.ascontiguousarray(F_test_n[i : i + 1], dtype=np.float32)
        )
    encode_latency_us = (time.time() - t0) / n_encode * 1e6

    return {
        "qss_memory_bytes": total_mem_bytes,
        "qss_memory_mb": total_mem_bytes / (1024 * 1024),
        "qss_predict_latency_us": predict_latency_us,
        "qss_encode_latency_us": encode_latency_us,
        "qss_total_latency_us": predict_latency_us + encode_latency_us,
    }


# ── Temperature scaling ──


def temperature_scale(
    scores: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
) -> tuple:
    """Find optimal temperature T that minimizes log loss.

    Returns (T_opt, calibrated_log_loss, uncalibrated_log_loss).
    """
    from scipy.optimize import minimize_scalar
    from sklearn.metrics import log_loss as sk_log_loss

    proba = np.clip(scores, 1e-15, 1 - 1e-15)
    logits = np.log(proba)
    uncal_ll = float(sk_log_loss(labels, proba, labels=np.arange(num_classes)))

    def neg_ll(T: float) -> float:
        scaled = logits / T
        scaled -= scaled.max(axis=1, keepdims=True)
        exp_scaled = np.exp(scaled)
        cal_proba = exp_scaled / exp_scaled.sum(axis=1, keepdims=True)
        cal_proba = np.clip(cal_proba, 1e-15, 1 - 1e-15)
        return float(sk_log_loss(labels, cal_proba, labels=np.arange(num_classes)))

    result = minimize_scalar(neg_ll, bounds=(0.01, 100.0), method="bounded")
    return result.x, result.fun, uncal_ll


# ── Evaluation metrics ──


def evaluate_label_residual(
    codes: np.ndarray,
    labels: np.ndarray,
    level_predictions: list,
    num_classes: int,
    k: int = 5,
) -> dict:
    """Compute accuracy, log loss, and PR-AUC for label-residual predictions."""
    from sklearn.metrics import average_precision_score, log_loss
    from sklearn.preprocessing import label_binarize

    combined = get_combined_scores(codes, level_predictions, num_classes)
    n = len(labels)

    predictions = np.argmax(combined, axis=1)
    top1_acc = float((predictions == labels).mean())

    if k < num_classes:
        topk_indices = np.argpartition(combined, -k, axis=1)[:, -k:]
        topk_correct = sum(1 for i in range(n) if labels[i] in topk_indices[i])
        topk_acc = topk_correct / n
    else:
        topk_acc = 1.0

    proba = np.clip(combined, 1e-15, 1 - 1e-15)
    logloss = float(log_loss(labels, proba, labels=np.arange(num_classes)))

    T_opt, cal_ll, _ = temperature_scale(combined, labels, num_classes)

    if num_classes == 2:
        prauc = float(average_precision_score(labels, proba[:, 1]))
    else:
        y_bin = label_binarize(labels, classes=np.arange(num_classes))
        prauc = float(average_precision_score(y_bin, proba, average="macro"))

    return {
        "top1_acc": top1_acc,
        "topk_acc": float(topk_acc),
        "log_loss": logloss,
        "log_loss_calibrated": cal_ll,
        "temperature": T_opt,
        "prauc": prauc,
    }
