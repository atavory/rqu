#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Standalone rebuild timing measurement for LGBM QSS pipeline.

Measures the FULL rebuild cost broken down by stage:
  1. g_only_train_s:    Train LightGBM on D°
  2. qss_lda_fit_s:     Compute centroid projection (LDA-K) on D°
  3. qss_rq_train_s:    Train RQ codebook on embeddings
  4. qss_encode_s:      Encode ALL N points with RQ
  5. qss_accumulate_s:  Fill label-residual accumulators for all N points
  6. qss_total_rebuild_s: Sum of all above

The true expected deletion cost is:
  E[cost] = (1 - rho) * t_RQ_encode_one + rho * qss_total_rebuild_s

Usage:
    buck2 run fbcode//mitra/projects/fi_unlearning/system:rebuild_timing_lgbm -- \
        --dataset adult --rho 0.01 --seed 0
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import Any

import faiss
import numpy as np
from mitra.projects.fi_unlearning.algs.lgbm import (
    extract_embeddings_with_stats,
    lgbm_predict_proba,
    train_lgbm,
)
from mitra.projects.fi_unlearning.algs.qss import (
    compute_centroid_features,
    encode_rq,
    l2_normalize,
    qss_predict_real,
)
from mitra.projects.fi_unlearning.system.optuna_lgbm_runner import (
    _load_pt_from_manifold,
    LGBM_DATASETS,
    lgbm_objective,
)
from mitra.projects.fi_unlearning.system.optuna_probe_runner import split_data


def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S", time.gmtime())
    print(f"[{ts}] {msg}", flush=True)


def _quick_optuna_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    task_type: str,
    K: int,
    n_trials: int = 30,
) -> dict[str, Any]:
    """Quick Optuna for LGBM hyperparameters."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: lgbm_objective(trial, X_train, y_train, task_type, K),
        n_trials=n_trials,
    )
    bp = study.best_params
    return {
        "n_estimators": bp["n_estimators"],
        "num_leaves": bp["num_leaves"],
        "learning_rate": bp["learning_rate"],
        "min_child_samples": bp["min_child_samples"],
        "subsample": bp["subsample"],
        "colsample_bytree": bp["colsample_bytree"],
        "reg_alpha": bp["reg_alpha"],
        "reg_lambda": bp["reg_lambda"],
    }


def measure_rebuild(
    dataset: str,
    seed: int,
    rho: float,
    X: np.ndarray,
    y: np.ndarray,
    task_type: str,
    K: int,
) -> dict[str, Any]:
    """Measure full rebuild timing for one (dataset, seed, rho)."""
    N = len(y)
    n_test = N // 4
    N_train = N - n_test
    n0 = max(50, round(rho * N_train))

    _log(f"=== {dataset} seed={seed} rho={rho} n0={n0} N={N} ===")

    # Split
    pub_idx, priv_idx, test_idx = split_data(X, y, seed, n0, "random")
    X_pub, y_pub = X[pub_idx], y[pub_idx]
    X_priv, y_priv = X[priv_idx], y[priv_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    X_all = np.concatenate([X_pub, X_priv], axis=0)
    y_all = np.concatenate([y_pub, y_priv])

    _log(f"  Split: pub={len(y_pub)}, priv={len(y_priv)}, test={len(y_test)}")

    # Quick Optuna
    _log(f"  Running quick Optuna (30 trials)...")
    hp = _quick_optuna_lgbm(X_pub, y_pub, task_type, K, 30)

    # ── Stage 1: Train g-only LGBM on D° ──
    _log(f"  Stage 1: Training g-only LGBM...")
    t0 = time.time()
    g_model = train_lgbm(X_pub, y_pub, task_type, K, **hp)
    g_only_train_s = time.time() - t0
    _log(f"    g_only_train_s = {g_only_train_s:.3f}s")

    # ── Stage 2: LDA fit (centroid projection on D°) ──
    _log(f"  Stage 2: LDA fit (centroid projection)...")
    t0 = time.time()
    F_pub, n_proj = compute_centroid_features(X_pub, y_pub, X_pub, K, task_type)
    # Also project all data and test
    F_priv, _ = compute_centroid_features(X_pub, y_pub, X_priv, K, task_type)
    F_test, _ = compute_centroid_features(X_pub, y_pub, X_test, K, task_type)
    qss_lda_fit_s = time.time() - t0
    _log(f"    qss_lda_fit_s = {qss_lda_fit_s:.3f}s")

    # For QSS-C-boost: extract LGBM embeddings instead
    _log(f"  Stage 2b: Extracting LGBM embeddings...")
    t0_emb = time.time()
    embs_all = extract_embeddings_with_stats(g_model, X_all, X_all)
    embs_test = extract_embeddings_with_stats(g_model, X_all, X_test)
    emb_extract_s = time.time() - t0_emb
    _log(f"    emb_extract_s = {emb_extract_s:.3f}s")

    # Use contrib_zscore for QSS-C-boost timing (most common winner)
    F_all_emb = embs_all["contrib_zscore"]
    F_test_emb = embs_test["contrib_zscore"]
    F_all_n = l2_normalize(F_all_emb)
    F_test_n = l2_normalize(F_test_emb)

    # Pick reasonable b, ell
    b = 128
    ell = 4
    D = F_all_n.shape[1]
    nbits = int(np.log2(b))

    # Subsample RQ training if needed
    MAX_RQ_TRAIN = 50_000
    F_rq_train = F_all_n
    if F_all_n.shape[0] > MAX_RQ_TRAIN:
        rng = np.random.RandomState(42)
        idx = rng.choice(F_all_n.shape[0], MAX_RQ_TRAIN, replace=False)
        F_rq_train = F_all_n[idx]

    # ── Stage 3: Train RQ codebook ──
    _log(f"  Stage 3: Training RQ codebook (b={b}, ell={ell}, D={D})...")
    t0 = time.time()
    rq = faiss.ResidualQuantizer(D, ell, nbits)
    rq.train_type = faiss.ResidualQuantizer.Train_default
    rq.max_beam_size = 1
    rq.train(np.ascontiguousarray(F_rq_train, dtype=np.float32))
    qss_rq_train_s = time.time() - t0
    _log(f"    qss_rq_train_s = {qss_rq_train_s:.3f}s")

    # ── Stage 4: Encode ALL N points ──
    _log(f"  Stage 4: Encoding all {F_all_n.shape[0]} points...")
    t0 = time.time()
    all_codes = encode_rq(rq, F_all_n, b, ell)
    test_codes = encode_rq(rq, F_test_n, b, ell)
    qss_encode_s = time.time() - t0
    _log(f"    qss_encode_s = {qss_encode_s:.3f}s")

    # ── Stage 5: Fill accumulators ──
    _log(f"  Stage 5: Filling accumulators...")
    t0 = time.time()

    # Get g-model predictions for residuals
    g_preds_all = lgbm_predict_proba(g_model, X_all, task_type, K)

    if task_type == "classification":
        y_oh = np.zeros((len(y_all), K), dtype=np.float64)
        y_oh[np.arange(len(y_all)), y_all] = 1.0
        residuals = y_oh - g_preds_all
        K_acc = K
    else:
        residuals = (y_all - g_preds_all).reshape(-1, 1)
        K_acc = 1

    # Build label-residual sufficient statistics (the accumulators)
    # This is what qss_predict_real does internally: S0, n, N_co, mu
    S0 = np.zeros((ell, b, K_acc), dtype=np.float64)
    n_counts = np.zeros((ell, b), dtype=np.int64)
    N_co: dict[tuple[int, int], np.ndarray] = {}

    for level in range(ell):
        np.add.at(S0[level], all_codes[:, level], residuals)
        np.add.at(n_counts[level], all_codes[:, level], 1)

    for m in range(ell):
        for k in range(m):
            joint = np.zeros((b, b), dtype=np.int64)
            np.add.at(joint, (all_codes[:, k], all_codes[:, m]), 1)
            N_co[(k, m)] = joint

    # Solve for mu (level predictions)
    mu = []
    for m in range(ell):
        S_m = S0[m].copy()
        for k in range(m):
            S_m -= N_co[(k, m)].T.astype(np.float64) @ mu[k]
        pred = np.zeros((b, K_acc), dtype=np.float64)
        occ = n_counts[m] > 0
        pred[occ] = S_m[occ] / n_counts[m][occ, np.newaxis]
        mu.append(pred)

    qss_accumulate_s = time.time() - t0
    _log(f"    qss_accumulate_s = {qss_accumulate_s:.3f}s")

    # ── Total ──
    qss_total_rebuild_s = (
        g_only_train_s + emb_extract_s + qss_rq_train_s + qss_encode_s + qss_accumulate_s
    )
    _log(f"  qss_total_rebuild_s = {qss_total_rebuild_s:.3f}s")
    _log(f"  Breakdown: g_train={g_only_train_s:.1f} + emb={emb_extract_s:.1f} + "
         f"rq_train={qss_rq_train_s:.1f} + encode={qss_encode_s:.1f} + "
         f"accum={qss_accumulate_s:.1f} = {qss_total_rebuild_s:.1f}s")

    # Also measure LDA-based pipeline (QSS-A)
    _log(f"  Measuring QSS-A rebuild timing...")
    F_all_lda = np.concatenate([F_pub, F_priv], axis=0).astype(np.float32)
    F_all_lda_n = l2_normalize(F_all_lda)
    D_lda = F_all_lda_n.shape[1]

    t0 = time.time()
    rq_a = faiss.ResidualQuantizer(D_lda, ell, nbits)
    rq_a.train_type = faiss.ResidualQuantizer.Train_default
    rq_a.max_beam_size = 1
    F_rq_a = F_all_lda_n[:MAX_RQ_TRAIN] if F_all_lda_n.shape[0] > MAX_RQ_TRAIN else F_all_lda_n
    rq_a.train(np.ascontiguousarray(F_rq_a, dtype=np.float32))
    qss_a_rq_train_s = time.time() - t0

    t0 = time.time()
    encode_rq(rq_a, F_all_lda_n, b, ell)
    qss_a_encode_s = time.time() - t0

    qss_a_total_rebuild_s = qss_lda_fit_s + qss_a_rq_train_s + qss_a_encode_s
    _log(f"  QSS-A rebuild: lda={qss_lda_fit_s:.1f} + rq={qss_a_rq_train_s:.1f} + "
         f"encode={qss_a_encode_s:.1f} = {qss_a_total_rebuild_s:.1f}s")

    return {
        "dataset": dataset,
        "seed": seed,
        "rho": rho,
        "n0": len(y_pub),
        "N": N,
        "K": K,
        "b": b,
        "ell": ell,
        "g_only_train_s": round(g_only_train_s, 3),
        "emb_extract_s": round(emb_extract_s, 3),
        "qss_lda_fit_s": round(qss_lda_fit_s, 3),
        "qss_rq_train_s": round(qss_rq_train_s, 3),
        "qss_encode_s": round(qss_encode_s, 3),
        "qss_accumulate_s": round(qss_accumulate_s, 3),
        "qss_total_rebuild_s": round(qss_total_rebuild_s, 3),
        "qss_a_rq_train_s": round(qss_a_rq_train_s, 3),
        "qss_a_encode_s": round(qss_a_encode_s, 3),
        "qss_a_total_rebuild_s": round(qss_a_total_rebuild_s, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild timing measurement for LGBM QSS")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--rho", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    dataset = args.dataset
    if dataset not in LGBM_DATASETS:
        print(f"Unknown dataset: {dataset}. Available: {list(LGBM_DATASETS.keys())}")
        sys.exit(1)

    info = LGBM_DATASETS[dataset]
    task_type = info["task"]
    K = info["K"]

    if args.output:
        out_path = args.output
    else:
        import socket
        hostname = socket.gethostname()
        out_dir = os.path.expanduser(f"~/fi_unlearning_results/{hostname}")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "rebuild_timing_lgbm.csv")

    _log(f"Loading {dataset} from Manifold...")
    X, y = _load_pt_from_manifold(dataset)
    K = info["K"]
    _log(f"Loaded: X={X.shape}, K={K}")

    result = measure_rebuild(dataset, args.seed, args.rho, X, y, task_type, K)

    columns = list(result.keys())
    write_header = not os.path.exists(out_path)
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if write_header:
            writer.writeheader()
        writer.writerow(result)

    _log(f"Written to {out_path}")
    _log(f"SUMMARY: {dataset} total_rebuild={result['qss_total_rebuild_s']:.1f}s "
         f"(g={result['g_only_train_s']:.1f} + emb={result['emb_extract_s']:.1f} + "
         f"rq={result['qss_rq_train_s']:.1f} + enc={result['qss_encode_s']:.1f} + "
         f"acc={result['qss_accumulate_s']:.1f})")


if __name__ == "__main__":
    main()
