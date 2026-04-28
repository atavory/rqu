#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Standalone runner: fixed-ρ LGBM QSS experiments with timing.

Runs the full LGBM-based QSS pipeline for fixed retention ratio (ρ = n0/N):
1. Load tabular dataset from Manifold
2. For each (dataset, seed, ρ):
   - Compute n0 = max(50, round(ρ * N_train))
   - Quick Optuna (50 trials) for LGBM hyperparameters on D°
   - Train g-only LGBM on D° (timed)
   - Extract leaf contribution embeddings
   - Grid-search QSS-C-boost over (b, ℓ, α) on validation split
   - Run SISA with S=5 shards (timed)
   - Also run QSS-A (centroid proj, RQ on D° only)
3. Write results CSV

Usage:
    buck2 run fbcode//mitra/projects/fi_unlearning/system:fixed_rho_lgbm -- \
        --dataset adult --rho 0.01 --seeds 0,1,2,3,4
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from typing import Any

import faiss
import numpy as np
from mitra.projects.fi_unlearning.algs.lgbm import (
    eval_sisa_lgbm,
    extract_embeddings_with_stats,
    lgbm_predict_proba,
    train_lgbm,
)
from mitra.projects.fi_unlearning.algs.qss import (
    compute_centroid_features,
    encode_rq,
    l2_normalize,
    qss_predict,
    qss_predict_real,
    run_qss_on_features,
)
from mitra.projects.fi_unlearning.system.optuna_lgbm_runner import (
    _load_pt_from_manifold,
    LGBM_DATASETS,
    lgbm_objective,
)
from mitra.projects.fi_unlearning.system.optuna_probe_runner import split_data
from scipy.stats import pearsonr

logger: logging.Logger = logging.getLogger(__name__)

# Grid search parameters for QSS
B_CANDIDATES = [64, 128, 256]
ELL_CANDIDATES = [2, 4, 8]
ALPHA_CANDIDATES = [0.1, 0.3, 0.5, 1.0, 2.0]

# For QSS-A (no alpha needed)
B_CANDIDATES_A = [64, 128, 256]
ELL_CANDIDATES_A = [2, 4, 8]


def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S", time.gmtime())
    print(f"[{ts}] {msg}", flush=True)


def _quick_optuna_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    task_type: str,
    K: int,
    n_trials: int = 50,
) -> dict[str, Any]:
    """Run a quick Optuna search for LGBM hyperparameters."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    t0 = time.time()
    study.optimize(
        lambda trial: lgbm_objective(trial, X_train, y_train, task_type, K),
        n_trials=n_trials,
        show_progress_bar=False,
    )
    elapsed = time.time() - t0

    bp = study.best_params
    _log(f"    Optuna: best CV={study.best_value:.4f} in {elapsed:.1f}s ({n_trials} trials)")
    return {
        "n_estimators": bp["n_estimators"],
        "num_leaves": bp["num_leaves"],
        "learning_rate": bp["learning_rate"],
        "min_child_samples": bp["min_child_samples"],
        "subsample": bp["subsample"],
        "colsample_bytree": bp["colsample_bytree"],
        "reg_alpha": bp["reg_alpha"],
        "reg_lambda": bp["reg_lambda"],
        "optuna_seconds": elapsed,
        "cv_score": study.best_value,
    }


def _run_qss_c_boost_grid(
    g_model: Any,
    X_pub: np.ndarray,
    y_pub: np.ndarray,
    X_priv: np.ndarray,
    y_priv: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task_type: str,
    K: int,
) -> dict[str, Any]:
    """Grid search QSS-C-boost over (emb_type, b, ℓ, α).

    Tries contrib_raw and contrib_zscore embeddings.
    Returns best result dict.
    """
    X_all = np.concatenate([X_pub, X_priv], axis=0)
    y_all = np.concatenate([y_pub, y_priv])

    # Extract embeddings (stats from X_all for z-scoring)
    embs_all = extract_embeddings_with_stats(g_model, X_all, X_all)
    embs_test = extract_embeddings_with_stats(g_model, X_all, X_test)

    # g-model predictions
    g_preds_all = lgbm_predict_proba(g_model, X_all, task_type, K)
    g_preds_test = lgbm_predict_proba(g_model, X_test, task_type, K)

    best_score = -1e9
    best_result: dict[str, Any] = {}

    for emb_type in ["contrib_zscore", "contrib_raw"]:
        F_all = embs_all[emb_type]
        F_test = embs_test[emb_type]
        F_all_n = l2_normalize(F_all)
        F_test_n = l2_normalize(F_test)
        D = F_all_n.shape[1]

        # Subsample RQ training data if too large
        MAX_RQ_TRAIN = 50_000
        F_rq_train = F_all_n
        if F_all_n.shape[0] > MAX_RQ_TRAIN:
            rng = np.random.RandomState(42)
            idx = rng.choice(F_all_n.shape[0], MAX_RQ_TRAIN, replace=False)
            F_rq_train = F_all_n[idx]

        for b in B_CANDIDATES:
            if b > F_rq_train.shape[0]:
                continue
            nbits = int(np.log2(b))
            for ell in ELL_CANDIDATES:
                # Train RQ
                rq = faiss.ResidualQuantizer(D, ell, nbits)
                rq.train_type = faiss.ResidualQuantizer.Train_default
                rq.max_beam_size = 1
                rq.train(np.ascontiguousarray(F_rq_train, dtype=np.float32))

                all_codes = encode_rq(rq, F_all_n, b, ell)
                test_codes = encode_rq(rq, F_test_n, b, ell)

                # Compute residuals
                if task_type == "classification":
                    y_oh = np.zeros((len(y_all), K), dtype=np.float64)
                    y_oh[np.arange(len(y_all)), y_all] = 1.0
                    residuals = y_oh - g_preds_all
                else:
                    residuals = (y_all - g_preds_all).reshape(-1, 1)

                residual_scores = qss_predict_real(
                    all_codes, residuals, test_codes,
                    K if task_type == "classification" else 1,
                    b, ell,
                )

                for alpha in ALPHA_CANDIDATES:
                    if task_type == "classification":
                        final_scores = g_preds_test + alpha * residual_scores
                        score = float(np.mean(final_scores.argmax(axis=1) == y_test))
                    else:
                        final_pred = g_preds_test + alpha * residual_scores.ravel()
                        score = float(pearsonr(y_test, final_pred)[0])

                    if score > best_score:
                        best_score = score
                        best_result = {
                            "qss_best": round(score, 6),
                            "qss_source": f"c_boost_{emb_type}",
                            "b": b,
                            "ell": ell,
                            "alpha": round(alpha, 4),
                            "emb_type": emb_type,
                        }

    return best_result


def _run_qss_a(
    X_pub: np.ndarray,
    y_pub: np.ndarray,
    X_priv: np.ndarray,
    y_priv: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task_type: str,
    K: int,
) -> dict[str, Any]:
    """Run QSS-A: centroid projection on D°, RQ on D° only."""
    # Centroid projection
    F_pub, n_proj = compute_centroid_features(X_pub, y_pub, X_pub, K, task_type)
    F_priv, _ = compute_centroid_features(X_pub, y_pub, X_priv, K, task_type)
    F_test, _ = compute_centroid_features(X_pub, y_pub, X_test, K, task_type)

    best_score = -1e9
    best_b, best_ell = 128, 4
    for b in B_CANDIDATES_A:
        if b > F_pub.shape[0]:
            continue
        for ell in ELL_CANDIDATES_A:
            score = run_qss_on_features(
                F_pub, y_pub, F_priv, y_priv, F_test, y_test,
                b, ell, task_type, K,
            )
            if score > best_score:
                best_score = score
                best_b, best_ell = b, ell

    return {"qss_a": round(best_score, 6), "qss_a_b": best_b, "qss_a_ell": best_ell}


def run_one(
    dataset: str,
    seed: int,
    rho: float,
    X: np.ndarray,
    y: np.ndarray,
    task_type: str,
    K: int,
    n_optuna_trials: int = 50,
) -> dict[str, Any] | None:
    """Run one (dataset, seed, ρ) experiment. Returns result dict."""
    N = len(y)
    # N_train is everything except test (25%)
    n_test = N // 4
    N_train = N - n_test
    n0 = max(50, round(rho * N_train))

    if n0 < 50:
        _log(f"  Skipping ρ={rho}: n0={n0} < 50")
        return None

    _log(f"  === {dataset} seed={seed} ρ={rho} n0={n0} N={N} ===")

    # Split data
    pub_idx, priv_idx, test_idx = split_data(X, y, seed, n0, "random")
    X_pub, y_pub = X[pub_idx], y[pub_idx]
    X_priv, y_priv = X[priv_idx], y[priv_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    _log(f"  Split: pub={len(y_pub)}, priv={len(y_priv)}, test={len(y_test)}")

    # ── Optuna for LGBM hyperparameters ──
    _log(f"  Running Optuna for LGBM ({n_optuna_trials} trials)...")
    lgbm_params = _quick_optuna_lgbm(X_pub, y_pub, task_type, K, n_optuna_trials)
    hp = {k: v for k, v in lgbm_params.items() if k not in ("optuna_seconds", "cv_score")}

    # ── g-only: train LGBM on D° ──
    _log(f"  Training g-only LGBM...")
    t0 = time.time()
    g_model = train_lgbm(X_pub, y_pub, task_type, K, **hp)
    g_only_train_s = time.time() - t0

    g_preds_test = lgbm_predict_proba(g_model, X_test, task_type, K)
    if task_type == "classification":
        g_only = float(np.mean(g_preds_test.argmax(axis=1) == y_test))
    else:
        g_only = float(pearsonr(y_test, g_preds_test)[0])
    _log(f"  g-only: {g_only:.4f} ({g_only_train_s:.1f}s)")

    # ── QSS-C-boost grid search ──
    _log(f"  Running QSS-C-boost grid search...")
    t0 = time.time()
    qss_result = _run_qss_c_boost_grid(
        g_model, X_pub, y_pub, X_priv, y_priv, X_test, y_test,
        task_type, K,
    )
    qss_time = time.time() - t0
    _log(f"  QSS-C-boost best: {qss_result.get('qss_best', 'N/A')} ({qss_time:.1f}s)")

    # ── QSS-A ──
    _log(f"  Running QSS-A...")
    t0 = time.time()
    qss_a_result = _run_qss_a(
        X_pub, y_pub, X_priv, y_priv, X_test, y_test, task_type, K,
    )
    qss_a_time = time.time() - t0
    _log(f"  QSS-A: {qss_a_result.get('qss_a', 'N/A')} ({qss_a_time:.1f}s)")

    # Pick overall best QSS
    if qss_result.get("qss_best", -1e9) >= qss_a_result.get("qss_a", -1e9):
        best_qss = qss_result.get("qss_best", 0)
        best_source = qss_result.get("qss_source", "c_boost")
        best_b = qss_result.get("b", 0)
        best_ell = qss_result.get("ell", 0)
        best_alpha = qss_result.get("alpha", 0)
    else:
        best_qss = qss_a_result["qss_a"]
        best_source = "a_centroid"
        best_b = qss_a_result["qss_a_b"]
        best_ell = qss_a_result["qss_a_ell"]
        best_alpha = 0.0

    # ── SISA ──
    _log(f"  Running SISA (5 shards)...")
    X_all = np.concatenate([X_pub, X_priv])
    y_all = np.concatenate([y_pub, y_priv])

    t0_sisa_total = time.time()
    sisa_result = eval_sisa_lgbm(
        X_all, y_all, X_test, y_test, task_type, K,
        n_shards=5, seed=seed, **hp,
    )
    sisa_total_train_s = time.time() - t0_sisa_total

    sisa_metric_key = "sisa_acc" if task_type == "classification" else "sisa_r"
    sisa_score = sisa_result[sisa_metric_key]
    avg_sisa_del_s = sisa_result["avg_sisa_del_s"]
    _log(f"  SISA: {sisa_score:.4f}, avg_del={avg_sisa_del_s:.2f}s")

    # ── Assemble result ──
    row = {
        "dataset": dataset,
        "seed": seed,
        "rho": rho,
        "n0": len(y_pub),
        "N": N,
        "K": K,
        "g_only": round(g_only, 6),
        "qss_best": round(best_qss, 6),
        "qss_source": best_source,
        "sisa": round(sisa_score, 6),
        "b": best_b,
        "ell": best_ell,
        "alpha": round(best_alpha, 4),
        "avg_sisa_del_s": round(avg_sisa_del_s, 3),
        "g_only_train_s": round(g_only_train_s, 3),
        "sisa_total_train_s": round(sisa_total_train_s, 3),
        "qss_a": round(qss_a_result.get("qss_a", 0), 6),
        "qss_c_boost": round(qss_result.get("qss_best", 0), 6),
    }

    gap = best_qss - sisa_score
    _log(
        f"  RESULT: g={g_only:.4f} QSS={best_qss:.4f}({best_source}) "
        f"SISA={sisa_score:.4f} gap={gap*100:.1f}pp "
        f"sisa_del={avg_sisa_del_s:.2f}s g_train={g_only_train_s:.1f}s"
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed-ρ LGBM QSS experiments")
    parser.add_argument("--dataset", required=True, help="Dataset name from LGBM_DATASETS")
    parser.add_argument("--rho", type=str, default="0.005,0.01,0.02,0.05",
                        help="Comma-separated retention ratios")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4",
                        help="Comma-separated seeds")
    parser.add_argument("--n-optuna-trials", type=int, default=50,
                        help="Optuna trials for LGBM params")
    parser.add_argument("--output", type=str, default="",
                        help="Output CSV path (default: ~/fi_unlearning_results/$(hostname)/fixed_rho_lgbm_results.csv)")
    args = parser.parse_args()

    dataset = args.dataset
    rhos = [float(r) for r in args.rho.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    if dataset not in LGBM_DATASETS:
        print(f"Unknown dataset: {dataset}. Available: {list(LGBM_DATASETS.keys())}")
        sys.exit(1)

    info = LGBM_DATASETS[dataset]
    task_type = info["task"]
    K = info["K"]

    # Output path
    if args.output:
        out_path = args.output
    else:
        import socket
        hostname = socket.gethostname()
        out_dir = os.path.expanduser(f"~/fi_unlearning_results/{hostname}")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "fixed_rho_lgbm_results.csv")

    _log(f"=== Fixed-ρ LGBM Runner ===")
    _log(f"Dataset: {dataset}, K={K}, task={task_type}")
    _log(f"ρ values: {rhos}")
    _log(f"Seeds: {seeds}")
    _log(f"Output: {out_path}")

    # Load dataset
    _log(f"Loading {dataset} from Manifold...")
    t0 = time.time()
    X, y = _load_pt_from_manifold(dataset)
    # Re-read K in case it was updated
    K = info["K"]
    _log(f"Loaded: X={X.shape}, K={K} in {time.time()-t0:.1f}s")

    # Check existing results to avoid redo
    existing_keys: set[tuple[str, int, float]] = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_keys.add((row["dataset"], int(row["seed"]), float(row["rho"])))
        _log(f"Found {len(existing_keys)} existing results in {out_path}")

    # CSV columns
    columns = [
        "dataset", "seed", "rho", "n0", "N", "K",
        "g_only", "qss_best", "qss_source", "sisa",
        "b", "ell", "alpha",
        "avg_sisa_del_s", "g_only_train_s", "sisa_total_train_s",
        "qss_a", "qss_c_boost",
    ]

    # Write header if file doesn't exist
    write_header = not os.path.exists(out_path)

    n_done = 0
    n_total = len(rhos) * len(seeds)

    for rho in rhos:
        for seed in seeds:
            if (dataset, seed, rho) in existing_keys:
                _log(f"  {dataset} seed={seed} ρ={rho}: already done, skip")
                n_done += 1
                continue

            result = run_one(dataset, seed, rho, X, y, task_type, K, args.n_optuna_trials)
            n_done += 1

            if result is not None:
                with open(out_path, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=columns)
                    if write_header:
                        writer.writeheader()
                        write_header = False
                    writer.writerow(result)

            _log(f"  Progress: {n_done}/{n_total}")

    _log(f"=== {dataset} complete ({n_done}/{n_total}) ===")
    _log(f"Results at: {out_path}")


if __name__ == "__main__":
    main()
