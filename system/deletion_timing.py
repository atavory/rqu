#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Timing benchmark: QSS vs SISA deletion cost.

All three scenarios run sequentially in one process, same GPU.
Output: /tmp/deletion_timing.csv + stdout summary.
"""

from __future__ import annotations

import csv
import sys
import time
from typing import Any

import json

import faiss
import numpy as np
import torch
from sklearn.linear_model import RidgeClassifierCV

from mitra.projects.fi_unlearning.algs.qss import (
    encode_rq,
    l2_normalize,
    train_final_model,
)
from mitra.projects.fi_unlearning.system.optuna_probe_runner import (
    DATASETS,
    load_dataset_from_manifold,
    split_data,
)

MF_CACHE = "fi_platform_ml_infra_fluent2_bucket/tree/fi_unlearning/optuna_cache_v5"


# v8 hyperparams (b, ell) per dataset — use most common
PARAMS = {
    "adult": {"b": 256, "ell": 8, "alpha": 1.0},
    "sst2": {"b": 256, "ell": 8, "alpha": 1.0},
    "celeba_smiling": {"b": 256, "ell": 8, "alpha": 1.0},
    "jigsaw": {"b": 256, "ell": 8, "alpha": 1.0},
}

TARGET_DATASETS = ["adult", "sst2", "celeba_smiling", "jigsaw"]
N_PUB_VALUES = [500, 1000]
N_REPS = 10
N_SISA_SHARDS = 5


def _timer_ms(fn, n_reps: int = 10) -> tuple[float, dict[str, float]]:
    """Run fn n_reps times, return median total_ms and median sub-timings."""
    all_results = []
    for _ in range(n_reps):
        all_results.append(fn())
    # Each result is dict of {name: ms}
    medians = {}
    for key in all_results[0]:
        vals = [r[key] for r in all_results]
        medians[key] = float(np.median(vals))
    return medians


def build_accumulators(
    codes: np.ndarray, residuals: np.ndarray, K: int, b: int, ell: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build QSS accumulators: S0, n, N_co, mu."""
    S0 = np.zeros((ell, b, K), dtype=np.float64)
    n = np.zeros((ell, b), dtype=np.int64)
    N_co: dict[tuple[int, int], np.ndarray] = {}
    for level in range(ell):
        np.add.at(S0[level], codes[:, level], residuals)
        np.add.at(n[level], codes[:, level], 1)
    for m in range(ell):
        for k in range(m):
            joint = np.zeros((b, b), dtype=np.int64)
            np.add.at(joint, (codes[:, k], codes[:, m]), 1)
            N_co[(k, m)] = joint
    return S0, n, N_co


def materialize_mu(
    S0: np.ndarray, n: np.ndarray, N_co: dict, K: int, b: int, ell: int,
) -> list[np.ndarray]:
    """Materialize level predictions mu from accumulators."""
    mu = []
    for m in range(ell):
        S_m = S0[m].copy()
        for k in range(m):
            S_m -= N_co[(k, m)].T.astype(np.float64) @ mu[k]
        pred = np.zeros((b, K), dtype=np.float64)
        occ = n[m] > 0
        pred[occ] = S_m[occ] / n[m][occ, np.newaxis]
        mu.append(pred)
    return mu


def predict_scores(
    codes_test: np.ndarray, mu: list[np.ndarray], K: int,
) -> np.ndarray:
    """Predict scores from mu and test codes."""
    nt = codes_test.shape[0]
    scores = np.zeros((nt, K), dtype=np.float64)
    for level in range(len(mu)):
        scores += mu[level][codes_test[:, level]]
    return scores


def scenario_d_priv_deletion(
    codes_priv: np.ndarray,
    residuals: np.ndarray,
    codes_test: np.ndarray,
    S0: np.ndarray,
    n: np.ndarray,
    N_co: dict,
    K: int, b: int, ell: int,
    del_idx: int,
) -> dict[str, float]:
    """Scenario 1: Delete one D• point — subtract from accumulators, re-materialize."""
    t_total = time.perf_counter()

    # Subtract this point's contribution from accumulators
    t0 = time.perf_counter()
    point_codes = codes_priv[del_idx]
    point_resid = residuals[del_idx]
    S0_new = S0.copy()
    n_new = n.copy()
    N_co_new = {k: v.copy() for k, v in N_co.items()}
    for level in range(ell):
        c = point_codes[level]
        S0_new[level, c] -= point_resid
        n_new[level, c] -= 1
    for m in range(ell):
        for k in range(m):
            N_co_new[(k, m)][point_codes[k], point_codes[m]] -= 1
    accum_ms = (time.perf_counter() - t0) * 1000

    # Re-materialize mu
    t0 = time.perf_counter()
    mu = materialize_mu(S0_new, n_new, N_co_new, K, b, ell)
    materialize_ms = (time.perf_counter() - t0) * 1000

    # Predict (for completeness)
    t0 = time.perf_counter()
    _ = predict_scores(codes_test, mu, K)
    predict_ms = (time.perf_counter() - t0) * 1000

    total_ms = (time.perf_counter() - t_total) * 1000
    return {
        "total_ms": total_ms,
        "g_refit_ms": 0.0,
        "residual_ms": 0.0,
        "encode_ms": 0.0,
        "accum_ms": accum_ms,
        "materialize_ms": materialize_ms,
        "predict_ms": predict_ms,
    }


def scenario_d_pub_deletion(
    X_pub: np.ndarray, y_pub: np.ndarray,
    X_priv: np.ndarray, y_priv: np.ndarray,
    X_test: np.ndarray,
    rq: faiss.ResidualQuantizer,
    codes_test: np.ndarray,
    g_proba_priv: np.ndarray,
    K: int, b: int, ell: int,
    ridge_alpha: float,
    del_idx: int,
) -> dict[str, float]:
    """Scenario 2: Delete one D° point — refit g, recompute residuals, re-encode, rebuild."""
    t_total = time.perf_counter()

    # Remove point from D°
    mask = np.ones(len(y_pub), dtype=bool)
    mask[del_idx] = False
    X_pub_new = X_pub[mask]
    y_pub_new = y_pub[mask]

    # Refit Ridge (closed-form)
    t0 = time.perf_counter()
    D = X_pub_new.shape[1]
    XtX = X_pub_new.T @ X_pub_new + ridge_alpha * np.eye(D, dtype=np.float64)
    # For binary: y is 0/1, use as-is
    Xty = X_pub_new.T @ y_pub_new.astype(np.float64)
    w = np.linalg.solve(XtX, Xty)
    g_refit_ms = (time.perf_counter() - t0) * 1000

    # Recompute residuals for all priv points
    t0 = time.perf_counter()
    g_new_priv = X_priv @ w  # raw decision values
    # Convert to proba for boosting residuals
    p = 1.0 / (1.0 + np.exp(-g_new_priv))
    g_proba_new = np.column_stack([1 - p, p])
    y_oh = np.zeros((len(y_priv), K), dtype=np.float64)
    y_oh[np.arange(len(y_priv)), y_priv] = 1.0
    residuals_new = y_oh - g_proba_new
    residual_ms = (time.perf_counter() - t0) * 1000

    # Re-encode all priv points through EXISTING RQ codebook
    t0 = time.perf_counter()
    codes_priv_new = encode_rq(rq, X_priv.astype(np.float32), b, ell)
    encode_ms = (time.perf_counter() - t0) * 1000

    # Rebuild accumulators
    t0 = time.perf_counter()
    S0, n, N_co = build_accumulators(codes_priv_new, residuals_new, K, b, ell)
    accum_ms = (time.perf_counter() - t0) * 1000

    # Re-materialize
    t0 = time.perf_counter()
    mu = materialize_mu(S0, n, N_co, K, b, ell)
    materialize_ms = (time.perf_counter() - t0) * 1000

    # Predict
    t0 = time.perf_counter()
    _ = predict_scores(codes_test, mu, K)
    predict_ms = (time.perf_counter() - t0) * 1000

    total_ms = (time.perf_counter() - t_total) * 1000
    return {
        "total_ms": total_ms,
        "g_refit_ms": g_refit_ms,
        "residual_ms": residual_ms,
        "encode_ms": encode_ms,
        "accum_ms": accum_ms,
        "materialize_ms": materialize_ms,
        "predict_ms": predict_ms,
    }


def load_sisa_params(ds_name: str, seed: int, n_pub: int) -> dict:
    """Load SISA Optuna params from Manifold cache."""
    import io
    from manifold.clients.python import ManifoldClient
    path = f"tree/fi_unlearning/optuna_cache_v5/{ds_name}_random_n{n_pub}_seed{seed}_sisa.json"
    buf = io.BytesIO()
    with ManifoldClient.get_client("fi_platform_ml_infra_fluent2_bucket") as mc:
        mc.sync_get(path=path, output=buf)
    buf.seek(0)
    return json.loads(buf.read())


def scenario_sisa_deletion(
    X_shard: np.ndarray, y_shard: np.ndarray,
    sisa_params: dict,
    task_type: str, K: int, device: str,
    del_idx: int,
) -> dict[str, float]:
    """Scenario 3: SISA deletion — retrain nn.Linear on affected shard."""
    t_total = time.perf_counter()

    # Remove point from shard
    mask = np.ones(len(y_shard), dtype=bool)
    mask[del_idx] = False
    X_shard_new = X_shard[mask]
    y_shard_new = y_shard[mask]

    # Retrain nn.Linear with SGD (same as accuracy experiments)
    t0 = time.perf_counter()
    _model, _scaler = train_final_model(
        X_shard_new, y_shard_new, sisa_params, task_type, K, device,
    )
    train_ms = (time.perf_counter() - t0) * 1000

    total_ms = (time.perf_counter() - t_total) * 1000
    return {
        "total_ms": total_ms,
        "g_refit_ms": train_ms,
        "residual_ms": 0.0,
        "encode_ms": 0.0,
        "accum_ms": 0.0,
        "materialize_ms": 0.0,
        "predict_ms": 0.0,
    }


def main() -> None:
    out_path = "/tmp/deletion_timing.csv"
    rows: list[dict[str, Any]] = []

    for ds_name in TARGET_DATASETS:
        info = DATASETS[ds_name]
        K = info["K"]
        params = PARAMS[ds_name]
        b, ell, alpha_boost = params["b"], params["ell"], params["alpha"]

        print(f"\n{'='*60}", file=sys.stderr)
        print(f"  {ds_name} (K={K})", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)

        E_raw, y = load_dataset_from_manifold(ds_name)
        E = l2_normalize(E_raw)
        N_total = len(y)
        print(f"  E={E.shape}", file=sys.stderr)

        for n_pub in N_PUB_VALUES:
            pub_idx, priv_idx, test_idx = split_data(E, y, 0, n_pub, "random")
            X_pub, y_pub = E[pub_idx], y[pub_idx]
            X_priv, y_priv = E[priv_idx], y[priv_idx]
            X_test, y_test = E[test_idx], y[test_idx]
            N_priv = len(y_priv)

            print(f"  n_pub={n_pub}: priv={N_priv} test={len(y_test)}", file=sys.stderr)

            # Train g (Ridge)
            g = RidgeClassifierCV(alphas=np.logspace(-3, 3, 7))
            g.fit(X_pub, y_pub)
            ridge_alpha = float(g.alpha_)

            # Get g probabilities for boosting
            df_priv = g.decision_function(X_priv)
            p_priv = 1.0 / (1.0 + np.exp(-df_priv))
            g_proba_priv = np.column_stack([1 - p_priv, p_priv])

            # Compute residuals
            y_oh = np.zeros((N_priv, K), dtype=np.float64)
            y_oh[np.arange(N_priv), y_priv] = 1.0
            residuals = y_oh - g_proba_priv

            # Train RQ on pub+priv (QSS-C style)
            X_all = np.concatenate([X_pub, X_priv], axis=0).astype(np.float32)
            MAX_RQ_TRAIN = 50_000
            if X_all.shape[0] > MAX_RQ_TRAIN:
                rng = np.random.RandomState(42)
                idx = rng.choice(X_all.shape[0], MAX_RQ_TRAIN, replace=False)
                X_rq_train = X_all[idx]
            else:
                X_rq_train = X_all
            D = X_rq_train.shape[1]
            nbits = int(np.log2(b))
            rq = faiss.ResidualQuantizer(D, ell, nbits)
            rq.train_type = faiss.ResidualQuantizer.Train_default
            rq.max_beam_size = 1
            rq.train(np.ascontiguousarray(X_rq_train, dtype=np.float32))

            # Encode priv + test
            codes_priv = encode_rq(rq, X_priv.astype(np.float32), b, ell)
            codes_test = encode_rq(rq, X_test.astype(np.float32), b, ell)

            # Build initial accumulators
            S0, n_counts, N_co = build_accumulators(codes_priv, residuals, K, b, ell)

            # Pick random deletion targets
            rng = np.random.RandomState(123)
            priv_del_indices = rng.choice(N_priv, N_REPS, replace=False)
            pub_del_indices = rng.choice(n_pub, N_REPS, replace=False)

            # Create SISA shards
            shard_size = N_priv // N_SISA_SHARDS
            shard_indices = [
                np.arange(i * shard_size, min((i + 1) * shard_size, N_priv))
                for i in range(N_SISA_SHARDS)
            ]

            # ── Scenario 1: D• deletion ──
            print(f"    S1: D• deletion...", file=sys.stderr)
            s1_results = []
            for di in priv_del_indices:
                r = scenario_d_priv_deletion(
                    codes_priv, residuals, codes_test,
                    S0, n_counts, N_co, K, b, ell, int(di),
                )
                s1_results.append(r)
            s1_med = {k: float(np.median([r[k] for r in s1_results])) for k in s1_results[0]}
            s1_med.update({"dataset": ds_name, "n_pub": n_pub, "n_priv": N_priv, "scenario": "D_priv"})
            rows.append(s1_med)

            # ── Scenario 2: D° deletion ──
            print(f"    S2: D° deletion...", file=sys.stderr)
            s2_results = []
            for di in pub_del_indices:
                r = scenario_d_pub_deletion(
                    X_pub, y_pub, X_priv, y_priv, X_test,
                    rq, codes_test, g_proba_priv,
                    K, b, ell, ridge_alpha, int(di),
                )
                s2_results.append(r)
            s2_med = {k: float(np.median([r[k] for r in s2_results])) for k in s2_results[0]}
            s2_med.update({"dataset": ds_name, "n_pub": n_pub, "n_priv": N_priv, "scenario": "D_pub"})
            rows.append(s2_med)

            # ── Scenario 3: SISA deletion ──
            print(f"    S3: SISA deletion (nn.Linear+SGD)...", file=sys.stderr)
            sisa_params = load_sisa_params(ds_name, 0, n_pub)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            task = "classification" if info["metric"] == "accuracy" else "regression"
            s3_results = []
            for di in range(N_REPS):
                # Pick random shard, delete random point from it
                shard_id = di % N_SISA_SHARDS
                si = shard_indices[shard_id]
                X_shard = X_priv[si]
                y_shard = y_priv[si]
                del_in_shard = rng.randint(0, len(y_shard))
                r = scenario_sisa_deletion(
                    X_shard, y_shard,
                    sisa_params, task, K, device,
                    del_in_shard,
                )
                s3_results.append(r)
            s3_med = {k: float(np.median([r[k] for r in s3_results])) for k in s3_results[0]}
            s3_med.update({"dataset": ds_name, "n_pub": n_pub, "n_priv": N_priv, "scenario": "SISA"})
            rows.append(s3_med)

            print(
                f"    D•={s1_med['total_ms']:.3f}ms  "
                f"D°={s2_med['total_ms']:.3f}ms  "
                f"SISA={s3_med['total_ms']:.3f}ms",
                file=sys.stderr,
            )

    # Write CSV
    fieldnames = [
        "dataset", "n_pub", "n_priv", "scenario",
        "total_ms", "g_refit_ms", "residual_ms", "encode_ms",
        "accum_ms", "materialize_ms", "predict_ms",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Summary
    print(f"\n{'='*90}")
    print(f"  DELETION TIMING SUMMARY (median of {N_REPS} reps)")
    print(f"{'='*90}")
    print(
        f"{'Dataset':<18} {'n_pub':>5} {'N_priv':>8} {'Scenario':>8} "
        f"{'Total':>8} {'g_refit':>8} {'resid':>8} {'encode':>8} "
        f"{'accum':>8} {'mater':>8} {'predict':>8}"
    )
    print("-" * 110)
    for r in rows:
        print(
            f"{r['dataset']:<18} {r['n_pub']:>5} {r['n_priv']:>8} {r['scenario']:>8} "
            f"{r['total_ms']:>8.3f} {r['g_refit_ms']:>8.3f} {r['residual_ms']:>8.3f} "
            f"{r['encode_ms']:>8.3f} {r['accum_ms']:>8.3f} {r['materialize_ms']:>8.3f} "
            f"{r['predict_ms']:>8.3f}"
        )

    print(f"\nWritten {len(rows)} rows to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
