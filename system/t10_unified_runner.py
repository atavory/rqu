#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict
"""MAST runner: T10 evaluation (6 methods) on GPU — v7.

Worker-loop pattern: each iteration reads existing results from Manifold,
finds one (n_pub, seed) combo to evaluate, runs it with step-by-step
logging, uploads the result, and loops. SISA is computed once per seed
(it doesn't depend on n_pub) and cached in memory.

Methods (matching paper table):
  QSS-A: centroid proj on D°, RQ on D° only, accumulators
  QSS-B: centroid proj on D°, RQ on all D, accumulators
  QSS-C: no LDA, RQ on all D, accumulators
  QSS-D: no LDA, RQ on all D + exp mech, accumulators
  SISA:  shard all D, LP per shard — exact via retraining
  g-only: LP on D° only — no private data used (accuracy floor)
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import threading
import time
from typing import Any, Generator

import numpy as np
import torch
from content_understanding.framework.utils.logging import MediaLog, TMediaLogger
from mitra.projects.fi_unlearning.algs.qss import (
    compute_centroid_features,
    eval_sisa_shards,
    l2_normalize,
    measure_qss_memory_and_latency,
    predict_with_model,
    run_qss_boosted,
    run_qss_on_features,
    train_final_mlp_model,
    train_final_model,
    train_mlp,
)
from mitra.projects.fi_unlearning.system.optuna_probe_runner import (
    _cache_key,
    _NoOpOutput,
    ALL_CONFIGS,
    CACHE_VERSION,
    DATASETS,
    load_dataset_from_manifold,
    MF_BUCKET,
    MF_CACHE,
    split_data,
)
from manifold.clients.python import ManifoldClient
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler

logger: TMediaLogger = MediaLog.get_logger(__name__)

MF_RESULTS = "tree/fi_unlearning/results/t10_v8"
MF_RESULTS_MLP = "tree/fi_unlearning/results/t10_v8_mlp"
MF_RQ_CACHE = "tree/fi_unlearning/optuna_rq_v1"
MF_RQ_CACHE_MLP = "tree/fi_unlearning/optuna_rq_mlp_v1"
RQ_CACHE_VERSION = "rq_v1"
MF_MLP_CACHE = "tree/fi_unlearning/optuna_mlp_cache_v1"
MLP_CACHE_VERSION = "mlp_v1"

# Fallback (b, ℓ) if RQ cache miss — used only as last resort
FALLBACK_B_ELL: dict[str, tuple[int, int]] = {
    "celeba_male": (256, 8),
    "celeba_young": (256, 8),
    "celeba_smiling": (256, 8),
    "celeba_eyeglasses": (256, 8),
    "celeba_attractive": (256, 8),
    "pets": (128, 8),
    "dtd": (256, 8),
    "mnist": (128, 8),
    "fmnist": (128, 8),
    "cifar100": (128, 8),
    "cub200": (256, 8),
    "imagenet": (256, 8),
    "covertype": (256, 4),
    "adult": (128, 4),
    "agnews": (128, 8),
    "sst2": (128, 8),
    "imdb": (128, 8),
    "dbpedia": (128, 8),
    "jigsaw": (128, 8),
    "utkface": (128, 8),
    "calhousing": (128, 4),
    "criteo": (128, 4),
    "news": (128, 4),
    "tabred_weather": (128, 8),
}


def _log(msg: str) -> None:
    """Print with timestamp and flush immediately."""
    ts = time.strftime("%H:%M:%S", time.gmtime())
    line = f"[{ts}] {msg}"
    logger.info(line)
    print(line, flush=True)


@contextlib.contextmanager
def _heartbeat(label: str, interval: int = 120) -> Generator[None, None, None]:
    """Context manager that prints a heartbeat every `interval` seconds."""
    stop = threading.Event()

    def _beat() -> None:
        start = time.time()
        while not stop.wait(interval):
            elapsed = time.time() - start
            _log(f"  {label}: still running ({elapsed:.0f}s)...")

    t = threading.Thread(target=_beat, daemon=True)
    t.start()
    try:
        yield
    finally:
        stop.set()
        t.join(timeout=5)


def _load_rq_params(
    dataset: str,
    seed: int,
    n_pub: int,
    qss_mode: str,
    split_type: str = "random",
    g_model: str = "lp",
) -> tuple[int, int, float]:
    """Load CV'd (b, ℓ, α) from RQ cache. Falls back to defaults if missing."""
    # Boost modes with MLP g use separate cache; non-boost modes reuse LP cache
    is_boost = qss_mode.endswith("_boost")
    cache_dir = MF_RQ_CACHE_MLP if (g_model == "mlp" and is_boost) else MF_RQ_CACHE
    path = f"{cache_dir}/{dataset}_{split_type}_n{n_pub}_seed{seed}_{qss_mode}.json"
    try:
        buf = io.BytesIO()
        with ManifoldClient.get_client(MF_BUCKET) as client:
            client.sync_get(path=path, output=buf)
        buf.seek(0)
        obj = json.loads(buf.read())
        if obj.get("version") == RQ_CACHE_VERSION:
            return obj["best_b"], obj["best_ell"], obj.get("best_alpha", 1.0)
    except Exception:
        pass
    # Fallback
    b, ell = FALLBACK_B_ELL.get(dataset, (256, 8))
    return b, ell, 1.0


# ── Manifold cache helpers ──


def _load_cached_params(
    dataset: str,
    seed: int,
    config: str,
    split_type: str = "random",
    n_pub: int = 500,
) -> dict[str, Any] | None:
    """Load cached Optuna params from Manifold."""
    # Try new key format first, fall back to old format (without n_pub)
    paths_to_try = [
        _cache_key(dataset, split_type, n_pub, seed, config),
        f"{MF_CACHE}/{dataset}_{split_type}_seed{seed}_{config}.json",
    ]
    for path in paths_to_try:
        try:
            buf = io.BytesIO()
            with ManifoldClient.get_client(MF_BUCKET) as client:
                client.sync_get(path=path, output=buf)
            buf.seek(0)
            obj = json.loads(buf.read())
            if obj.get("version") == CACHE_VERSION:
                return obj
        except Exception:
            pass
    return None


def _load_sisa_params(
    dataset: str,
    seed: int,
    split_type: str,
    n_pub_values: list[int],
) -> dict[str, Any] | None:
    """Load SISA Optuna params — try any n_pub key since SISA is n_pub-independent."""
    for n_pub in n_pub_values:
        result = _load_cached_params(dataset, seed, "sisa", split_type, n_pub)
        if result is not None:
            return result
    # Also try legacy key (no n_pub)
    path = f"{MF_CACHE}/{dataset}_{split_type}_seed{seed}_sisa.json"
    try:
        buf = io.BytesIO()
        with ManifoldClient.get_client(MF_BUCKET) as client:
            client.sync_get(path=path, output=buf)
        buf.seek(0)
        obj = json.loads(buf.read())
        if obj.get("version") == CACHE_VERSION:
            return obj
    except Exception:
        pass
    return None


def _load_mlp_cached_params(
    dataset: str,
    seed: int,
    config: str,
    split_type: str = "random",
    n_pub: int = 5000,
) -> dict[str, Any] | None:
    """Load cached MLP Optuna params from Manifold."""
    path = f"{MF_MLP_CACHE}/{dataset}_{split_type}_n{n_pub}_seed{seed}_{config}.json"
    try:
        buf = io.BytesIO()
        with ManifoldClient.get_client(MF_BUCKET) as client:
            client.sync_get(path=path, output=buf)
        buf.seek(0)
        obj = json.loads(buf.read())
        if obj.get("version") == MLP_CACHE_VERSION:
            return obj
    except Exception:
        pass
    return None


def _load_mlp_sisa_params(
    dataset: str,
    seed: int,
    split_type: str,
    n_pub_values: list[int],
) -> dict[str, Any] | None:
    """Load MLP SISA Optuna params — try any n_pub key since SISA is n_pub-independent."""
    for n_pub in n_pub_values:
        result = _load_mlp_cached_params(dataset, seed, "sisa", split_type, n_pub)
        if result is not None:
            return result
    return None


def _upload_to_manifold(path: str, data: bytes) -> None:
    """Upload bytes to Manifold (overwrites existing)."""
    with ManifoldClient.get_client(MF_BUCKET) as client:
        client.sync_put(
            path, data, predicate=ManifoldClient.Predicates.AllowOverwrite
        )


MF_ROWS = f"{MF_RESULTS}/rows"


def _row_path(dataset: str, n_pub: int, seed: int) -> str:
    return f"{MF_ROWS}/{dataset}_n{n_pub}_s{seed}.json"


def _upload_row_atomic(
    dataset: str, n_pub: int, seed: int, row: dict[str, Any],
    rows_dir: str = MF_ROWS,
) -> None:
    """Write one result row as an atomic JSON file on Manifold.

    Path: rows_dir/<dataset>_n<npub>_s<seed>.json
    This is safe for concurrent workers — each combo gets its own file.
    """
    path = f"{rows_dir}/{dataset}_n{n_pub}_s{seed}.json"
    data = json.dumps(row).encode("utf-8")
    _upload_to_manifold(path, data)


def _parse_row_filename(name: str) -> tuple[int, int] | None:
    """Parse 'imagenet_n500_s0.json' → (500, 0) or None."""
    if not name.endswith(".json"):
        return None
    parts = name.replace(".json", "").split("_")
    n_str = [p for p in parts if p.startswith("n") and p[1:].isdigit()]
    s_str = [p for p in parts if p.startswith("s") and p[1:].isdigit()]
    if n_str and s_str:
        return (int(n_str[-1][1:]), int(s_str[-1][1:]))
    return None


def _read_done_keys(dataset: str, rows_dir: str = MF_ROWS) -> set[tuple[int, int]]:
    """List existing per-combo JSON files on Manifold to find done combos."""
    try:
        with ManifoldClient.get_client(MF_BUCKET) as client:
            entries = list(client.sync_ls(rows_dir))
        done: set[tuple[int, int]] = set()
        prefix = f"{dataset}_n"
        for name, _entry in entries:
            if not name.startswith(prefix):
                continue
            parsed = _parse_row_filename(name)
            if parsed is not None:
                done.add(parsed)
        return done
    except Exception:
        return set()


def _read_existing_rows(dataset: str, rows_dir: str = MF_ROWS) -> list[dict[str, Any]]:
    """Read all per-combo JSON files for a dataset from Manifold."""
    rows: list[dict[str, Any]] = []
    try:
        with ManifoldClient.get_client(MF_BUCKET) as client:
            entries = list(client.sync_ls(rows_dir))
            prefix = f"{dataset}_n"
            for name, _entry in entries:
                if not name.startswith(prefix) or not name.endswith(".json"):
                    continue
                buf = io.BytesIO()
                client.sync_get(path=f"{rows_dir}/{name}", output=buf)
                row = json.loads(buf.getvalue().decode("utf-8"))
                rows.append(row)
    except Exception:
        pass
    return rows


# ── Compute SISA for one seed (n_pub-independent) ──


def _eval_sisa_mlp(
    X_shards: list[np.ndarray],
    y_shards: list[np.ndarray],
    X_test: np.ndarray,
    y_test: np.ndarray,
    task_type: str,
    K: int,
    device: str,
    mlp_params: dict[str, Any] | None = None,
) -> tuple[float, list[dict[str, float]]]:
    """SISA with MLP per shard. Uses Optuna params from mlp_params if provided."""
    shard_preds = []
    shard_times = []
    for s, (Xs, ys) in enumerate(zip(X_shards, y_shards)):
        t0 = time.time()
        if mlp_params is not None:
            # Apply scaler if Optuna selected it
            X_s = Xs.copy()
            if mlp_params.get("use_scaler", False):
                scaler = StandardScaler().fit(Xs)
                X_s = scaler.transform(Xs)
                X_test_s = scaler.transform(X_test)
            else:
                X_test_s = X_test
            model = train_mlp(
                X_s, ys, task_type, K,
                lr=mlp_params["best_lr"],
                weight_decay=mlp_params["best_weight_decay"],
                epochs=mlp_params["best_epochs"],
                batch_size=mlp_params["best_batch_size"],
                hidden_dim=mlp_params["best_hidden_dim"],
                patience=mlp_params["best_patience"],
                device=device,
            )
            preds = predict_with_model(model, X_test_s, task_type=task_type, device=device)
        else:
            model = train_mlp(Xs, ys, task_type, K, device=device)
            preds = predict_with_model(model, X_test, task_type=task_type, device=device)
        train_s = time.time() - t0
        shard_times.append({"shard": float(s), "train_s": train_s})
        shard_preds.append(preds)
    if task_type == "classification":
        avg_proba = np.mean(shard_preds, axis=0)
        score = float((avg_proba.argmax(axis=1) == y_test).mean())
    else:
        avg_pred = np.mean(shard_preds, axis=0)
        from scipy.stats import pearsonr as _pr
        score = float(_pr(y_test, avg_pred)[0])
    return score, shard_times


def compute_sisa(
    name: str,
    E: np.ndarray,
    y: np.ndarray,
    seed: int,
    task_type: str,
    K: int,
    split_type: str,
    n_pub_values: list[int],
    device: str,
    g_model: str = "lp",
) -> tuple[float, float, float] | None:
    """Compute SISA score for a seed. Returns (score, optuna_s, avg_del_s) or None."""
    if g_model == "lp":
        sisa_params = _load_sisa_params(name, seed, split_type, n_pub_values)
        if sisa_params is None:
            _log(f"  SISA seed={seed}: no LP cache found, skipping")
            return None
        mlp_sisa_params = None
    else:
        sisa_params = None  # LP params not needed for MLP
        mlp_sisa_params = _load_mlp_sisa_params(name, seed, split_type, n_pub_values)
        if mlp_sisa_params is None:
            _log(f"  SISA seed={seed}: no MLP cache found, skipping")
            return None

    # Use any n_pub to get the split — SISA uses all non-test data
    # which is the same regardless of n_pub
    n_pub_any = n_pub_values[0]
    pub_idx, priv_idx, test_idx = split_data(E, y, seed, n_pub_any, split_type)
    X_all = E[np.concatenate([pub_idx, priv_idx])]
    y_all = y[np.concatenate([pub_idx, priv_idx])]
    X_test, y_test = E[test_idx], y[test_idx]

    n_shards = 5
    N_all = len(y_all)
    shard_size = N_all // n_shards
    X_shards = []
    y_shards = []
    for s in range(n_shards):
        start = s * shard_size
        end = start + shard_size if s < n_shards - 1 else N_all
        X_shards.append(X_all[start:end])
        y_shards.append(y_all[start:end])

    _log(f"  SISA seed={seed} (g={g_model}): evaluating {n_shards} shards ({N_all} samples)...")
    t0 = time.time()
    with _heartbeat(f"SISA seed={seed}"):
        if g_model == "mlp":
            sisa_score, sisa_times = _eval_sisa_mlp(
                X_shards, y_shards, X_test, y_test, task_type, K, device,
                mlp_params=mlp_sisa_params,
            )
        else:
            sisa_score, sisa_times = eval_sisa_shards(
                X_shards, y_shards, X_test, y_test,
                sisa_params, task_type, K, device,
            )
    elapsed = time.time() - t0
    avg_del_s = float(np.mean([t["train_s"] for t in sisa_times]))
    if sisa_params is not None:
        optuna_s = sisa_params.get("optuna_seconds", 0.0)
    elif mlp_sisa_params is not None:
        optuna_s = mlp_sisa_params.get("optuna_seconds", 0.0)
    else:
        optuna_s = 0.0
    _log(f"  SISA seed={seed}: score={sisa_score:.4f} ({elapsed:.0f}s)")
    return sisa_score, optuna_s, avg_del_s


# ── Run one (n_pub, seed) combo ──


def run_one_combo(
    name: str,
    E: np.ndarray,
    y: np.ndarray,
    seed: int,
    task_type: str,
    K: int,
    n_pub: int,
    split_type: str,
    device: str,
    sisa_score: float,
    sisa_optuna_s: float,
    avg_sisa_del_s: float,
    g_model: str = "lp",
) -> dict[str, Any] | None:
    """Run QSS methods + g-only for one (n_pub, seed). SISA score is passed in."""
    needs_scaler_flag = DATASETS[name].get("needs_scaler", False)

    mlp_g_only_params = None
    if g_model == "lp":
        # Load cached LP params (g_only only — SISA handled separately)
        _log(f"  Loading g_only LP cache...")
        g_only_params = _load_cached_params(name, seed, "g_only", split_type, n_pub)
        if g_only_params is None:
            _log(f"  seed={seed} n_pub={n_pub}: missing g_only LP cache, skipping")
            return None
    else:
        g_only_params = None
        _log(f"  Loading g_only MLP cache...")
        mlp_g_only_params = _load_mlp_cached_params(
            name, seed, "g_only", split_type, n_pub,
        )
        if mlp_g_only_params is None:
            _log(f"  seed={seed} n_pub={n_pub}: missing g_only MLP cache, skipping")
            return None

    _log(f"  Loading RQ caches...")
    rq_a = _load_rq_params(name, seed, n_pub, "a_centroid", split_type, g_model)
    rq_b = _load_rq_params(name, seed, n_pub, "b_centroid", split_type, g_model)
    rq_c = _load_rq_params(name, seed, n_pub, "c", split_type, g_model)
    rq_b_boost = _load_rq_params(name, seed, n_pub, "b_boost", split_type, g_model)
    rq_c_boost = _load_rq_params(name, seed, n_pub, "c_boost", split_type, g_model)

    _log(f"  Splitting data (n_pub={n_pub})...")
    pub_idx, priv_idx, test_idx = split_data(E, y, seed, n_pub, split_type)
    X_pub, y_pub = E[pub_idx], y[pub_idx]
    X_priv, y_priv = E[priv_idx], y[priv_idx]
    X_test, y_test = E[test_idx], y[test_idx]
    _log(f"  pub={len(y_pub)} priv={len(y_priv)} test={len(y_test)}")

    needs_scaler = DATASETS[name].get("needs_scaler", False)

    # ── g-only baseline ──
    _log(f"  g-only ({g_model}): training model...")
    t0 = time.time()
    if g_model == "mlp":
        g_trained, g_scaler = train_final_mlp_model(
            X_pub, y_pub, task_type, K, device,
            needs_scaler=needs_scaler_flag,
            mlp_params=mlp_g_only_params,
        )
    else:
        g_trained, g_scaler = train_final_model(
            X_pub, y_pub, g_only_params, task_type, K, device,
        )
    g_preds_test = predict_with_model(g_trained, X_test, g_scaler, task_type, device)
    g_preds_priv = predict_with_model(g_trained, X_priv, g_scaler, task_type, device)
    g_train_s = time.time() - t0

    if task_type == "classification":
        g_only_score = float((g_preds_test.argmax(axis=1) == y_test).mean())
    else:
        g_only_score = float(pearsonr(y_test, g_preds_test)[0])
    _log(f"  g-only: {g_only_score:.4f} ({g_train_s:.1f}s)")

    # ── Prepare features ──
    _log(f"  Preparing features (scaler={needs_scaler})...")
    if needs_scaler:
        feat_scaler = StandardScaler().fit(X_pub)
        X_pub_s = feat_scaler.transform(X_pub).astype(np.float32)
        X_priv_s = feat_scaler.transform(X_priv).astype(np.float32)
        X_test_s = feat_scaler.transform(X_test).astype(np.float32)
    else:
        X_pub_s = X_pub
        X_priv_s = X_priv
        X_test_s = X_test

    # Centroid projection
    _log(f"  Computing centroid features (K={K})...")
    t0 = time.time()
    F_pub, n_proj_dims = compute_centroid_features(
        X_pub_s, y_pub, X_pub_s, K, task_type
    )
    _log(f"  Centroid F_pub done ({time.time()-t0:.1f}s), projecting priv+test...")
    F_priv, _ = compute_centroid_features(X_pub_s, y_pub, X_priv_s, K, task_type)
    F_test, _ = compute_centroid_features(X_pub_s, y_pub, X_test_s, K, task_type)
    _log(f"  Centroid features done: F_pub={F_pub.shape}")

    # Precompute shared data
    F_all_proj = np.concatenate([F_pub, F_priv], axis=0).astype(np.float32)
    X_all_raw = np.concatenate([X_pub_s, X_priv_s], axis=0).astype(np.float32)
    _log(f"  Concatenated: F_all={F_all_proj.shape} X_all={X_all_raw.shape}")

    # ── QSS-A: centroid proj, RQ on D° only ──
    _log(f"  QSS-A: running...")
    t0 = time.time()
    b_a, ell_a, _ = rq_a
    b_a = min(b_a, len(y_pub))
    with _heartbeat("QSS-A"):
        qss_a_score = run_qss_on_features(
            F_pub, y_pub, F_priv, y_priv, F_test, y_test,
            b_a, ell_a, task_type, K,
        )
    _log(f"  QSS-A: {qss_a_score:.4f} ({time.time()-t0:.1f}s)")

    # ── QSS-B: centroid proj, RQ on all D ──
    _log(f"  QSS-B: running...")
    t0 = time.time()
    b_b, ell_b, _ = rq_b
    with _heartbeat("QSS-B"):
        qss_b_score = run_qss_on_features(
            F_pub, y_pub, F_priv, y_priv, F_test, y_test,
            b_b, ell_b, task_type, K, F_rq_train=F_all_proj,
        )
    _log(f"  QSS-B: {qss_b_score:.4f} ({time.time()-t0:.1f}s)")

    # ── QSS-C: no proj, RQ on all D ──
    _log(f"  QSS-C: running...")
    t0 = time.time()
    b_c, ell_c, _ = rq_c
    with _heartbeat("QSS-C"):
        qss_c_score = run_qss_on_features(
            X_pub_s, y_pub, X_priv_s, y_priv, X_test_s, y_test,
            b_c, ell_c, task_type, K, F_rq_train=X_all_raw,
        )
    _log(f"  QSS-C: {qss_c_score:.4f} ({time.time()-t0:.1f}s)")

    # ── QSS-D: no proj, RQ on all D + ε-DP ──
    _log(f"  QSS-D: running...")
    t0 = time.time()
    with _heartbeat("QSS-D"):
        qss_d_score = run_qss_on_features(
            X_pub_s, y_pub, X_priv_s, y_priv, X_test_s, y_test,
            b_c, ell_c, task_type, K, F_rq_train=X_all_raw, dp_epsilon=1.0,
        )
    _log(f"  QSS-D: {qss_d_score:.4f} ({time.time()-t0:.1f}s)")

    # ── B-boost ──
    _log(f"  QSS-B-boost: running...")
    t0 = time.time()
    b_bb, ell_bb, alpha_bb = rq_b_boost
    with _heartbeat("QSS-B-boost"):
        qss_b_boost_score = run_qss_boosted(
            F_all_proj, F_priv, y_priv, F_test, y_test,
            g_preds_priv, g_preds_test, b_bb, ell_bb, task_type, K, alpha=alpha_bb,
        )
    _log(f"  QSS-B-boost: {qss_b_boost_score:.4f} ({time.time()-t0:.1f}s)")

    # ── C-boost ──
    _log(f"  QSS-C-boost: running...")
    t0 = time.time()
    b_cb, ell_cb, alpha_cb = rq_c_boost
    with _heartbeat("QSS-C-boost"):
        qss_c_boost_score = run_qss_boosted(
            X_all_raw, X_priv_s, y_priv, X_test_s, y_test,
            g_preds_priv, g_preds_test, b_cb, ell_cb, task_type, K, alpha=alpha_cb,
        )
    _log(f"  QSS-C-boost: {qss_c_boost_score:.4f} ({time.time()-t0:.1f}s)")

    # ── Memory + latency for C ──
    _log(f"  Measuring QSS-C memory+latency...")
    with _heartbeat("QSS-C measurement"):
        mem_lat = measure_qss_memory_and_latency(
            X_all_raw, y_priv, X_priv_s, y_priv, X_test_s, y_test,
            b_c, ell_c, task_type, K,
        )

    if g_only_params is not None:
        g_only_optuna_s = g_only_params.get("optuna_seconds", 0.0)
    elif mlp_g_only_params is not None:
        g_only_optuna_s = mlp_g_only_params.get("optuna_seconds", 0.0)
    else:
        g_only_optuna_s = 0.0

    _log(
        f"  DONE: g={g_only_score:.4f} A={qss_a_score:.4f} B={qss_b_score:.4f} "
        f"C={qss_c_score:.4f} D={qss_d_score:.4f} "
        f"Bb={qss_b_boost_score:.4f} Cb={qss_c_boost_score:.4f} "
        f"SISA={sisa_score:.4f}"
    )

    row: dict[str, Any] = {
        "dataset": name,
        "seed": seed,
        "g_model": g_model,
        "split_type": split_type,
        "n_pub_requested": n_pub,
        "n_pub_actual": len(y_pub),
        "n_priv": len(y_priv),
        "n_test": len(y_test),
        "ab_proj": "centroid",
        "n_proj_dims": n_proj_dims,
        "feature_dim": F_pub.shape[1],
        "b_a": b_a,
        "ell_a": ell_a,
        "b_b": b_b,
        "ell_b": ell_b,
        "b_c": b_c,
        "ell_c": ell_c,
        "b_b_boost": b_bb,
        "ell_b_boost": ell_bb,
        "alpha_b_boost": f"{alpha_bb:.4f}",
        "b_c_boost": b_cb,
        "ell_c_boost": ell_cb,
        "alpha_c_boost": f"{alpha_cb:.4f}",
    }
    scores = {
        "g_only": g_only_score,
        "qss_a": qss_a_score,
        "qss_b": qss_b_score,
        "qss_c": qss_c_score,
        "qss_d": qss_d_score,
        "qss_b_boost": qss_b_boost_score,
        "qss_c_boost": qss_c_boost_score,
        "sisa": sisa_score,
    }
    suffix = "acc" if task_type == "classification" else "r"
    for method, score in scores.items():
        row[f"{method}_{suffix}"] = f"{score:.6f}"
    row.update(
        {
            "g_only_train_s": f"{g_train_s:.1f}",
            "g_only_optuna_s": f"{g_only_optuna_s:.1f}",
            "sisa_optuna_s": f"{sisa_optuna_s:.1f}",
            "avg_sisa_del_s": f"{avg_sisa_del_s:.1f}",
            "qss_c_memory_bytes": mem_lat["qss_memory_bytes"],
            "qss_c_memory_mb": f"{mem_lat['qss_memory_mb']:.4f}",
            "qss_c_predict_latency_us": f"{mem_lat['qss_predict_latency_us']:.2f}",
            "qss_c_encode_latency_us": f"{mem_lat['qss_encode_latency_us']:.2f}",
            "qss_c_total_latency_us": f"{mem_lat['qss_total_latency_us']:.2f}",
        }
    )
    return row


# ── Runner ──


class T10UnifiedRunner:
    """Worker-loop T10 evaluation runner.

    Each iteration: read existing CSV → find one combo → run it → upload → repeat.
    SISA is computed once per seed and cached (it doesn't depend on n_pub).
    """

    def __init__(
        self,
        dataset: str,
        n_seeds: int = 5,
        n_pub: int = 500,
        n_pub_list: str = "",
        split_type: str = "random",
        g_model: str = "lp",
    ) -> None:
        self.dataset = dataset
        self.n_seeds = n_seeds
        if n_pub_list:
            self.n_pub_values = [int(x) for x in n_pub_list.split(",")]
        else:
            self.n_pub_values = [n_pub]
        self.split_type = split_type
        self.g_model = g_model

    def start(self, job_config: Any = None) -> Any:
        self.run()
        return _NoOpOutput()

    def run(self) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Select Manifold results path based on g model type
        results_base = MF_RESULTS_MLP if self.g_model == "mlp" else MF_RESULTS
        rows_dir = f"{results_base}/rows"
        _log(
            f"T10UnifiedRunner: dataset={self.dataset} "
            f"n_seeds={self.n_seeds} n_pub_values={self.n_pub_values} "
            f"split_type={self.split_type} g_model={self.g_model} device={device}"
        )

        if self.dataset not in DATASETS:
            _log(f"Unknown dataset: {self.dataset}")
            return

        info = DATASETS[self.dataset]
        task_type = "classification" if info["metric"] == "accuracy" else "regression"
        K = info["K"]
        needs_scaler = info.get("needs_scaler", False)

        _log(f"Loading dataset {self.dataset} from Manifold...")
        E_raw, y = load_dataset_from_manifold(self.dataset)
        E = E_raw if needs_scaler else l2_normalize(E_raw)
        _log(f"Loaded: E={E.shape}, task={task_type}, K={K}")

        # SISA cache: seed -> (score, optuna_s, avg_del_s)
        sisa_cache: dict[int, tuple[float, float, float]] = {}

        # ── Worker loop ──
        while True:
            # Check which combos are done (via atomic per-combo files)
            _log("Checking done combos on Manifold...")
            done_keys = _read_done_keys(self.dataset, rows_dir)
            _log(f"Found {len(done_keys)} done combos")

            # Find todo
            todo = [
                (n_pub, seed)
                for n_pub in self.n_pub_values
                for seed in range(self.n_seeds)
                if (n_pub, seed) not in done_keys
            ]
            if not todo:
                _log(f"All {len(done_keys)} combos done for {self.dataset}!")
                break

            # Pick a random combo to avoid all workers racing on the same one
            import random
            n_pub, seed = random.choice(todo)
            _log(
                f"=== Starting n_pub={n_pub} seed={seed} "
                f"({len(done_keys)}/{len(done_keys)+len(todo)} done) ==="
            )

            # Compute SISA for this seed if not cached
            if seed not in sisa_cache:
                _log(f"Computing SISA for seed={seed} (once per seed)...")
                sisa_result = compute_sisa(
                    self.dataset, E, y, seed, task_type, K,
                    self.split_type, self.n_pub_values, device,
                    g_model=self.g_model,
                )
                if sisa_result is None:
                    _log(f"SISA cache miss for seed={seed}, skipping all combos with this seed")
                    # Skip all n_pub for this seed — mark as unevaluable
                    continue
                sisa_cache[seed] = sisa_result

            sisa_score, sisa_optuna_s, avg_sisa_del_s = sisa_cache[seed]

            # Run the combo
            row = run_one_combo(
                self.dataset, E, y, seed, task_type, K, n_pub,
                self.split_type, device, sisa_score, sisa_optuna_s, avg_sisa_del_s,
                g_model=self.g_model,
            )
            if row is None:
                _log(f"Combo n_pub={n_pub} seed={seed} returned None (cache miss), skipping")
                continue

            # Upload atomic per-combo file (no race condition)
            try:
                _upload_row_atomic(self.dataset, n_pub, seed, row, rows_dir)
                _log(f"Uploaded row {self.dataset}_n{n_pub}_s{seed}.json to {rows_dir}")
            except Exception as e:
                _log(f"WARNING: failed to upload row: {e}")

        # Final summary
        _log(f"=== All done for {self.dataset} ===")
        existing_rows = _read_existing_rows(self.dataset, rows_dir)
        metric = "acc" if task_type == "classification" else "r"
        for n_pub in self.n_pub_values:
            rows_np = [
                r for r in existing_rows if int(r["n_pub_requested"]) == n_pub
            ]
            if not rows_np:
                continue

            def _mean(key: str) -> float:
                return float(np.mean([float(r[key]) for r in rows_np]))

            _log(
                f"SUMMARY {self.dataset} (D°={n_pub}): "
                f"g={_mean(f'g_only_{metric}'):.4f} "
                f"A={_mean(f'qss_a_{metric}'):.4f} "
                f"C={_mean(f'qss_c_{metric}'):.4f} "
                f"Cb={_mean(f'qss_c_boost_{metric}'):.4f} "
                f"SISA={_mean(f'sisa_{metric}'):.4f}"
            )
