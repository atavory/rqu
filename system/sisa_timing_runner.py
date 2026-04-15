#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict
"""MAST runner: SISA uncached timing measurement.

For each dataset, runs Optuna from scratch (no cache) + retrain for each shard,
measuring full wall-clock deletion cost. This gives the true SISA unlearning
latency without any caching shortcuts.
"""

from __future__ import annotations

import csv
import io
import time
from typing import Any

import numpy as np
import optuna
import torch
import torch.nn as nn
from content_understanding.framework.utils.logging import MediaLog, TMediaLogger
    _NoOpOutput,
    DATASETS,
    get_shard_data,
    l2_normalize,
    load_dataset_from_manifold,
    MF_BUCKET,
    probe_objective,
    split_data,
    train_probe_pytorch,
)
from manifold.clients.python import ManifoldClient

logger: TMediaLogger = MediaLog.get_logger(__name__)

MF_RESULTS = "tree/fi_unlearning/results/sisa_timing"


def _upload_csv_to_manifold(path: str, content: str) -> None:
    data = content.encode("utf-8")
    with ManifoldClient.get_client(MF_BUCKET) as client:
        client.sync_put(path, data)


def train_linear_probe_timed(
    X_np: np.ndarray,
    y_np: np.ndarray,
    task_type: str,
    K: int,
    lr: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    device: str = "cpu",
) -> tuple[nn.Module, float]:
    """Train linear probe and return (model, wall_seconds)."""
    t0 = time.time()
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
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
    elapsed = time.time() - t0
    return model, elapsed


class SISATimingRunner:
    """Measures uncached SISA deletion cost on GPU.

    For each (dataset, seed, shard), runs Optuna from scratch + retrain,
    measuring full wall-clock. This is the "deletion cost" for SISA.
    Hydra-instantiable. Called by mitra framework via runner.start().
    """

    def __init__(
        self,
        dataset: str,
        n_seeds: int = 5,
        n_trials: int = 200,
    ) -> None:
        self.dataset = dataset
        self.n_seeds = n_seeds
        self.n_trials = n_trials

    def start(self, job_config: Any = None) -> Any:
        self.run()
        return _NoOpOutput()

    def run(self) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        msg = (
            f"SISATimingRunner: dataset={self.dataset} "
            f"n_seeds={self.n_seeds} n_trials={self.n_trials} device={device}"
        )
        logger.info(msg)
        print(msg, flush=True)

        info = DATASETS[self.dataset]
        task_type = "classification" if info["metric"] == "accuracy" else "regression"
        K = info["K"]
        needs_scaler = info.get("needs_scaler", False)

        E_raw, y = load_dataset_from_manifold(self.dataset)
        E = E_raw if needs_scaler else l2_normalize(E_raw)
        msg = f"Loaded: E={E.shape}, task={task_type}, K={K}"
        logger.info(msg)
        print(msg, flush=True)

        all_rows: list[dict[str, Any]] = []
        n_shards = 5

        for seed in range(self.n_seeds):
            pub_idx, priv_idx, _test_idx = split_data(E, y, seed)
            X_pub, y_pub = E[pub_idx], y[pub_idx]
            X_priv, y_priv = E[priv_idx], y[priv_idx]

            # Time each shard type
            for shard_type in ["sisa_full", "sisa_priv"]:
                for s in range(n_shards):
                    config = f"{shard_type}_s{s}"
                    X_train, y_train = get_shard_data(
                        X_pub,
                        y_pub,
                        X_priv,
                        y_priv,
                        config,
                        n_shards,
                    )

                    # Optuna from scratch (uncached)
                    optuna.logging.set_verbosity(optuna.logging.WARNING)
                    study = optuna.create_study(direction="maximize")
                    t0_optuna = time.time()
                    study.optimize(
                        lambda trial: probe_objective(
                            trial, X_train, y_train, task_type, K, device
                        ),
                        n_trials=self.n_trials,
                        show_progress_bar=False,
                    )
                    optuna_s = time.time() - t0_optuna

                    bp = study.best_params
                    X_s = X_train.copy()
                    if bp["use_scaler"]:
                        mean = X_s.mean(axis=0)
                        std = X_s.std(axis=0) + 1e-10
                        X_s = (X_s - mean) / std

                    _, train_s = train_linear_probe_timed(
                        X_s,
                        y_train,
                        task_type,
                        K,
                        lr=bp["lr"],
                        weight_decay=bp["weight_decay"],
                        epochs=bp["epochs"],
                        batch_size=bp["batch_size"],
                        device=device,
                    )

                    total_s = optuna_s + train_s
                    msg = (
                        f"  seed={seed} {config}: optuna={optuna_s:.1f}s "
                        f"train={train_s:.3f}s total={total_s:.1f}s "
                        f"n_train={len(y_train)} cv={study.best_value:.4f}"
                    )
                    logger.info(msg)
                    print(msg, flush=True)

                    all_rows.append(
                        {
                            "dataset": self.dataset,
                            "seed": seed,
                            "config": config,
                            "shard_type": shard_type,
                            "shard_idx": s,
                            "n_train": len(y_train),
                            "optuna_seconds": f"{optuna_s:.3f}",
                            "train_seconds": f"{train_s:.6f}",
                            "total_seconds": f"{total_s:.3f}",
                            "cv_score": f"{study.best_value:.6f}",
                            "device": device,
                        }
                    )

        if all_rows:
            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
            csv_content = buf.getvalue()

            csv_path = f"{MF_RESULTS}/{self.dataset}.csv"
            try:
                _upload_csv_to_manifold(csv_path, csv_content)
                msg = f"Uploaded {len(all_rows)} rows to {csv_path}"
            except Exception as e:
                msg = f"Upload failed: {e}"
                print("\n=== CSV RESULTS ===", flush=True)
                print(csv_content, flush=True)
                print("=== END CSV ===", flush=True)
            logger.info(msg)
            print(msg, flush=True)

        print(f"All done for {self.dataset}", flush=True)
