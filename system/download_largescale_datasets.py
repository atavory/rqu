#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Download large-scale datasets and upload to Manifold as .npy files.

Datasets:
  - criteo: Criteo Click Logs (day 0, ~45M rows, D=13 numeric, K=2)
  - news: Online News Popularity (UCI, ~40K rows, D=58 numeric, K=2 binarized)
  - tabred_weather: TabRed Weather (Yandex, ~1M rows, D=125 numeric, regression→K=10 bucketed)

Usage:
  buck2 run fbcode//mitra/projects/fi_unlearning/system:download_largescale_datasets -- DATASET

Each dataset is downloaded, preprocessed to (X: float32, y: int64), and
uploaded to Manifold at tree/fi_unlearning/datasets/{name}_features.npy
and tree/fi_unlearning/datasets/{name}_labels.npy.

NOTE: HF Hub downloads are blocked from Claude Code sessions. Run this
from a regular terminal or MAST job.
"""

from __future__ import annotations

import io
import sys
import tempfile

import numpy as np


MF_BUCKET = "fi_platform_ml_infra_fluent2_bucket"
MF_PREFIX = "tree/fi_unlearning/datasets"


def _log(msg: str) -> None:
    print(f"[DL] {msg}", file=sys.stderr, flush=True)


def _mf_put_npy(path: str, arr: np.ndarray) -> None:
    from manifold.clients.python import ManifoldClient

    buf = io.BytesIO()
    np.save(buf, arr)
    buf.seek(0)
    with ManifoldClient.get_client(MF_BUCKET) as mc:
        mc.sync_put(path, buf, predicate=ManifoldClient.Predicates.AllowOverwrite)
    _log(f"  Uploaded {path} ({arr.shape}, {arr.dtype})")


def download_criteo() -> tuple[np.ndarray, np.ndarray]:
    """Criteo Display Ads Challenge — day_0 from the 1TB dataset.

    13 numeric features (I1-I13) + label (click/no-click).
    We use only the 13 numeric features (ignore 26 categorical hashes).
    Missing numeric values → 0.
    """
    _log("Downloading Criteo day_0...")
    import urllib.request
    import gzip

    url = "https://storage.googleapis.com/criteo-cail-datasets/day_0.gz"
    with tempfile.NamedTemporaryFile(suffix=".gz") as tmp:
        urllib.request.urlretrieve(url, tmp.name)
        _log("  Downloaded, parsing...")
        X_rows = []
        y_rows = []
        with gzip.open(tmp.name, "rt") as f:
            for i, line in enumerate(f):
                parts = line.strip().split("\t")
                label = int(parts[0])
                # 13 numeric features are columns 1-13
                feats = []
                for j in range(1, 14):
                    val = parts[j] if j < len(parts) else ""
                    feats.append(float(val) if val else 0.0)
                X_rows.append(feats)
                y_rows.append(label)
                if (i + 1) % 5_000_000 == 0:
                    _log(f"  {i + 1} rows...")
    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.int64)
    _log(f"  Criteo: X={X.shape}, y={y.shape}, K={len(np.unique(y))}")
    return X, y


def download_news() -> tuple[np.ndarray, np.ndarray]:
    """Online News Popularity (UCI).

    58 predictive features, target = number of shares.
    We binarize: shares > median → class 1, else → class 0.
    """
    _log("Downloading Online News Popularity...")
    import urllib.request
    import zipfile
    import csv

    url = "https://archive.ics.uci.edu/static/public/332/online+news+popularity.zip"
    with tempfile.NamedTemporaryFile(suffix=".zip") as tmp:
        urllib.request.urlretrieve(url, tmp.name)
        with zipfile.ZipFile(tmp.name) as zf:
            csv_name = [n for n in zf.namelist() if n.endswith(".csv")][0]
            with zf.open(csv_name) as f:
                reader = csv.reader(io.TextIOWrapper(f))
                header = next(reader)
                data = []
                for row in reader:
                    data.append([float(x.strip()) for x in row])
    arr = np.array(data, dtype=np.float32)
    # First 2 cols are url + timedelta (non-predictive), last col is shares
    X = arr[:, 2:-1]  # 58 features
    shares = arr[:, -1]
    median_shares = np.median(shares)
    y = (shares > median_shares).astype(np.int64)
    _log(f"  News: X={X.shape}, y={y.shape}, K={len(np.unique(y))}, median_shares={median_shares}")
    return X, y


def download_tabred_weather() -> tuple[np.ndarray, np.ndarray]:
    """TabRed Weather dataset (Yandex Research).

    Regression target (temperature). We bucket into 10 equal-frequency bins.
    """
    _log("Downloading TabRed Weather...")
    try:
        from datasets import load_dataset
    except ImportError:
        _log("ERROR: `datasets` package not available. Install or use HF_DATASETS_OFFLINE=1")
        sys.exit(1)

    ds = load_dataset("yandex-research/tabred", "weather", trust_remote_code=True)
    # Combine all splits
    X_parts, y_parts = [], []
    for split in ds:
        df = ds[split]
        # All columns except target
        col_names = [c for c in df.column_names if c != "target"]
        X_split = np.array([df[c] for c in col_names], dtype=np.float32).T
        y_split = np.array(df["target"], dtype=np.float32)
        X_parts.append(X_split)
        y_parts.append(y_split)
    X = np.concatenate(X_parts, axis=0)
    y_raw = np.concatenate(y_parts, axis=0)

    # Bucket into 10 equal-frequency bins
    K = 10
    percentiles = np.percentile(y_raw, np.linspace(0, 100, K + 1))
    y = np.digitize(y_raw, percentiles[1:-1]).astype(np.int64)
    _log(f"  Weather: X={X.shape}, y={y.shape}, K={len(np.unique(y))}")
    return X, y


DOWNLOADERS = {
    "criteo": download_criteo,
    "news": download_news,
    "tabred_weather": download_tabred_weather,
}


def _parse_criteo_gz(path: str) -> tuple[list, list]:
    """Parse one Criteo gz file, return (X_rows, y_rows)."""
    import gzip

    _log(f"  Parsing {path}...")
    opener = gzip.open if path.endswith(".gz") else open
    X_rows, y_rows = [], []
    try:
        with opener(path, "rt") as f:
            for i, line in enumerate(f):
                parts = line.strip().split("\t")
                if not parts or not parts[0]:
                    continue
                label = int(parts[0])
                feats = []
                for j in range(1, 14):
                    val = parts[j] if j < len(parts) else ""
                    feats.append(float(val) if val else 0.0)
                X_rows.append(feats)
                y_rows.append(label)
                if (i + 1) % 5_000_000 == 0:
                    _log(f"    {len(X_rows)} rows...")
    except EOFError:
        _log(f"    WARNING: truncated gz at {len(X_rows)} rows")
    _log(f"    {path}: {len(X_rows)} rows")
    return X_rows, y_rows


def load_criteo_from_file(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load Criteo from a local .gz file or directory of .gz files."""
    import glob
    import os

    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "day_*.gz")))
        _log(f"Loading Criteo from {len(files)} files in {path}...")
        all_X, all_y = [], []
        for f in files:
            xr, yr = _parse_criteo_gz(f)
            all_X.extend(xr)
            all_y.extend(yr)
            _log(f"  Running total: {len(all_X)} rows")
        X = np.array(all_X, dtype=np.float32)
        y = np.array(all_y, dtype=np.int64)
    else:
        _log(f"Loading Criteo from {path}...")
        xr, yr = _parse_criteo_gz(path)
        X = np.array(xr, dtype=np.float32)
        y = np.array(yr, dtype=np.int64)
    _log(f"  Criteo total: X={X.shape}, y={y.shape}, K={len(np.unique(y))}")
    return X, y


def load_news_from_file(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load News from a local .zip or .csv file."""
    import csv
    import zipfile

    _log(f"Loading News from {path}...")
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            csv_name = [n for n in zf.namelist() if n.endswith(".csv")][0]
            with zf.open(csv_name) as f:
                reader = csv.reader(io.TextIOWrapper(f))
                header = next(reader)
                data = [[float(x.strip()) for x in row[2:]] for row in reader]
    else:
        with open(path) as f:
            reader = csv.reader(f)
            header = next(reader)
            data = [[float(x.strip()) for x in row[2:]] for row in reader]
    arr = np.array(data, dtype=np.float32)
    # row[2:] already skipped URL + timedelta, so features are all but last col
    X = arr[:, :-1]
    shares = arr[:, -1]
    y = (shares > np.median(shares)).astype(np.int64)
    _log(f"  News: X={X.shape}, y={y.shape}, K={len(np.unique(y))}")
    return X, y


def load_tabred_from_file(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load TabRed Weather from local .parquet or .csv files.

    Expects a directory containing train.parquet, val.parquet, test.parquet
    (or a single CSV).
    """
    import os

    _log(f"Loading TabRed Weather from {path}...")
    if os.path.isdir(path):
        import pyarrow.parquet as pq

        X_parts, y_parts = [], []
        for split in ("train", "val", "test"):
            pf = os.path.join(path, f"{split}.parquet")
            if not os.path.exists(pf):
                continue
            table = pq.read_table(pf)
            df = table.to_pandas()
            cols = [c for c in df.columns if c != "target"]
            X_parts.append(df[cols].values.astype(np.float32))
            y_parts.append(df["target"].values.astype(np.float32))
        X = np.concatenate(X_parts, axis=0)
        y_raw = np.concatenate(y_parts, axis=0)
    else:
        import csv
        with open(path) as f:
            reader = csv.reader(f)
            header = next(reader)
            data = [[float(x) for x in row] for row in reader]
        arr = np.array(data, dtype=np.float32)
        X = arr[:, :-1]
        y_raw = arr[:, -1]
    K = 10
    percentiles = np.percentile(y_raw, np.linspace(0, 100, K + 1))
    y = np.digitize(y_raw, percentiles[1:-1]).astype(np.int64)
    _log(f"  Weather: X={X.shape}, y={y.shape}, K={len(np.unique(y))}")
    return X, y


LOCAL_LOADERS = {
    "criteo": load_criteo_from_file,
    "news": load_news_from_file,
    "tabred_weather": load_tabred_from_file,
}


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <{'|'.join(DOWNLOADERS)}> [LOCAL_PATH]")
        print("  If LOCAL_PATH given, reads from local file instead of downloading.")
        sys.exit(1)

    name = sys.argv[1]
    local_path = sys.argv[2] if len(sys.argv) > 2 else None

    _log(f"=== Dataset: {name} ===")
    if local_path:
        if name not in LOCAL_LOADERS:
            print(f"Unknown dataset: {name}")
            sys.exit(1)
        X, y = LOCAL_LOADERS[name](local_path)
    else:
        if name not in DOWNLOADERS:
            print(f"Unknown dataset: {name}")
            sys.exit(1)
        X, y = DOWNLOADERS[name]()

    # Normalize features (zero-mean, unit-variance)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    X = ((X - mean) / std).astype(np.float32)

    # Upload to both paths:
    # 1) T10 pipeline: manifold_datasets/{name}_X.npy + {name}_y.npy
    # 2) MLP runner:   {name}_features.npy + {name}_labels.npy
    t10_X = f"{MF_PREFIX}/manifold_datasets/{name}_X.npy"
    t10_y = f"{MF_PREFIX}/manifold_datasets/{name}_y.npy"
    mlp_feat = f"{MF_PREFIX}/{name}_features.npy"
    mlp_lab = f"{MF_PREFIX}/{name}_labels.npy"
    _mf_put_npy(t10_X, X)
    _mf_put_npy(t10_y, y)
    _mf_put_npy(mlp_feat, X)
    _mf_put_npy(mlp_lab, y)
    _log(f"=== Done: {name} → T10: {t10_X}, MLP: {mlp_feat} ===")


class DownloadDatasetsRunner:
    """MAST runner that downloads datasets on a machine with internet access."""

    def __init__(self, datasets: str = "all") -> None:
        self.datasets = datasets

    def start(self, job_config=None):
        self.run()

        class _NoOp:
            def upload_to_manifold(self) -> None:
                pass

        return _NoOp()

    def run(self) -> None:
        import time
        import threading

        def _heartbeat(stop):
            while not stop.is_set():
                print(f"[heartbeat] download alive at {time.strftime('%H:%M:%S')}", flush=True)
                stop.wait(120)

        stop = threading.Event()
        threading.Thread(target=_heartbeat, args=(stop,), daemon=True).start()

        targets = list(DOWNLOADERS.keys()) if self.datasets == "all" else self.datasets.split(",")
        for name in targets:
            if name not in DOWNLOADERS:
                _log(f"Unknown dataset: {name}, skipping")
                continue
            _log(f"=== Dataset: {name} ===")
            X, y = DOWNLOADERS[name]()
            mean = X.mean(axis=0)
            std = X.std(axis=0)
            std[std == 0] = 1.0
            X = ((X - mean) / std).astype(np.float32)
            t10_X = f"{MF_PREFIX}/manifold_datasets/{name}_X.npy"
            t10_y = f"{MF_PREFIX}/manifold_datasets/{name}_y.npy"
            mlp_feat = f"{MF_PREFIX}/{name}_features.npy"
            mlp_lab = f"{MF_PREFIX}/{name}_labels.npy"
            _mf_put_npy(t10_X, X)
            _mf_put_npy(t10_y, y)
            _mf_put_npy(mlp_feat, X)
            _mf_put_npy(mlp_lab, y)
            _log(f"=== Done: {name} ===")

        stop.set()
        _log("All downloads complete!")


if __name__ == "__main__":
    main()
