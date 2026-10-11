"""Shared AutoML plumbing: same group-aware splits and metric code as the core pipeline."""

from __future__ import annotations

import json
import time
from pathlib import Path

from src.data.cleaning import clean_dataframe
from src.data.ingestion import load_raw
from src.features.text_features import build_text_feature_frame
from src.models.splits import group_train_val_test_split
from src.models.threshold import select_threshold
from src.models.train import compute_metrics


class AutoMLUnavailable(RuntimeError):
    """Raised when the optional AutoML dependency is not installed."""


def prepare_splits(csv_path, seed: int = 42):
    """Handcrafted text features (tabular) with the same train/val/test split as the core models."""
    df, _ = clean_dataframe(load_raw(csv_path))
    df = df[df["label"].notna()].reset_index(drop=True)
    y = df["label"].astype(int).to_numpy()
    if len(set(y)) < 2:
        raise ValueError("Need both classes in 'label'.")
    X = build_text_feature_frame(df["review_text"].tolist())
    tr, va, te = group_train_val_test_split(df, seed=seed)
    return {
        k: (X.iloc[i].reset_index(drop=True), y[i])
        for k, i in (("train", tr), ("val", va), ("test", te))
    }


def finalize(name: str, p_val, p_test, splits, train_seconds: float, out_dir, extra=None) -> dict:
    thr = select_threshold(splits["val"][1], p_val, "f1")
    res = {
        "model": name,
        "status": "trained",
        "features": "handcrafted text features only (tabular)",
        "threshold": thr,
        "validation": compute_metrics(splits["val"][1], p_val, thr),
        "test": compute_metrics(splits["test"][1], p_test, thr),
        "train_seconds": train_seconds,
        **(extra or {}),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(res, indent=2))
    return res


def timed(fn):
    t0 = time.perf_counter()
    r = fn()
    return r, time.perf_counter() - t0
