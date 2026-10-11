"""Leakage-safe train/validation/test splitting using group-aware splits."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def make_groups(df: pd.DataFrame) -> np.ndarray:
    """Group key: user_id when available (prevents same reviewer in train and test),
    otherwise the normalized review text (prevents duplicate leakage)."""
    if "user_id" in df.columns and df["user_id"].notna().any():
        base = df["user_id"].astype("string")
        fallback = df["review_text"].astype("string").str.lower().str.strip()
        return base.fillna(fallback).to_numpy()
    return df["review_text"].astype("string").str.lower().str.strip().to_numpy()


def group_train_val_test_split(
    df: pd.DataFrame, val_size: float = 0.15, test_size: float = 0.15, seed: int = 42
):
    """Return (train_idx, val_idx, test_idx) as positional index arrays with disjoint groups."""
    groups = make_groups(df)
    idx = np.arange(len(df))
    outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval, test = next(outer.split(idx, groups=groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=val_size / (1 - test_size), random_state=seed)
    tr, va = next(inner.split(trainval, groups=groups[trainval]))
    return trainval[tr], trainval[va], test
