"""Feature drift via Population Stability Index (PSI) against a training-time baseline.
Conventional rules of thumb (not universal): <0.1 stable, 0.1-0.25 moderate, >0.25 significant."""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-4


def make_baseline(feature_df: pd.DataFrame, bins: int = 10) -> dict:
    """Decile edges and bin proportions per feature, computed on TRAINING data only."""
    base = {}
    for col in feature_df.columns:
        v = feature_df[col].dropna().to_numpy(dtype=float)
        edges = (
            np.unique(np.quantile(v, np.linspace(0, 1, bins + 1)[1:-1])) if len(v) else np.array([])
        )
        props = np.bincount(np.digitize(v, edges), minlength=len(edges) + 1) / max(len(v), 1)
        base[col] = {"edges": edges.tolist(), "proportions": props.tolist()}
    return base


def psi(edges, expected, values) -> float:
    v = np.asarray(values, dtype=float)
    actual = np.bincount(np.digitize(v, edges), minlength=len(edges) + 1) / max(len(v), 1)
    e, a = np.clip(np.asarray(expected), EPS, None), np.clip(actual, EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def level(value: float) -> str:
    return "stable" if value < 0.1 else "moderate" if value < 0.25 else "significant"


def drift_report(baseline: dict, live: pd.DataFrame, min_samples: int = 30) -> dict:
    if not baseline:
        return {"status": "no_baseline", "n_live": len(live), "features": {}}
    if len(live) < min_samples:
        return {
            "status": "insufficient_data",
            "n_live": len(live),
            "min_samples": min_samples,
            "features": {},
        }
    feats = {}
    for name, b in baseline.items():
        if name in live:
            val = psi(b["edges"], b["proportions"], live[name].to_numpy())
            feats[name] = {"psi": val, "level": level(val)}
    mx = max((f["psi"] for f in feats.values()), default=0.0)
    return {
        "status": "ok",
        "n_live": len(live),
        "max_psi": mx,
        "overall_level": level(mx),
        "features": feats,
    }
