"""Measure how model predictions change under perturbations. Reports measured values only."""

from __future__ import annotations

import numpy as np

from robustness.perturbations import PERTURBATIONS


def evaluate_robustness(predict_fake, texts, threshold: float = 0.5, max_texts: int = 500) -> dict:
    """predict_fake: list[str] -> array of fake probabilities. Perturbations that leave a text unchanged are skipped."""
    texts = [t for t in texts if isinstance(t, str) and t.strip()][:max_texts]
    if not texts:
        raise ValueError("No non-empty texts to evaluate.")
    base = np.asarray(predict_fake(texts), dtype=float)
    out = {"n_texts": len(texts), "threshold": threshold, "perturbations": {}}
    for name, fn in PERTURBATIONS.items():
        idx = [i for i, t in enumerate(texts) if fn(t) != t and fn(t).strip()]
        if not idx:
            out["perturbations"][name] = {"n_changed": 0}
            continue
        new = np.asarray(predict_fake([fn(texts[i]) for i in idx]), dtype=float)
        old = base[idx]
        diff = new - old
        flips = (old >= threshold) != (new >= threshold)
        out["perturbations"][name] = {
            "n_changed": len(idx),
            "mean_abs_probability_change": float(np.abs(diff).mean()),
            "max_abs_probability_change": float(np.abs(diff).max()),
            "mean_signed_change": float(diff.mean()),
            "classification_flip_rate": float(flips.mean()),
        }
    return out
