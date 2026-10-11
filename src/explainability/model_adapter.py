"""Adapters turning the saved text model into plain functions for explainers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def proba_fn(model):
    """texts -> (n, 2) array [genuine, fake]."""

    def f(texts):
        clean = [str(t) if t is not None else "" for t in texts]
        return model.predict_proba(pd.DataFrame({"review_text": clean}))

    return f


def fake_fn(model):
    """texts -> (n,) array of fake probability."""
    p = proba_fn(model)
    classes = getattr(model, "classes_", None)
    fake_idx = 1
    if classes is not None and len(classes) == 2:
        if 1 in classes:
            fake_idx = list(classes).index(1)
    return lambda texts: np.asarray(p(texts))[:, fake_idx]
