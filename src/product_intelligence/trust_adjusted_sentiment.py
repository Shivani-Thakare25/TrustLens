"""Trust-adjusted rating ESTIMATE: ratings weighted by (1 - fake_probability). Not a 'true rating'."""

from __future__ import annotations

import numpy as np
import pandas as pd


def trust_adjusted_rating(ratings: pd.Series, fake_probs: pd.Series, min_reviews: int = 5):
    """Return estimate or None when fewer than min_reviews rated reviews or weights sum to ~0."""
    m = ratings.notna() & fake_probs.notna()
    if m.sum() < min_reviews:
        return None
    w = 1.0 - fake_probs[m].to_numpy(dtype=float)
    if w.sum() < 1e-9:
        return None
    return float(np.average(ratings[m].to_numpy(dtype=float), weights=w))
