"""Product-level profile built from per-review model outputs."""

from __future__ import annotations

import pandas as pd

from src.product_intelligence.trust_adjusted_sentiment import trust_adjusted_rating


def product_profile(
    df: pd.DataFrame,
    threshold: float = 0.5,
    min_reviews: int = 5,
    suspicious_cluster_count: int | None = None,
    burst_days: int | None = None,
) -> dict:
    """df needs columns: fake_probability; optional: rating. Fields unavailable are None."""
    n = len(df)
    fp = df["fake_probability"]
    rating = df["rating"] if "rating" in df else pd.Series([None] * n, dtype="float")
    raw = float(rating.mean()) if rating.notna().any() else None
    adj = trust_adjusted_rating(rating, fp, min_reviews)
    dist = (
        rating.dropna().astype(int).value_counts().sort_index().to_dict()
        if rating.notna().any()
        else None
    )
    return {
        "total_reviews": n,
        "rating_distribution": {int(k): int(v) for k, v in dist.items()} if dist else None,
        "raw_rating": raw,
        "suspicious_percentage": float((fp >= threshold).mean() * 100) if n else None,
        "suspicious_cluster_count": suspicious_cluster_count,
        "temporal_anomaly_days": burst_days,
        "trust_adjusted_rating_estimate": adj,
        "note": (
            "Trust-adjusted rating estimate; not a verified true rating."
            if adj is not None
            else f"Insufficient data (<{min_reviews} rated reviews) for an estimate."
        ),
    }
