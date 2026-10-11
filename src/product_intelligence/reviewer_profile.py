"""Reviewer behavioral profile. reviewer_behavioral_risk is a documented HEURISTIC in [0,1]:
mean of available components (cluster share, burst share, rating extremity, product repetition)."""

from __future__ import annotations

import pandas as pd

ELEVATED = "Elevated reviewer behavioral risk based on available signals"
NOT_ELEVATED = "Reviewer behavioral risk not elevated based on available signals"


def reviewer_profile(
    df: pd.DataFrame,
    user_id,
    clustered_ids: set | None = None,
    burst_dates: set | None = None,
    elevated_at: float = 0.6,
) -> dict:
    sub = df[df["user_id"] == user_id]
    n = len(sub)
    if n == 0:
        return {
            "user_id": user_id,
            "review_count": 0,
            "reviewer_behavioral_risk": None,
            "assessment": "No reviews found for this reviewer.",
        }
    out = {
        "user_id": user_id,
        "review_count": n,
        "average_rating": float(sub["rating"].mean()) if sub["rating"].notna().any() else None,
        "rating_variance": (
            float(sub["rating"].var(ddof=0)) if sub["rating"].notna().sum() > 1 else None
        ),
        "product_diversity": (
            float(sub["product_id"].nunique() / n) if sub["product_id"].notna().any() else None
        ),
        "average_review_length": float(sub["review_text"].str.len().mean()),
    }
    dates = pd.to_datetime(sub["review_date"], errors="coerce").dropna()
    out["reviews_per_day"] = (
        float(n / max((dates.max() - dates.min()).days, 1)) if len(dates) > 1 else None
    )
    comps = {}
    if clustered_ids is not None:
        comps["cluster_share"] = float(sub["review_id"].isin(clustered_ids).mean())
    if burst_dates is not None and len(dates):
        comps["burst_share"] = float(dates.dt.strftime("%Y-%m-%d").isin(burst_dates).mean())
    if n >= 3 and sub["rating"].notna().any():
        comps["rating_extremity"] = float(sub["rating"].isin([1, 5]).mean())
    if n >= 2 and out["product_diversity"] is not None:
        comps["product_repetition"] = 1.0 - out["product_diversity"]
    out["risk_components"] = comps
    risk = sum(comps.values()) / len(comps) if comps else None
    out["reviewer_behavioral_risk"] = risk
    out["assessment"] = (
        "Insufficient data for a behavioral assessment."
        if risk is None
        else ELEVATED if risk >= elevated_at else NOT_ELEVATED
    )
    return out
