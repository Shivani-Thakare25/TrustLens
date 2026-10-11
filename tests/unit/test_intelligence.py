import pandas as pd
import pytest

from src.product_intelligence.product_profile import product_profile
from src.product_intelligence.reviewer_profile import (
    ELEVATED,
    NOT_ELEVATED,
    reviewer_profile,
)
from src.product_intelligence.trust_adjusted_sentiment import trust_adjusted_rating


def test_trust_adjusted_rating_math_and_guards():
    r = pd.Series([5, 5, 5, 1, 1.0])
    p = pd.Series([0.9, 0.9, 0.9, 0.0, 0.0])
    est = trust_adjusted_rating(r, p, min_reviews=5)
    assert est == pytest.approx((5 * 0.3 + 1 * 2) / (0.3 + 2))
    assert est < r.mean()
    assert trust_adjusted_rating(r[:3], p[:3], min_reviews=5) is None
    assert trust_adjusted_rating(r, pd.Series([1.0] * 5)) is None


def test_product_profile_fields():
    df = pd.DataFrame(
        {"rating": [5, 5, 4, 2, 1, 5], "fake_probability": [0.9, 0.8, 0.1, 0.1, 0.1, 0.7]}
    )
    pr = product_profile(df, threshold=0.5)
    assert pr["total_reviews"] == 6 and pr["suspicious_percentage"] == pytest.approx(50.0)
    assert pr["rating_distribution"] == {1: 1, 2: 1, 4: 1, 5: 3}
    assert pr["trust_adjusted_rating_estimate"] < pr["raw_rating"]
    no_rating = product_profile(pd.DataFrame({"fake_probability": [0.1] * 6}))
    assert no_rating["raw_rating"] is None and no_rating["trust_adjusted_rating_estimate"] is None


def _rv():
    return pd.DataFrame(
        {
            "review_id": list("abcd") + ["e"],
            "user_id": ["u1"] * 4 + ["u2"],
            "product_id": ["p1"] * 4 + ["p2"],
            "review_text": ["x" * 10] * 5,
            "rating": [5, 5, 5, 5, 3],
            "review_date": pd.to_datetime(["2024-01-01"] * 4 + ["2024-02-01"]),
        }
    )


def test_reviewer_risk_elevated_and_normal():
    df = _rv()
    hi = reviewer_profile(df, "u1", clustered_ids={"a", "b", "c", "d"}, burst_dates={"2024-01-01"})
    assert hi["assessment"] == ELEVATED and hi["reviewer_behavioral_risk"] > 0.6
    lo = reviewer_profile(df, "u2", clustered_ids=set(), burst_dates=set())
    assert lo["assessment"] == NOT_ELEVATED
    assert reviewer_profile(df, "nobody")["reviewer_behavioral_risk"] is None


def test_reviewer_insufficient_data():
    r = reviewer_profile(_rv(), "u2")
    assert r["reviewer_behavioral_risk"] is None and "Insufficient" in r["assessment"]
