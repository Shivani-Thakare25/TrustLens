"""Model probability -> TrustLens application score and decision bands.

Bands are APPLICATION-LEVEL defaults (not scientific truth) and can be overridden from config.
"""

from __future__ import annotations

DEFAULT_BANDS = [
    (0, 29, "High suspicion"),
    (30, 49, "Moderate-high suspicion"),
    (50, 69, "Uncertain"),
    (70, 84, "Likely genuine"),
    (85, 100, "High-confidence genuine"),
]


def decision_band(trust_score: int, bands=None) -> str:
    for lo, hi, name in bands or DEFAULT_BANDS:
        if lo <= trust_score <= hi:
            return name
    raise ValueError(f"trust_score out of range: {trust_score}")


def score_from_probability(fake_probability: float, threshold: float = 0.5, bands=None) -> dict:
    """fake_probability is the calibrated MODEL probability; risk/trust score is derived from it."""
    if not 0.0 <= fake_probability <= 1.0:
        raise ValueError("fake_probability must be in [0, 1]")
    risk = round(fake_probability * 100)
    trust = 100 - risk
    return {
        "fake_probability": fake_probability,
        "genuine_probability": 1 - fake_probability,
        "risk_score": risk,
        "trust_score": trust,
        "classification": (
            "Predicted suspicious" if fake_probability >= threshold else "Likely genuine"
        ),
        "confidence": max(fake_probability, 1 - fake_probability),
        "decision_band": decision_band(trust, bands),
    }
