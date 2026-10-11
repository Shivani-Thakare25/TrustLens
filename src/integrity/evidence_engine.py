"""Evidence engine: converts computed values into signals. Unavailable inputs are marked
unavailable; no value is ever fabricated."""

from __future__ import annotations

from typing import Optional

from src.integrity.evidence_schema import Signal
from src.integrity.signal_registry import SIGNALS


def _strength(score: float) -> str:
    return (
        "high" if score >= 0.7 else "medium" if score >= 0.4 else "low" if score >= 0.15 else "none"
    )


def _sig(name, value, score, explain_hi, explain_lo):
    if score is None:
        return Signal(
            name, SIGNALS[name], None, None, "unavailable", "Input not available", "none", False
        )
    direction = "increases_risk" if score >= 0.4 else "neutral"
    return Signal(
        name,
        SIGNALS[name],
        float(value),
        float(score),
        direction,
        explain_hi if direction == "increases_risk" else explain_lo,
        _strength(score),
        True,
    )


def build_evidence(
    text_features: dict,
    similar: Optional[list] = None,
    aspect_check: Optional[dict] = None,
    verified_purchase: Optional[bool] = None,
    graph_row: Optional[dict] = None,
    burst_flag: Optional[bool] = None,
    cluster: Optional[dict] = None,
    image_signals: Optional[dict] = None,
) -> list[Signal]:
    """Normalization caps (documented heuristics): promo_ratio 0.2, exclamations 4, generic phrases 2."""
    f = text_features
    out = [
        _sig(
            "promotional_language",
            f["promo_ratio"],
            min(1.0, f["promo_ratio"] / 0.2),
            "Review contains a high share of promotional wording",
            "Promotional wording is within a normal range",
        ),
        _sig(
            "exclamation_intensity",
            f["exclaim_count"],
            min(1.0, f["exclaim_count"] / 4),
            "Review uses many exclamation marks",
            "Exclamation usage is normal",
        ),
        _sig(
            "generic_phrases",
            f["generic_phrase_count"],
            min(1.0, f["generic_phrase_count"] / 2),
            "Review relies on generic stock phrases",
            "Few generic stock phrases",
        ),
    ]
    if similar is None:
        out.append(_sig("review_similarity", None, None, "", ""))
    else:
        top = similar[0]["similarity_score"] if similar else 0.0
        out.append(
            _sig(
                "review_similarity",
                top,
                top,
                f"Review is highly similar to an existing review (score {top:.2f})",
                "No highly similar existing reviews",
            )
        )
    if not aspect_check or not aspect_check.get("available"):
        out.append(_sig("rating_aspect_consistency", None, None, "", ""))
    else:
        v = 1.0 if aspect_check["inconsistent"] else 0.0
        out.append(
            _sig(
                "rating_aspect_consistency",
                v,
                v,
                aspect_check["explanation"],
                aspect_check["explanation"],
            )
        )
    if verified_purchase is None:
        out.append(_sig("unverified_purchase", None, None, "", ""))
    else:
        v = 0.0 if verified_purchase else 1.0
        out.append(
            _sig(
                "unverified_purchase",
                v,
                0.5 * v,
                "Purchase is not verified (weak signal on its own)",
                "Verified purchase",
            )
        )
    freq = (graph_row or {}).get("reviewer_product_frequency")
    if freq is None or freq != freq:
        out.append(_sig("reviewer_product_repeat", None, None, "", ""))
    else:
        out.append(
            _sig(
                "reviewer_product_repeat",
                freq,
                min(1.0, (freq - 1) / 2),
                f"Same reviewer posted {int(freq)} reviews of this product",
                "Single review of this product",
            )
        )
    out.append(
        _sig(
            "temporal_burst",
            None if burst_flag is None else float(burst_flag),
            None if burst_flag is None else float(burst_flag),
            "Review date falls on a statistically unusual burst day",
            "No temporal burst on this date",
        )
    )
    if cluster is None:
        out.append(_sig("cluster_membership", None, None, "", ""))
    else:
        s = cluster["cluster_risk_score"] / 100
        out.append(
            _sig(
                "cluster_membership",
                cluster["cluster_risk_score"],
                s,
                "Part of a potentially coordinated review cluster",
                "Member of a low-risk similarity group",
            )
        )
    if not image_signals:
        out.append(_sig("image_evidence", None, None, "", ""))
    else:
        v = 1.0 if image_signals.get("duplicate_image_flag") else 0.0
        out.append(
            _sig("image_evidence", v, v, "Duplicate image detected", "No duplicate image detected")
        )
    return out


def why_flagged(signals: list[Signal], max_items: int = 5) -> list[str]:
    """Human-readable reasons generated only from signals that actually increase risk."""
    risky = sorted(
        (s for s in signals if s.direction == "increases_risk"), key=lambda s: -s.normalized_score
    )
    return [s.explanation for s in risky[:max_items]]


def evidence_breakdown(signals: list[Signal]) -> dict:
    """Per-category mean normalized score over AVAILABLE signals; None when a category has none."""
    cats: dict = {}
    for s in signals:
        cats.setdefault(s.category, [])
        if s.availability:
            cats[s.category].append(s.normalized_score)
    return {c: (sum(v) / len(v) if v else None) for c, v in cats.items()}
