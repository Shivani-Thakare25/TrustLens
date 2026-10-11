import pandas as pd
import pytest

from src.clustering.coordination import LABEL, score_clusters
from src.clustering.review_clusters import find_clusters
from src.features.text_features import extract_text_features
from src.graph.graph_builder import build_graph
from src.graph.graph_features import graph_features
from src.graph.graph_visualization import graph_figure
from src.integrity.evidence_engine import (
    build_evidence,
    evidence_breakdown,
    why_flagged,
)
from src.integrity.risk_scoring import decision_band, score_from_probability
from src.temporal.burst_detection import detect_bursts, temporal_concentration


def test_burst_detected_on_spike_day_only():
    days = pd.date_range("2024-01-01", periods=30)
    dates = list(days) + [pd.Timestamp("2024-01-15")] * 14
    res = detect_bursts(pd.Series(dates))
    assert [r["date"] for r in res] == ["2024-01-15"]
    assert res[0]["count"] == 15


def test_no_burst_on_uniform_or_short_data():
    assert detect_bursts(pd.Series(pd.date_range("2024-01-01", periods=30))) == []
    assert detect_bursts(pd.Series(pd.date_range("2024-01-01", periods=3))) == []
    assert detect_bursts(pd.Series([None, None])) == []


def test_temporal_concentration_values():
    d = pd.Series(pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-03-01"]))
    assert temporal_concentration(d, 7) == 0.75
    assert temporal_concentration(pd.Series([None])) is None


def _df():
    t = "Best product ever, highly recommend it to everyone, five stars"
    return pd.DataFrame(
        {
            "review_id": ["a", "b", "c", "d"],
            "user_id": ["u1", "u2", "u3", "u4"],
            "product_id": ["p1", "p1", "p1", "p2"],
            "review_text": [t, t + "!", t + " really", "Battery drained fast and the case cracked"],
            "rating": [5, 5, 5, 2],
            "review_date": pd.to_datetime(["2024-05-01", "2024-05-02", "2024-05-02", "2024-08-01"]),
        }
    )


def test_clusters_group_near_duplicates_and_score():
    df = _df()
    cl = find_clusters(df["review_text"], threshold=0.6)
    assert len(cl) == 1 and sorted(cl[0]["indices"]) == [0, 1, 2]
    scored = score_clusters(df, cl)[0]
    assert scored["label"] == LABEL and scored["review_count"] == 3
    assert scored["reviewer_count"] == 3 and scored["product_count"] == 1
    assert scored["rating_concentration"] == 1.0
    assert scored["temporal_concentration"] == 1.0
    assert 0 < scored["cluster_risk_score"] <= 100


def test_cluster_edge_cases():
    assert find_clusters(["only one"]) == []
    with pytest.raises(ValueError):
        find_clusters(["x"] * 5001)


def test_cluster_score_without_optional_columns():
    df = _df()[["review_id", "review_text"]]
    scored = score_clusters(df, find_clusters(df["review_text"]))[0]
    assert scored["temporal_concentration"] is None and scored["reviewer_count"] is None


def test_graph_features_and_unavailable():
    df = _df()
    df.loc[4] = ["e", "u1", "p1", "again", 5, pd.Timestamp("2024-05-03")]
    gf = graph_features(df)
    assert gf.loc[0, "reviewer_product_frequency"] == 2
    assert gf.loc[3, "product_degree"] == 1
    nogr = graph_features(df[["review_id", "review_text"]])
    assert nogr.isna().all().all()
    assert graph_figure(build_graph(df)).data and not graph_figure(build_graph(pd.DataFrame())).data


def test_decision_bands_and_scores():
    assert decision_band(0) == "High suspicion" and decision_band(29) == "High suspicion"
    assert (
        decision_band(30) == "Moderate-high suspicion"
        and decision_band(85) == "High-confidence genuine"
    )
    s = score_from_probability(0.84, threshold=0.5)
    assert s["risk_score"] == 84 and s["trust_score"] == 16
    assert s["classification"] == "Predicted suspicious" and s["decision_band"] == "High suspicion"
    with pytest.raises(ValueError):
        score_from_probability(1.2)


def test_evidence_unavailable_is_not_fabricated():
    f = extract_text_features("Best product ever!!! Highly recommend, must buy!")
    sigs = {s.signal_name: s for s in build_evidence(f)}
    for name in (
        "review_similarity",
        "temporal_burst",
        "cluster_membership",
        "image_evidence",
        "unverified_purchase",
        "reviewer_product_repeat",
        "rating_aspect_consistency",
    ):
        assert not sigs[name].availability and sigs[name].signal_value is None
    assert sigs["promotional_language"].direction == "increases_risk"


def test_why_flagged_uses_only_computed_signals():
    f = extract_text_features("Best product ever!!! Highly recommend, must buy!")
    sims = [{"review_id": "x", "similarity_score": 0.91, "matching_phrases": []}]
    sigs = build_evidence(
        f,
        similar=sims,
        verified_purchase=True,
        burst_flag=False,
        aspect_check={"available": True, "inconsistent": False, "explanation": "agree"},
    )
    reasons = why_flagged(sigs)
    assert any("0.91" in r for r in reasons)
    assert not any("burst" in r.lower() for r in reasons)
    bd = evidence_breakdown(sigs)
    assert bd["similarity"] == 0.91 and bd["image"] is None
