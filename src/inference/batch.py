"""Batch analysis of a review DataFrame: validate -> clean -> predict -> add dataset-level context."""

from __future__ import annotations

import json

import pandas as pd

from src.clustering.coordination import score_clusters
from src.clustering.review_clusters import MAX_REVIEWS, find_clusters
from src.data.cleaning import clean_dataframe
from src.data.validation import validate_dataframe
from src.graph.graph_features import graph_features
from src.temporal.burst_detection import detect_bursts

_BOOL = {
    "true": True,
    "yes": True,
    "y": True,
    "1": True,
    "false": False,
    "no": False,
    "n": False,
    "0": False,
}


def _bool(v):
    if isinstance(v, bool):
        return v
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return None
    return _BOOL.get(str(v).strip().lower())


def _to_reviews(df: pd.DataFrame) -> list[dict]:
    out = []
    for rec in df.to_dict("records"):
        r = {
            k: (None if (not isinstance(v, (list, dict)) and pd.isna(v)) else v)
            for k, v in rec.items()
        }
        r["verified_purchase"] = _bool(r.get("verified_purchase"))
        out.append(r)
    return out


def analyze_dataframe(predictor, raw_df: pd.DataFrame, max_rows: int = MAX_REVIEWS) -> dict:
    """Returns {results, clusters, burst_days, stats}. Raises ValueError on unusable input."""
    report = validate_dataframe(raw_df)
    if not report.ok:
        raise ValueError(f"Missing required columns: {report.missing_required_columns}")
    if len(raw_df) > max_rows:
        raise ValueError(f"At most {max_rows} rows can be analyzed per batch.")
    df, clean_stats = clean_dataframe(raw_df)
    if df.empty:
        raise ValueError("No usable rows after cleaning.")
    analyses = predictor.analyze_batch(_to_reviews(df))

    res = df[["review_id", "user_id", "product_id", "rating", "review_date", "review_text"]].copy()
    res["fake_probability"] = [a["fake_probability"] for a in analyses]
    res["trust_score"] = [a["trust_score"] for a in analyses]
    res["risk_score"] = [a["risk_score"] for a in analyses]
    res["classification"] = [a["classification"] for a in analyses]
    res["decision_band"] = [a["decision_band"] for a in analyses]
    res["overall_sentiment"] = [a["overall_sentiment"] for a in analyses]
    res["aspects"] = [json.dumps(a["aspects"]) for a in analyses]
    res["top_reasons"] = ["; ".join(a["explanation"]["reasons"]) for a in analyses]

    clusters = score_clusters(df, find_clusters(df["review_text"]))
    by_review = {rid: c["cluster_id"] for c in clusters for rid in c["review_ids"]}
    risk_by_cluster = {c["cluster_id"]: c["cluster_risk_score"] for c in clusters}
    res["cluster_id"] = res["review_id"].map(by_review)
    res["cluster_risk_score"] = res["cluster_id"].map(risk_by_cluster)
    res["reviewer_product_frequency"] = graph_features(df)["reviewer_product_frequency"].to_numpy()

    burst_days: dict = {}
    if df["product_id"].notna().any() and df["review_date"].notna().any():
        for pid, g in df.dropna(subset=["product_id"]).groupby("product_id"):
            b = detect_bursts(g["review_date"])
            if b:
                burst_days[str(pid)] = [x["date"] for x in b]
    day = pd.to_datetime(res["review_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    res["burst_day"] = [
        (str(p) in burst_days and d in burst_days[str(p)]) for p, d in zip(res["product_id"], day)
    ]
    return {
        "results": res,
        "clusters": clusters,
        "burst_days": burst_days,
        "stats": {"validation": report.to_dict(), "cleaning": clean_stats, "analyzed": len(res)},
    }
