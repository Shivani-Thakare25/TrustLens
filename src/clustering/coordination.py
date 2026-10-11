"""Cluster-level coordination signals. Output wording: 'Potentially coordinated review cluster'.

cluster_risk_score is a documented HEURISTIC (not a trained or validated model): weighted mean of
available components in [0,1], renormalized over components whose data exist, scaled to 0-100.
"""

from __future__ import annotations

import pandas as pd

from src.temporal.burst_detection import temporal_concentration

WEIGHTS = {"similarity": 0.40, "size": 0.20, "temporal": 0.25, "rating": 0.15}
LABEL = "Potentially coordinated review cluster"


def score_clusters(df: pd.DataFrame, clusters: list[dict]) -> list[dict]:
    """Attach statistics, risk score and evidence strings to each cluster."""
    out = []
    for c in clusters:
        sub = df.iloc[c["indices"]]
        n = len(sub)
        comps, evidence = {"similarity": c["average_similarity"], "size": min(1.0, (n - 1) / 9)}, []
        evidence.append(
            f"{n} reviews with average pairwise text similarity {c['average_similarity']:.2f}"
        )

        reviewers = (
            int(sub["user_id"].nunique())
            if "user_id" in sub and sub["user_id"].notna().any()
            else None
        )
        products = (
            int(sub["product_id"].nunique())
            if "product_id" in sub and sub["product_id"].notna().any()
            else None
        )
        conc = temporal_concentration(sub["review_date"]) if "review_date" in sub else None
        if conc is not None:
            comps["temporal"] = conc
            evidence.append(f"{conc:.0%} of cluster reviews fall within a 7-day window")
        rating_conc = None
        if "rating" in sub and sub["rating"].notna().any():
            rating_conc = float(sub["rating"].value_counts(normalize=True).iloc[0])
            comps["rating"] = rating_conc
            evidence.append(f"{rating_conc:.0%} of cluster reviews share the same rating")

        wsum = sum(WEIGHTS[k] for k in comps)
        risk = 100 * sum(WEIGHTS[k] * v for k, v in comps.items()) / wsum
        out.append(
            {
                "cluster_id": c["cluster_id"],
                "label": LABEL,
                "review_ids": sub["review_id"].tolist(),
                "review_count": n,
                "reviewer_count": reviewers,
                "product_count": products,
                "average_similarity": c["average_similarity"],
                "temporal_concentration": conc,
                "rating_concentration": rating_conc,
                "cluster_risk_score": round(risk, 1),
                "evidence": evidence,
            }
        )
    return sorted(out, key=lambda x: -x["cluster_risk_score"])
