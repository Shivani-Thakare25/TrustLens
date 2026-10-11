"""Per-review graph features; NaN when user_id/product_id are unavailable."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.graph.graph_builder import build_graph

COLS = ["reviewer_degree", "product_degree", "reviewer_product_frequency"]


def graph_features(df: pd.DataFrame, cluster_by_review: dict | None = None) -> pd.DataFrame:
    g = build_graph(df)
    rows = []
    for u, p in zip(df.get("user_id", [None] * len(df)), df.get("product_id", [None] * len(df))):
        a, b = ("user", u), ("product", p)
        if pd.isna(u) or pd.isna(p) or a not in g or b not in g:
            rows.append([np.nan] * 3)
        else:
            rows.append([g.degree(a), g.degree(b), g[a][b]["weight"]])
    out = pd.DataFrame(rows, columns=COLS, index=df.index)
    if cluster_by_review is not None:
        out["cluster_id"] = df["review_id"].map(cluster_by_review)
    return out
