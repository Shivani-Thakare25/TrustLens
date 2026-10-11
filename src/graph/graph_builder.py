"""Reviewer -> Product graph. Edge weight = number of reviews by that reviewer of that product."""

from __future__ import annotations

import networkx as nx
import pandas as pd


def build_graph(df: pd.DataFrame) -> nx.Graph:
    """Requires non-null user_id and product_id; rows lacking either are skipped (not fabricated)."""
    g = nx.Graph()
    if "user_id" not in df or "product_id" not in df:
        return g
    for u, p in df[["user_id", "product_id"]].dropna().itertuples(index=False):
        a, b = ("user", u), ("product", p)
        g.add_node(a, kind="user")
        g.add_node(b, kind="product")
        w = g[a][b]["weight"] + 1 if g.has_edge(a, b) else 1
        g.add_edge(a, b, weight=w)
    return g
