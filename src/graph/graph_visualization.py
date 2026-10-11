"""Plotly view of the reviewer-product graph."""

from __future__ import annotations

import networkx as nx
import plotly.graph_objects as go


def graph_figure(g: nx.Graph) -> go.Figure:
    """Spring-layout figure. Empty graph returns a figure with an explanatory title."""
    if g.number_of_nodes() == 0:
        return go.Figure(layout={"title": "No reviewer/product ids available to build a graph"})
    pos = nx.spring_layout(g, seed=42)
    ex, ey = [], []
    for a, b in g.edges():
        ex += [pos[a][0], pos[b][0], None]
        ey += [pos[a][1], pos[b][1], None]
    colors = ["#1f77b4" if g.nodes[n]["kind"] == "user" else "#d62728" for n in g.nodes]
    return go.Figure(
        [
            go.Scatter(
                x=ex, y=ey, mode="lines", line={"width": 1, "color": "#999"}, hoverinfo="none"
            ),
            go.Scatter(
                x=[pos[n][0] for n in g.nodes],
                y=[pos[n][1] for n in g.nodes],
                mode="markers",
                marker={"size": 9, "color": colors},
                text=[f"{n[0]}: {n[1]} (degree {g.degree(n)})" for n in g.nodes],
                hoverinfo="text",
            ),
        ],
        layout={"title": "Reviewer (blue) – Product (red) graph", "showlegend": False},
    )
