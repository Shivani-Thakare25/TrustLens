"""Plotly charts. All inputs are computed outputs; empty inputs yield a titled empty figure."""

from __future__ import annotations

import plotly.graph_objects as go

SENT_COLORS = {"Positive": "#2ca02c", "Negative": "#d62728", "Neutral": "#7f7f7f"}


def _empty(title: str) -> go.Figure:
    return go.Figure(layout={"title": title})


def evidence_breakdown_chart(breakdown: dict) -> go.Figure:
    items = {k: v for k, v in breakdown.items() if v is not None}
    if not items:
        return _empty("No evidence categories available")
    fig = go.Figure(
        go.Bar(
            x=list(items.values()),
            y=list(items.keys()),
            orientation="h",
            marker_color=["#d62728" if v >= 0.4 else "#2ca02c" for v in items.values()],
        )
    )
    fig.update_layout(
        title="Evidence breakdown (0 = no risk evidence, 1 = strong)", xaxis={"range": [0, 1]}
    )
    return fig


def aspect_chart(aspects: list[dict]) -> go.Figure:
    if not aspects:
        return _empty("No aspects detected")
    return go.Figure(
        go.Bar(
            x=[a["aspect"] for a in aspects],
            y=[a["confidence"] for a in aspects],
            marker_color=[SENT_COLORS[a["sentiment"]] for a in aspects],
            text=[a["sentiment"] for a in aspects],
        ),
        layout={"title": "Aspect sentiment (bar = confidence)"},
    )


def token_chart(contribs: list[dict], value_key: str, title: str) -> go.Figure:
    if not contribs:
        return _empty(title + " (no data)")
    c = sorted(contribs, key=lambda x: x[value_key])
    return go.Figure(
        go.Bar(
            x=[x[value_key] for x in c],
            y=[x["token"] for x in c],
            orientation="h",
            marker_color=["#d62728" if x[value_key] > 0 else "#2ca02c" for x in c],
        ),
        layout={"title": title + " (red raises fake probability)"},
    )


def rating_distribution_chart(dist: dict | None) -> go.Figure:
    if not dist:
        return _empty("No ratings available")
    return go.Figure(
        go.Bar(x=[str(k) for k in dist], y=list(dist.values())),
        layout={"title": "Rating distribution", "xaxis": {"title": "Rating"}},
    )


def cluster_chart(clusters: list[dict]) -> go.Figure:
    if not clusters:
        return _empty("No potentially coordinated review clusters found")
    return go.Figure(
        go.Scatter(
            x=[c["review_count"] for c in clusters],
            y=[c["average_similarity"] for c in clusters],
            mode="markers+text",
            text=[c["cluster_id"] for c in clusters],
            marker={
                "size": [10 + c["cluster_risk_score"] / 4 for c in clusters],
                "color": [c["cluster_risk_score"] for c in clusters],
                "colorscale": "Reds",
                "showscale": True,
                "colorbar": {"title": "Risk"},
            },
        ),
        layout={
            "title": "Potentially coordinated review clusters",
            "xaxis": {"title": "Reviews"},
            "yaxis": {"title": "Average similarity"},
        },
    )


def distribution_chart(values, title: str, xlabel: str) -> go.Figure:
    return go.Figure(
        go.Histogram(x=list(values)), layout={"title": title, "xaxis": {"title": xlabel}}
    )


def model_comparison_chart(results: dict) -> go.Figure:
    trained = {k: v for k, v in results.items() if v.get("status") == "trained"}
    if not trained:
        return _empty("No trained models")
    fig = go.Figure()
    for metric in ("f1", "pr_auc", "roc_auc"):
        fig.add_bar(
            name=metric, x=list(trained), y=[v["test"].get(metric) for v in trained.values()]
        )
    fig.update_layout(barmode="group", title="Test-set metrics by model")
    return fig


def sentiment_distribution_chart(counts: dict | None) -> go.Figure:
    if not counts:
        return _empty("No sentiment distribution available")
    return go.Figure(
        go.Bar(
            x=list(counts.keys()),
            y=list(counts.values()),
            marker_color=[SENT_COLORS.get(k, "#1f77b4") for k in counts.keys()],
        ),
        layout={
            "title": "Overall sentiment distribution",
            "xaxis": {"title": "Sentiment"},
            "yaxis": {"title": "Count"},
        },
    )
