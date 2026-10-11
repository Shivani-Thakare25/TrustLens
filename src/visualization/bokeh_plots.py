"""Bokeh interactive risk exploration."""

from __future__ import annotations

import pandas as pd
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure

COLORS = {"Predicted suspicious": "#d62728", "Likely genuine": "#1f77b4"}


def risk_scatter(df: pd.DataFrame):
    """Fake probability vs review length (words); hover shows ids. Requires fake_probability, review_text."""
    d = df.copy()
    d["words"] = d["review_text"].str.split().str.len()
    d["color"] = d["classification"].map(COLORS).fillna("#7f7f7f")
    d["review_id"] = d["review_id"].astype(str)
    src = ColumnDataSource(
        d[["words", "fake_probability", "color", "review_id", "classification", "trust_score"]]
    )
    p = figure(
        title="Risk exploration: fake probability vs review length",
        x_axis_label="Words",
        y_axis_label="Model fake probability",
        height=400,
        width=700,
        tools="pan,wheel_zoom,box_zoom,reset",
    )
    p.scatter("words", "fake_probability", source=src, size=7, color="color", alpha=0.7)
    p.add_tools(
        HoverTool(
            tooltips=[
                ("review", "@review_id"),
                ("class", "@classification"),
                ("trust", "@trust_score"),
                ("p(fake)", "@fake_probability{0.00}"),
            ]
        )
    )
    return p
