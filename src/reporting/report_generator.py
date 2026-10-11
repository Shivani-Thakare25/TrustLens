"""Downloadable analysis reports (JSON / CSV / HTML). Every value comes from a computed analysis."""

from __future__ import annotations

import csv
import html
import io
import json
from datetime import datetime, timezone

LIMITATIONS = [
    "Predictions are probabilistic model outputs. 'Predicted suspicious' does not mean proven fake.",
    "Signals such as similarity or reviewer behavior are indicators, not proof of misconduct.",
    "Unavailable signals are shown as unavailable and are not estimated.",
    "SHAP/LIME explain this model's behavior, not real-world causation.",
    "Counterfactuals show model sensitivity to hypothetical changes, not authenticity.",
]


def build_report(
    analysis: dict,
    model_metadata: dict | None = None,
    review: dict | None = None,
    explanation_extra: dict | None = None,
    include_text: bool = True,
) -> dict:
    review = dict(review or {})
    if not include_text:
        review.pop("review_text", None)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "review": review,
        "prediction": {
            k: analysis.get(k)
            for k in (
                "review_id",
                "classification",
                "fake_probability",
                "genuine_probability",
                "risk_score",
                "trust_score",
                "confidence",
                "decision_band",
            )
        },
        "overall_sentiment": analysis.get("overall_sentiment"),
        "aspects": analysis.get("aspects", []),
        "evidence": analysis.get("evidence", {}),
        "similar_reviews": analysis.get("similar_reviews", []),
        "explanation": explanation_extra or analysis.get("explanation", {}),
        "counterfactuals": analysis.get("counterfactuals", {}),
        "model": model_metadata or {"model_version": analysis.get("model_version")},
        "limitations": LIMITATIONS,
    }


def to_json(report: dict) -> str:
    return json.dumps(report, indent=2, default=str)


def to_csv(report: dict) -> str:
    """One row per evidence signal with the headline prediction repeated for context."""
    buf = io.StringIO()
    w = csv.writer(buf)
    p = report["prediction"]
    w.writerow(
        [
            "review_id",
            "classification",
            "fake_probability",
            "trust_score",
            "signal_name",
            "category",
            "normalized_score",
            "direction",
            "availability",
            "explanation",
        ]
    )
    for s in report.get("evidence", {}).get("signals", []):
        w.writerow(
            [
                p["review_id"],
                p["classification"],
                p["fake_probability"],
                p["trust_score"],
                s["signal_name"],
                s["category"],
                s["normalized_score"],
                s["direction"],
                s["availability"],
                s["explanation"],
            ]
        )
    return buf.getvalue()


def _table(rows, headers):
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>" for r in rows
    )
    return f"<table border='1' cellpadding='4' style='border-collapse:collapse'><tr>{head}</tr>{body}</table>"


def to_html(report: dict) -> str:
    p, e = report["prediction"], report.get("evidence", {})
    parts = [
        "<html><head><meta charset='utf-8'><title>TrustLens Report</title></head>"
        "<body style='font-family:sans-serif;max-width:900px;margin:auto'>",
        "<h1>TrustLens Analysis Report</h1>",
        f"<p>Generated: {html.escape(report['generated_at'])}</p>",
        f"<h2>Prediction</h2><p><b>{html.escape(str(p['classification']))}</b> &mdash; Trust score "
        f"{p['trust_score']}/100 ({html.escape(str(p['decision_band']))}); model fake probability "
        f"{p['fake_probability']:.1%}; confidence {p['confidence']:.1%}.</p>",
    ]
    if report["review"].get("review_text"):
        parts.append(
            f"<h2>Review</h2><blockquote>{html.escape(report['review']['review_text'])}</blockquote>"
        )
    parts.append(f"<p>Overall sentiment: {html.escape(str(report['overall_sentiment']))}</p>")
    if report["aspects"]:
        parts.append(
            "<h2>Aspect sentiment</h2>"
            + _table(
                [
                    (a["aspect"], a["sentiment"], a["confidence"], a["supporting_phrase"])
                    for a in report["aspects"]
                ],
                ["Aspect", "Sentiment", "Confidence", "Supporting phrase"],
            )
        )
    parts.append(
        "<h2>Evidence signals</h2>"
        + _table(
            [
                (
                    s["signal_name"],
                    s["direction"],
                    s["normalized_score"] if s["availability"] else "unavailable",
                    s["explanation"],
                )
                for s in e.get("signals", [])
            ],
            ["Signal", "Direction", "Score", "Explanation"],
        )
    )
    reasons = report["explanation"].get("reasons", [])
    if reasons:
        parts.append(
            "<h2>Why did TrustLens flag this?</h2><ol>"
            + "".join(f"<li>{html.escape(r)}</li>" for r in reasons)
            + "</ol>"
        )
    cf = report.get("counterfactuals") or {}
    if cf.get("changes"):
        parts.append(
            f"<h2>{html.escape(cf['title'])}</h2><p><i>{html.escape(cf['disclaimer'])}</i></p>"
            + _table(
                [
                    (c["change"], f"{c['original_probability']:.1%}", f"{c['new_probability']:.1%}")
                    for c in cf["changes"]
                ],
                ["Hypothetical change", "Original", "New"],
            )
        )
    if report["similar_reviews"]:
        parts.append(
            "<h2>Similar reviews</h2>"
            + _table(
                [
                    (s["review_id"], s["similarity_score"], ", ".join(s["matching_phrases"]))
                    for s in report["similar_reviews"]
                ],
                ["Review id", "Similarity", "Matching phrases"],
            )
        )
    m = report["model"]
    parts.append("<h2>Model</h2>" + _table([(k, v) for k, v in m.items()], ["Field", "Value"]))
    parts.append(
        "<h2>Limitations</h2><ul>"
        + "".join(f"<li>{html.escape(x)}</li>" for x in report["limitations"])
        + "</ul>"
    )
    parts.append("</body></html>")
    return "".join(parts)
