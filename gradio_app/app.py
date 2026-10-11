"""Gradio demo. Run: python -m gradio_app.app (port from GRADIO_PORT, default 7860)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.services.trustlens_service import (  # noqa: E402
    ModelNotLoadedError,
    get_service,
)


def analyze_review(text: str, rating, verified: str):
    """Returns (classification, fake_probability, trust_score, sentiment, aspect_rows, evidence_md, explanation_md)."""
    payload = {"review_text": text or ""}
    if rating:
        payload["rating"] = float(rating)
    if verified in ("Yes", "No"):
        payload["verified_purchase"] = verified == "Yes"
    try:
        r = get_service().predict(payload)
    except ModelNotLoadedError as e:
        return (f"Error: {e}", None, None, "", [], "", "")
    except ValueError as e:
        return (f"Error: {e}", None, None, "", [], "", "")
    rows = [
        [a["aspect"], a["sentiment"], a["confidence"], a["supporting_phrase"]] for a in r["aspects"]
    ]
    ev = "\n".join(
        f"- **{s['signal_name']}**: "
        + (
            f"{s['normalized_score']:.2f} ({s['direction']}) — {s['explanation']}"
            if s["availability"]
            else "unavailable"
        )
        for s in r["evidence"]["signals"]
    )
    why = "\n".join(f"{i}. {x}" for i, x in enumerate(r["explanation"]["reasons"], 1))
    return (
        r["classification"],
        round(r["fake_probability"], 4),
        r["trust_score"],
        r["overall_sentiment"],
        rows,
        ev,
        "**WHY DID TRUSTLENS FLAG THIS?**\n\n" + why,
    )


def build_demo():
    import gradio as gr

    with gr.Blocks(title="TrustLens") as demo:
        gr.Markdown(
            "# TrustLens\nExplainable review integrity analysis. Outputs are model predictions, not proof."
        )
        text = gr.Textbox(label="Review text", lines=5)
        with gr.Row():
            rating = gr.Dropdown([0, 1, 2, 3, 4, 5], value=0, label="Rating (0 = not provided)")
            verified = gr.Dropdown(
                ["Not provided", "Yes", "No"], value="Not provided", label="Verified purchase"
            )
        btn = gr.Button("Analyze Review")
        cls, prob, trust, sent = (
            gr.Textbox(label="Classification"),
            gr.Number(label="Fake probability"),
            gr.Number(label="Trust score"),
            gr.Textbox(label="Overall sentiment"),
        )
        aspects = gr.Dataframe(
            headers=["aspect", "sentiment", "confidence", "supporting phrase"], label="Aspects"
        )
        evidence, expl = gr.Markdown(), gr.Markdown()
        btn.click(
            analyze_review,
            [text, rating, verified],
            [cls, prob, trust, sent, aspects, evidence, expl],
        )
    return demo


if __name__ == "__main__":
    build_demo().launch(server_name="0.0.0.0", server_port=int(os.getenv("GRADIO_PORT", "7860")))
