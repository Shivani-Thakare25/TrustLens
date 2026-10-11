"""Model counterfactual analysis: apply text edits, re-score with the REAL model, report the change.

Only changes to the model's actual input (the review text) are evaluated. Similarity, behavioral
and temporal signals are not model inputs in this version, so no counterfactual is claimed for them.
"""

from __future__ import annotations

import re

from src.features.text_features import GENERIC_PHRASES, PROMO

TITLE = "Model Counterfactual Analysis"
DISCLAIMER = (
    "This shows how the model prediction changes under hypothetical input changes. "
    "It does not prove that the review would become authentic."
)

_PROMO_RE = re.compile(
    r"\b(?:"
    + "|".join(map(re.escape, sorted(PROMO | set(GENERIC_PHRASES), key=len, reverse=True)))
    + r")\b",
    re.I,
)


def _edits(text: str) -> dict:
    return {
        "Replace exclamation marks with periods": re.sub(r"!+", ".", text),
        "Remove promotional words and stock phrases": re.sub(
            r"\s+", " ", _PROMO_RE.sub("", text)
        ).strip(),
        "Lowercase all text": text.lower(),
        "Keep only the first sentence": re.split(r"(?<=[.!?])\s+", text.strip())[0],
    }


def counterfactuals(predict_fake, text: str) -> dict:
    """Return original probability and per-edit new probability/delta (edits that change nothing are skipped)."""
    base = float(predict_fake([text])[0])
    rows = []
    for name, new in _edits(text).items():
        if new and new != text:
            p = float(predict_fake([new])[0])
            rows.append(
                {
                    "change": name,
                    "original_probability": base,
                    "new_probability": p,
                    "delta": p - base,
                }
            )
    rows.sort(key=lambda r: r["delta"])
    return {"title": TITLE, "disclaimer": DISCLAIMER, "original_probability": base, "changes": rows}
