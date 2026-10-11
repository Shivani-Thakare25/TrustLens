"""Token-level SHAP on the real model's fake-probability output (Partition explainer, text masker)."""

from __future__ import annotations

import re


def explain_shap(predict_fake, text: str, top_k: int = 10, max_evals: int | None = None) -> dict:
    """Returns {available, base_value, contributions:[{token, shap_value}]} or {available: False, reason}."""
    try:
        import shap
    except ImportError:
        return {"available": False, "reason": "shap is not installed"}
    if not text or not text.strip():
        return {"available": False, "reason": "empty text"}
    n_tokens = len(re.split(r"\W+", text))
    explainer = shap.Explainer(lambda xs: predict_fake(list(xs)), shap.maskers.Text(r"\W+"))
    sv = explainer([text], max_evals=max_evals or max(300, 6 * n_tokens))
    pairs = [(re.sub(r"^\W+|\W+$", "", t), float(v)) for t, v in zip(sv.data[0], sv.values[0])]
    pairs = [(t, v) for t, v in pairs if t]
    pairs.sort(key=lambda x: -abs(x[1]))
    return {
        "available": True,
        "base_value": float(sv.base_values[0]),
        "contributions": [{"token": t, "shap_value": v} for t, v in pairs[:top_k]],
    }
