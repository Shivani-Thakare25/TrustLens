"""LIME text explanation for the fake-probability class."""

from __future__ import annotations


def explain_lime(
    predict_proba, text: str, num_features: int = 10, num_samples: int = 500, seed: int = 42
) -> dict:
    try:
        from lime.lime_text import LimeTextExplainer
    except ImportError:
        return {"available": False, "reason": "lime is not installed"}
    if not text or not text.strip():
        return {"available": False, "reason": "empty text"}
    exp = LimeTextExplainer(class_names=["genuine", "fake"], random_state=seed)
    res = exp.explain_instance(
        text, predict_proba, labels=(1,), num_features=num_features, num_samples=num_samples
    )
    return {
        "available": True,
        "contributions": [{"token": w, "weight": float(x)} for w, x in res.as_list(label=1)],
    }
