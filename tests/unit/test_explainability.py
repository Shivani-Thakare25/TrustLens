import numpy as np

from src.explainability.counterfactual import DISCLAIMER, counterfactuals
from src.explainability.explanation_generator import build_explanation
from src.explainability.feature_importance import global_importance
from src.explainability.lime_explainer import explain_lime
from src.explainability.model_adapter import fake_fn, proba_fn
from src.explainability.shap_explainer import explain_shap
from src.features.text_features import extract_text_features
from src.integrity.evidence_engine import build_evidence

PROMO = "Best product ever!!! Highly recommend, must buy now!"


def test_adapters_shapes(trained_model):
    p = proba_fn(trained_model)([PROMO, "battery died"])
    assert p.shape == (2, 2) and np.allclose(p.sum(axis=1), 1)
    assert fake_fn(trained_model)([PROMO]).shape == (1,)


def test_counterfactual_uses_real_model(trained_model):
    f = fake_fn(trained_model)
    cf = counterfactuals(f, PROMO)
    assert cf["disclaimer"] == DISCLAIMER and cf["changes"]
    for c in cf["changes"]:
        assert abs(c["delta"] - (c["new_probability"] - c["original_probability"])) < 1e-9
    assert cf["changes"][0]["delta"] <= cf["changes"][-1]["delta"]


def test_counterfactual_skips_noop_edits(trained_model):
    cf = counterfactuals(fake_fn(trained_model), "plain words here")
    assert [c["change"] for c in cf["changes"]] == []


def test_shap_and_lime_return_tokens_from_the_text(trained_model):
    s = explain_shap(fake_fn(trained_model), PROMO, top_k=5)
    assert s["available"] and 0 < len(s["contributions"]) <= 5
    words = {w.lower() for w in PROMO.replace("!", " ").replace(",", " ").split()}
    assert all(c["token"].lower() in words for c in s["contributions"])
    lm = explain_lime(proba_fn(trained_model), PROMO, num_samples=200)
    assert lm["available"] and all(c["token"].lower() in words for c in lm["contributions"])
    assert explain_shap(fake_fn(trained_model), "  ")["available"] is False
    assert explain_lime(proba_fn(trained_model), "")["available"] is False


def test_global_importance_named_and_sorted(trained_model):
    imp = global_importance(trained_model, top_k=10)
    assert len(imp) == 10
    assert [i["importance"] for i in imp] == sorted((i["importance"] for i in imp), reverse=True)
    assert all(isinstance(i["feature"], str) for i in imp)


def test_explanation_from_signals(trained_model):
    f = fake_fn(trained_model)
    sigs = build_evidence(extract_text_features(PROMO))
    ex = build_explanation(sigs, explain_shap(f, PROMO), None, counterfactuals(f, PROMO))
    assert ex["title"] == "WHY DID TRUSTLENS FLAG THIS?"
    assert any("promotional" in r.lower() for r in ex["reasons"])
    assert build_explanation(build_evidence(extract_text_features("ok item")))["reasons"] == [
        "No measured signal indicated elevated risk."
    ]
