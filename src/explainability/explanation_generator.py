"""Assemble the 'WHY DID TRUSTLENS FLAG THIS?' block strictly from computed outputs."""

from __future__ import annotations

from src.integrity.evidence_engine import why_flagged


def build_explanation(
    signals,
    shap_result: dict | None = None,
    lime_result: dict | None = None,
    cf_result: dict | None = None,
    top_n: int = 5,
) -> dict:
    reasons = why_flagged(signals, top_n)
    out = {
        "title": "WHY DID TRUSTLENS FLAG THIS?",
        "reasons": reasons or ["No measured signal indicated elevated risk."],
        "increases_risk_tokens": [],
        "decreases_risk_tokens": [],
        "counterfactual": cf_result,
    }
    if shap_result and shap_result.get("available"):
        c = shap_result["contributions"]
        out["increases_risk_tokens"] = [x["token"] for x in c if x["shap_value"] > 0][:top_n]
        out["decreases_risk_tokens"] = [x["token"] for x in c if x["shap_value"] < 0][:top_n]
    out["shap"], out["lime"] = shap_result, lime_result
    return out
