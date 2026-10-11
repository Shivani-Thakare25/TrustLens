"""Regression tests for TrustLens Final Scope Cleanup & Bug Fixes.

Ensures:
- Model artifacts load safely without retraining.
- Positive, negative, and neutral sentiments are handled accurately.
- Neutral reviews are not mislabeled as positive or negative.
- Fake-review probability, trust scores, and decision bands are strictly consistent.
- Empty review inputs and missing metadata are handled safely without crashing.
- Aspect-level sentiment extraction is backed by the trained sentiment model.
- Report exports (JSON, CSV, HTML) work properly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from api.services.trustlens_service import get_service, reset_service
from src.absa.absa import extract_aspect_sentiment, overall_sentiment
from src.inference.predictor import TrustLensPredictor
from src.inference.sentiment_predictor import SentimentPredictor
from src.integrity.risk_scoring import score_from_probability
from src.product_intelligence.trust_adjusted_sentiment import trust_adjusted_rating
from src.reporting.report_generator import build_report, to_csv, to_html, to_json


@pytest.fixture(scope="module")
def predictor():
    reset_service()
    p = TrustLensPredictor("artifacts/model")
    assert p.loaded, "Trained fake-review model must load from artifacts/model"
    return p


@pytest.fixture(scope="module")
def sentiment_predictor():
    sp = SentimentPredictor()
    assert sp.loaded, "Trained sentiment model must load from models/"
    return sp


def test_models_loaded_without_retraining(predictor, sentiment_predictor):
    assert predictor.loaded
    assert predictor.model is not None
    assert sentiment_predictor.loaded
    assert set(sentiment_predictor.classes) == {"negative", "neutral", "positive"}


def test_positive_negative_neutral_sentiment_accuracy(sentiment_predictor):
    # Positive case
    pos_res = sentiment_predictor.analyze_one("This product is absolutely fantastic and works wonderfully!")
    assert pos_res["sentiment"] == "Positive"
    assert pos_res["confidence"] > 0.60

    # Negative case
    neg_res = sentiment_predictor.analyze_one("Terrible experience, completely defective and broke on the first day.")
    assert neg_res["sentiment"] == "Negative"
    assert neg_res["confidence"] > 0.60

    # Neutral case
    neu_res = sentiment_predictor.analyze_one("The product is okay, average quality nothing special.")
    assert neu_res["sentiment"] == "Neutral", f"Neutral text misclassified as: {neu_res['sentiment']}"
    assert neu_res["confidence"] > 0.50


def test_neutral_review_not_forced_to_positive_or_negative(sentiment_predictor):
    neutral_samples = [
        "The product is okay, average quality nothing special.",
        "Average battery life, works as expected.",
        "It is an ordinary item, neither good nor bad.",
    ]
    for sample in neutral_samples:
        pred = sentiment_predictor.analyze_one(sample)
        assert pred["sentiment"] == "Neutral", f"'{sample}' was labeled '{pred['sentiment']}' instead of Neutral"


def test_decision_threshold_and_classification_consistency():
    # Low fake probability -> Likely genuine band -> Classification must be 'Likely genuine'
    low_prob = 0.15
    res_low = score_from_probability(low_prob, threshold=0.50)
    assert res_low["trust_score"] == 85
    assert res_low["classification"] == "Likely genuine"
    assert "genuine" in res_low["decision_band"].lower()

    # High fake probability -> High suspicion band -> Classification must be 'Predicted suspicious'
    high_prob = 0.85
    res_high = score_from_probability(high_prob, threshold=0.50)
    assert res_high["trust_score"] == 15
    assert res_high["classification"] == "Predicted suspicious"
    assert "suspicion" in res_high["decision_band"].lower()

    # Genuine probability and trust score math
    assert res_low["genuine_probability"] == pytest.approx(0.85)
    assert res_low["risk_score"] + res_low["trust_score"] == 100
    assert res_high["genuine_probability"] == pytest.approx(0.15)
    assert res_high["risk_score"] + res_high["trust_score"] == 100


def test_empty_review_input_handled_safely(predictor):
    res_empty = predictor.analyze({"review_text": ""})
    assert res_empty["classification"] == "Likely genuine"
    assert res_empty["trust_score"] == 100
    assert res_empty["overall_sentiment"] == "Neutral"
    assert res_empty["aspects"] == []

    res_spaces = predictor.analyze({"review_text": "   \n\t  "})
    assert res_spaces["classification"] == "Likely genuine"
    assert res_spaces["trust_score"] == 100


def test_missing_optional_metadata_handled_safely(predictor):
    # Review with no rating and no verified_purchase
    res = predictor.analyze({
        "review_text": "Good phone with fast processor and decent battery.",
        "rating": None,
        "verified_purchase": None,
    })
    assert res["trust_score"] >= 0
    assert res["overall_sentiment"] in ("Positive", "Neutral", "Negative")
    assert isinstance(res["aspects"], list)


def test_aspect_level_sentiment_extraction(sentiment_predictor):
    text = "The camera is fantastic and battery lasts all day, but delivery was disappointing."
    aspects = extract_aspect_sentiment(text, sentiment_predictor=sentiment_predictor)
    aspect_map = {a["aspect"]: a["sentiment"] for a in aspects}

    assert "camera" in aspect_map
    assert aspect_map["camera"] == "Positive"
    assert "delivery" in aspect_map
    assert aspect_map["delivery"] == "Negative"
    assert all(0.0 <= a["confidence"] <= 1.0 for a in aspects)


def test_counterfactual_edits_work(predictor):
    text = "Best product ever! Highly recommend to everyone, must buy now, absolutely love it!"
    res = predictor.analyze({"review_text": text}, with_counterfactual=True)
    cf = res.get("counterfactuals")
    assert cf and "changes" in cf
    assert len(cf["changes"]) > 0
    for change in cf["changes"]:
        assert "delta" in change
        assert "new_probability" in change


def test_report_downloads_generation(predictor):
    res = predictor.analyze({"review_text": "Decent laptop, battery is average but screen is bright."})
    meta = {"model_name": "logistic_regression", "model_version": "test_v1"}
    report = build_report(res, meta, {"review_text": "Decent laptop..."})

    # JSON export
    json_str = to_json(report)
    parsed = json.loads(json_str)
    assert "prediction" in parsed
    assert "overall_sentiment" in parsed

    # CSV export
    csv_str = to_csv(report)
    assert "review_id" in csv_str
    assert "fake_probability" in csv_str

    # HTML export
    html_str = to_html(report)
    assert "TrustLens Analysis Report" in html_str
    assert "Prediction" in html_str


def test_trust_adjusted_rating_calculation():
    import pandas as pd
    ratings = pd.Series([5, 5, 5, 1, 2])
    fake_probs = pd.Series([0.9, 0.9, 0.9, 0.1, 0.1])
    # The three 5-star reviews have high fake probability (low weight)
    # The 1-star and 2-star reviews have low fake probability (high weight)
    # Therefore trust-adjusted rating should be noticeably lower than raw average
    adj = trust_adjusted_rating(ratings, fake_probs, min_reviews=5)
    assert adj is not None
    assert adj < ratings.mean()
