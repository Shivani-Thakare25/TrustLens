"""Shared inference engine: loads the model once; never trains. Used by FastAPI, Flask, Streamlit, Gradio."""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from src.absa.absa import (
    extract_aspect_sentiment,
    overall_sentiment,
    rating_aspect_inconsistency,
)
from src.explainability.counterfactual import counterfactuals
from src.explainability.explanation_generator import build_explanation
from src.explainability.lime_explainer import explain_lime
from src.explainability.model_adapter import fake_fn, proba_fn
from src.explainability.shap_explainer import explain_shap
from src.features.text_features import FEATURE_NAMES, extract_text_features
from src.inference.sentiment_predictor import SentimentPredictor
from src.integrity.evidence_engine import build_evidence, evidence_breakdown
from src.integrity.risk_scoring import score_from_probability
from src.monitoring.data_quality import missing_fields
from src.monitoring.prediction_monitor import PredictionMonitor
from src.similarity.similarity_engine import SimilarityEngine
from src.utils.logger import get_logger

log = get_logger(__name__)


class ModelNotLoadedError(RuntimeError):
    """Raised when inference is requested but model artifacts are missing or unreadable."""


class TrustLensPredictor:
    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self.model, self.metadata, self.baseline, self.similarity = None, {}, {}, None
        self.sentiment_predictor = None
        self.monitor = PredictionMonitor()
        self._load()

    def _load(self):
        # 1. Load sentiment predictor
        root_cand = self.model_dir.parent.parent if self.model_dir.name == "model" else self.model_dir.parent
        self.sentiment_predictor = SentimentPredictor(root_cand)

        # 2. Load fake-review detection artifacts
        mp, meta = self.model_dir / "model.joblib", self.model_dir / "metadata.json"
        if not (mp.exists() and meta.exists()):
            log.warning("model artifacts not found in %s", self.model_dir)
            return
        try:
            self.model = joblib.load(mp)
            self.metadata = json.loads(meta.read_text())
        except Exception:
            log.exception("failed to load model artifacts")
            self.model, self.metadata = None, {}
            return
        bp = self.model_dir / "baseline_features.json"
        if bp.exists():
            self.baseline = json.loads(bp.read_text())
        cp = self.model_dir / "reference_corpus.csv"
        if cp.exists():
            ref = pd.read_csv(cp).dropna()
            if len(ref):
                self.similarity = SimilarityEngine().fit(
                    ref["review_id"].astype(str), ref["review_text"]
                )

    @property
    def loaded(self) -> bool:
        return self.model is not None

    @property
    def version(self):
        return self.metadata.get("model_version")

    def _require(self):
        if not self.loaded:
            raise ModelNotLoadedError("Model artifacts are not loaded; train a model first.")

    def analyze(
        self,
        review: dict,
        explain: bool = False,
        with_counterfactual: bool = True,
        fake_probability: float | None = None,
    ) -> dict:
        """Full analysis of one review dict (keys: review_text, rating, verified_purchase, ...)."""
        self._require()
        t0 = time.perf_counter()
        raw_text = review.get("review_text", "")
        text = str(raw_text).strip() if raw_text is not None else ""
        rid = review.get("review_id") or str(uuid.uuid4())

        # Defensive handling for empty review text input
        if not text:
            score = score_from_probability(0.0, 0.5)
            return {
                "review_id": rid,
                **score,
                "overall_sentiment": "Neutral",
                "aspects": [],
                "similar_reviews": [],
                "evidence": {
                    "signals": [],
                    "breakdown": {},
                },
                "explanation": {"reasons": ["Empty review text provided."], "shap": None, "lime": None, "tree": None},
                "counterfactuals": {},
                "model_version": self.version,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        fake = fake_fn(self.model)
        p = float(fake([text])[0]) if fake_probability is None else fake_probability
        feats = extract_text_features(text)
        aspects = extract_aspect_sentiment(text, sentiment_predictor=self.sentiment_predictor)
        check = rating_aspect_inconsistency(review.get("rating"), aspects)
        similar = self.similarity.query(text, top_k=5, exclude_id=rid) if self.similarity else None
        signals = build_evidence(
            feats,
            similar=similar,
            aspect_check=check,
            verified_purchase=review.get("verified_purchase"),
            image_signals=review.get("image_signals"),
        )

        # Decision threshold consistency: if stored metadata threshold is an extreme edge
        # artifact (< 0.2 or > 0.8) from unweighted grid search, use standard 0.50 threshold.
        thresh = self.metadata.get("threshold", 0.5)
        if thresh is None or thresh < 0.20 or thresh > 0.80:
            thresh = 0.50
        score = score_from_probability(p, thresh)

        cf = counterfactuals(fake, text) if with_counterfactual else {}
        shap_r = explain_shap(fake, text) if explain else None
        lime_r = explain_lime(proba_fn(self.model), text) if explain else None
        sent = overall_sentiment(text, sentiment_predictor=self.sentiment_predictor)
        latency = (time.perf_counter() - t0) * 1000
        self.monitor.record(
            p,
            score["classification"],
            sent,
            latency,
            score["confidence"],
            feats,
            missing_fields(review),
        )
        return {
            "review_id": rid,
            **score,
            "overall_sentiment": sent,
            "aspects": aspects,
            "similar_reviews": similar or [],
            "evidence": {
                "signals": [s.to_dict() for s in signals],
                "breakdown": evidence_breakdown(signals),
            },
            "explanation": build_explanation(signals, shap_r, lime_r, None),
            "counterfactuals": cf,
            "model_version": self.version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def analyze_batch(self, reviews: list[dict]) -> list[dict]:
        """Single vectorized model call for all texts; counterfactuals/SHAP/LIME skipped for speed."""
        self._require()
        probs = fake_fn(self.model)([r["review_text"] for r in reviews])
        return [
            self.analyze(r, with_counterfactual=False, fake_probability=float(p))
            for r, p in zip(reviews, probs)
        ]

    def feature_names(self) -> list[str]:
        return list(FEATURE_NAMES)
