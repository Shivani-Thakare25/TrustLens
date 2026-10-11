"""Inference engine for trained TrustLens sentiment analysis models.

Loads pre-trained sentiment artifacts (neutral-optimized model or baseline model).
Never trains or retrains models at runtime.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
from scipy.sparse import hstack

from src.utils.logger import get_logger

log = get_logger(__name__)


def _normalize_text(val: object) -> str:
    if val is None:
        return ""
    text = str(val).strip()
    return re.sub(r"\s+", " ", text)


class SentimentPredictor:
    """Wraps pre-trained sentiment models for single and batch predictions."""

    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path(__file__).resolve().parents[2]
        self.model = None
        self.word_vectorizer = None
        self.char_vectorizer = None
        self.tfidf_vectorizer = None
        self.classes: List[str] = ["negative", "neutral", "positive"]
        self.model_type: Optional[str] = None
        self._load()

    def _load(self):
        """Attempt loading trained sentiment artifacts without running training scripts."""
        # 1. Try sentiment_neutral_optimized (dual TF-IDF model)
        opt_dir = self.project_root / "models" / "sentiment_neutral_optimized"
        opt_model = opt_dir / "sentiment_neutral_optimized_model.joblib"
        opt_word = opt_dir / "word_tfidf.joblib"
        opt_char = opt_dir / "char_tfidf.joblib"

        if opt_model.exists() and opt_word.exists() and opt_char.exists():
            try:
                self.model = joblib.load(opt_model)
                self.word_vectorizer = joblib.load(opt_word)
                self.char_vectorizer = joblib.load(opt_char)
                self.classes = [str(c).lower() for c in getattr(self.model, "classes_", self.classes)]
                self.model_type = "neutral_optimized"
                log.info("Loaded neutral-optimized sentiment model from %s", opt_dir)
                return
            except Exception:
                log.exception("Failed to load neutral-optimized sentiment model")

        # 2. Fall back to models/sentiment (single TF-IDF baseline model)
        base_dir = self.project_root / "models" / "sentiment"
        base_model = base_dir / "sentiment_model.joblib"
        base_tfidf = base_dir / "sentiment_tfidf.joblib"

        if base_model.exists() and base_tfidf.exists():
            try:
                self.model = joblib.load(base_model)
                self.tfidf_vectorizer = joblib.load(base_tfidf)
                self.classes = [str(c).lower() for c in getattr(self.model, "classes_", self.classes)]
                self.model_type = "baseline"
                log.info("Loaded baseline sentiment model from %s", base_dir)
                return
            except Exception:
                log.exception("Failed to load baseline sentiment model")

        log.warning("No trained sentiment model found in %s/models", self.project_root)

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def _transform(self, texts: List[str]):
        clean_texts = [_normalize_text(t) for t in texts]
        if self.model_type == "neutral_optimized":
            w = self.word_vectorizer.transform(clean_texts)
            c = self.char_vectorizer.transform(clean_texts)
            return hstack([w, c], format="csr")
        elif self.model_type == "baseline":
            return self.tfidf_vectorizer.transform(clean_texts)
        raise RuntimeError("No sentiment model loaded to transform features.")

    def predict(self, texts: List[str]) -> List[str]:
        """Return list of capitalized sentiment labels ('Negative', 'Neutral', 'Positive')."""
        if not self.loaded:
            raise RuntimeError("Sentiment model is not loaded.")
        if not texts:
            return []

        # Empty texts should default to Neutral safely
        indices_to_predict = []
        out = ["Neutral"] * len(texts)
        texts_to_predict = []

        for idx, t in enumerate(texts):
            if _normalize_text(t):
                indices_to_predict.append(idx)
                texts_to_predict.append(t)

        if not texts_to_predict:
            return out

        matrix = self._transform(texts_to_predict)
        raw_preds = self.model.predict(matrix)

        for idx, pred in zip(indices_to_predict, raw_preds):
            out[idx] = str(pred).strip().capitalize()

        return out

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Return (n, num_classes) array of probabilities."""
        if not self.loaded:
            raise RuntimeError("Sentiment model is not loaded.")
        if not texts:
            return np.empty((0, len(self.classes)))

        matrix = self._transform(texts)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(matrix)
        elif hasattr(self.model, "decision_function"):
            decision = self.model.decision_function(matrix)
            if decision.ndim == 1:
                p = 1.0 / (1.0 + np.exp(-decision))
                return np.vstack([1.0 - p, p]).T
            shifted = decision - decision.max(axis=1, keepdims=True)
            exp_scores = np.exp(shifted)
            return exp_scores / exp_scores.sum(axis=1, keepdims=True)
        return np.ones((len(texts), len(self.classes))) / len(self.classes)

    def analyze_one(self, text: str) -> dict:
        """Detailed analysis of a single review text string."""
        norm = _normalize_text(text)
        if not norm:
            return {
                "sentiment": "Neutral",
                "confidence": 1.0,
                "probabilities": {c.capitalize(): 1.0 / len(self.classes) for c in self.classes},
            }
        if not self.loaded:
            return {
                "sentiment": "Neutral",
                "confidence": 0.5,
                "probabilities": {},
            }
        probs = self.predict_proba([norm])[0]
        max_idx = int(np.argmax(probs))
        pred_label = self.classes[max_idx].capitalize()
        prob_dict = {
            self.classes[i].capitalize(): float(probs[i])
            for i in range(len(self.classes))
        }
        return {
            "sentiment": pred_label,
            "confidence": float(probs[max_idx]),
            "probabilities": prob_dict,
        }
