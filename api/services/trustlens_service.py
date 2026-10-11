"""Service layer shared by FastAPI and Flask. All ML logic lives in src/inference."""

from __future__ import annotations

import base64
import binascii
import os
from datetime import datetime, timezone

from api.services.validation import validate_batch, validate_review_payload
from src.absa.absa import extract_aspect_sentiment, overall_sentiment
from src.explainability.counterfactual import counterfactuals
from src.explainability.model_adapter import fake_fn
from src.inference.predictor import ModelNotLoadedError, TrustLensPredictor

_service = None
_image_index = None


class TrustLensService:
    def __init__(self, predictor: TrustLensPredictor):
        self.p = predictor

    def health(self) -> dict:
        ok = self.p.loaded
        return {
            "status": "healthy" if ok else "unhealthy",
            "model_loaded": ok,
            "model_version": self.p.version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _analyze_image(self, b64: str) -> dict:
        """Optional image evidence. Bad images raise ValueError; missing OpenCV degrades gracefully."""
        from src.computer_vision import image_analysis as ia

        if not ia.cv_available():
            return {
                "available": False,
                "reason": "OpenCV is not installed (pip install -r requirements-cv.txt)",
            }
        try:
            data = base64.b64decode(b64, validate=True)
        except (binascii.Error, ValueError):
            raise ValueError("image_base64 is not valid base64.")
        global _image_index
        if _image_index is None:
            _image_index = ia.ImageHashIndex()
        info = ia.analyze_image(
            data, _image_index, run_yolo=os.getenv("TRUSTLENS_ENABLE_YOLO") == "1"
        )
        return {"available": True, **info}

    def predict(self, payload: dict) -> dict:
        review = validate_review_payload(payload)
        b64 = review.pop("image_base64", None)
        image = self._analyze_image(b64) if b64 else None
        if image and image["available"]:
            review["image_signals"] = image
        res = self.p.analyze(review)
        if image is not None:
            res["image_analysis"] = image
        return res

    def batch_predict(self, payload: dict) -> dict:
        res = self.p.analyze_batch(validate_batch(payload))
        return {"count": len(res), "results": res}

    def aspect_sentiment(self, payload: dict) -> dict:
        text = validate_review_payload(payload)["review_text"]
        return {
            "overall_sentiment": overall_sentiment(text, sentiment_predictor=self.p.sentiment_predictor),
            "aspects": extract_aspect_sentiment(text, sentiment_predictor=self.p.sentiment_predictor),
        }

    def explain(self, payload: dict) -> dict:
        r = self.p.analyze(
            validate_review_payload(payload), explain=True, with_counterfactual=False
        )
        return {
            "review_id": r["review_id"],
            "fake_probability": r["fake_probability"],
            "explanation": r["explanation"],
            "evidence": r["evidence"],
            "model_version": r["model_version"],
        }

    def counterfactual(self, payload: dict) -> dict:
        self.p._require()
        return counterfactuals(
            fake_fn(self.p.model), validate_review_payload(payload)["review_text"]
        )

    def similar_reviews(self, payload: dict, top_k: int = 5) -> dict:
        text = validate_review_payload(payload)["review_text"]
        if self.p.similarity is None:
            return {
                "available": False,
                "reason": "No reference corpus saved with this model "
                "(train with save_reference_corpus=True).",
                "similar_reviews": [],
            }
        return {"available": True, "similar_reviews": self.p.similarity.query(text, top_k=top_k)}

    def model_info(self) -> dict:
        self.p._require()
        m = self.p.metadata
        return {
            k: m.get(k)
            for k in (
                "model_name",
                "model_version",
                "training_date",
                "dataset_version",
                "label_semantics",
                "feature_version",
                "random_seed",
                "git_commit",
                "threshold",
                "split_sizes",
                "selection_criterion",
            )
        }

    def features(self) -> dict:
        return {
            "handcrafted_features": self.p.feature_names(),
            "text_representation": ["word TF-IDF (1-2 grams)", "char TF-IDF (3-5 grams)"],
        }

    def metrics(self) -> dict:
        return self.p.monitor.summary()

    def drift(self) -> dict:
        return self.p.monitor.drift(self.p.baseline)


def get_service() -> TrustLensService:
    global _service
    if _service is None:
        _service = TrustLensService(
            TrustLensPredictor(os.getenv("TRUSTLENS_MODEL_DIR", "artifacts/model"))
        )
    return _service


def reset_service() -> None:
    global _service, _image_index
    _service, _image_index = None, None


__all__ = ["TrustLensService", "get_service", "reset_service", "ModelNotLoadedError"]
