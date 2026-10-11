"""In-memory rolling monitor of predictions (no raw review text is stored)."""

from __future__ import annotations

import threading
from collections import Counter, deque

import numpy as np
import pandas as pd

from src.monitoring.data_quality import OPTIONAL_FIELDS
from src.monitoring.drift import drift_report


class PredictionMonitor:
    def __init__(self, maxlen: int = 10000):
        self.records = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def record(
        self, fake_probability, classification, sentiment, latency_ms, confidence, features, missing
    ):
        with self._lock:
            self.records.append(
                {
                    "fp": fake_probability,
                    "cls": classification,
                    "sent": sentiment,
                    "lat": latency_ms,
                    "conf": confidence,
                    "feat": features,
                    "missing": missing,
                }
            )

    def summary(self) -> dict:
        with self._lock:
            recs = list(self.records)
        n = len(recs)
        if n == 0:
            return {"predictions_recorded": 0}
        lat = np.array([r["lat"] for r in recs])
        return {
            "predictions_recorded": n,
            "classification_distribution": dict(Counter(r["cls"] for r in recs)),
            "sentiment_distribution": dict(Counter(r["sent"] for r in recs)),
            "mean_fake_probability": float(np.mean([r["fp"] for r in recs])),
            "mean_confidence": float(np.mean([r["conf"] for r in recs])),
            "latency_ms": {
                "average": float(lat.mean()),
                "p50": float(np.percentile(lat, 50)),
                "p95": float(np.percentile(lat, 95)),
            },
            "missing_field_rate": {
                f: sum(f in r["missing"] for r in recs) / n for f in OPTIONAL_FIELDS
            },
        }

    def drift(self, baseline: dict, min_samples: int = 30) -> dict:
        with self._lock:
            live = pd.DataFrame([r["feat"] for r in self.records])
        return drift_report(baseline, live, min_samples)
