"""Measure latency and throughput. Reports MEASURED values only.
In-process:  python scripts/benchmark.py [--model-dir artifacts/model] [--n 200] [--out reports/benchmark.json]
HTTP:        python scripts/benchmark.py --url http://localhost:8000 [--n 100]   (end-to-end incl. network/JSON)
"""

import argparse
import json
import os
import platform
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

SAMPLES = [
    "The battery is excellent and the camera quality is amazing. Best product ever! Highly recommended!!!",
    "Delivery took six days and the box was dented, though the screen is fine.",
    "I returned the first unit because the hinge squeaked; the replacement works well.",
    "Terrible value. The price is too high and the sound quality is poor.",
]


def stats(lat_ms):
    a = np.asarray(lat_ms)
    return {
        "n": len(a),
        "average_ms": float(a.mean()),
        "p50_ms": float(np.percentile(a, 50)),
        "p95_ms": float(np.percentile(a, 95)),
        "max_ms": float(a.max()),
    }


def bench_inprocess(model_dir, n, batch_size):
    from src.inference.predictor import TrustLensPredictor

    pred = TrustLensPredictor(model_dir)
    if not pred.loaded:
        sys.exit(f"No trained model in {model_dir}; train one first.")
    reviews = [{"review_text": SAMPLES[i % len(SAMPLES)]} for i in range(n)]
    for r in reviews[:5]:  # warm-up, excluded from measurements
        pred.analyze(r, with_counterfactual=False)
    lat = []
    for r in reviews:
        t = time.perf_counter()
        pred.analyze(r, with_counterfactual=False)
        lat.append((time.perf_counter() - t) * 1000)
    batch = [{"review_text": SAMPLES[i % len(SAMPLES)]} for i in range(batch_size)]
    t = time.perf_counter()
    pred.analyze_batch(batch)
    secs = time.perf_counter() - t
    return {
        "mode": "in-process",
        "model_version": pred.version,
        "single_review": stats(lat),
        "batch": {
            "batch_size": len(batch),
            "seconds": secs,
            "reviews_per_second": len(batch) / secs,
        },
    }


def bench_http(url, n):
    def post(path, body):
        req = urllib.request.Request(
            url + path, json.dumps(body).encode(), {"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status

    for i in range(3):
        post("/api/v1/predict", {"review_text": SAMPLES[i]})
    lat = []
    for i in range(n):
        t = time.perf_counter()
        assert post("/api/v1/predict", {"review_text": SAMPLES[i % len(SAMPLES)]}) == 200
        lat.append((time.perf_counter() - t) * 1000)
    reviews = [{"review_text": SAMPLES[i % len(SAMPLES)]} for i in range(min(n, 100))]
    t = time.perf_counter()
    post("/api/v1/batch-predict", {"reviews": reviews})
    secs = time.perf_counter() - t
    return {
        "mode": "http",
        "url": url,
        "predict_endpoint": stats(lat),
        "batch": {
            "batch_size": len(reviews),
            "seconds": secs,
            "reviews_per_second": len(reviews) / secs,
        },
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=os.getenv("TRUSTLENS_MODEL_DIR", "artifacts/model"))
    ap.add_argument("--url", default=None)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--out", default="reports/benchmark.json")
    a = ap.parse_args()
    res = (
        bench_http(a.url.rstrip("/"), a.n)
        if a.url
        else bench_inprocess(a.model_dir, a.n, a.batch_size)
    )
    res["environment"] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
