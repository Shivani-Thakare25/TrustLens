"""Usage: python scripts/robustness.py <data.csv> [--model-dir artifacts/model] [--out reports/robustness.json]"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402

from robustness.robustness_evaluation import evaluate_robustness  # noqa: E402
from src.data.ingestion import load_raw  # noqa: E402
from src.explainability.model_adapter import fake_fn  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--model-dir", default="artifacts/model")
    ap.add_argument("--out", default="reports/robustness.json")
    ap.add_argument("--max-texts", type=int, default=500)
    a = ap.parse_args()
    meta = json.loads((Path(a.model_dir) / "metadata.json").read_text())
    model = joblib.load(Path(a.model_dir) / "model.joblib")
    res = evaluate_robustness(
        fake_fn(model),
        load_raw(a.data)["review_text"].astype(str).tolist(),
        meta["threshold"],
        a.max_texts,
    )
    res["model_version"] = meta["model_version"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
