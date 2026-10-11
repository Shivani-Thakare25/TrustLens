"""Usage: python scripts/evaluate.py [--model-dir artifacts/model] [--out reports/evaluation]"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.evaluate import evaluate_run  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="artifacts/model")
    ap.add_argument("--out", default="reports/evaluation")
    a = ap.parse_args()
    print(json.dumps(evaluate_run(a.model_dir, a.out), indent=2))
