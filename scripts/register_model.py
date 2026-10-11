"""Usage: python scripts/register_model.py [--model-dir artifacts/model] [--registry artifacts/registry] [--stage candidate]"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.registry import register_model  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="artifacts/model")
    ap.add_argument("--registry", default="artifacts/registry")
    ap.add_argument("--stage", default="candidate")
    a = ap.parse_args()
    print(json.dumps(register_model(a.model_dir, a.registry, a.stage), indent=2))
