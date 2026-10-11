"""Run the full local pipeline.
Usage: python scripts/run_all.py <raw.csv> --label-semantics <true|weak|proxy|synthetic> [--workspace .] [--models ...]
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    cmd = [sys.executable, *map(str, args)]
    print(">>", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        sys.exit(f"step failed: {args[0]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--label-semantics", required=True)
    ap.add_argument("--workspace", default=".")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--no-tracking", action="store_true")
    a = ap.parse_args()
    ws = Path(a.workspace).resolve()
    processed, model = ws / "data/processed/reviews.csv", ws / "artifacts/model"
    run("scripts/validate_data.py", a.data, "--out", ws / "reports/data_validation.json")
    run(
        "scripts/preprocess.py",
        a.data,
        "--out",
        processed,
        "--stats",
        ws / "reports/cleaning_stats.json",
    )
    run("scripts/build_features.py", processed, "--out", ws / "data/processed/text_features.csv")
    train = ["scripts/train.py", processed, "--out", model, "--label-semantics", a.label_semantics]
    if a.models:
        train += ["--models", *a.models]
    if a.tune:
        train.append("--tune")
    if a.no_tracking:
        train.append("--no-tracking")
    run(*train)
    run("scripts/evaluate.py", "--model-dir", model, "--out", ws / "reports/evaluation")
    run("scripts/register_model.py", "--model-dir", model, "--registry", ws / "artifacts/registry")
    print("Pipeline complete. Start the API: uvicorn api.fastapi_app.main:app --port 8000")
