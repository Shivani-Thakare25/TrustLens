"""Usage: python scripts/train.py <data.csv> [--out artifacts/model] [--label-semantics true|weak|proxy|synthetic]
[--models logistic_regression random_forest ...] [--tune] [--no-tracking] [--save-reference-corpus]
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from src.data.cleaning import clean_dataframe  # noqa: E402
from src.data.ingestion import load_raw  # noqa: E402
from src.models.evaluate import evaluate_run  # noqa: E402
from src.models.train import train_all  # noqa: E402
from src.utils.tracking import log_run  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--out", default=None)
    ap.add_argument("--label-semantics", default=None)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--no-tracking", action="store_true")
    ap.add_argument("--save-reference-corpus", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "configs/config.yaml"))
    dcfg = yaml.safe_load(open(ROOT / "configs/data_config.yaml"))
    mcfg = yaml.safe_load(open(ROOT / "configs/model_config.yaml"))
    out = Path(a.out or cfg["model_dir"])
    df, _ = clean_dataframe(load_raw(a.data))
    version = "sha256:" + hashlib.sha256(Path(a.data).read_bytes()).hexdigest()[:12]
    meta = train_all(
        df,
        out,
        models=a.models or mcfg["models"],
        seed=cfg["random_seed"],
        threshold_objective=mcfg["threshold_objective"],
        dataset_version=version,
        label_semantics=a.label_semantics or dcfg["label_semantics"],
        save_reference_corpus=a.save_reference_corpus or mcfg["save_reference_corpus"],
        tune=a.tune or mcfg["tune"],
        tune_n_iter=mcfg["tune_n_iter"],
    )
    evaluate_run(out, out / "evaluation")
    tracking = (
        {}
        if a.no_tracking
        else log_run(meta, out, [out / "evaluation"], str(ROOT / "configs/tracking.yaml"))
    )
    print(
        json.dumps(
            {
                **{
                    k: meta[k]
                    for k in (
                        "model_name",
                        "model_version",
                        "threshold",
                        "split_sizes",
                        "dataset_version",
                        "label_semantics",
                    )
                },
                "tracking": tracking,
            },
            indent=2,
        )
    )
