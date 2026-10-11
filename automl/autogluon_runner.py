"""AutoGluon Tabular runner. Requires: pip install -r requirements-automl.txt
Usage: python automl/autogluon_runner.py <processed.csv> [--out artifacts/automl] [--time-limit 120]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from automl.common import (  # noqa: E402
    AutoMLUnavailable,
    finalize,
    prepare_splits,
    timed,
)


def run(csv_path, out_dir="artifacts/automl", time_limit: int = 120, seed: int = 42) -> dict:
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError as e:
        raise AutoMLUnavailable(
            "autogluon.tabular is not installed (see requirements-automl.txt)"
        ) from e
    s = prepare_splits(csv_path, seed)
    train = s["train"][0].assign(label=s["train"][1])
    model_path = Path(out_dir) / "autogluon_model"
    pred = TabularPredictor(label="label", eval_metric="average_precision", path=str(model_path))
    _, secs = timed(lambda: pred.fit(train, time_limit=time_limit, presets="medium_quality"))
    p = lambda X: pred.predict_proba(X)[1].to_numpy()  # noqa: E731
    return finalize(
        "autogluon",
        p(s["val"][0]),
        p(s["test"][0]),
        s,
        secs,
        out_dir,
        {"time_limit": time_limit, "seed": seed},
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--out", default="artifacts/automl")
    ap.add_argument("--time-limit", type=int, default=120)
    a = ap.parse_args()
    try:
        print(run(a.data, a.out, a.time_limit))
    except AutoMLUnavailable as e:
        sys.exit(f"AutoGluon unavailable: {e}")
