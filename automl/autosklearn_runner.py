"""auto-sklearn runner. Linux/WSL2 only and needs an older Python; install separately
(see AUTOML.md). Usage: python automl/autosklearn_runner.py <processed.csv> [--out artifacts/automl] [--time-limit 120]
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
        import autosklearn.classification as ask
    except ImportError as e:
        raise AutoMLUnavailable(
            "auto-sklearn is not installed (Linux/WSL2 only; see AUTOML.md)"
        ) from e
    s = prepare_splits(csv_path, seed)
    clf = ask.AutoSklearnClassifier(time_left_for_this_task=time_limit, seed=seed, n_jobs=1)
    _, secs = timed(lambda: clf.fit(s["train"][0], s["train"][1]))
    p = lambda X: clf.predict_proba(X)[:, 1]  # noqa: E731
    return finalize(
        "autosklearn",
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
        sys.exit(f"auto-sklearn unavailable: {e}")
