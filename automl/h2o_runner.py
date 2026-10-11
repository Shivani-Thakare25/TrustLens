"""H2O AutoML runner. Requires Java and: pip install -r requirements-automl.txt
Usage: python automl/h2o_runner.py <processed.csv> [--out artifacts/automl] [--max-runtime 120]"""

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


def run(csv_path, out_dir="artifacts/automl", max_runtime_secs: int = 120, seed: int = 42) -> dict:
    try:
        import h2o
        from h2o.automl import H2OAutoML
    except ImportError as e:
        raise AutoMLUnavailable(
            "h2o is not installed (see requirements-automl.txt; Java is also required)"
        ) from e
    s = prepare_splits(csv_path, seed)
    h2o.init(verbose=False)
    try:

        def frame(k):
            f = h2o.H2OFrame(s[k][0].assign(label=s[k][1]))
            f["label"] = f["label"].asfactor()
            return f

        tr, va, te = frame("train"), frame("val"), frame("test")
        feats = list(s["train"][0].columns)
        aml = H2OAutoML(max_runtime_secs=max_runtime_secs, seed=seed, sort_metric="aucpr")
        _, secs = timed(lambda: aml.train(x=feats, y="label", training_frame=tr))
        p = lambda f: aml.leader.predict(f)["p1"].as_data_frame().to_numpy().ravel()  # noqa: E731
        return finalize(
            "h2o",
            p(va),
            p(te),
            s,
            secs,
            out_dir,
            {"leader": aml.leader.model_id, "max_runtime_secs": max_runtime_secs, "seed": seed},
        )
    finally:
        h2o.cluster().shutdown(prompt=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--out", default="artifacts/automl")
    ap.add_argument("--max-runtime", type=int, default=120)
    a = ap.parse_args()
    try:
        print(run(a.data, a.out, a.max_runtime))
    except AutoMLUnavailable as e:
        sys.exit(f"H2O unavailable: {e}")
