import numpy as np
import pandas as pd

from src.monitoring.drift import drift_report, make_baseline, psi


def test_psi_zero_for_same_distribution_and_large_for_shift():
    rng = np.random.default_rng(0)
    base = pd.DataFrame({"f": rng.normal(0, 1, 5000)})
    b = make_baseline(base)["f"]
    assert psi(b["edges"], b["proportions"], rng.normal(0, 1, 5000)) < 0.05
    assert psi(b["edges"], b["proportions"], rng.normal(2, 1, 5000)) > 0.25


def test_drift_report_states():
    base = make_baseline(pd.DataFrame({"f": np.arange(100.0)}))
    assert drift_report({}, pd.DataFrame({"f": [1.0]}))["status"] == "no_baseline"
    assert drift_report(base, pd.DataFrame({"f": [1.0] * 5}))["status"] == "insufficient_data"
    rep = drift_report(base, pd.DataFrame({"f": np.arange(100.0)}))
    assert rep["status"] == "ok" and rep["overall_level"] == "stable"
    assert (
        drift_report(base, pd.DataFrame({"f": np.full(100, 500.0)}))["overall_level"]
        == "significant"
    )
