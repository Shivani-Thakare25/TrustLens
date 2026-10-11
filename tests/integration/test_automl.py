import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from api.services.trustlens_service import reset_service
from automl import autogluon_runner, autosklearn_runner, h2o_runner
from automl.common import AutoMLUnavailable, prepare_splits
from src.models.train import compute_metrics
from tests.conftest import make_synthetic_df

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def csv(tmp_path):
    p = tmp_path / "d.csv"
    make_synthetic_df(240).to_csv(p, index=False)
    return p


@pytest.mark.parametrize(
    "runner,mod",
    [(autogluon_runner, "autogluon"), (h2o_runner, "h2o"), (autosklearn_runner, "autosklearn")],
)
def test_runners_report_unavailable_instead_of_crashing(runner, mod, csv, tmp_path, monkeypatch):
    for m in (mod, f"{mod}.tabular", f"{mod}.automl", f"{mod}.classification"):
        monkeypatch.setitem(sys.modules, m, None)
    with pytest.raises(AutoMLUnavailable):
        runner.run(csv, tmp_path)


def test_cli_exits_with_message_when_unavailable(csv):
    r = subprocess.run(
        [sys.executable, "automl/autogluon_runner.py", str(csv)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PYTHONPATH": ""},
    )
    if "autogluon" in r.stderr.lower() and "unavailable" in r.stderr.lower():
        assert r.returncode != 0
    else:
        pytest.skip("autogluon is installed here")


class _StubPredictor:
    def __init__(self, label, eval_metric, path):
        self.label = label

    def fit(self, train, time_limit, presets):
        assert self.label in train.columns and "exclaim_count" in train

    def predict_proba(self, X):
        p = np.clip(X["exclaim_count"].to_numpy() / 5, 0, 1)
        return pd.DataFrame({0: 1 - p, 1: p})


def _install_stub(monkeypatch):
    tab = types.ModuleType("autogluon.tabular")
    tab.TabularPredictor = _StubPredictor
    pkg = types.ModuleType("autogluon")
    pkg.tabular = tab
    monkeypatch.setitem(sys.modules, "autogluon", pkg)
    monkeypatch.setitem(sys.modules, "autogluon.tabular", tab)


def test_autogluon_runner_logic_with_stub(csv, tmp_path, monkeypatch):
    _install_stub(monkeypatch)
    res = autogluon_runner.run(csv, tmp_path / "out", time_limit=1)
    saved = json.loads((tmp_path / "out" / "autogluon.json").read_text())
    assert saved["status"] == "trained" and "handcrafted" in saved["features"]
    s = prepare_splits(csv)
    p_te = np.clip(s["test"][0]["exclaim_count"].to_numpy() / 5, 0, 1)
    expected = compute_metrics(s["test"][1], p_te, res["threshold"])
    assert res["test"]["pr_auc"] == pytest.approx(expected["pr_auc"]) and res["test"][
        "f1"
    ] == pytest.approx(expected["f1"])
    assert 0.05 <= res["threshold"] <= 0.95


def test_prepare_splits_requires_two_classes(tmp_path):
    df = make_synthetic_df(80)
    df["label"] = 1
    p = tmp_path / "one.csv"
    df.to_csv(p, index=False)
    with pytest.raises(ValueError):
        prepare_splits(p)


def test_performance_page_shows_automl_results(model_dir, csv, tmp_path, monkeypatch):
    _install_stub(monkeypatch)
    autogluon_runner.run(csv, tmp_path / "automl", time_limit=1)
    monkeypatch.setenv("TRUSTLENS_MODEL_DIR", str(model_dir))
    monkeypatch.setenv("TRUSTLENS_AUTOML_DIR", str(tmp_path / "automl"))
    reset_service()
    app = str(ROOT / "streamlit_app" / "app.py")
    at = AppTest.from_file(app, default_timeout=60).run()
    at.sidebar.radio(key="page").set_value("Model Performance").run()
    assert not at.exception
    assert any("AutoML results" in s.value for s in at.subheader)
    reset_service()
