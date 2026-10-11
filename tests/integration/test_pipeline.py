"""Pipeline, tracking, registry, evaluation and DVC-consistency tests (SYNTHETIC data; validates code paths)."""

import json
import subprocess
import sys
import types
from pathlib import Path

import pytest
import yaml

from src.models.evaluate import evaluate_run
from src.models.registry import register_model
from src.models.train import train_all
from src.utils.tracking import log_run
from tests.conftest import make_synthetic_df

ROOT = Path(__file__).resolve().parents[2]


def _py(*args, cwd=ROOT):
    return subprocess.run(
        [sys.executable, *map(str, args)], cwd=cwd, capture_output=True, text=True
    )


def test_run_all_end_to_end(tmp_path):
    raw = tmp_path / "raw.csv"
    make_synthetic_df(240).to_csv(raw, index=False)
    ws = tmp_path / "ws"
    r = _py(
        "scripts/run_all.py",
        raw,
        "--label-semantics",
        "synthetic",
        "--workspace",
        ws,
        "--models",
        "logistic_regression",
        "--no-tracking",
    )
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    meta = json.loads((ws / "artifacts/model/metadata.json").read_text())
    assert meta["label_semantics"] == "synthetic" and meta["dataset_version"].startswith("sha256:")
    for f in (
        "data/processed/reviews.csv",
        "data/processed/text_features.csv",
        "reports/data_validation.json",
        "reports/cleaning_stats.json",
        "reports/evaluation/metrics.json",
        "reports/evaluation/roc.png",
        "reports/evaluation/calibration.png",
        "artifacts/registry/registry.json",
        "artifacts/model/feature_importance.json",
        "artifacts/model/baseline_features.json",
    ):
        assert (ws / f).exists(), f
    assert (
        json.loads((ws / "artifacts/registry/registry.json").read_text())[0]["model_version"]
        == meta["model_version"]
    )


def test_scripts_fail_cleanly_on_bad_input(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("rating\n5\n")
    r = _py("scripts/validate_data.py", bad)
    assert r.returncode == 1 and "review_text" in r.stdout
    assert _py("scripts/register_model.py", "--model-dir", tmp_path / "none").returncode != 0
    assert _py("scripts/download_data.py", "--url", "http://insecure.example/x.csv").returncode != 0
    assert "Candidate public sources" in _py("scripts/download_data.py", "--list-sources").stdout


def test_setup_script_runs():
    r = _py("scripts/setup.py")
    assert r.returncode == 0 and "core packages" in r.stdout


def test_tuning_records_group_cv_results(tmp_path):
    meta = train_all(
        make_synthetic_df(240), tmp_path, models=["logistic_regression"], tune=True, tune_n_iter=2
    )
    t = meta["all_model_results"]["logistic_regression"]["tuning"]
    assert "clf__C" in t["best_params"] and 0 <= t["cv_pr_auc"] <= 1


def test_evaluate_run_writes_plots_and_matches_stored_metrics(tmp_path):
    m = tmp_path / "m"
    meta = train_all(make_synthetic_df(), m, models=["logistic_regression"])
    res = evaluate_run(m, tmp_path / "eval")
    assert set(res["plots"]) == {"confusion_matrix", "roc", "pr", "calibration"}
    assert all(Path(p).stat().st_size > 1000 for p in res["plots"].values())
    stored = meta["all_model_results"]["logistic_regression"]["test"]
    assert res["metrics"]["f1"] == pytest.approx(stored["f1"]) and res["metrics"][
        "pr_auc"
    ] == pytest.approx(stored["pr_auc"])


def test_registry_versions_and_errors(tmp_path):
    m = tmp_path / "m"
    train_all(make_synthetic_df(), m, models=["logistic_regression"])
    e = register_model(m, tmp_path / "reg")
    assert Path(e["path"], "model.joblib").exists()
    with pytest.raises(FileExistsError):
        register_model(m, tmp_path / "reg")
    with pytest.raises(FileNotFoundError):
        register_model(tmp_path / "empty", tmp_path / "reg")


def _cfg(tmp_path, mlflow=True, wandb=False):
    p = tmp_path / "tracking.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "mlflow": {"enabled": mlflow, "experiment_name": "t"},
                "wandb": {"enabled": wandb, "project": "t"},
            }
        )
    )
    return str(p)


def _model(tmp_path):
    m = tmp_path / "m"
    return m, train_all(make_synthetic_df(), m, models=["logistic_regression"])


def test_mlflow_logs_params_metrics_artifacts(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("MLFLOW_ARTIFACT_ROOT", (tmp_path / "artifacts").as_uri())
    m, meta = _model(tmp_path)
    res = log_run(meta, m, [], _cfg(tmp_path))
    assert res["mlflow"]
    run = mlflow.tracking.MlflowClient(uri).get_run(res["mlflow"])
    assert run.data.metrics["best.test.f1"] == pytest.approx(
        meta["all_model_results"]["logistic_regression"]["test"]["f1"]
    )
    assert run.data.params["label_semantics"] == "unknown"
    arts = [
        a.path for a in mlflow.tracking.MlflowClient(uri).list_artifacts(res["mlflow"], "model")
    ]
    assert "model/model.joblib" in arts and "model/metadata.json" in arts


def test_tracking_never_fails_training_when_unconfigured(tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    m, meta = _model(tmp_path)
    assert log_run(meta, m, [], _cfg(tmp_path, mlflow=False, wandb=True)) == {
        "mlflow": None,
        "wandb": False,
    }
    assert log_run(meta, m, [], str(tmp_path / "missing.yaml")) == {"mlflow": None, "wandb": False}


def test_mlflow_failure_is_swallowed(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:1")  # unreachable
    m, meta = _model(tmp_path)
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(
        "src.utils.tracking._log_mlflow",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")),
    )
    assert log_run(meta, m, [], cfg)["mlflow"] is None


def test_wandb_used_only_with_key(tmp_path, monkeypatch):
    calls = []
    fake = types.SimpleNamespace(
        init=lambda **k: calls.append("init")
        or types.SimpleNamespace(finish=lambda: calls.append("finish")),
        log=lambda d: calls.append("log"),
        finish=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "wandb", fake)
    m, meta = _model(tmp_path)
    monkeypatch.setenv("WANDB_API_KEY", "test-key-not-real")
    assert log_run(meta, m, [], _cfg(tmp_path, mlflow=False, wandb=True))["wandb"] is True
    assert calls == ["init", "log", "finish"]


def test_dvc_stages_reference_real_scripts_and_dirs():
    stages = yaml.safe_load((ROOT / "dvc.yaml").read_text())["stages"]
    assert list(stages) == ["validate", "preprocess", "features", "train", "evaluate"]
    for name, st in stages.items():
        script = st["cmd"].split()[1]
        assert (ROOT / script).is_file(), f"{name}: {script}"
        for dep in st["deps"]:
            if dep.startswith(("scripts/", "src/", "configs/")):
                assert (ROOT / dep).exists(), f"{name}: {dep}"
    assert yaml.safe_load((ROOT / "configs/config.yaml").read_text())["random_seed"] == 42


def test_config_keys_used_by_train_script_exist():
    mcfg = yaml.safe_load((ROOT / "configs/model_config.yaml").read_text())
    for k in ("models", "threshold_objective", "tune", "tune_n_iter", "save_reference_corpus"):
        assert k in mcfg
    dcfg = yaml.safe_load((ROOT / "configs/data_config.yaml").read_text())
    for k in ("min_rating", "max_rating", "max_text_chars", "label_semantics"):
        assert k in dcfg
