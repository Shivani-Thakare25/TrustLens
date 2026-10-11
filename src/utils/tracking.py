"""Experiment tracking. MLflow and W&B are OPTIONAL and can never make training fail."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from src.utils.logger import get_logger

log = get_logger(__name__)


def _flat_metrics(meta: dict) -> dict:
    out = {}
    for name, r in meta["all_model_results"].items():
        if r.get("status") != "trained":
            continue
        for split in ("validation", "test"):
            for k, v in r[split].items():
                out[f"{name}.{split}.{k}"] = float(v)
        out[f"{name}.train_seconds"] = float(r["train_seconds"])
    for k, v in meta["all_model_results"][meta["model_name"]]["test"].items():
        out[f"best.test.{k}"] = float(v)
    return out


def _params(meta: dict) -> dict:
    keys = (
        "model_name",
        "model_version",
        "random_seed",
        "threshold",
        "label_semantics",
        "dataset_version",
        "feature_version",
        "git_commit",
        "selection_criterion",
    )
    p = {k: meta.get(k) for k in keys}
    p.update({f"split_{k}": v for k, v in meta["split_sizes"].items()})
    return p


def _log_mlflow(cfg: dict, meta: dict, model_dir: Path, extra_dirs) -> str:
    import mlflow

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    name = cfg.get("experiment_name", "trustlens")
    if mlflow.get_experiment_by_name(name) is None and os.getenv("MLFLOW_ARTIFACT_ROOT"):
        mlflow.create_experiment(name, artifact_location=os.environ["MLFLOW_ARTIFACT_ROOT"])
    mlflow.set_experiment(name)
    with mlflow.start_run(run_name=f"{meta['model_name']}-{meta['model_version']}") as run:
        mlflow.log_params({k: str(v) for k, v in _params(meta).items()})
        mlflow.log_metrics(_flat_metrics(meta))
        if meta.get("git_commit"):
            mlflow.set_tag("git_commit", meta["git_commit"])
        mlflow.log_artifacts(
            str(model_dir), artifact_path="model"
        )  # model, metadata, importance, predictions
        for d in extra_dirs:
            if Path(d).exists():
                mlflow.log_artifacts(
                    str(d), artifact_path="evaluation"
                )  # ROC/PR/calibration/confusion plots
        return run.info.run_id


def _log_wandb(cfg: dict, meta: dict, model_dir: Path) -> bool:
    import wandb

    run = wandb.init(project=cfg.get("project", "trustlens"), config=_params(meta), reinit=True)
    wandb.log(_flat_metrics(meta))
    run.finish() if hasattr(run, "finish") else wandb.finish()
    return True


def log_run(
    meta: dict, model_dir, extra_dirs=(), config_path: str = "configs/tracking.yaml"
) -> dict:
    """Returns {'mlflow': run_id|None, 'wandb': bool}. Every failure is logged and swallowed."""
    result = {"mlflow": None, "wandb": False}
    try:
        cfg = yaml.safe_load(open(config_path))
    except Exception:
        log.warning("tracking config unreadable; tracking disabled")
        return result
    if cfg.get("mlflow", {}).get("enabled"):
        try:
            result["mlflow"] = _log_mlflow(cfg["mlflow"], meta, Path(model_dir), extra_dirs)
        except Exception as e:
            log.warning("MLflow tracking skipped: %s: %s", type(e).__name__, str(e)[:200])
    w = cfg.get("wandb", {})
    if w.get("enabled") and os.getenv("WANDB_API_KEY"):
        try:
            result["wandb"] = _log_wandb(w, meta, Path(model_dir))
        except Exception as e:
            log.warning("W&B tracking skipped: %s: %s", type(e).__name__, str(e)[:200])
    elif w.get("enabled"):
        log.warning("W&B enabled in config but WANDB_API_KEY is not set; skipping")
    return result
