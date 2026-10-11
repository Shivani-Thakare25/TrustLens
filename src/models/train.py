"""Training pipeline: split -> fit (train only) -> calibrate -> threshold on val -> report on test."""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.explainability.feature_importance import global_importance
from src.features.text_features import FEATURE_NAMES, build_text_feature_frame
from src.models.factory import ALL_MODELS, build_estimator
from src.models.splits import group_train_val_test_split, make_groups
from src.models.threshold import select_threshold
from src.models.tuning import tune_pipeline
from src.monitoring.drift import make_baseline
from src.utils.logger import get_logger

log = get_logger(__name__)
FEATURE_VERSION = "v1"


def _handcrafted(texts):
    return build_text_feature_frame(
        texts.tolist() if hasattr(texts, "tolist") else list(texts)
    ).to_numpy()


def _hand_names(transformer, input_features):
    return np.array(FEATURE_NAMES, dtype=object)


def build_pipeline(estimator, seed: int = 42) -> Pipeline:
    """Preprocessing is inside the pipeline, so it is fit on training folds only (no leakage)."""
    features = ColumnTransformer(
        [
            (
                "word",
                TfidfVectorizer(ngram_range=(1, 2), max_features=5000, min_df=2, sublinear_tf=True),
                "review_text",
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    max_features=5000,
                    min_df=2,
                    sublinear_tf=True,
                ),
                "review_text",
            ),
            (
                "hand",
                Pipeline(
                    [
                        ("f", FunctionTransformer(_handcrafted, feature_names_out=_hand_names)),
                        ("s", StandardScaler()),
                    ]
                ),
                "review_text",
            ),
        ]
    )
    return Pipeline([("features", features), ("clf", estimator)])


def compute_metrics(y, p, threshold: float) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    pred = (p >= threshold).astype(int)
    out = {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "brier": float(brier_score_loss(y, p)),
    }
    if len(np.unique(y)) == 2:
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["pr_auc"] = float(average_precision_score(y, p))
    return out


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def train_all(
    df: pd.DataFrame,
    out_dir,
    models=None,
    seed: int = 42,
    threshold_objective: str = "f1",
    dataset_version: str = "unversioned",
    label_semantics: str = "unknown",
    save_reference_corpus: bool = False,
    tune: bool = False,
    tune_n_iter: int = 8,
) -> dict:
    """Train each available model, select the best by validation PR-AUC, and save artifacts."""
    df = df[df["label"].notna()].reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    if df["label"].nunique() < 2:
        raise ValueError("Need both classes in 'label' to train.")
    tr, va, te = group_train_val_test_split(df, seed=seed)
    X, y = df[["review_text"]], df["label"].to_numpy()
    spw = float((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1))

    groups_tr = make_groups(df)[tr]
    results, fitted, preds = {}, {}, {}
    for name in models or ALL_MODELS:
        est = build_estimator(name, seed=seed, scale_pos_weight=spw)
        if est is None:
            results[name] = {"status": "skipped_unavailable"}
            continue
        t0 = time.perf_counter()
        pipe = build_pipeline(est, seed)
        tuning = None
        if tune:
            tuning = tune_pipeline(name, pipe, X.iloc[tr], y[tr], groups_tr, seed, tune_n_iter)
            pipe.set_params(**tuning["best_params"])
        model = CalibratedClassifierCV(pipe, method="sigmoid", cv=3)
        model.fit(X.iloc[tr], y[tr])
        train_s = time.perf_counter() - t0
        p_val = model.predict_proba(X.iloc[va])[:, 1]
        thr = select_threshold(y[va], p_val, threshold_objective)
        t1 = time.perf_counter()
        p_te = model.predict_proba(X.iloc[te])[:, 1]
        infer_ms = (time.perf_counter() - t1) / max(len(te), 1) * 1000
        preds[name] = (y[te], p_te)
        results[name] = {
            "status": "trained",
            "tuning": tuning,
            "train_seconds": train_s,
            "inference_ms_per_review": infer_ms,
            "threshold": thr,
            "validation": compute_metrics(y[va], p_val, thr),
            "test": compute_metrics(y[te], p_te, thr),
        }
        fitted[name] = model
        log.info("trained %s in %.1fs", name, train_s)

    if not fitted:
        raise RuntimeError("No models were trained.")
    best = max(fitted, key=lambda n: results[n]["validation"].get("pr_auc", 0.0))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[best], out / "model.joblib")
    meta = {
        "model_name": best,
        "model_version": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
        "training_date": datetime.now(timezone.utc).isoformat(),
        "dataset_version": dataset_version,
        "label_semantics": label_semantics,
        "feature_version": FEATURE_VERSION,
        "random_seed": seed,
        "git_commit": _git_commit(),
        "threshold": results[best]["threshold"],
        "split_sizes": {"train": len(tr), "val": len(va), "test": len(te)},
        "selection_criterion": "validation PR-AUC",
        "all_model_results": results,
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    pd.DataFrame({"y_true": preds[best][0], "fake_probability": preds[best][1]}).to_csv(
        out / "test_predictions.csv", index=False
    )
    (out / "feature_importance.json").write_text(
        json.dumps(global_importance(fitted[best], 30), indent=2)
    )
    baseline = make_baseline(build_text_feature_frame(df["review_text"].iloc[tr].tolist()))
    (out / "baseline_features.json").write_text(json.dumps(baseline))
    if save_reference_corpus:  # opt-in: stores raw training review text for similarity lookups
        df.iloc[tr][["review_id", "review_text"]].to_csv(out / "reference_corpus.csv", index=False)
    return meta
