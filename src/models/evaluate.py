"""Evaluation artifacts from saved test-set predictions (never recomputes on training data)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from src.models.train import compute_metrics  # noqa: E402


def _save(fig, path: Path) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)


def evaluate_run(model_dir, out_dir) -> dict:
    """Writes metrics.json and confusion_matrix/roc/pr/calibration PNGs to out_dir; returns metrics + paths."""
    model_dir, out = Path(model_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((model_dir / "metadata.json").read_text())
    preds = pd.read_csv(model_dir / "test_predictions.csv")
    y, p, thr = preds["y_true"].to_numpy(), preds["fake_probability"].to_numpy(), meta["threshold"]
    metrics = compute_metrics(y, p, thr)
    paths = {}

    cm = confusion_matrix(y, (p >= thr).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, int(v), ha="center", va="center")
    ax.set_xticks([0, 1], ["genuine-labelled", "fake-labelled"], rotation=20)
    ax.set_yticks([0, 1], ["genuine-labelled", "fake-labelled"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix (threshold {thr:.2f})")
    paths["confusion_matrix"] = _save(fig, out / "confusion_matrix.png")

    if len(np.unique(y)) == 2:
        fpr, tpr, _ = roc_curve(y, p)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(fpr, tpr, label=f"AUC={metrics['roc_auc']:.3f}")
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
        ax.legend()
        paths["roc"] = _save(fig, out / "roc.png")

        prec, rec, _ = precision_recall_curve(y, p)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(rec, prec, label=f"PR-AUC={metrics['pr_auc']:.3f}")
        ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve")
        ax.legend()
        paths["pr"] = _save(fig, out / "pr.png")

        bins = int(min(10, max(2, len(y) // 20)))
        frac, mean_pred = calibration_curve(y, p, n_bins=bins, strategy="quantile")
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(mean_pred, frac, "o-", label=f"Brier={metrics['brier']:.3f}")
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set(
            xlabel="Mean predicted probability",
            ylabel="Observed fraction positive",
            title="Reliability diagram",
        )
        ax.legend()
        paths["calibration"] = _save(fig, out / "calibration.png")
    result = {
        "model_name": meta["model_name"],
        "label_semantics": meta["label_semantics"],
        "n_test": int(len(y)),
        "metrics": metrics,
        "plots": paths,
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    return result
