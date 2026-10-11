"""Uses a tiny SYNTHETIC demonstration dataset. Results here validate code paths, not real-world performance."""

import json

import numpy as np
import pandas as pd

from src.models.splits import group_train_val_test_split, make_groups
from src.models.threshold import select_threshold
from src.models.train import train_all


def synthetic_df(n=300, seed=0):
    rng = np.random.default_rng(seed)
    promo = [
        "Best product ever!!! Highly recommend, must buy now!",
        "Amazing perfect awesome!!! Love it, highly recommend!",
    ]
    plain = [
        "The battery lasted two days but the strap frayed after a week of use.",
        "Delivery took six days and the box was dented, though the screen is fine.",
        "I returned the first unit because the hinge squeaked; the replacement works.",
    ]
    rows = []
    for i in range(n):
        fake = int(rng.random() < 0.4)
        base = rng.choice(promo if fake else plain)
        rows.append(
            {
                "review_id": f"s{i}",
                "user_id": f"u{i % 120}",
                "review_text": f"{base} variant {i % 7}",
                "label": fake,
            }
        )
    return pd.DataFrame(rows)


def test_split_has_disjoint_groups_and_covers_all():
    df = synthetic_df()
    tr, va, te = group_train_val_test_split(df, seed=1)
    g = make_groups(df)
    assert not set(g[tr]) & set(g[te])
    assert not set(g[tr]) & set(g[va])
    assert not set(g[va]) & set(g[te])
    assert len(tr) + len(va) + len(te) == len(df)


def test_threshold_uses_given_data_and_respects_constraint():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.6, 0.5, 0.7, 0.9])
    t = select_threshold(y, p, "f1")
    assert 0.05 <= t <= 0.95
    t2 = select_threshold(y, p, "recall_constraint", min_recall=1.0)
    assert ((p >= t2).astype(int)[y == 1]).all()


def test_train_all_writes_artifacts(tmp_path):
    train_all(
        synthetic_df(),
        tmp_path,
        models=["logistic_regression", "random_forest"],
        label_semantics="synthetic",
    )
    assert (tmp_path / "model.joblib").exists()
    saved = json.loads((tmp_path / "metadata.json").read_text())
    assert saved["label_semantics"] == "synthetic"
    assert saved["model_name"] in ("logistic_regression", "random_forest")
    assert sum(saved["split_sizes"].values()) == 300
    best = saved["all_model_results"][saved["model_name"]]
    assert 0.0 <= best["test"]["brier"] <= 1.0
    assert 0.05 <= saved["threshold"] <= 0.95


def test_train_requires_two_classes(tmp_path):
    import pytest

    df = synthetic_df(60)
    df["label"] = 0
    with pytest.raises(ValueError):
        train_all(df, tmp_path, models=["logistic_regression"])
