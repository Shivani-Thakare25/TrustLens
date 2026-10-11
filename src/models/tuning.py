"""Hyperparameter search with group-aware stratified CV on TRAINING data only (scored by PR-AUC)."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold

PARAM_GRIDS = {
    "logistic_regression": {"clf__C": [0.01, 0.1, 1.0, 10.0, 100.0]},
    "random_forest": {
        "clf__n_estimators": [100, 200, 400],
        "clf__max_depth": [None, 10, 30],
        "clf__min_samples_leaf": [1, 2, 5],
    },
    "xgboost": {
        "clf__max_depth": [3, 5, 7],
        "clf__learning_rate": [0.03, 0.1, 0.3],
        "clf__n_estimators": [100, 200, 400],
    },
    "lightgbm": {
        "clf__num_leaves": [15, 31, 63],
        "clf__learning_rate": [0.03, 0.1, 0.3],
        "clf__n_estimators": [100, 200, 400],
    },
    "catboost": {
        "clf__depth": [4, 6, 8],
        "clf__learning_rate": [0.03, 0.1, 0.3],
        "clf__iterations": [100, 200, 400],
    },
}


def tune_pipeline(
    name: str, pipeline, X, y, groups, seed: int = 42, n_iter: int = 8, folds: int = 3
) -> dict:
    """Return {'best_params', 'cv_pr_auc'}; params are applied by the caller via pipeline.set_params."""
    grid = PARAM_GRIDS[name]
    n_combos = int(np.prod([len(v) for v in grid.values()]))
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    search = RandomizedSearchCV(
        pipeline,
        grid,
        n_iter=min(n_iter, n_combos),
        scoring="average_precision",
        cv=cv,
        random_state=seed,
        n_jobs=1,
        refit=False,
    )
    search.fit(X, y, groups=groups)
    return {
        "best_params": {
            k: (v.item() if hasattr(v, "item") else v) for k, v in search.best_params_.items()
        },
        "cv_pr_auc": float(search.best_score_),
    }
