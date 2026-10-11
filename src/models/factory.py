"""Model factory. Boosting libraries are optional; unavailable ones are skipped, not fatal."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.utils.logger import get_logger

log = get_logger(__name__)
ALL_MODELS = ["logistic_regression", "random_forest", "xgboost", "lightgbm", "catboost"]


def build_estimator(name: str, seed: int = 42, scale_pos_weight: float = 1.0):
    """Return an unfitted estimator or None if its library is not installed."""
    if name == "logistic_regression":
        return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=seed
        )
    try:
        if name == "xgboost":
            from xgboost import XGBClassifier

            return XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                random_state=seed,
                n_jobs=-1,
            )
        if name == "lightgbm":
            from lightgbm import LGBMClassifier

            return LGBMClassifier(
                n_estimators=200,
                learning_rate=0.1,
                class_weight="balanced",
                random_state=seed,
                verbose=-1,
            )
        if name == "catboost":
            from catboost import CatBoostClassifier

            return CatBoostClassifier(
                iterations=200,
                learning_rate=0.1,
                auto_class_weights="Balanced",
                random_seed=seed,
                verbose=0,
            )
    except ImportError:
        log.warning("optional model %s unavailable (library not installed); skipping", name)
        return None
    raise ValueError(f"Unknown model: {name}")
