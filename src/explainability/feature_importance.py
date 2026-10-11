"""Global feature importance from the fitted (calibrated) pipeline's classifier."""

from __future__ import annotations

import numpy as np

from src.features.text_features import FEATURE_NAMES


def _names(features) -> list[str]:
    out = []
    for n in features.get_feature_names_out():
        if n.startswith("hand__x") and n[7:].isdigit():
            out.append("hand__" + FEATURE_NAMES[int(n[7:])])
        else:
            out.append(n)
    return out


def global_importance(model, top_k: int = 20) -> list[dict]:
    """Mean |coefficient| (linear) or feature_importances_ (trees), averaged across calibration folds."""
    scores, names = None, None
    for cc in getattr(model, "calibrated_classifiers_", []):
        pipe = cc.estimator
        clf, feats = pipe.named_steps["clf"], pipe.named_steps["features"]
        if hasattr(clf, "coef_"):
            s = np.abs(np.ravel(clf.coef_))
        elif hasattr(clf, "feature_importances_"):
            s = np.asarray(clf.feature_importances_, dtype=float)
        else:
            continue
        names = _names(feats)
        scores = s if scores is None else scores + s
    if scores is None:
        return []
    order = np.argsort(-scores)[:top_k]
    return [
        {"feature": names[i], "importance": float(scores[i] / len(model.calibrated_classifiers_))}
        for i in order
    ]
