"""Threshold selection on VALIDATION data only (never the test set)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def select_threshold(
    y_val, p_val, objective: str = "f1", min_recall: float = 0.8, min_precision: float = 0.8
) -> float:
    """objective: 'f1' | 'recall_constraint' (max precision s.t. recall>=min_recall)
    | 'precision_constraint' (max recall s.t. precision>=min_precision)."""
    y_val, p_val = np.asarray(y_val), np.asarray(p_val)
    best_t, best_score = 0.5, -1.0
    for t in np.unique(np.round(np.linspace(0.05, 0.95, 91), 2)):
        pred = (p_val >= t).astype(int)
        if objective == "f1":
            score = f1_score(y_val, pred, zero_division=0)
        elif objective == "recall_constraint":
            score = (
                precision_score(y_val, pred, zero_division=0)
                if recall_score(y_val, pred, zero_division=0) >= min_recall
                else -1.0
            )
        elif objective == "precision_constraint":
            score = (
                recall_score(y_val, pred, zero_division=0)
                if precision_score(y_val, pred, zero_division=0) >= min_precision
                else -1.0
            )
        else:
            raise ValueError(f"Unknown objective: {objective}")
        if score > best_score:
            best_t, best_score = float(t), score
    return best_t
