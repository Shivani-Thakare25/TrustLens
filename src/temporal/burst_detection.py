"""Temporal anomaly detection.

Burst test: for each day, compare its review count to the leave-one-out mean daily rate using a
Poisson upper-tail probability, with Bonferroni correction across days. A minimum count guards
against flagging tiny absolute numbers. Flags are statistical anomalies, not proof of manipulation.
"""

from __future__ import annotations

import pandas as pd
from scipy.stats import poisson


def detect_bursts(
    dates: pd.Series, alpha: float = 0.01, min_days: int = 7, min_count: int = 3
) -> list[dict]:
    """Return days with a significantly elevated review count. Empty list if data is insufficient."""
    d = pd.to_datetime(dates, errors="coerce").dropna()
    if d.empty:
        return []
    daily = d.dt.floor("D").value_counts().sort_index()
    full = daily.reindex(
        pd.date_range(daily.index.min(), daily.index.max(), freq="D"), fill_value=0
    )
    n, total = len(full), int(full.sum())
    if n < min_days:
        return []
    out = []
    for day, count in full.items():
        lam = max((total - count) / (n - 1), 1e-3)
        p = float(poisson.sf(count - 1, lam))
        if count >= min_count and p < alpha / n:
            out.append(
                {
                    "date": str(day.date()),
                    "count": int(count),
                    "expected": round(lam, 3),
                    "p_value": p,
                }
            )
    return out


def temporal_concentration(dates: pd.Series, window_days: int = 7):
    """Largest fraction of reviews falling inside any `window_days` window. None if no valid dates."""
    d = pd.to_datetime(dates, errors="coerce").dropna().sort_values().reset_index(drop=True)
    if d.empty:
        return None
    best, j = 1, 0
    for i in range(len(d)):
        while (d[i] - d[j]).days >= window_days:
            j += 1
        best = max(best, i - j + 1)
    return best / len(d)
