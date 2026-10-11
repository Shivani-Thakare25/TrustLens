"""Load raw CSV/JSON/JSONL files into a DataFrame."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)


def load_raw(path) -> pd.DataFrame:
    """Load .csv, .json, or .jsonl. Malformed CSV lines are skipped and counted in the log."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Raw data file not found: {p}")
    suffix = p.suffix.lower()
    if suffix == ".csv":
        bad = []

        def _skip(line):
            bad.append(1)
            return None

        df = pd.read_csv(p, on_bad_lines=_skip, engine="python")
        if bad:
            log.warning("skipped %d malformed CSV lines", len(bad))
        return df
    if suffix == ".jsonl":
        return pd.read_json(p, lines=True)
    if suffix == ".json":
        return pd.read_json(p)
    raise ValueError(f"Unsupported file type: {suffix}")
