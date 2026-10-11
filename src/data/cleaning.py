"""Cleaning: invalid values become null; only unusable rows are removed, and all counts are logged."""

from __future__ import annotations

import pandas as pd

from src.data.schema import ALL_COLUMNS
from src.utils.logger import get_logger

log = get_logger(__name__)


def clean_dataframe(
    df: pd.DataFrame, min_rating: int = 1, max_rating: int = 5, max_text_chars: int = 20000
):
    """Return (cleaned_df, stats). Light normalization preserves linguistic signals."""
    out = df.copy()
    stats = {"input_rows": len(out)}

    for col in ALL_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA  # nullable, never fabricated

    txt = out["review_text"].astype("string").str.replace(r"\s+", " ", regex=True).str.strip()
    out["review_text"] = txt.str.slice(0, max_text_chars)

    r = pd.to_numeric(out["rating"], errors="coerce")
    bad_r = (r < min_rating) | (r > max_rating)
    stats["ratings_set_null"] = int((bad_r | (r.isna() & out["rating"].notna())).sum())
    out["rating"] = r.mask(bad_r)

    d = pd.to_datetime(out["review_date"], errors="coerce")
    stats["dates_set_null"] = int((d.isna() & out["review_date"].notna()).sum())
    out["review_date"] = d

    lab = pd.to_numeric(out["label"], errors="coerce")
    stats["labels_set_null"] = int((~lab.isin([0, 1]) & out["label"].notna()).sum())
    out["label"] = lab.where(lab.isin([0, 1]))

    empty = out["review_text"].isna() | (out["review_text"] == "")
    stats["rows_removed_empty_text"] = int(empty.sum())
    out = out[~empty]

    dup = out.duplicated(subset=["review_text", "user_id", "product_id"], keep="first")
    stats["rows_removed_exact_duplicates"] = int(dup.sum())
    out = out[~dup].reset_index(drop=True)

    if out["review_id"].isna().all():
        out["review_id"] = [f"r{i}" for i in range(len(out))]
    stats["output_rows"] = len(out)
    log.info("cleaning stats: %s", stats)
    return out[ALL_COLUMNS], stats
