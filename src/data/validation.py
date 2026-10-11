"""Validation: reports counts of affected records; does not silently drop data."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.data.schema import REQUIRED_COLUMNS, VALID_LABELS


@dataclass
class ValidationReport:
    n_rows: int
    missing_required_columns: list = field(default_factory=list)
    missing_text: int = 0
    invalid_ratings: int = 0
    invalid_dates: int = 0
    invalid_labels: int = 0
    duplicate_rows: int = 0

    @property
    def ok(self) -> bool:
        return not self.missing_required_columns

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def validate_dataframe(
    df: pd.DataFrame, min_rating: int = 1, max_rating: int = 5
) -> ValidationReport:
    """Count problems in df. Never raises; the caller decides what to do."""
    report = ValidationReport(n_rows=len(df))
    report.missing_required_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if report.missing_required_columns:
        return report

    text = df["review_text"]
    report.missing_text = int((text.isna() | (text.astype(str).str.strip() == "")).sum())

    if "rating" in df.columns:
        r = pd.to_numeric(df["rating"], errors="coerce")
        out_of_range = (r < min_rating) | (r > max_rating)
        unparseable = r.isna() & df["rating"].notna()
        report.invalid_ratings = int((out_of_range | unparseable).sum())
    if "review_date" in df.columns:
        d = pd.to_datetime(df["review_date"], errors="coerce")
        report.invalid_dates = int((d.isna() & df["review_date"].notna()).sum())
    if "label" in df.columns:
        lab = pd.to_numeric(df["label"], errors="coerce")
        report.invalid_labels = int((~lab.isin(VALID_LABELS) & df["label"].notna()).sum())
    report.duplicate_rows = int(df.duplicated(subset=["review_text"]).sum())
    return report
