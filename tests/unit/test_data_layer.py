import pandas as pd
import pytest

from src.data.cleaning import clean_dataframe
from src.data.ingestion import load_raw
from src.data.validation import validate_dataframe


def _df():
    return pd.DataFrame(
        {
            "review_text": ["Great phone!", "  ", "Great phone!", "ok  item"],
            "rating": [5, 3, 9, "abc"],
            "review_date": ["2024-01-02", "bad", None, "2024-02-01"],
            "label": [0, 1, 7, None],
        }
    )


def test_validation_counts_problems():
    r = validate_dataframe(_df())
    assert r.n_rows == 4 and r.ok
    assert r.missing_text == 1
    assert r.invalid_ratings == 2
    assert r.invalid_dates == 1
    assert r.invalid_labels == 1
    assert r.duplicate_rows == 1


def test_validation_missing_required_column():
    r = validate_dataframe(pd.DataFrame({"rating": [1]}))
    assert not r.ok and r.missing_required_columns == ["review_text"]


def test_cleaning_reports_and_nulls_not_fabricates():
    out, stats = clean_dataframe(_df())
    assert stats["rows_removed_empty_text"] == 1
    assert stats["rows_removed_exact_duplicates"] == 1
    assert stats["ratings_set_null"] == 2
    assert out["product_id"].isna().all()  # absent field stays null
    assert out["review_text"].tolist() == ["Great phone!", "ok item"]
    assert out["review_id"].notna().all()


def test_load_raw_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw(tmp_path / "nope.csv")
    p = tmp_path / "x.txt"
    p.write_text("a")
    with pytest.raises(ValueError):
        load_raw(p)


def test_load_raw_csv_skips_malformed(tmp_path):
    p = tmp_path / "r.csv"
    p.write_text("review_text,rating\nfine,5\nbad,1,extra,cols\nalso fine,4\n")
    df = load_raw(p)
    assert len(df) == 2
