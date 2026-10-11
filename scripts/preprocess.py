"""Usage: python scripts/preprocess.py <raw.csv> [--out data/processed/reviews.csv] [--stats reports/cleaning_stats.json]"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from src.data.cleaning import clean_dataframe  # noqa: E402
from src.data.ingestion import load_raw  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--out", default="data/processed/reviews.csv")
    ap.add_argument("--stats", default="reports/cleaning_stats.json")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(Path(__file__).resolve().parents[1] / "configs/data_config.yaml"))
    df, stats = clean_dataframe(
        load_raw(a.data), cfg["min_rating"], cfg["max_rating"], cfg["max_text_chars"]
    )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.stats).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    Path(a.stats).write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
