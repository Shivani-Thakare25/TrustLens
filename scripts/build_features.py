"""Extract handcrafted text features from cleaned review text.
Usage: python scripts/build_features.py <processed.csv> [--out data/processed/text_features.csv]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.text_features import build_text_feature_frame  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--out", default="data/processed/text_features.csv")
    a = ap.parse_args()

    df = pd.read_csv(a.data)
    text_col = "review_text" if "review_text" in df.columns else df.columns[0]
    feats = build_text_feature_frame(df[text_col].fillna(""))

    out_p = Path(a.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(out_p, index=False)
    print(f"wrote {len(feats)} feature rows to {out_p}")


if __name__ == "__main__":
    main()