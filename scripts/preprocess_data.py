from pathlib import Path
import pandas as pd
import re
import json

ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

PROCESSED.mkdir(parents=True, exist_ok=True)


def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def process_fake_reviews():
    path = RAW / "fake reviews dataset.csv"

    df = pd.read_csv(path)

    required = {"category", "rating", "label", "text_"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns in fake-review dataset: {missing}")

    df["text"] = df["text_"].apply(clean_text)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].str.len() > 0]

    df = df.drop_duplicates()

    output = df[
        ["category", "rating", "label", "text"]
    ].reset_index(drop=True)

    output.to_parquet(
        PROCESSED / "fake_reviews_processed.parquet",
        index=False
    )

    return {
        "rows": len(output),
        "columns": list(output.columns)
    }


def process_sentiment_reviews():
    path = RAW / "Dataset-SA.csv"

    df = pd.read_csv(path)

    required = {
        "product_name",
        "product_price",
        "Rate",
        "Review",
        "Summary",
        "Sentiment"
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns in sentiment dataset: {missing}")

    df["Review"] = df["Review"].apply(clean_text)
    df["Summary"] = df["Summary"].apply(clean_text)

    df["Rate"] = pd.to_numeric(df["Rate"], errors="coerce")

    df = df.dropna(subset=["Sentiment"])

    # Keep rows having either review text or summary
    df = df[
        (df["Review"].str.len() > 0) |
        (df["Summary"].str.len() > 0)
    ]

    df = df.drop_duplicates()

    output = df[
        [
            "product_name",
            "product_price",
            "Rate",
            "Review",
            "Summary",
            "Sentiment"
        ]
    ].reset_index(drop=True)

    output.to_parquet(
        PROCESSED / "sentiment_reviews_processed.parquet",
        index=False
    )

    return {
        "rows": len(output),
        "columns": list(output.columns)
    }


def main():
    report = {
        "fake_reviews": process_fake_reviews(),
        "sentiment_reviews": process_sentiment_reviews()
    }

    with open(
        PROCESSED / "dataset_quality_report.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(report, f, indent=4)

    print("Data preprocessing completed.")
    print(json.dumps(report, indent=4))


if __name__ == "__main__":
    main()