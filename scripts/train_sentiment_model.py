from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data" / "processed"

MODEL_DIR = ROOT / "models" / "sentiment"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def main():

    # ============================================================
    # 1. LOAD DATA
    # ============================================================

    path = DATA / "sentiment_reviews_features.parquet"

    df = pd.read_parquet(path)

    print(f"Loaded {len(df)} sentiment reviews.")


    # ============================================================
    # 2. VALIDATE REQUIRED COLUMNS
    # ============================================================

    required_columns = [
        "combined_text",
        "Sentiment"
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )


    # ============================================================
    # 3. CLEAN DATA
    # ============================================================

    df["combined_text"] = (
        df["combined_text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["Sentiment"] = (
        df["Sentiment"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df = df[
        (df["combined_text"].str.len() > 0) &
        (df["Sentiment"].str.len() > 0)
    ].copy()

    print(
        f"Usable reviews: {len(df)}"
    )


    # ============================================================
    # 4. REMOVE EXACT DUPLICATES
    # ============================================================

    before = len(df)

    df = df.drop_duplicates(
        subset=["combined_text", "Sentiment"]
    )

    removed = before - len(df)

    print(
        f"Removed duplicate text-label pairs: {removed}"
    )

    print("\nSentiment distribution:")

    print(
        df["Sentiment"].value_counts()
    )


    # ============================================================
    # 5. FEATURES AND TARGET
    # ============================================================

    X = df["combined_text"]

    y = df["Sentiment"]


    # ============================================================
    # 6. TRAIN / TEST SPLIT
    # ============================================================

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )


    # ============================================================
    # 7. TF-IDF
    # ============================================================

    print("\nBuilding sentiment TF-IDF...")

    vectorizer = TfidfVectorizer(
        max_features=60000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        strip_accents="unicode"
    )

    X_train_tfidf = vectorizer.fit_transform(
        X_train
    )

    X_test_tfidf = vectorizer.transform(
        X_test
    )

    print(
        f"Training matrix: {X_train_tfidf.shape}"
    )

    print(
        f"Testing matrix:  {X_test_tfidf.shape}"
    )


    # ============================================================
    # 8. SENTIMENT MODEL
    # ============================================================

    print("\nTraining Logistic Regression...")

    model = LogisticRegression(
        max_iter=1500,
        C=2.0,
        class_weight="balanced",
        random_state=42
    )

    model.fit(
        X_train_tfidf,
        y_train
    )


    # ============================================================
    # 9. PREDICTIONS
    # ============================================================

    predictions = model.predict(
        X_test_tfidf
    )


    # ============================================================
    # 10. EVALUATION
    # ============================================================

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )


    print("\n" + "=" * 50)
    print("SENTIMENT MODEL RESULTS")
    print("=" * 50)

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1       : {f1:.4f}"
    )

    print("\nClassification report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )


    # ============================================================
    # 11. SAVE MODEL
    # ============================================================

    joblib.dump(
        model,
        MODEL_DIR / "sentiment_model.joblib"
    )

    joblib.dump(
        vectorizer,
        MODEL_DIR / "sentiment_tfidf.joblib"
    )


    # ============================================================
    # 12. SAVE EVALUATION
    # ============================================================

    evaluation = {
        "model": "logistic_regression",
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "dataset_rows": int(len(df)),
        "random_state": 42,
        "test_size": 0.20,
        "features": {
            "tfidf": True,
            "ngram_range": [1, 2],
            "max_features": 60000
        },
        "sentiment_classes": sorted(
            df["Sentiment"].unique().tolist()
        )
    }

    with open(
        MODEL_DIR / "evaluation.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            evaluation,
            f,
            indent=4
        )


    # ============================================================
    # 13. COMPLETION
    # ============================================================

    print("\n" + "=" * 50)
    print("PHASE 5 SENTIMENT MODEL COMPLETED")
    print("=" * 50)

    print(
        f"Saved to: {MODEL_DIR}"
    )


if __name__ == "__main__":
    main()
    