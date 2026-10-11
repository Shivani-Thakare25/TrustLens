from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models" / "fake_review"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


def main():

    path = DATA / "fake_reviews_features.parquet"

    df = pd.read_parquet(path)

    print(f"Loaded {len(df)} reviews.")

    # Preserve original labels.
    # We do not assume CG/OR means fake/genuine here.
    X = df["clean_text"].fillna("")
    y = df["label"]

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    models = {
        "logistic_regression": Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=50000,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True
                )
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42
                )
            )
        ]),

        "random_forest": Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=20000,
                    ngram_range=(1, 2),
                    min_df=2
                )
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    n_jobs=-1
                )
            )
        ])
    }

    results = {}

    for name, model in models.items():

        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
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

        results[name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1       : {f1:.4f}")

        print("\nClassification report:")
        print(classification_report(
            y_test,
            predictions,
            zero_division=0
        ))

        joblib.dump(
            model,
            MODEL_DIR / f"{name}.joblib"
        )

    # Select model using F1
    best_model = max(
        results,
        key=lambda name: results[name]["f1"]
    )

    results["best_model"] = best_model

    with open(
        MODEL_DIR / "evaluation.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(results, f, indent=4)

    print("\n==============================")
    print("PHASE 4 COMPLETED")
    print("==============================")
    print(f"Best model: {best_model}")
    print(f"Saved to: {MODEL_DIR}")


if __name__ == "__main__":
    main()