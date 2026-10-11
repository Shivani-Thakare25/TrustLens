from pathlib import Path
import json
import joblib
import pandas as pd

from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data" / "processed"

MODEL_DIR = ROOT / "models" / "fake_review_advanced"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


def main():

    # ============================================================
    # 1. LOAD DATA
    # ============================================================

    path = DATA / "fake_reviews_features.parquet"

    df = pd.read_parquet(path)

    print(f"Loaded {len(df)} reviews.")

    # Remove rows with empty review text
    df = df[
        df["clean_text"].fillna("").str.len() > 0
    ].copy()

    X_text = df["clean_text"].fillna("")

    y = df["label"]


    # ============================================================
    # 2. NUMERIC TEXT FEATURES
    # ============================================================

    numeric_columns = [
        "char_count",
        "word_count",
        "sentence_count",
        "avg_word_length",
        "exclamation_count",
        "question_count"
    ]

    numeric = (
        df[numeric_columns]
        .fillna(0)
        .astype(float)
    )

    # Normalize numeric features
    numeric = numeric / (numeric.max() + 1e-8)


    # ============================================================
    # 3. TRAIN / TEST SPLIT
    # ============================================================

    (
        X_train_text,
        X_test_text,
        X_train_num,
        X_test_num,
        y_train,
        y_test
    ) = train_test_split(
        X_text,
        numeric,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )


    # ============================================================
    # 4. WORD-LEVEL TF-IDF
    # ============================================================

    print("\nBuilding word TF-IDF...")

    word_vectorizer = TfidfVectorizer(
        max_features=60000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        strip_accents="unicode"
    )

    X_train_word = word_vectorizer.fit_transform(
        X_train_text
    )

    X_test_word = word_vectorizer.transform(
        X_test_text
    )


    # ============================================================
    # 5. CHARACTER-LEVEL TF-IDF
    # ============================================================

    print("Building character TF-IDF...")

    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        max_features=40000,
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True
    )

    X_train_char = char_vectorizer.fit_transform(
        X_train_text
    )

    X_test_char = char_vectorizer.transform(
        X_test_text
    )


    # ============================================================
    # 6. COMBINE ALL FEATURES
    # ============================================================

    X_train = hstack([
        X_train_word,
        X_train_char,
        csr_matrix(X_train_num.values)
    ]).tocsr()

    X_test = hstack([
        X_test_word,
        X_test_char,
        csr_matrix(X_test_num.values)
    ]).tocsr()

    print(
        f"\nFinal training matrix: {X_train.shape}"
    )

    print(
        f"Final testing matrix:  {X_test.shape}"
    )


    # ============================================================
    # 7. CREATE MODELS
    # ============================================================

    models = {}


    # Logistic Regression
    models["logistic_regression"] = LogisticRegression(
        max_iter=1500,
        C=2.0,
        random_state=42
    )


    # Random Forest
    models["random_forest"] = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        max_features="sqrt"
    )


    # XGBoost
    if XGBOOST_AVAILABLE:

        print("\nXGBoost detected.")

        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss"
        )

    else:

        print(
            "\nXGBoost is not installed. "
            "Skipping XGBoost."
        )


    # ============================================================
    # 8. XGBOOST LABEL ENCODING
    # ============================================================

    # Original labels remain CG / OR.
    # XGBoost receives 0 / 1 internally.

    sorted_labels = sorted(
        y.unique()
    )

    xgb_label_map = {
        sorted_labels[0]: 0,
        sorted_labels[1]: 1
    }

    xgb_inverse_label_map = {
        0: sorted_labels[0],
        1: sorted_labels[1]
    }


    # ============================================================
    # 9. TRAIN AND EVALUATE MODELS
    # ============================================================

    results = {}

    for name, model in models.items():

        print("\n" + "=" * 50)
        print(f"Training {name}")
        print("=" * 50)


        # --------------------------------------------------------
        # XGBoost
        # --------------------------------------------------------

        if name == "xgboost":

            y_train_encoded = y_train.map(
                xgb_label_map
            )

            model.fit(
                X_train,
                y_train_encoded
            )

            predictions_encoded = model.predict(
                X_test
            )

            predictions = [
                xgb_inverse_label_map[
                    int(value)
                ]
                for value in predictions_encoded
            ]


        # --------------------------------------------------------
        # Logistic Regression / Random Forest
        # --------------------------------------------------------

        else:

            model.fit(
                X_train,
                y_train
            )

            predictions = model.predict(
                X_test
            )


        # ========================================================
        # METRICS
        # ========================================================

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


        results[name] = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1)
        }


        # ========================================================
        # PRINT RESULTS
        # ========================================================

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


        # ========================================================
        # SAVE MODEL
        # ========================================================

        joblib.dump(
            model,
            MODEL_DIR / f"{name}.joblib"
        )


    # ============================================================
    # 10. SELECT BEST MODEL
    # ============================================================

    best_model = max(
        results,
        key=lambda name: results[name]["f1"]
    )


    # ============================================================
    # 11. SAVE VECTORIZERS
    # ============================================================

    joblib.dump(
        word_vectorizer,
        MODEL_DIR / "word_tfidf.joblib"
    )

    joblib.dump(
        char_vectorizer,
        MODEL_DIR / "char_tfidf.joblib"
    )


    # ============================================================
    # 12. SAVE LABEL MAPPING
    # ============================================================

    with open(
        MODEL_DIR / "label_mapping.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "original_labels": sorted_labels,
                "xgboost_encoding": xgb_label_map
            },
            f,
            indent=4
        )


    # ============================================================
    # 13. SAVE EVALUATION RESULTS
    # ============================================================

    with open(
        MODEL_DIR / "evaluation.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "models": results,
                "best_model": best_model,
                "random_state": 42,
                "test_size": 0.20,
                "dataset_rows": len(df),
                "features": {
                    "word_tfidf": True,
                    "character_tfidf": True,
                    "numeric_text_features": numeric_columns
                },
                "label_mapping": xgb_label_map
            },
            f,
            indent=4
        )


    # ============================================================
    # 14. FINAL OUTPUT
    # ============================================================

    print("\n" + "=" * 50)
    print("ADVANCED PHASE 4 COMPLETED")
    print("=" * 50)

    print(
        f"Best model: {best_model}"
    )

    print(
        f"Best F1: "
        f"{results[best_model]['f1']:.4f}"
    )

    print(
        f"Saved to: {MODEL_DIR}"
    )


if __name__ == "__main__":
    main()