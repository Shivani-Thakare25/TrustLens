from pathlib import Path
import json
import re
import sys

import joblib
import pandas as pd
from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)
from sklearn.model_selection import StratifiedGroupKFold


# =========================================================
# PROJECT PATHS
# =========================================================

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "sentiment_reviews_features.parquet"
)

OUTPUT_DIR = ROOT / "models" / "sentiment_compare"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# HELPERS
# =========================================================

def normalize_group_text(value):
    if pd.isna(value):
        return ""

    value = str(value).lower()
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def calculate_metrics(y_true, y_pred):

    precision_w, recall_w, f1_w, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )
    )

    precision_m, recall_m, f1_m, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )

    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "weighted_precision": float(precision_w),
        "weighted_recall": float(recall_w),
        "weighted_f1": float(f1_w),
        "macro_precision": float(precision_m),
        "macro_recall": float(recall_m),
        "macro_f1": float(f1_m),
    }


# =========================================================
# LOAD DATA
# =========================================================

print("Loading processed sentiment data...")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_PATH}\n"
        "Run scripts/build_features.py first."
    )

df = pd.read_parquet(DATA_PATH)

required = {
    "combined_text",
    "Sentiment",
}

missing = required - set(df.columns)

if missing:
    raise ValueError(
        f"Missing columns: {sorted(missing)}"
    )

print(f"Loaded rows: {len(df)}")


# =========================================================
# CLEAN
# =========================================================

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
    (df["combined_text"] != "")
    & (df["Sentiment"] != "")
].copy()

print(f"Usable rows: {len(df)}")


# =========================================================
# REMOVE EXACT DUPLICATE TEXT-LABEL PAIRS
# =========================================================

before = len(df)

df = df.drop_duplicates(
    subset=[
        "combined_text",
        "Sentiment",
    ]
).reset_index(drop=True)

duplicates_removed = before - len(df)

print(
    f"Removed exact text-label duplicates: "
    f"{duplicates_removed}"
)


# =========================================================
# LEAKAGE CONTROL GROUP
# =========================================================

if "Review" in df.columns:

    df["_review_group"] = (
        df["Review"]
        .apply(normalize_group_text)
    )

else:

    print(
        "WARNING: Review column unavailable. "
        "Using combined_text as group."
    )

    df["_review_group"] = (
        df["combined_text"]
        .apply(normalize_group_text)
    )


# Empty review → combined text
empty_groups = (
    df["_review_group"] == ""
)

df.loc[empty_groups, "_review_group"] = (
    df.loc[empty_groups, "combined_text"]
    .apply(normalize_group_text)
)


# =========================================================
# DATA DISTRIBUTION
# =========================================================

print("\nSentiment distribution:")

print(
    df["Sentiment"]
    .value_counts()
)


# =========================================================
# SAME 80/20 LEAKAGE-CONTROLLED SPLIT
# =========================================================

X = df["combined_text"]
y = df["Sentiment"]
groups = df["_review_group"]

splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

train_idx, test_idx = next(
    splitter.split(
        X,
        y,
        groups,
    )
)

train_df = df.iloc[train_idx].copy()
test_df = df.iloc[test_idx].copy()

print("\nSplit:")
print(
    f"Training rows: {len(train_df)}"
)
print(
    f"Testing rows : {len(test_df)}"
)

print(
    f"Training percentage: "
    f"{len(train_df) / len(df) * 100:.2f}%"
)

print(
    f"Testing percentage: "
    f"{len(test_df) / len(df) * 100:.2f}%"
)


# =========================================================
# LEAKAGE CHECK
# =========================================================

train_groups = set(
    train_df["_review_group"]
)

test_groups = set(
    test_df["_review_group"]
)

group_overlap = (
    train_groups
    .intersection(test_groups)
)

print(
    f"\nReview-group overlap: "
    f"{len(group_overlap)}"
)

if group_overlap:
    raise RuntimeError(
        "DATA LEAKAGE DETECTED."
    )


train_texts = set(
    train_df["combined_text"]
    .map(normalize_group_text)
)

test_texts = set(
    test_df["combined_text"]
    .map(normalize_group_text)
)

text_overlap = (
    train_texts
    .intersection(test_texts)
)

print(
    f"Exact combined-text overlap: "
    f"{len(text_overlap)}"
)

if text_overlap:
    raise RuntimeError(
        "EXACT TEXT LEAKAGE DETECTED."
    )

print("Leakage checks: PASS")


# =========================================================
# TEXT DATA
# =========================================================

X_train = (
    train_df["combined_text"]
    .values
)

X_test = (
    test_df["combined_text"]
    .values
)

y_train = (
    train_df["Sentiment"]
    .values
)

y_test = (
    test_df["Sentiment"]
    .values
)


# =========================================================
# WORD TF-IDF
# =========================================================

print("\nBuilding word TF-IDF...")

word_vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
    strip_accents="unicode",
)

X_train_word = (
    word_vectorizer
    .fit_transform(X_train)
)

X_test_word = (
    word_vectorizer
    .transform(X_test)
)

print(
    f"Word matrix: "
    f"{X_train_word.shape}"
)


# =========================================================
# CHARACTER TF-IDF
# =========================================================

print("\nBuilding character TF-IDF...")

char_vectorizer = TfidfVectorizer(
    analyzer="char",
    max_features=30000,
    ngram_range=(3, 5),
    min_df=2,
    sublinear_tf=True,
)

X_train_char = (
    char_vectorizer
    .fit_transform(X_train)
)

X_test_char = (
    char_vectorizer
    .transform(X_test)
)

print(
    f"Character matrix: "
    f"{X_train_char.shape}"
)


# =========================================================
# COMBINE
# =========================================================

X_train_final = hstack(
    [
        X_train_word,
        X_train_char,
    ]
).tocsr()

X_test_final = hstack(
    [
        X_test_word,
        X_test_char,
    ]
).tocsr()

print(
    f"\nFinal matrix: "
    f"{X_train_final.shape}"
)


# =========================================================
# MODELS
# =========================================================

models = {

    "logistic_C_0.5": LogisticRegression(
        C=0.5,
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    ),

    "logistic_C_1.0": LogisticRegression(
        C=1.0,
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    ),

    "logistic_C_2.0": LogisticRegression(
        C=2.0,
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    ),

    "logistic_C_4.0": LogisticRegression(
        C=4.0,
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    ),

    "linear_svc_C_0.5": LinearSVC(
        C=0.5,
        class_weight="balanced",
        max_iter=5000,
    ),

    "linear_svc_C_1.0": LinearSVC(
        C=1.0,
        class_weight="balanced",
        max_iter=5000,
    ),

    "linear_svc_C_2.0": LinearSVC(
        C=2.0,
        class_weight="balanced",
        max_iter=5000,
    ),

    "linear_svc_C_4.0": LinearSVC(
        C=4.0,
        class_weight="balanced",
        max_iter=5000,
    ),
}


# =========================================================
# TRAIN + EVALUATE
# =========================================================

results = {}

best_model_name = None
best_model = None
best_macro_f1 = -1

for name, model in models.items():

    print("\n" + "=" * 60)

    print(
        f"Training: {name}"
    )

    print("=" * 60)

    model.fit(
        X_train_final,
        y_train,
    )

    predictions = model.predict(
        X_test_final
    )

    metrics = calculate_metrics(
        y_test,
        predictions,
    )

    print(
        f"Accuracy          : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Weighted F1       : "
        f"{metrics['weighted_f1']:.4f}"
    )

    print(
        f"Macro F1          : "
        f"{metrics['macro_f1']:.4f}"
    )

    print("\nClassification report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    results[name] = metrics

    # Primary selection criterion:
    # Macro F1
    if metrics["macro_f1"] > best_macro_f1:

        best_macro_f1 = (
            metrics["macro_f1"]
        )

        best_model_name = name
        best_model = model


# =========================================================
# BEST MODEL
# =========================================================

print("\n" + "=" * 60)

print(
    f"BEST MODEL: "
    f"{best_model_name}"
)

print("=" * 60)

print(
    json.dumps(
        results[best_model_name],
        indent=2,
    )
)


# =========================================================
# SAVE BEST MODEL
# =========================================================

joblib.dump(
    best_model,
    OUTPUT_DIR
    / "best_sentiment_model.joblib",
)

joblib.dump(
    word_vectorizer,
    OUTPUT_DIR
    / "word_tfidf.joblib",
)

joblib.dump(
    char_vectorizer,
    OUTPUT_DIR
    / "char_tfidf.joblib",
)


# =========================================================
# SAVE ALL RESULTS
# =========================================================

evaluation = {

    "experiment":
        "sentiment_model_comparison",

    "split": {

        "method":
            "StratifiedGroupKFold",

        "target_train_ratio":
            0.80,

        "target_test_ratio":
            0.20,

        "random_state":
            42,

        "group_column":
            "_review_group",
    },

    "data": {

        "rows_after_cleaning":
            len(df),

        "exact_duplicates_removed":
            duplicates_removed,

        "training_rows":
            len(train_df),

        "testing_rows":
            len(test_df),

        "unique_review_groups":
            df["_review_group"].nunique(),

        "review_group_overlap":
            len(group_overlap),

        "exact_text_overlap":
            len(text_overlap),
    },

    "best_model":
        best_model_name,

    "best_macro_f1":
        best_macro_f1,

    "results":
        results,
}


with open(
    OUTPUT_DIR / "comparison_results.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        evaluation,
        f,
        indent=2,
    )


# =========================================================
# FINISHED
# =========================================================

print("\n" + "=" * 60)
print("SENTIMENT MODEL COMPARISON COMPLETED")
print("=" * 60)

print(
    f"Best model: "
    f"{best_model_name}"
)

print(
    f"Best Macro F1: "
    f"{best_macro_f1:.4f}"
)

print(
    f"\nResults saved to:"
    f"\n{OUTPUT_DIR}"
)