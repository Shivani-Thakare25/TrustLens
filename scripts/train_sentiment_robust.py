from pathlib import Path
import json
import re
import sys

import joblib
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)
from sklearn.model_selection import StratifiedGroupKFold


# ---------------------------------------------------------
# Project root
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA_PATH = ROOT / "data" / "processed" / "sentiment_reviews_features.parquet"
OUTPUT_DIR = ROOT / "models" / "sentiment_robust"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def normalize_group_text(value):
    """Normalize text used to identify repeated-review groups."""
    if pd.isna(value):
        return ""

    value = str(value).lower()
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def calculate_metrics(y_true, y_pred):
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_precision": float(precision_w),
        "weighted_recall": float(recall_w),
        "weighted_f1": float(f1_w),
        "macro_precision": float(precision_m),
        "macro_recall": float(recall_m),
        "macro_f1": float(f1_m),
    }


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
print("Loading processed sentiment data...")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Processed dataset not found:\n{DATA_PATH}\n"
        "Run scripts/build_features.py first."
    )

df = pd.read_parquet(DATA_PATH)

required = {"combined_text", "Sentiment"}

missing = required - set(df.columns)

if missing:
    raise ValueError(
        f"Missing required columns: {sorted(missing)}"
    )

print(f"Loaded rows: {len(df)}")


# ---------------------------------------------------------
# Clean data
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# Remove exact text-label duplicates
# ---------------------------------------------------------
before = len(df)

df = df.drop_duplicates(
    subset=["combined_text", "Sentiment"]
).reset_index(drop=True)

removed_duplicates = before - len(df)

print(f"Removed exact text-label duplicates: {removed_duplicates}")


# ---------------------------------------------------------
# Create leakage-control groups
#
# Prefer the original Review column.
# This prevents the same review text from appearing
# in both training and testing even if Summary differs.
# ---------------------------------------------------------
if "Review" in df.columns:

    df["_review_group"] = df["Review"].apply(
        normalize_group_text
    )

else:

    print(
        "WARNING: Review column not found. "
        "Using combined_text for grouping."
    )

    df["_review_group"] = df["combined_text"].apply(
        normalize_group_text
    )


# Empty Review values fall back to combined text.
empty_groups = df["_review_group"] == ""

df.loc[empty_groups, "_review_group"] = (
    df.loc[empty_groups, "combined_text"]
    .apply(normalize_group_text)
)

print(
    f"Unique leakage-control groups: "
    f"{df['_review_group'].nunique()}"
)


# ---------------------------------------------------------
# Display sentiment distribution
# ---------------------------------------------------------
print("\nOverall sentiment distribution:")
print(df["Sentiment"].value_counts())


# ---------------------------------------------------------
# Leakage-controlled 80/20 split
# ---------------------------------------------------------
X = df["combined_text"]
y = df["Sentiment"]
groups = df["_review_group"]

splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

train_idx, test_idx = next(
    splitter.split(X, y, groups)
)

train_df = df.iloc[train_idx].copy()
test_df = df.iloc[test_idx].copy()

print("\nLeakage-controlled split:")
print(f"Training rows: {len(train_df)}")
print(f"Testing rows : {len(test_df)}")

print(
    f"Training percentage: "
    f"{len(train_df) / len(df) * 100:.2f}%"
)

print(
    f"Testing percentage : "
    f"{len(test_df) / len(df) * 100:.2f}%"
)


# ---------------------------------------------------------
# Verify ZERO group leakage
# ---------------------------------------------------------
train_groups = set(train_df["_review_group"])
test_groups = set(test_df["_review_group"])

group_overlap = train_groups.intersection(test_groups)

print(
    f"\nTrain/test review-group overlap: "
    f"{len(group_overlap)}"
)

if group_overlap:
    raise RuntimeError(
        "DATA LEAKAGE DETECTED: "
        "review groups occur in both train and test."
    )

print("Leakage check: PASS")


# ---------------------------------------------------------
# Additional exact-text leakage check
# ---------------------------------------------------------
train_texts = set(
    train_df["combined_text"]
    .map(normalize_group_text)
)

test_texts = set(
    test_df["combined_text"]
    .map(normalize_group_text)
)

text_overlap = train_texts.intersection(test_texts)

print(
    f"Train/test exact combined-text overlap: "
    f"{len(text_overlap)}"
)

if text_overlap:
    raise RuntimeError(
        "DATA LEAKAGE DETECTED: "
        "exact combined review text occurs in both "
        "train and test."
    )

print("Exact-text leakage check: PASS")


# ---------------------------------------------------------
# Prepare text
# ---------------------------------------------------------
X_train = train_df["combined_text"].values
X_test = test_df["combined_text"].values

y_train = train_df["Sentiment"].values
y_test = test_df["Sentiment"].values


# ---------------------------------------------------------
# Word TF-IDF
# ---------------------------------------------------------
print("\nBuilding word TF-IDF...")

word_vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
    strip_accents="unicode",
)

X_train_word = word_vectorizer.fit_transform(X_train)
X_test_word = word_vectorizer.transform(X_test)

print(f"Word training matrix: {X_train_word.shape}")
print(f"Word testing matrix : {X_test_word.shape}")


# ---------------------------------------------------------
# Character TF-IDF
# ---------------------------------------------------------
print("\nBuilding character TF-IDF...")

char_vectorizer = TfidfVectorizer(
    analyzer="char",
    max_features=30000,
    ngram_range=(3, 5),
    min_df=2,
    sublinear_tf=True,
)

X_train_char = char_vectorizer.fit_transform(X_train)
X_test_char = char_vectorizer.transform(X_test)

print(f"Char training matrix: {X_train_char.shape}")
print(f"Char testing matrix : {X_test_char.shape}")


# ---------------------------------------------------------
# Combine features
# ---------------------------------------------------------
print("\nCombining word + character features...")

X_train_final = hstack(
    [X_train_word, X_train_char]
).tocsr()

X_test_final = hstack(
    [X_test_word, X_test_char]
).tocsr()

print(f"Final training matrix: {X_train_final.shape}")
print(f"Final testing matrix : {X_test_final.shape}")


# ---------------------------------------------------------
# Train Logistic Regression
# ---------------------------------------------------------
print("\nTraining Logistic Regression...")

model = LogisticRegression(
    C=2.0,
    max_iter=2000,
    class_weight="balanced",
    solver="lbfgs",
)

model.fit(X_train_final, y_train)


# ---------------------------------------------------------
# Evaluate
# ---------------------------------------------------------
print("\nEvaluating...")

y_pred = model.predict(X_test_final)

metrics = calculate_metrics(
    y_test,
    y_pred,
)

report = classification_report(
    y_test,
    y_pred,
    zero_division=0,
)


# ---------------------------------------------------------
# Print results
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("ROBUST SENTIMENT MODEL RESULTS")
print("=" * 60)

print(f"Accuracy          : {metrics['accuracy']:.4f}")
print(f"Weighted Precision : {metrics['weighted_precision']:.4f}")
print(f"Weighted Recall    : {metrics['weighted_recall']:.4f}")
print(f"Weighted F1       : {metrics['weighted_f1']:.4f}")
print(f"Macro Precision   : {metrics['macro_precision']:.4f}")
print(f"Macro Recall      : {metrics['macro_recall']:.4f}")
print(f"Macro F1          : {metrics['macro_f1']:.4f}")

print("\nClassification report:")
print(report)


# ---------------------------------------------------------
# Save model artifacts
# ---------------------------------------------------------
print("\nSaving model artifacts...")

joblib.dump(
    model,
    OUTPUT_DIR / "sentiment_robust_model.joblib",
)

joblib.dump(
    word_vectorizer,
    OUTPUT_DIR / "word_tfidf.joblib",
)

joblib.dump(
    char_vectorizer,
    OUTPUT_DIR / "char_tfidf.joblib",
)


# ---------------------------------------------------------
# Save evaluation
# ---------------------------------------------------------
evaluation = {
    "model": "logistic_regression_word_char_tfidf",
    "split": {
        "method": "StratifiedGroupKFold",
        "train_ratio_target": 0.80,
        "test_ratio_target": 0.20,
        "random_state": 42,
        "group_column": "_review_group",
    },
    "data": {
        "rows_after_cleaning": int(len(df)),
        "exact_duplicates_removed": int(removed_duplicates),
        "unique_review_groups": int(
            df["_review_group"].nunique()
        ),
        "training_rows": int(len(train_df)),
        "testing_rows": int(len(test_df)),
        "train_test_group_overlap": int(
            len(group_overlap)
        ),
        "train_test_exact_text_overlap": int(
            len(text_overlap)
        ),
    },
    "metrics": metrics,
    "classification_report": report,
}

with open(
    OUTPUT_DIR / "evaluation.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        evaluation,
        f,
        indent=2,
    )


# ---------------------------------------------------------
# Save integrity report
# ---------------------------------------------------------
integrity = {
    "status": "PASS",
    "leakage_checks": {
        "review_group_overlap": len(group_overlap),
        "exact_combined_text_overlap": len(text_overlap),
    },
    "train_rows": len(train_df),
    "test_rows": len(test_df),
    "train_percentage": (
        len(train_df) / len(df) * 100
    ),
    "test_percentage": (
        len(test_df) / len(df) * 100
    ),
}

with open(
    OUTPUT_DIR / "data_integrity_report.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        integrity,
        f,
        indent=2,
    )


print("\n" + "=" * 60)
print("ROBUST SENTIMENT TRAINING COMPLETED")
print("=" * 60)

print(
    f"Model saved to: "
    f"{OUTPUT_DIR / 'sentiment_robust_model.joblib'}"
)

print(
    f"Evaluation saved to: "
    f"{OUTPUT_DIR / 'evaluation.json'}"
)

print(
    f"Integrity report saved to: "
    f"{OUTPUT_DIR / 'data_integrity_report.json'}"
)