"""
TrustLens - Final Leakage-Safe Sentiment Optimization

This experiment:
- Uses sentiment_reviews_features.parquet (Phase 3 output).
- Groups by exact normalized combined_text.
- Guarantees identical review text cannot cross train/test.
- Uses a training-only validation split for model selection.
- Keeps the real test set untouched until final evaluation.
- Tests Logistic Regression Neutral class weights.
- Also evaluates baseline Logistic C=2 and LinearSVC C=0.5
  on THIS SAME leakage-safe split.
- Does not generate synthetic data.
- Does not overwrite sentiment_robust or sentiment_compare.
"""

from pathlib import Path
import json
import re

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "sentiment_reviews_features.parquet"
)

OUTPUT_DIR = (
    ROOT
    / "models"
    / "sentiment_neutral_optimized"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

OUTER_SPLITS = 5
INNER_SPLITS = 5

WORD_MAX_FEATURES = 50_000
CHAR_MAX_FEATURES = 30_000

CLASS_ORDER = [
    "negative",
    "neutral",
    "positive",
]

NEUTRAL_WEIGHTS = [
    1.00,
    1.15,
    1.30,
    1.50,
    1.75,
    2.00,
]

MAX_VALIDATION_ACCURACY_DROP = 0.005


def normalize_text(value):
    if pd.isna(value):
        return ""

    value = str(value).lower()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[^\w\s]", "", value)

    return value.strip()


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            indent=2,
            ensure_ascii=False,
        )


def build_features(fit_texts, transform_texts):

    print("\nBuilding word TF-IDF...")

    word_vectorizer = TfidfVectorizer(
        max_features=WORD_MAX_FEATURES,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        max_df=0.98,
    )

    X_fit_word = word_vectorizer.fit_transform(
        fit_texts
    )

    X_transform_word = word_vectorizer.transform(
        transform_texts
    )

    print(
        f"Word matrix: {X_fit_word.shape}"
    )

    print(
        "\nBuilding character TF-IDF..."
    )

    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        max_features=CHAR_MAX_FEATURES,
        ngram_range=(3, 5),
        sublinear_tf=True,
        min_df=2,
        max_df=0.98,
    )

    X_fit_char = char_vectorizer.fit_transform(
        fit_texts
    )

    X_transform_char = char_vectorizer.transform(
        transform_texts
    )

    print(
        f"Character matrix: {X_fit_char.shape}"
    )

    X_fit = hstack(
        [
            X_fit_word,
            X_fit_char,
        ],
        format="csr",
    )

    X_transform = hstack(
        [
            X_transform_word,
            X_transform_char,
        ],
        format="csr",
    )

    print(
        f"Final matrix: {X_fit.shape}"
    )

    return (
        X_fit,
        X_transform,
        word_vectorizer,
        char_vectorizer,
    )


def calculate_metrics(y_true, y_pred):

    report = classification_report(
        y_true,
        y_pred,
        labels=CLASS_ORDER,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),

        "weighted_precision": float(
            precision_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),

        "weighted_recall": float(
            recall_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),

        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),

        "macro_precision": float(
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),

        "macro_recall": float(
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),

        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),

        "neutral_precision": float(
            precision_score(
                y_true,
                y_pred,
                labels=["neutral"],
                average="macro",
                zero_division=0,
            )
        ),

        "neutral_recall": float(
            recall_score(
                y_true,
                y_pred,
                labels=["neutral"],
                average="macro",
                zero_division=0,
            )
        ),

        "neutral_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=["neutral"],
                average="macro",
                zero_division=0,
            )
        ),

        "classification_report": report,
    }


print("=" * 70)
print(
    "TrustLens - Final Leakage-Safe "
    "Sentiment Optimization"
)
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

if not DATA_PATH.exists():

    raise FileNotFoundError(
        f"Required Phase 3 file not found:\n"
        f"{DATA_PATH}"
    )


print(
    "\nLoading Phase 3 sentiment features..."
)

df = pd.read_parquet(
    DATA_PATH
)

required_columns = {
    "combined_text",
    "Sentiment",
}

missing_columns = (
    required_columns
    - set(df.columns)
)

if missing_columns:

    raise ValueError(
        "Missing required columns: "
        f"{sorted(missing_columns)}"
    )


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
    .str.lower()
)

df = df[
    df["combined_text"].str.len().gt(0)
    & df["Sentiment"].isin(CLASS_ORDER)
].copy()

print(
    f"Loaded rows: {len(df)}"
)


# ============================================================
# REMOVE EXACT TEXT + LABEL DUPLICATES
# ============================================================

before = len(df)

df = df.drop_duplicates(
    subset=[
        "combined_text",
        "Sentiment",
    ]
).reset_index(drop=True)

removed = before - len(df)

print(
    "Removed exact text-label duplicates: "
    f"{removed}"
)


print(
    "\nSentiment distribution:"
)

print(
    df["Sentiment"].value_counts()
)


# ============================================================
# EXACT TEXT GROUP
# ============================================================

# This is the critical leakage fix.
#
# Every identical normalized combined review gets the
# SAME group ID.

df["_group"] = (
    df["combined_text"]
    .map(normalize_text)
)

print(
    "\nRows sharing an identical normalized "
    "text group: "
    f"{df['_group'].duplicated().sum()}"
)


# ============================================================
# OUTER TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print(
    "CREATING EXACT-TEXT GROUPED "
    "TRAIN/TEST SPLIT"
)
print("=" * 70)


outer_splitter = StratifiedGroupKFold(
    n_splits=OUTER_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

dummy = np.zeros(
    len(df)
)

y = df["Sentiment"].to_numpy()

groups = df["_group"].to_numpy()

train_indices, test_indices = next(
    outer_splitter.split(
        dummy,
        y,
        groups,
    )
)


train_df = (
    df.iloc[train_indices]
    .reset_index(drop=True)
)

test_df = (
    df.iloc[test_indices]
    .reset_index(drop=True)
)


train_groups = set(
    train_df["_group"]
)

test_groups = set(
    test_df["_group"]
)

group_overlap = (
    train_groups
    & test_groups
)


train_texts = set(
    train_df["combined_text"]
)

test_texts = set(
    test_df["combined_text"]
)

exact_text_overlap = (
    train_texts
    & test_texts
)


print(
    f"Training rows: {len(train_df)}"
)

print(
    f"Testing rows : {len(test_df)}"
)

print(
    "Training percentage: "
    f"{len(train_df) / len(df) * 100:.2f}%"
)

print(
    "Testing percentage: "
    f"{len(test_df) / len(df) * 100:.2f}%"
)

print(
    f"Review-group overlap: "
    f"{len(group_overlap)}"
)

print(
    f"Exact combined-text overlap: "
    f"{len(exact_text_overlap)}"
)


if group_overlap or exact_text_overlap:

    raise RuntimeError(
        "CRITICAL: Exact-text leakage remains "
        "after grouped split."
    )


print(
    "OUTER LEAKAGE CHECK: PASS"
)


# ============================================================
# INNER TRAIN / VALIDATION SPLIT
# ============================================================

print("\n" + "=" * 70)
print(
    "CREATING TRAINING-ONLY "
    "VALIDATION SPLIT"
)
print("=" * 70)


inner_splitter = StratifiedGroupKFold(
    n_splits=INNER_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

inner_dummy = np.zeros(
    len(train_df)
)

inner_y = (
    train_df["Sentiment"]
    .to_numpy()
)

inner_groups = (
    train_df["_group"]
    .to_numpy()
)

inner_train_indices, validation_indices = next(
    inner_splitter.split(
        inner_dummy,
        inner_y,
        inner_groups,
    )
)


inner_train_df = (
    train_df.iloc[inner_train_indices]
    .reset_index(drop=True)
)

validation_df = (
    train_df.iloc[validation_indices]
    .reset_index(drop=True)
)


inner_group_overlap = (
    set(inner_train_df["_group"])
    & set(validation_df["_group"])
)

inner_text_overlap = (
    set(
        inner_train_df["combined_text"]
    )
    &
    set(
        validation_df["combined_text"]
    )
)


print(
    f"Inner training rows: "
    f"{len(inner_train_df)}"
)

print(
    f"Validation rows    : "
    f"{len(validation_df)}"
)

print(
    f"Inner group overlap: "
    f"{len(inner_group_overlap)}"
)

print(
    f"Inner exact-text overlap: "
    f"{len(inner_text_overlap)}"
)


if (
    inner_group_overlap
    or inner_text_overlap
):

    raise RuntimeError(
        "CRITICAL: Leakage detected "
        "between training and validation."
    )


print(
    "INNER LEAKAGE CHECK: PASS"
)


# ============================================================
# BUILD VALIDATION FEATURES
# ============================================================

(
    X_inner_train,
    X_validation,
    _,
    _,
) = build_features(
    inner_train_df[
        "combined_text"
    ].tolist(),

    validation_df[
        "combined_text"
    ].tolist(),
)


y_inner_train = (
    inner_train_df["Sentiment"]
    .to_numpy()
)

y_validation = (
    validation_df["Sentiment"]
    .to_numpy()
)


# ============================================================
# VALIDATION SEARCH
# ============================================================

print("\n" + "=" * 70)
print(
    "VALIDATION-ONLY MODEL SELECTION"
)
print("=" * 70)


validation_results = []


for neutral_weight in NEUTRAL_WEIGHTS:

    class_weights = {
        "negative": 1.0,
        "neutral": neutral_weight,
        "positive": 1.0,
    }


    model = LogisticRegression(
        C=2.0,
        class_weight=class_weights,
        max_iter=2000,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )


    model.fit(
        X_inner_train,
        y_inner_train,
    )


    predictions = model.predict(
        X_validation
    )


    result = calculate_metrics(
        y_validation,
        predictions,
    )


    result[
        "neutral_weight"
    ] = float(
        neutral_weight
    )


    validation_results.append(
        result
    )


    print(
        f"\nNeutral weight = "
        f"{neutral_weight:.2f}"
    )

    print(
        f"Accuracy       : "
        f"{result['accuracy']:.4f}"
    )

    print(
        f"Macro F1       : "
        f"{result['macro_f1']:.4f}"
    )

    print(
        f"Weighted F1    : "
        f"{result['weighted_f1']:.4f}"
    )

    print(
        f"Neutral F1     : "
        f"{result['neutral_f1']:.4f}"
    )

    print(
        f"Neutral Recall : "
        f"{result['neutral_recall']:.4f}"
    )


# ============================================================
# SELECT CONFIGURATION
# ============================================================

best_validation_accuracy = max(
    result["accuracy"]
    for result in validation_results
)


accuracy_floor = (
    best_validation_accuracy
    - MAX_VALIDATION_ACCURACY_DROP
)


eligible_results = [
    result
    for result in validation_results
    if result["accuracy"]
    >= accuracy_floor
]


selected = max(
    eligible_results,
    key=lambda result: (
        result["macro_f1"],
        result["neutral_f1"],
        result["accuracy"],
    ),
)


selected_weight = (
    selected["neutral_weight"]
)


print("\n" + "=" * 70)
print(
    "SELECTED NEUTRAL WEIGHT"
)
print("=" * 70)

print(
    f"Neutral weight: "
    f"{selected_weight:.2f}"
)

print(
    f"Validation accuracy: "
    f"{selected['accuracy']:.4f}"
)

print(
    f"Validation Macro F1: "
    f"{selected['macro_f1']:.4f}"
)

print(
    f"Validation Neutral F1: "
    f"{selected['neutral_f1']:.4f}"
)


# ============================================================
# FAIR SAME-SPLIT BASELINES
# ============================================================

print("\n" + "=" * 70)
print(
    "SAME-SPLIT BASELINE COMPARISON"
)
print("=" * 70)


benchmark_results = {}


baseline_logistic = LogisticRegression(
    C=2.0,
    class_weight="balanced",
    max_iter=2000,
    solver="lbfgs",
    random_state=RANDOM_STATE,
)


baseline_logistic.fit(
    X_inner_train,
    y_inner_train,
)


baseline_predictions = (
    baseline_logistic.predict(
        X_validation
    )
)


benchmark_results[
    "logistic_C2_balanced"
] = calculate_metrics(
    y_validation,
    baseline_predictions,
)


baseline_svc = LinearSVC(
    C=0.5,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    max_iter=5000,
)


baseline_svc.fit(
    X_inner_train,
    y_inner_train,
)


svc_predictions = (
    baseline_svc.predict(
        X_validation
    )
)


benchmark_results[
    "linear_svc_C0.5_balanced"
] = calculate_metrics(
    y_validation,
    svc_predictions,
)


for name, result in (
    benchmark_results.items()
):

    print(
        f"\n{name}"
    )

    print(
        f"Accuracy   : "
        f"{result['accuracy']:.4f}"
    )

    print(
        f"Macro F1   : "
        f"{result['macro_f1']:.4f}"
    )

    print(
        f"Neutral F1 : "
        f"{result['neutral_f1']:.4f}"
    )


# ============================================================
# FINAL TRAINING
# ============================================================

print("\n" + "=" * 70)
print(
    "FINAL REFIT ON COMPLETE "
    "REAL TRAINING SET"
)
print("=" * 70)


(
    X_final_train,
    X_final_test,
    final_word_vectorizer,
    final_char_vectorizer,
) = build_features(
    train_df[
        "combined_text"
    ].tolist(),

    test_df[
        "combined_text"
    ].tolist(),
)


y_final_train = (
    train_df["Sentiment"]
    .to_numpy()
)

y_final_test = (
    test_df["Sentiment"]
    .to_numpy()
)


final_model = LogisticRegression(
    C=2.0,
    class_weight={
        "negative": 1.0,
        "neutral": selected_weight,
        "positive": 1.0,
    },
    max_iter=2000,
    solver="lbfgs",
    random_state=RANDOM_STATE,
)


final_model.fit(
    X_final_train,
    y_final_train,
)


# ============================================================
# FINAL TEST
# ============================================================

print("\n" + "=" * 70)
print(
    "FINAL EVALUATION ON "
    "UNTOUCHED REAL TEST SET"
)
print("=" * 70)


final_predictions = (
    final_model.predict(
        X_final_test
    )
)


final_metrics = calculate_metrics(
    y_final_test,
    final_predictions,
)


print(
    f"\nAccuracy           : "
    f"{final_metrics['accuracy']:.4f}"
)

print(
    f"Weighted Precision : "
    f"{final_metrics['weighted_precision']:.4f}"
)

print(
    f"Weighted Recall    : "
    f"{final_metrics['weighted_recall']:.4f}"
)

print(
    f"Weighted F1        : "
    f"{final_metrics['weighted_f1']:.4f}"
)

print(
    f"Macro Precision    : "
    f"{final_metrics['macro_precision']:.4f}"
)

print(
    f"Macro Recall       : "
    f"{final_metrics['macro_recall']:.4f}"
)

print(
    f"Macro F1           : "
    f"{final_metrics['macro_f1']:.4f}"
)

print(
    f"Neutral Precision  : "
    f"{final_metrics['neutral_precision']:.4f}"
)

print(
    f"Neutral Recall     : "
    f"{final_metrics['neutral_recall']:.4f}"
)

print(
    f"Neutral F1         : "
    f"{final_metrics['neutral_f1']:.4f}"
)


print(
    "\nClassification report:"
)

print(
    classification_report(
        y_final_test,
        final_predictions,
        labels=CLASS_ORDER,
        zero_division=0,
    )
)


# ============================================================
# SAVE ARTIFACTS
# ============================================================

joblib.dump(
    final_model,
    OUTPUT_DIR
    / "sentiment_neutral_optimized_model.joblib",
)


joblib.dump(
    final_word_vectorizer,
    OUTPUT_DIR
    / "word_tfidf.joblib",
)


joblib.dump(
    final_char_vectorizer,
    OUTPUT_DIR
    / "char_tfidf.joblib",
)


evaluation = {
    "model": "LogisticRegression",
    "C": 2.0,
    "selected_neutral_weight": float(
        selected_weight
    ),
    "training_rows": int(
        len(train_df)
    ),
    "testing_rows": int(
        len(test_df)
    ),
    "accuracy": final_metrics[
        "accuracy"
    ],
    "weighted_precision": final_metrics[
        "weighted_precision"
    ],
    "weighted_recall": final_metrics[
        "weighted_recall"
    ],
    "weighted_f1": final_metrics[
        "weighted_f1"
    ],
    "macro_precision": final_metrics[
        "macro_precision"
    ],
    "macro_recall": final_metrics[
        "macro_recall"
    ],
    "macro_f1": final_metrics[
        "macro_f1"
    ],
    "neutral_precision": final_metrics[
        "neutral_precision"
    ],
    "neutral_recall": final_metrics[
        "neutral_recall"
    ],
    "neutral_f1": final_metrics[
        "neutral_f1"
    ],
    "classification_report": final_metrics[
        "classification_report"
    ],
    "test_set_used_for_tuning": False,
    "synthetic_data_used": False,
}


save_json(
    OUTPUT_DIR / "evaluation.json",
    evaluation,
)


save_json(
    OUTPUT_DIR / "validation_results.json",
    {
        "selected_neutral_weight": float(
            selected_weight
        ),
        "best_validation_accuracy": float(
            best_validation_accuracy
        ),
        "accuracy_floor": float(
            accuracy_floor
        ),
        "results": validation_results,
        "same_split_benchmarks":
            benchmark_results,
    },
)


save_json(
    OUTPUT_DIR
    / "data_integrity_report.json",
    {
        "dataset": str(
            DATA_PATH
        ),
        "random_state": RANDOM_STATE,
        "grouping_key":
            "normalized_combined_text",
        "outer_split":
            "StratifiedGroupKFold",
        "inner_split":
            "StratifiedGroupKFold",
        "outer_group_overlap":
            len(group_overlap),
        "outer_exact_text_overlap":
            len(exact_text_overlap),
        "inner_group_overlap":
            len(inner_group_overlap),
        "inner_exact_text_overlap":
            len(inner_text_overlap),
        "test_set_used_for_tuning":
            False,
        "test_set_used_for_model_selection":
            False,
        "synthetic_data_used":
            False,
        "tfidf_fitted_on_test":
            False,
        "final_test_evaluated_after_selection":
            True,
    },
)


print("\n" + "=" * 70)
print("COMPLETED")
print("=" * 70)

print(
    f"Artifacts: {OUTPUT_DIR}"
)

print(
    "Integrity checks: PASS"
)

print(
    "Test data used for tuning: NO"
)

print(
    "Synthetic data used: NO"
)

print(
    "Exact-text leakage: NO"
)

print("=" * 70)