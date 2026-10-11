"""
TrustLens Phase 5 - Aspect-Based Sentiment Analysis (ABSA)

Design:
- The Flipkart Dataset-SA source has overall sentiment labels but no ground-truth
  aspect annotations. Therefore this module does NOT claim supervised ABSA.
- Aspects are extracted with a controlled review-domain lexicon and synonym map.
- The existing TrustLens leakage-safe sentiment model is reused to classify the
  sentence/context containing each detected aspect.
- Confidence comes from the sentiment model probability, not from the aspect
  extractor.
- If no aspect is detected, the review remains valid but produces no aspect row.
- No test labels are used, no synthetic data is generated, and no model is
  retrained in this phase.

Expected project files:
  data/processed/sentiment_reviews_features.parquet
  models/sentiment_neutral_optimized/sentiment_neutral_optimized_model.joblib
  models/sentiment_neutral_optimized/word_tfidf.joblib
  models/sentiment_neutral_optimized/char_tfidf.joblib
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack


# Canonical aspect -> phrases that indicate that aspect.
# Phrases are deliberately conservative: they should identify an aspect,
# not express sentiment by themselves.
ASPECT_LEXICON: Dict[str, Tuple[str, ...]] = {
    "price": ("price", "pricing", "cost", "expensive", "cheap", "affordable"),
    "value_for_money": ("value for money", "value-for-money", "worth the money", "worth it"),
    "quality": ("quality", "build quality", "product quality"),
    "design": ("design", "look", "looks", "appearance", "style"),
    "build": ("build", "build quality", "construction"),
    "material": ("material", "materials", "fabric", "plastic", "metal"),
    "durability": ("durability", "durable", "lasts", "long lasting", "long-lasting"),
    "performance": ("performance", "performs", "speed", "fast", "slow"),
    "features": ("feature", "features", "function", "functions", "functionality"),
    "ease_of_use": ("easy to use", "ease of use", "simple to use", "user friendly", "user-friendly"),
    "usability": ("usability", "controls", "control"),
    "battery": ("battery", "battery life", "backup", "power backup"),
    "charging": ("charging", "charge", "charger", "charging speed"),
    "camera": ("camera", "cameras", "photo", "photos", "photography", "selfie"),
    "display": ("display", "screen", "screens", "touchscreen", "touch screen"),
    "audio": ("sound", "audio", "speaker", "speakers", "volume", "bass"),
    "connectivity": ("wifi", "wi-fi", "bluetooth", "connectivity", "network", "signal"),
    "software": ("software", "app", "application", "apps", "update", "updates"),
    "storage": ("storage", "memory", "internal memory", "ram"),
    "processor": ("processor", "cpu", "chipset", "processor speed"),
    "size": ("size", "dimension", "dimensions"),
    "weight": ("weight", "heavy", "lightweight", "light weight"),
    "comfort": ("comfort", "comfortable", "uncomfortable"),
    "fit": ("fit", "fitting", "fits", "size fit"),
    "packaging": ("packaging", "package", "packing", "box"),
    "delivery": ("delivery", "shipping", "shipment", "delivered", "delivery time"),
    "service": ("service", "customer service", "support", "after sales", "after-sales"),
    "installation": ("installation", "install", "installed"),
    "replacement": ("replacement", "replace", "replaced"),
    "return": ("return", "returns", "returned", "refund"),
    "taste": ("taste", "flavour", "flavor"),
    "smell": ("smell", "fragrance", "odor", "odour"),
    "quantity": ("quantity", "amount", "portion"),
    "freshness": ("fresh", "freshness"),
}

# Longest phrase first prevents a shorter phrase from stealing a match.
COMPILED_ASPECTS = {
    aspect: tuple(
        re.compile(r"(?<![a-z0-9])" + re.escape(phrase.lower()) + r"(?![a-z0-9])")
        for phrase in sorted(phrases, key=len, reverse=True)
    )
    for aspect, phrases in ASPECT_LEXICON.items()
}

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).replace("\r", " ").replace("\n", " ")
    return WHITESPACE_RE.sub(" ", text).strip()


def split_sentences(text: str) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    return [s.strip() for s in SENTENCE_RE.split(text) if s.strip()]


def detect_aspects(text: str) -> List[Tuple[str, str]]:
    """
    Return (canonical_aspect, matched_phrase), de-duplicated by aspect.
    """
    text_lower = text.lower()
    found: List[Tuple[str, str]] = []

    for aspect, patterns in COMPILED_ASPECTS.items():
        for pattern in patterns:
            match = pattern.search(text_lower)
            if match:
                found.append((aspect, match.group(0)))
                break

    return found


def find_aspect_context(text: str, aspect: str) -> str:
    """
    Return the sentence containing a phrase belonging to the canonical aspect.
    If the aspect appears in multiple sentences, combine the matching sentences
    up to a small fixed context limit.
    """
    sentences = split_sentences(text)
    patterns = COMPILED_ASPECTS[aspect]

    matches = []
    for sentence in sentences:
        lowered = sentence.lower()
        if any(pattern.search(lowered) for pattern in patterns):
            matches.append(sentence)

    if matches:
        return " ".join(matches[:2])

    # This should be rare because detect_aspects() already found the aspect.
    return text[:500]


def load_sentiment_components(project_root: Path):
    model_dir = project_root / "models" / "sentiment_neutral_optimized"

    required = [
        model_dir / "sentiment_neutral_optimized_model.joblib",
        model_dir / "word_tfidf.joblib",
        model_dir / "char_tfidf.joblib",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Required Phase 5 sentiment artifacts are missing:\n"
            + "\n".join(missing)
            + "\nRun the completed sentiment training pipeline first."
        )

    model = joblib.load(required[0])
    word_vectorizer = joblib.load(required[1])
    char_vectorizer = joblib.load(required[2])
    return model, word_vectorizer, char_vectorizer


def predict_contexts(
    contexts: List[str],
    model,
    word_vectorizer,
    char_vectorizer,
    batch_size: int = 2048,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict sentiment and confidence for contexts.

    The vectorizers are already fitted by the leakage-safe sentiment pipeline;
    no fitting occurs here.
    """
    labels: List[np.ndarray] = []
    confidences: List[np.ndarray] = []

    for start in range(0, len(contexts), batch_size):
        batch = contexts[start : start + batch_size]

        word_matrix = word_vectorizer.transform(batch)
        char_matrix = char_vectorizer.transform(batch)
        matrix = hstack([word_matrix, char_matrix], format="csr")

        predictions = model.predict(matrix)

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(matrix)
            confidence = probabilities.max(axis=1)
        else:
            # The final TrustLens sentiment model is LogisticRegression, so this
            # branch is defensive only and must not fabricate probabilities.
            decision = model.decision_function(matrix)
            if decision.ndim == 1:
                confidence = 1.0 / (1.0 + np.exp(-np.abs(decision)))
            else:
                shifted = decision - decision.max(axis=1, keepdims=True)
                exp_scores = np.exp(shifted)
                probabilities = exp_scores / exp_scores.sum(axis=1, keepdims=True)
                confidence = probabilities.max(axis=1)

        labels.append(np.asarray(predictions))
        confidences.append(np.asarray(confidence, dtype=float))

    return np.concatenate(labels), np.concatenate(confidences)


def process(
    project_root: Path,
    input_path: Path,
    output_dir: Path,
    limit: int | None = None,
) -> dict:
    print("=" * 70)
    print("TrustLens - Phase 5 Aspect-Based Sentiment Analysis")
    print("=" * 70)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Loading: {input_path}")
    df = pd.read_parquet(input_path)

    required = {"combined_text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if limit is not None:
        df = df.head(limit).copy()

    df["combined_text"] = df["combined_text"].map(normalize_text)
    df = df[df["combined_text"].str.len() > 0].copy()

    print(f"Rows to analyze: {len(df)}")

    model, word_vectorizer, char_vectorizer = load_sentiment_components(project_root)

    # Deduplicate contexts before inference, then restore them to all rows.
    # This reduces repeated computation without changing the output.
    context_records = []
    row_records = []
    unique_contexts: Dict[Tuple[str, str], int] = {}

    for row_id, text in enumerate(df["combined_text"].tolist()):
        aspects = detect_aspects(text)

        for aspect, matched_phrase in aspects:
            context = find_aspect_context(text, aspect)
            key = (aspect, context)

            if key not in unique_contexts:
                unique_contexts[key] = len(context_records)
                context_records.append(
                    {
                        "aspect": aspect,
                        "matched_phrase": matched_phrase,
                        "context": context,
                    }
                )

            row_records.append(
                {
                    "source_row": row_id,
                    "aspect": aspect,
                    "matched_phrase": matched_phrase,
                    "context_key": key,
                    "context_index": unique_contexts[key],
                }
            )

    print(f"Unique aspect-contexts: {len(context_records)}")
    print(f"Aspect occurrences: {len(row_records)}")

    aspect_rows = []

    if context_records:
        contexts = [x["context"] for x in context_records]
        predicted, confidence = predict_contexts(
            contexts, model, word_vectorizer, char_vectorizer
        )

        for item in row_records:
            idx = item["context_index"]
            aspect_rows.append(
                {
                    "source_row": int(item["source_row"]),
                    "aspect": item["aspect"],
                    "matched_phrase": item["matched_phrase"],
                    "aspect_context": item["context_key"][1],
                    "sentiment": str(predicted[idx]),
                    "confidence": round(float(confidence[idx]), 6),
                    "method": "lexicon_context_with_trained_sentiment_model",
                }
            )

    aspect_df = pd.DataFrame(aspect_rows)

    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "absa_results.parquet"
    aspect_df.to_parquet(output_path, index=False)

    summary = {
        "input_rows": int(len(df)),
        "rows_with_at_least_one_aspect": int(
            aspect_df["source_row"].nunique() if not aspect_df.empty else 0
        ),
        "rows_without_detected_aspect": int(
            len(df) - (aspect_df["source_row"].nunique() if not aspect_df.empty else 0)
        ),
        "aspect_occurrences": int(len(aspect_df)),
        "unique_aspects": int(aspect_df["aspect"].nunique() if not aspect_df.empty else 0),
        "aspect_distribution": (
            aspect_df["aspect"].value_counts().to_dict()
            if not aspect_df.empty
            else {}
        ),
        "aspect_sentiment_distribution": (
            aspect_df["sentiment"].value_counts().to_dict()
            if not aspect_df.empty
            else {}
        ),
        "mean_confidence": (
            round(float(aspect_df["confidence"].mean()), 6)
            if not aspect_df.empty
            else None
        ),
        "method": (
            "weakly supervised/rule-assisted ABSA: controlled aspect lexicon "
            "plus sentence-context sentiment inference using the validated "
            "TrustLens sentiment model"
        ),
        "ground_truth_aspect_labels_available": False,
        "synthetic_data_used": False,
        "test_labels_used_for_tuning": False,
    }

    with open(output_dir / "absa_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Human-readable aspect summary.
    if not aspect_df.empty:
        pivot = (
            aspect_df.groupby(["aspect", "sentiment"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        pivot.to_csv(output_dir / "aspect_sentiment_summary.csv", index=False)

    print()
    print("=" * 70)
    print("PHASE 5 ABSA COMPLETED")
    print("=" * 70)
    print(f"ABSA results       : {output_path}")
    print(f"Summary            : {output_dir / 'absa_summary.json'}")
    print(f"Unique aspects     : {summary['unique_aspects']}")
    print(f"Aspect occurrences : {summary['aspect_occurrences']}")
    print(f"Mean confidence    : {summary['mean_confidence']}")
    print("Ground-truth aspects: NOT available in source dataset")
    print("Method             : rule-assisted aspect extraction + trained sentiment model")
    print("Synthetic data     : NO")
    print("Test tuning        : NO")
    print("=" * 70)

    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Run TrustLens Phase 5 ABSA.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="TrustLens project root.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input sentiment features parquet.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for a smoke test. Omit for the complete dataset.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    project_root = args.project_root.resolve()
    input_path = (
        args.input.resolve()
        if args.input
        else project_root / "data" / "processed" / "sentiment_reviews_features.parquet"
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else project_root / "models" / "absa"
    )

    try:
        process(project_root, input_path, output_dir, args.limit)
    except Exception as exc:
        print(f"\nPHASE 5 ABSA FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
