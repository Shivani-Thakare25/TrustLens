"""
TrustLens - Phase 6 Similarity Engine
Final scalable, leakage-safe review similarity analysis.

Purpose:
- Detect exact duplicate review text.
- Detect highly similar review pairs without O(n^2) all-pairs comparison.
- Produce similarity evidence for downstream suspicion-risk scoring.
- Avoid using labels or test-set information.

Method:
1. Normalize review text.
2. Detect exact normalized-text duplicates.
3. Build deterministic character n-gram TF-IDF signatures.
4. Use sparse nearest-neighbor search in batches.
5. Keep only high-similarity pairs.
6. Aggregate per-review similarity statistics.

This is an evidence engine, not a fraud classifier.
A high similarity score means reviews are similar; it does NOT prove fraud.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


DEFAULT_INPUT = "data/processed/sentiment_reviews_features.parquet"
DEFAULT_OUTPUT = "models/similarity"

# High threshold intentionally favors strong near-duplicate evidence.
SIMILARITY_THRESHOLD = 0.82
NGRAM_RANGE = (3, 5)
MAX_FEATURES = 100_000
TOP_K = 5
BATCH_SIZE = 2_000


def clean_text(text: object) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_review_text(df: pd.DataFrame) -> pd.Series:
    review = df["Review"].fillna("").astype(str) if "Review" in df else pd.Series("", index=df.index)
    summary = df["Summary"].fillna("").astype(str) if "Summary" in df else pd.Series("", index=df.index)

    combined = (summary.str.strip() + " " + review.str.strip()).str.strip()
    return combined.map(clean_text)


def exact_duplicate_groups(texts: pd.Series) -> tuple[pd.Series, pd.DataFrame]:
    group_sizes = texts.groupby(texts).transform("size")
    duplicate_mask = (texts != "") & (group_sizes > 1)

    duplicate_rows = pd.DataFrame({
        "text": texts[duplicate_mask],
    })

    if duplicate_rows.empty:
        return group_sizes, pd.DataFrame(
            columns=["normalized_text", "duplicate_count"]
        )

    summary = (
        duplicate_rows.groupby("text")
        .size()
        .reset_index(name="duplicate_count")
        .rename(columns={"text": "normalized_text"})
        .sort_values("duplicate_count", ascending=False)
        .reset_index(drop=True)
    )
    return group_sizes, summary


def batched_similarity_search(
    matrix,
    threshold: float,
    top_k: int,
    batch_size: int,
) -> pd.DataFrame:
    """
    Sparse cosine nearest-neighbor search.

    NearestNeighbors returns cosine distance, so:
        similarity = 1 - distance

    Self-pairs are removed and each pair is stored once.
    """
    n = matrix.shape[0]

    if n < 2:
        return pd.DataFrame(
            columns=["review_index_a", "review_index_b", "similarity"]
        )

    # cosine metric works directly with sparse TF-IDF matrices.
    nn = NearestNeighbors(
        n_neighbors=min(top_k + 1, n),
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )
    nn.fit(matrix)

    rows = []

    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        distances, indices = nn.kneighbors(matrix[start:stop])

        for local_i in range(stop - start):
            i = start + local_i

            for distance, j in zip(distances[local_i], indices[local_i]):
                j = int(j)

                if i == j:
                    continue

                similarity = 1.0 - float(distance)

                if similarity < threshold:
                    continue

                a, b = sorted((i, j))
                rows.append((a, b, similarity))

    if not rows:
        return pd.DataFrame(
            columns=["review_index_a", "review_index_b", "similarity"]
        )

    result = pd.DataFrame(
        rows,
        columns=["review_index_a", "review_index_b", "similarity"],
    )

    # The same pair can appear from both directions.
    result = (
        result.groupby(["review_index_a", "review_index_b"], as_index=False)
        ["similarity"]
        .max()
        .sort_values("similarity", ascending=False)
        .reset_index(drop=True)
    )

    return result


def aggregate_review_evidence(
    n_reviews: int,
    pairs: pd.DataFrame,
) -> pd.DataFrame:
    evidence = pd.DataFrame({
        "review_index": np.arange(n_reviews, dtype=np.int64),
        "similar_review_count": np.zeros(n_reviews, dtype=np.int32),
        "max_similarity": np.zeros(n_reviews, dtype=np.float32),
        "mean_similarity": np.zeros(n_reviews, dtype=np.float32),
    })

    if pairs.empty:
        return evidence

    counts = np.zeros(n_reviews, dtype=np.int32)
    max_sim = np.zeros(n_reviews, dtype=np.float32)
    sum_sim = np.zeros(n_reviews, dtype=np.float64)

    a = pairs["review_index_a"].to_numpy(dtype=np.int64)
    b = pairs["review_index_b"].to_numpy(dtype=np.int64)
    s = pairs["similarity"].to_numpy(dtype=np.float64)

    for idx in range(len(pairs)):
        i = a[idx]
        j = b[idx]
        sim = s[idx]

        counts[i] += 1
        counts[j] += 1
        max_sim[i] = max(max_sim[i], sim)
        max_sim[j] = max(max_sim[j], sim)
        sum_sim[i] += sim
        sum_sim[j] += sim

    evidence["similar_review_count"] = counts
    evidence["max_similarity"] = max_sim

    nonzero = counts > 0
    evidence.loc[nonzero, "mean_similarity"] = (
        sum_sim[nonzero] / counts[nonzero]
    ).astype(np.float32)

    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--threshold",
        type=float,
        default=SIMILARITY_THRESHOLD,
        help="Minimum cosine similarity to retain as a strong similarity pair.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help="Nearest neighbors considered per review.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
    )
    args = parser.parse_args()

    project_root = (
        Path(args.project_root).resolve()
        if args.project_root
        else Path(__file__).resolve().parents[1]
    )

    input_path = (project_root / args.input).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TrustLens - Phase 6 Similarity Engine")
    print("=" * 70)
    print(f"Loading: {input_path}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    df = pd.read_parquet(input_path)

    required = {"Review", "Summary"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    review_text = build_review_text(df)
    valid_mask = review_text.str.len() > 0

    working = df.loc[valid_mask].copy().reset_index(drop=True)
    working_text = review_text.loc[valid_mask].reset_index(drop=True)

    print(f"Rows with usable review text: {len(working)}")

    _, duplicate_summary = exact_duplicate_groups(working_text)

    exact_duplicate_review_count = int(
        duplicate_summary["duplicate_count"].sum()
        if not duplicate_summary.empty
        else 0
    )
    exact_duplicate_group_count = int(len(duplicate_summary))

    print(f"Exact duplicate groups: {exact_duplicate_group_count}")
    print(f"Rows belonging to exact duplicate groups: {exact_duplicate_review_count}")

    print("\nBuilding character TF-IDF representation...")
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=NGRAM_RANGE,
        min_df=2,
        max_features=MAX_FEATURES,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )

    matrix = vectorizer.fit_transform(working_text)

    print(f"TF-IDF shape: {matrix.shape}")
    print(f"Searching top-{args.top_k} nearest neighbors...")
    print(f"Similarity threshold: {args.threshold:.2f}")

    pairs = batched_similarity_search(
        matrix=matrix,
        threshold=args.threshold,
        top_k=args.top_k,
        batch_size=args.batch_size,
    )

    # Remove exact-text pairs from the near-duplicate output because they are
    # already represented separately as exact duplicates.
    if not pairs.empty:
        same_text = (
            working_text.iloc[pairs["review_index_a"].to_numpy()].to_numpy()
            == working_text.iloc[pairs["review_index_b"].to_numpy()].to_numpy()
        )
        pairs = pairs.loc[~same_text].reset_index(drop=True)

    evidence = aggregate_review_evidence(len(working), pairs)

    # Attach stable source row identifiers where available.
    if "product_name" in working.columns:
        evidence["product_name"] = working["product_name"].astype(str).to_numpy()

    results = pairs.copy()
    if not results.empty:
        results["text_a"] = (
            working_text.iloc[results["review_index_a"].to_numpy()].to_numpy()
        )
        results["text_b"] = (
            working_text.iloc[results["review_index_b"].to_numpy()].to_numpy()
        )

    pairs_path = output_dir / "similarity_pairs.parquet"
    evidence_path = output_dir / "review_similarity_evidence.parquet"
    duplicate_path = output_dir / "exact_duplicate_groups.csv"
    vectorizer_path = output_dir / "similarity_vectorizer.joblib"
    summary_path = output_dir / "similarity_summary.json"

    results.to_parquet(pairs_path, index=False)
    evidence.to_parquet(evidence_path, index=False)
    duplicate_summary.to_csv(duplicate_path, index=False)

    import joblib
    joblib.dump(vectorizer, vectorizer_path)

    reviews_with_strong_similarity = int(
        (evidence["similar_review_count"] > 0).sum()
    )

    summary = {
        "phase": 6,
        "method": "character TF-IDF cosine nearest-neighbor similarity",
        "input_rows": int(len(df)),
        "usable_text_rows": int(len(working)),
        "exact_duplicate_groups": exact_duplicate_group_count,
        "rows_in_exact_duplicate_groups": exact_duplicate_review_count,
        "strong_similarity_pairs": int(len(results)),
        "reviews_with_strong_similarity": reviews_with_strong_similarity,
        "similarity_threshold": float(args.threshold),
        "top_k": int(args.top_k),
        "ngram_range": list(NGRAM_RANGE),
        "max_features": MAX_FEATURES,
        "labels_used": False,
        "test_labels_used_for_tuning": False,
        "synthetic_data_used": False,
        "fraud_claim": False,
        "interpretation": (
            "Similarity is evidence of textual overlap only. "
            "High similarity does not prove that reviews are fake, "
            "coordinated, or fraudulent."
        ),
        "artifacts": {
            "similarity_pairs": str(pairs_path),
            "review_similarity_evidence": str(evidence_path),
            "exact_duplicate_groups": str(duplicate_path),
            "vectorizer": str(vectorizer_path),
        },
    }

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("PHASE 6 SIMILARITY ENGINE COMPLETED")
    print("=" * 70)
    print(f"Strong similarity pairs : {len(results)}")
    print(f"Reviews with similarity  : {reviews_with_strong_similarity}")
    print(f"Exact duplicate groups   : {exact_duplicate_group_count}")
    print(f"Pairs                    : {pairs_path}")
    print(f"Evidence                 : {evidence_path}")
    print(f"Duplicates               : {duplicate_path}")
    print(f"Vectorizer               : {vectorizer_path}")
    print(f"Summary                  : {summary_path}")
    print("Labels used              : NO")
    print("Synthetic data           : NO")
    print("Test tuning              : NO")
    print("=" * 70)


if __name__ == "__main__":
    main()
