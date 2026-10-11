"""Text-similarity clusters via connected components of a thresholded cosine-similarity graph.
Dense similarity matrix: suitable for up to a few thousand reviews per call (guarded)."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MAX_REVIEWS = 5000


def find_clusters(texts, threshold: float = 0.6, min_size: int = 2) -> list[dict]:
    """Return [{cluster_id, indices, average_similarity}] for groups of mutually-linked similar reviews."""
    texts = list(texts)
    if len(texts) > MAX_REVIEWS:
        raise ValueError(f"find_clusters supports at most {MAX_REVIEWS} reviews per call.")
    if len(texts) < min_size:
        return []
    X = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit_transform(texts)
    sim = cosine_similarity(X)
    np.fill_diagonal(sim, 0.0)
    n, labels = connected_components(csr_matrix(sim >= threshold), directed=False)
    clusters = []
    for lab in range(n):
        idx = np.where(labels == lab)[0]
        if len(idx) < min_size:
            continue
        sub = sim[np.ix_(idx, idx)]
        avg = float(sub.sum() / (len(idx) * (len(idx) - 1)))
        clusters.append(
            {"cluster_id": f"c{len(clusters)}", "indices": idx.tolist(), "average_similarity": avg}
        )
    return clusters
