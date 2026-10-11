"""TF-IDF word + character n-gram cosine similarity against a reference corpus."""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.similarity.phrase_detector import matching_phrases


class SimilarityEngine:
    """Fit on a corpus of reviews, then query for the most similar existing reviews."""

    def __init__(self):
        self.word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.ids, self.texts = [], []

    def fit(self, review_ids, texts):
        self.ids, self.texts = list(review_ids), list(texts)
        if not self.texts:
            raise ValueError("Cannot fit similarity engine on an empty corpus.")
        self.Xw = self.word.fit_transform(self.texts)
        self.Xc = self.char.fit_transform(self.texts)
        return self

    def query(
        self, text: str, top_k: int = 5, min_score: float = 0.0, exclude_id=None
    ) -> list[dict]:
        """Return [{review_id, similarity_score, matching_phrases}] sorted by score desc."""
        sw = cosine_similarity(self.word.transform([text]), self.Xw)[0]
        sc = cosine_similarity(self.char.transform([text]), self.Xc)[0]
        score = (sw + sc) / 2
        out = []
        for i in np.argsort(-score):
            if exclude_id is not None and self.ids[i] == exclude_id:
                continue
            if score[i] < min_score or len(out) >= top_k:
                break
            out.append(
                {
                    "review_id": self.ids[i],
                    "similarity_score": round(float(score[i]), 4),
                    "matching_phrases": matching_phrases(text, self.texts[i]),
                }
            )
        return out
