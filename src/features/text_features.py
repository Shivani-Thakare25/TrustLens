"""Handcrafted stylometric/linguistic features. Text is not over-cleaned; case and punctuation matter."""

from __future__ import annotations

import re

import pandas as pd

PROMO = {
    "best",
    "amazing",
    "perfect",
    "recommend",
    "recommended",
    "must",
    "buy",
    "love",
    "awesome",
    "incredible",
    "fantastic",
    "excellent",
    "unbelievable",
    "life-changing",
}
URGENCY = {"now", "hurry", "limited", "today", "immediately", "asap", "don't miss", "dont miss"}
SUPERLATIVE_RE = re.compile(r"\b(?:best|worst|greatest|most|perfect|ever)\b", re.I)
FIRST_PERSON = {"i", "me", "my", "mine", "myself", "we", "our", "us"}
GENERIC_PHRASES = (
    "highly recommend",
    "best product ever",
    "must buy",
    "five stars",
    "works great",
    "love it",
    "great product",
    "good quality",
)
WORD_RE = re.compile(r"[A-Za-z0-9']+")

FEATURE_NAMES = [
    "char_len",
    "word_count",
    "sentence_count",
    "punct_ratio",
    "caps_ratio",
    "repeated_word_ratio",
    "repeated_bigram_ratio",
    "exclaim_count",
    "question_count",
    "lexical_diversity",
    "promo_ratio",
    "urgency_count",
    "superlative_count",
    "first_person_ratio",
    "generic_phrase_count",
]


def extract_text_features(text: str) -> dict:
    """Return a dict of numeric features for one review. Empty text yields zeros."""
    text = text or ""
    words = [w.lower() for w in WORD_RE.findall(text)]
    n = len(words)
    chars = len(text)
    letters = [c for c in text if c.isalpha()]
    lower = text.lower()

    def ratio(a, b):
        return a / b if b else 0.0

    bigrams = list(zip(words, words[1:]))
    return {
        "char_len": float(chars),
        "word_count": float(n),
        "sentence_count": float(len([s for s in re.split(r"[.!?]+", text) if s.strip()])),
        "punct_ratio": ratio(sum(1 for c in text if c in "!?.,;:-"), chars),
        "caps_ratio": ratio(sum(1 for c in letters if c.isupper()), len(letters)),
        "repeated_word_ratio": ratio(n - len(set(words)), n),
        "repeated_bigram_ratio": ratio(len(bigrams) - len(set(bigrams)), len(bigrams)),
        "exclaim_count": float(text.count("!")),
        "question_count": float(text.count("?")),
        "lexical_diversity": ratio(len(set(words)), n),
        "promo_ratio": ratio(sum(1 for w in words if w in PROMO), n),
        "urgency_count": float(sum(1 for u in URGENCY if u in lower)),
        "superlative_count": float(len(SUPERLATIVE_RE.findall(text))),
        "first_person_ratio": ratio(sum(1 for w in words if w in FIRST_PERSON), n),
        "generic_phrase_count": float(sum(lower.count(g) for g in GENERIC_PHRASES)),
    }


def build_text_feature_frame(texts) -> pd.DataFrame:
    """Vectorize an iterable of texts into a DataFrame with FEATURE_NAMES columns."""
    return pd.DataFrame([extract_text_features(t) for t in texts], columns=FEATURE_NAMES)
