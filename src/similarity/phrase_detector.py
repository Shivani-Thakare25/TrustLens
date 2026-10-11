"""Shared n-gram phrase detection between two reviews."""

from __future__ import annotations

import re

TOKEN = re.compile(r"[a-z0-9']+")


def _ngrams(tokens, n):
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def matching_phrases(a: str, b: str, n: int = 4, limit: int = 5) -> list[str]:
    """Return up to `limit` word n-grams shared by both texts."""
    ta, tb = TOKEN.findall(a.lower()), TOKEN.findall(b.lower())
    return sorted(_ngrams(ta, n) & _ngrams(tb, n))[:limit]
