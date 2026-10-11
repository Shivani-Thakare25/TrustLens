"""Exact duplicate detection on normalized text."""

from __future__ import annotations

import re
from collections import defaultdict


def normalize(text: str) -> str:
    return re.sub(r"\W+", " ", (text or "").lower()).strip()


def find_exact_duplicates(review_ids, texts) -> list[list]:
    """Return groups (size>=2) of review_ids whose normalized text is identical."""
    buckets = defaultdict(list)
    for rid, t in zip(review_ids, texts):
        buckets[normalize(t)].append(rid)
    return [ids for key, ids in buckets.items() if key and len(ids) > 1]
