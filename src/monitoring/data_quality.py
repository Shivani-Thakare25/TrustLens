"""Input quality checks for monitoring."""

from __future__ import annotations

OPTIONAL_FIELDS = ["rating", "verified_purchase", "review_date", "product_id", "user_id"]


def missing_fields(review: dict) -> list[str]:
    return [f for f in OPTIONAL_FIELDS if review.get(f) is None]
