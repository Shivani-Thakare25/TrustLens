"""Supported review schema. Optional fields are nullable and never fabricated."""

from __future__ import annotations

REQUIRED_COLUMNS = ["review_text"]

ALL_COLUMNS = [
    "review_id",
    "product_id",
    "user_id",
    "review_text",
    "review_title",
    "rating",
    "review_date",
    "verified_purchase",
    "helpful_votes",
    "total_votes",
    "product_category",
    "product_price",
    "reviewer_review_count",
    "reviewer_average_rating",
    "review_images",
    "image_urls",
    "label",
]

LABEL_SEMANTICS = ("true", "weak", "proxy", "synthetic", "unknown")
VALID_LABELS = (0, 1)  # 1 = suspicious-labelled, 0 = genuine-labelled
