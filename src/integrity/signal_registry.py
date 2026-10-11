"""Registry of signals: name -> category. Single source of truth for evidence breakdown."""

SIGNALS = {
    "promotional_language": "textual",
    "exclamation_intensity": "textual",
    "generic_phrases": "textual",
    "review_similarity": "similarity",
    "rating_aspect_consistency": "aspect",
    "unverified_purchase": "behavioral",
    "reviewer_product_repeat": "graph",
    "temporal_burst": "temporal",
    "cluster_membership": "cluster",
    "image_evidence": "image",
}
CATEGORIES = [
    "textual",
    "behavioral",
    "similarity",
    "temporal",
    "aspect",
    "graph",
    "cluster",
    "image",
]
