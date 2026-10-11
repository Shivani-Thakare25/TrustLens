"""Framework-independent payload validation shared by FastAPI and Flask."""

from __future__ import annotations

MAX_TEXT = 20000
MAX_BATCH = 500
MAX_IMAGE_B64 = 7_000_000  # ~5 MB of image bytes


def validate_review_payload(d) -> dict:
    """Return a cleaned review dict or raise ValueError with a safe message (no review text echoed)."""
    if not isinstance(d, dict):
        raise ValueError("Request body must be a JSON object.")
    text = d.get("review_text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("review_text must be a non-empty string.")
    if len(text) > MAX_TEXT:
        raise ValueError(f"review_text exceeds {MAX_TEXT} characters.")
    out = dict(d)
    out["review_text"] = text.strip()
    r = d.get("rating")
    if r is not None:
        if isinstance(r, bool) or not isinstance(r, (int, float)) or not 1 <= r <= 5:
            raise ValueError("rating must be a number between 1 and 5.")
    vp = d.get("verified_purchase")
    if vp is not None and not isinstance(vp, bool):
        raise ValueError("verified_purchase must be true or false.")
    img = d.get("image_base64")
    if img is not None and (not isinstance(img, str) or len(img) > MAX_IMAGE_B64):
        raise ValueError("image_base64 must be a base64 string of at most 5 MB.")
    return out


def validate_batch(d) -> list[dict]:
    items = d.get("reviews") if isinstance(d, dict) else None
    if not isinstance(items, list) or not items:
        raise ValueError("reviews must be a non-empty list.")
    if len(items) > MAX_BATCH:
        raise ValueError(f"A batch may contain at most {MAX_BATCH} reviews.")
    if any(isinstance(i, dict) and i.get("image_base64") for i in items):
        raise ValueError("image_base64 is not supported in batch requests.")
    return [validate_review_payload(i) for i in items]
