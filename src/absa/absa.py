"""Rule/lexicon-based ABSA baseline. Not evaluated against any benchmark; no accuracy claimed."""

from __future__ import annotations

import re

ASPECTS = {
    "battery": ["battery", "charge", "charging", "battery life"],
    "camera": ["camera", "photo", "photos", "picture", "pictures", "lens"],
    "delivery": ["delivery", "shipping", "shipped", "arrived", "packaging", "package"],
    "price": ["price", "cost", "value", "expensive", "cheap", "worth"],
    "screen": ["screen", "display", "brightness"],
    "performance": ["performance", "speed", "fast", "slow", "lag", "laggy"],
    "quality": ["quality", "build", "durable", "material", "broke", "broken"],
    "service": ["service", "support", "customer service", "refund", "seller"],
}
POS = {
    "excellent",
    "amazing",
    "great",
    "good",
    "love",
    "perfect",
    "fantastic",
    "awesome",
    "fast",
    "durable",
    "worth",
    "helpful",
    "impressive",
    "sharp",
    "bright",
    "best",
    "nice",
    "solid",
}
NEG = {
    "bad",
    "poor",
    "terrible",
    "awful",
    "disappointing",
    "disappointed",
    "slow",
    "late",
    "broken",
    "broke",
    "worst",
    "cheap",
    "useless",
    "horrible",
    "lag",
    "laggy",
    "weak",
    "damaged",
    "overpriced",
}
NEGATORS = {"not", "no", "never", "n't", "isn't", "wasn't", "don't", "doesn't", "didn't", "hardly"}
CLAUSE_SPLIT = re.compile(r"[.!?;]+|\bbut\b|\bhowever\b|\balthough\b|\band\b|,", re.I)
TOKEN = re.compile(r"[a-z']+")


def _polarity(tokens: list[str]) -> float:
    score = 0.0
    for i, t in enumerate(tokens):
        s = 1.0 if t in POS else -1.0 if t in NEG else 0.0
        if s and any(
            n in tokens[max(0, i - 3) : i]
            or any(x.endswith("n't") for x in tokens[max(0, i - 3) : i])
            for n in NEGATORS
        ):
            s = -s
        score += s
    return score


def extract_aspect_sentiment(text: str, sentiment_predictor=None) -> list[dict]:
    """Return one entry per detected aspect: aspect, sentiment, confidence, supporting_phrase.
    Uses trained sentiment model when provided; falls back to lexicon baseline.
    """
    found: dict[str, list[tuple[float, str]]] = {}
    for clause in CLAUSE_SPLIT.split(text or ""):
        clause = clause.strip()
        if not clause:
            continue
        tokens = TOKEN.findall(clause.lower())
        low = " ".join(tokens)
        for aspect, kws in ASPECTS.items():
            if any(re.search(rf"\b{re.escape(k)}\b", low) for k in kws):
                found.setdefault(aspect, []).append((_polarity(tokens), clause))
    out = []
    for aspect, items in found.items():
        total = sum(s for s, _ in items)
        strongest = max(items, key=lambda x: abs(x[0]))
        clause = strongest[1]

        if sentiment_predictor is not None and getattr(sentiment_predictor, "loaded", False):
            res = sentiment_predictor.analyze_one(clause)
            sentiment = res["sentiment"]
            confidence = round(float(res["confidence"]), 2)
        else:
            sentiment = "Positive" if total > 0 else "Negative" if total < 0 else "Neutral"
            confidence = round(min(1.0, 0.5 + 0.25 * abs(total)), 2) if total else 0.5

        out.append(
            {
                "aspect": aspect,
                "sentiment": sentiment,
                "confidence": confidence,
                "supporting_phrase": clause,
            }
        )
    return out


def overall_sentiment(text: str, sentiment_predictor=None) -> str:
    """Return overall sentiment ('Positive', 'Negative', 'Neutral').
    Uses trained sentiment model when provided; falls back to lexicon baseline.
    """
    if not text or not str(text).strip():
        return "Neutral"
    if sentiment_predictor is not None and getattr(sentiment_predictor, "loaded", False):
        return sentiment_predictor.analyze_one(text)["sentiment"]
    s = _polarity(TOKEN.findall(str(text).lower()))
    return "Positive" if s > 0 else "Negative" if s < 0 else "Neutral"


def rating_aspect_inconsistency(rating, aspects: list[dict]) -> dict:
    """Auxiliary signal only: high rating with negative aspects (or reverse). Not proof of anything."""
    if rating is None or not aspects:
        return {
            "available": False,
            "inconsistent": False,
            "explanation": "rating or aspects unavailable",
        }
    neg = [a["aspect"] for a in aspects if a["sentiment"] == "Negative"]
    pos = [a["aspect"] for a in aspects if a["sentiment"] == "Positive"]
    if rating >= 4 and neg and len(neg) >= len(pos):
        return {
            "available": True,
            "inconsistent": True,
            "explanation": f"Potential rating-aspect inconsistency: rating {rating} but negative on {', '.join(neg)}",
        }
    if rating <= 2 and pos and len(pos) >= len(neg):
        return {
            "available": True,
            "inconsistent": True,
            "explanation": f"Potential rating-aspect inconsistency: rating {rating} but positive on {', '.join(pos)}",
        }
    return {
        "available": True,
        "inconsistent": False,
        "explanation": "Rating and aspect sentiment agree",
    }
