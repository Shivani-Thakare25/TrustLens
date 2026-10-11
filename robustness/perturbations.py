"""Deterministic text perturbations for robustness testing. Paraphrasing is NOT implemented
(no reliable offline paraphraser is bundled); only the perturbations below are measured."""

from __future__ import annotations

import re

from src.explainability.counterfactual import _PROMO_RE

SYNONYMS = {
    "great": "good",
    "amazing": "impressive",
    "terrible": "poor",
    "bad": "poor",
    "love": "like",
    "best": "top",
    "awesome": "fine",
    "excellent": "very good",
    "perfect": "suitable",
    "fantastic": "nice",
    "horrible": "unpleasant",
    "recommend": "suggest",
}
FILLER = " By the way, the weather was mild this week and traffic was light."


def _syn(text: str) -> str:
    def rep(m):
        w = m.group(0)
        s = SYNONYMS[w.lower()]
        return s.capitalize() if w[0].isupper() else s

    return re.sub(r"\b(" + "|".join(SYNONYMS) + r")\b", rep, text, flags=re.I)


def _shorten(text: str) -> str:
    words = text.split()
    return " ".join(words[: max(3, len(words) // 2)])


PERTURBATIONS = {
    "remove_exclamations": lambda t: re.sub(r"!+", ".", t),
    "add_exclamations": lambda t: re.sub(r"\.", "!!", t) if "." in t else t + "!!",
    "lowercase": lambda t: t.lower(),
    "uppercase": lambda t: t.upper(),
    "synonym_replacement": _syn,
    "remove_promotional_phrases": lambda t: re.sub(r"\s+", " ", _PROMO_RE.sub("", t)).strip(),
    "shorten_half": _shorten,
    "insert_irrelevant_text": lambda t: t + FILLER,
}
