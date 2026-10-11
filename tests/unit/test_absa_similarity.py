from src.absa.absa import (
    extract_aspect_sentiment,
    overall_sentiment,
    rating_aspect_inconsistency,
)
from src.similarity.duplicate_detector import find_exact_duplicates
from src.similarity.similarity_engine import SimilarityEngine

TEXT = "The phone has an excellent camera but the battery is disappointing and delivery was late."


def test_absa_example_from_spec():
    r = {a["aspect"]: a["sentiment"] for a in extract_aspect_sentiment(TEXT)}
    assert r == {"camera": "Positive", "battery": "Negative", "delivery": "Negative"}


def test_absa_negation_flips():
    r = {a["aspect"]: a["sentiment"] for a in extract_aspect_sentiment("The battery is not good.")}
    assert r["battery"] == "Negative"


def test_absa_empty_and_no_aspects():
    assert extract_aspect_sentiment("") == []
    assert extract_aspect_sentiment("Hello there") == []
    assert overall_sentiment("") == "Neutral"


def test_rating_inconsistency_flag_and_unavailable():
    asp = extract_aspect_sentiment("The battery is terrible.")
    assert rating_aspect_inconsistency(5, asp)["inconsistent"] is True
    assert rating_aspect_inconsistency(1, asp)["inconsistent"] is False
    assert rating_aspect_inconsistency(None, asp)["available"] is False


def test_exact_duplicates_ignore_case_and_punctuation():
    g = find_exact_duplicates(["a", "b", "c"], ["Great item!", "great   item", "other"])
    assert g == [["a", "b"]]


def test_similarity_ranks_near_duplicate_first_with_phrases():
    corpus = [
        "Best product ever, highly recommend it to everyone",
        "Battery died after two weeks of normal use",
        "Shipping was slow but the screen looks great",
    ]
    eng = SimilarityEngine().fit(["r1", "r2", "r3"], corpus)
    res = eng.query("Best product ever, highly recommend it to all", top_k=2)
    assert res[0]["review_id"] == "r1"
    assert res[0]["similarity_score"] > res[1]["similarity_score"]
    assert res[0]["matching_phrases"]
    assert eng.query(corpus[0], exclude_id="r1")[0]["review_id"] != "r1"
