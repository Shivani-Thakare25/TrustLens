import numpy as np
import pytest

from robustness.perturbations import FILLER, PERTURBATIONS
from robustness.robustness_evaluation import evaluate_robustness
from src.explainability.model_adapter import fake_fn

TEXTS = [
    "Best product ever!!! Highly recommend, must buy now!",
    "The battery is great. Delivery was bad.",
    "Hinge squeaked and the case cracked.",
]


def test_perturbations_behave():
    assert PERTURBATIONS["synonym_replacement"]("Great phone, I love it") == "Good phone, I like it"
    assert PERTURBATIONS["remove_exclamations"]("Wow!!!") == "Wow."
    assert PERTURBATIONS["lowercase"]("ABC") == "abc" and PERTURBATIONS["uppercase"]("abc") == "ABC"
    assert PERTURBATIONS["insert_irrelevant_text"]("x") == "x" + FILLER
    assert len(PERTURBATIONS["shorten_half"]("a b c d e f g h").split()) == 4
    assert (
        "recommend"
        not in PERTURBATIONS["remove_promotional_phrases"]("I recommend it, best buy").lower()
    )


def test_measured_values_match_hand_computation():
    predict = lambda ts: np.array([len(t) / 1000 for t in ts])  # noqa: E731
    r = evaluate_robustness(predict, TEXTS, threshold=0.5)
    ins = r["perturbations"]["insert_irrelevant_text"]
    assert ins["mean_signed_change"] == pytest.approx(len(FILLER) / 1000)
    assert ins["mean_abs_probability_change"] == pytest.approx(len(FILLER) / 1000)
    assert ins["classification_flip_rate"] == 0.0 and ins["n_changed"] == 3
    assert r["perturbations"]["lowercase"]["mean_abs_probability_change"] == pytest.approx(0.0)


def test_flip_rate_detected():
    predict = lambda ts: np.array([0.9 if "!" in t else 0.1 for t in ts])  # noqa: E731
    r = evaluate_robustness(predict, ["Great!", "Nice!"], threshold=0.5)
    assert r["perturbations"]["remove_exclamations"]["classification_flip_rate"] == 1.0


def test_unchanged_texts_skipped_and_empty_rejected():
    r = evaluate_robustness(lambda ts: np.zeros(len(ts)), ["abc"], 0.5)
    assert r["perturbations"]["remove_exclamations"] == {"n_changed": 0}
    with pytest.raises(ValueError):
        evaluate_robustness(lambda ts: np.zeros(len(ts)), ["", "  ", None])


def test_with_real_model(trained_model):
    r = evaluate_robustness(fake_fn(trained_model), TEXTS, 0.5)
    for v in r["perturbations"].values():
        if v["n_changed"]:
            assert 0 <= v["classification_flip_rate"] <= 1 and v["max_abs_probability_change"] <= 1
