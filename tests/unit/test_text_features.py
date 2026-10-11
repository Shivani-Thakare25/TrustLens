from src.features.text_features import (
    FEATURE_NAMES,
    build_text_feature_frame,
    extract_text_features,
)


def test_empty_text_is_all_zero():
    f = extract_text_features("")
    assert set(f) == set(FEATURE_NAMES)
    assert all(v == 0.0 for v in f.values())


def test_promotional_text_scores_higher_than_neutral():
    promo = extract_text_features(
        "Best product ever!!! Highly recommend to everyone! Must buy now!"
    )
    plain = extract_text_features(
        "The battery lasted two days, but the case scratched after a week."
    )
    assert promo["exclaim_count"] == 5 or promo["exclaim_count"] > plain["exclaim_count"]
    assert promo["promo_ratio"] > plain["promo_ratio"]
    assert promo["generic_phrase_count"] >= 2
    assert plain["first_person_ratio"] == 0.0


def test_repetition_and_diversity():
    f = extract_text_features("good good good good")
    assert f["repeated_word_ratio"] == 0.75
    assert f["lexical_diversity"] == 0.25


def test_frame_shape():
    df = build_text_feature_frame(["a b c", "Hello world!"])
    assert df.shape == (2, len(FEATURE_NAMES))
