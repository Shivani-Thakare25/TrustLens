import joblib
import numpy as np
import pandas as pd
import pytest

from src.models.train import train_all


def make_synthetic_df(n=300, seed=0):
    """SYNTHETIC demonstration data: validates code paths, not real-world performance."""
    rng = np.random.default_rng(seed)
    promo = [
        "Best product ever!!! Highly recommend, must buy now!",
        "Amazing perfect awesome!!! Love it, highly recommend!",
    ]
    plain = [
        "The battery lasted two days but the strap frayed after a week of use.",
        "Delivery took six days and the box was dented, though the screen is fine.",
        "I returned the first unit because the hinge squeaked; the replacement works.",
    ]
    rows = []
    for i in range(n):
        fake = int(rng.random() < 0.4)
        rows.append(
            {
                "review_id": f"s{i}",
                "user_id": f"u{i % 120}",
                "review_text": f"{rng.choice(promo if fake else plain)} variant {i % 7}",
                "label": fake,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("model")
    train_all(
        make_synthetic_df(),
        d,
        models=["logistic_regression"],
        label_semantics="synthetic",
        save_reference_corpus=True,
    )
    return d


@pytest.fixture(scope="session")
def trained_model(model_dir):
    return joblib.load(model_dir / "model.joblib")
