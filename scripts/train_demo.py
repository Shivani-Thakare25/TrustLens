"""Initializes a demonstration baseline model for offline presentation and testing.
Uses synthetic demonstration data with label_semantics='synthetic' to ensure
academic honesty while allowing full end-to-end functionality without manual training.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.evaluate import evaluate_run
from src.models.registry import register_model
from src.models.train import train_all


def generate_demo_corpus(n: int = 400, seed: int = 42) -> pd.DataFrame:
    """Generate representative demonstration reviews for offline inspection and viva demo."""
    rng = np.random.default_rng(seed)
    promo_templates = [
        "Best product ever! Highly recommend to everyone, must buy now, absolutely love it!",
        "Incredible purchase! Amazing quality, fastest shipping ever, 10/10 perfect!",
        "Life-changing product! Don't miss out, buy immediately, best decision ever!",
        "Awesome experience! Highly recommended, wonderful quality and fantastic service!",
        "Must buy today! Unbelievable performance and top notch design, five stars all the way!",
    ]
    plain_templates = [
        "The battery lasted two days with moderate use, but the wrist strap felt slightly stiff.",
        "Delivery took six business days and the box was dented, though the internal screen is fine.",
        "Camera works well in daylight, but low light photos have noticeable noise and slow shutter.",
        "The build quality is acceptable for the price. Not the fastest unit, but dependable so far.",
        "Sound is clear on calls, but the volume controls on the cable are too small to press easily.",
        "Good value for money. Minor lag when opening heavy apps, but overall battery is reliable.",
        "Returned the first device due to a loose hinge, but the replacement works as expected.",
    ]
    products = [f"prod_{i % 8 + 1:02d}" for i in range(n)]
    users = [f"user_{i % 80 + 1:03d}" for i in range(n)]
    dates = [d.strftime("%Y-%m-%d") for d in pd.date_range("2024-01-01", periods=180, freq="D")]

    rows = []
    for i in range(n):
        is_fake = int(rng.random() < 0.40)
        template = rng.choice(promo_templates if is_fake else plain_templates)
        review_text = f"{template} Note variant {i % 11}."
        rating = int(rng.choice([5, 5, 4, 1] if is_fake else [3, 4, 2, 5, 1]))
        verified = bool(rng.random() < (0.25 if is_fake else 0.85))
        review_date = str(rng.choice(dates))

        rows.append(
            {
                "review_id": f"rev_{i + 1:04d}",
                "user_id": users[i],
                "product_id": products[i],
                "review_text": review_text,
                "rating": rating,
                "verified_purchase": verified,
                "review_date": review_date,
                "label": is_fake,
            }
        )
    return pd.DataFrame(rows)


def main():
    model_dir = ROOT / "artifacts/model"
    registry_dir = ROOT / "artifacts/registry"
    eval_dir = ROOT / "reports/evaluation"

    print("Generating baseline demonstration corpus...")
    df = generate_demo_corpus(n=400, seed=42)

    # Save demo raw data for optional inspection
    demo_data_path = ROOT / "data/processed/demo_reviews.csv"
    demo_data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(demo_data_path, index=False)
    print(f"Saved demo dataset to {demo_data_path}")

    print("Training baseline model (Logistic Regression & Random Forest)...")
    meta = train_all(
        df,
        model_dir,
        models=["logistic_regression", "random_forest"],
        seed=42,
        label_semantics="synthetic",
        save_reference_corpus=True,
    )

    print("Generating evaluation artifacts (confusion matrix, ROC, PR, calibration)...")
    evaluate_run(model_dir, eval_dir)

    print("Registering model in artifact registry...")
    register_model(model_dir, registry_dir)

    print("Demonstration baseline model successfully initialized.")
    print(f"Model: {meta.get('model_name')} (threshold: {meta.get('threshold'):.2f})")
    print(f"Artifacts saved in: {model_dir}")


if __name__ == "__main__":
    main()
