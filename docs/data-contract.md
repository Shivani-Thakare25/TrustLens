# Data Contract

## Required input

The canonical pipeline expects a CSV review dataset at:

`data/raw/reviews.csv`

The exact accepted columns and coercion rules are implemented in `src/data/schema.py` and validated by `src/data/validation.py`.

## Label semantics

Every training run must explicitly declare one of:

- `true` — verified/ground-truth labels
- `weak` — weak supervision
- `proxy` — proxy labels
- `synthetic` — synthetic labels

These semantics are important when interpreting evaluation results.

## Data quality rules

Validation should happen before preprocessing and training. Invalid rows, missing required fields and unsupported values must be surfaced by the validation layer rather than silently converted into evidence.

## Privacy

Do not place credentials, personal secrets or unnecessary personal data into the repository. Use `.env` only for local environment configuration and keep `.env.example` as the shareable template.
