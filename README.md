# TrustLens

**TrustLens** is an explainable multi-signal review-integrity platform for detecting **predicted suspicious reviews**, analysing aspect-level sentiment, surfacing similarity/temporal/graph coordination signals, and producing evidence-first explanations.

> TrustLens is a decision-support system. A prediction of suspiciousness is **not proof of fraud**. Reviewer-level outputs describe elevated behavioural risk based on available signals, not a declaration that a reviewer is fake.

## Current release

This repository is the integrated TrustLens Phase 1–26 deployment baseline, completed with the Phase 27–29 release/documentation/consistency layer.

- Release target: `phase1-29-final`
- Random seed: `42`
- Primary API: FastAPI on port `8000`
- Secondary API: Flask on port `5000`
- Streamlit UI: port `8501`
- Gradio UI: port `7860`
- MLflow profile: port `5000` when enabled
- Kubernetes namespace: see `k8s/namespace.yaml`

The repository intentionally does **not** contain a fabricated production dataset, trained model, benchmark result, latency claim, cluster count, or accuracy claim. Supply a permitted dataset and train/evaluate before making empirical claims.

## Architecture

```text
                         ┌─────────────────────┐
                         │  Raw Review Dataset  │
                         └──────────┬──────────┘
                                    │
                         validation / cleaning
                                    │
                     ┌──────────────▼──────────────┐
                     │ preprocessing + text signals │
                     └──────────────┬──────────────┘
                                    │
      ┌───────────────┬─────────────┼───────────────┬───────────────┐
      ▼               ▼             ▼               ▼               ▼
   ML models         ABSA       similarity      temporal       graph signals
      │               │             │               │               │
      └───────────────┴─────────────┼───────────────┴───────────────┘
                                    ▼
                         Evidence / Trust engine
                                    │
                 ┌──────────────────┼──────────────────┐
                 ▼                  ▼                  ▼
             XAI / SHAP         Product            Reviewer
             LIME / CF       intelligence        intelligence
                 │                  │                  │
                 └──────────────────┼──────────────────┘
                                    ▼
                         FastAPI / Flask services
                           │                 │
                      Streamlit           Gradio
                           │                 │
                      Consumer /        Interactive
                     Investigation       analysis
```

## Repository map

- `src/data/` — ingestion, schema and validation
- `src/preprocessing/` — preprocessing layer
- `src/features/` — feature engineering
- `src/nlp/` — NLP components
- `src/models/` — training, evaluation, tuning, model registry and thresholding
- `src/absa/` — aspect-based sentiment analysis
- `src/similarity/` — duplicate and phrase similarity
- `src/temporal/` — temporal burst detection
- `src/graph/` — reviewer/product graph features and visualisation
- `src/clustering/` — review clusters and coordination signals
- `src/integrity/` — evidence schema, signal registry and risk scoring
- `src/explainability/` — SHAP, LIME, feature importance and counterfactuals
- `src/product_intelligence/` — product/reviewer intelligence and trust-adjusted rating
- `src/computer_vision/` — image analysis / CV hooks
- `src/inference/` — prediction and batch inference
- `src/monitoring/` — data quality, drift and prediction monitoring
- `api/` — FastAPI, Flask and service layer
- `streamlit_app/` — primary dashboard
- `gradio_app/` — interactive inference UI
- `automl/` — AutoGluon, H2O and auto-sklearn runners
- `robustness/` — adversarial/perturbation evaluation
- `k8s/` — Kubernetes manifests
- `kubeflow/` — Kubeflow pipeline
- `.github/workflows/` — CI workflows
- `tests/` — unit, integration and API tests
- `scripts/` — operational and pipeline commands
- `configs/` — runtime/model/tracking configuration
- `docs/` — release, architecture, operations, data and model documentation

## Windows setup and quickstart

TrustLens includes automated Windows PowerShell scripts in the project root for a seamless setup and launch experience:

### Quickstart (2-Step Execution)
```powershell
.\setup.ps1
.\run.ps1
```

> **PowerShell ExecutionPolicy Note**: If Windows restricts PowerShell script execution on your laptop, bypass it for the current session without making permanent system changes:
> ```powershell
> powershell -ExecutionPolicy Bypass -File .\setup.ps1
> powershell -ExecutionPolicy Bypass -File .\run.ps1
> ```

### What `setup.ps1` does:
1. Verifies Python 3.10+ installation.
2. Creates an isolated local virtual environment in `.venv`.
3. Upgrades pip and installs all development dependencies (`requirements-dev.txt`).
4. Initializes project directories (`data/`, `artifacts/`, `reports/`) and `.env`.
5. Prepares the offline demonstration baseline model in `artifacts/model` with `label_semantics='synthetic'`.

### What `run.ps1` does:
1. Validates that `.venv` is present.
2. Checks that model artifacts exist (without triggering unnecessary retraining).
3. Launches the integrated Streamlit Dashboard on port 8501 (`http://localhost:8501`).
4. To launch the FastAPI REST service instead, run: `.\run.ps1 -Mode api`.

### Manual setup (alternative)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
python scripts/setup.py
```

Optional capability sets:
```powershell
pip install -r requirements-cv.txt
pip install -r requirements-mlops.txt
pip install -r requirements-automl.txt
```

`auto-sklearn` has platform/Python constraints and is intentionally not pinned in the general requirements file. Use a supported Linux/WSL2 environment if that runner is required.

## Data and training

Place a permitted review dataset at:

```text
data/raw/reviews.csv
```

The dataset contract is documented in `docs/data-contract.md`. **Do not invent labels.** When labels are weak, proxy or synthetic, pass the correct semantics to the pipeline.

Example:

```powershell
python scripts/run_all.py data/raw/reviews.csv --label-semantics true
```

Other accepted semantics are `weak`, `proxy`, and `synthetic`.

The pipeline produces validation, preprocessing, feature, model and evaluation artifacts under `reports/`, `data/processed/` and `artifacts/`.

## Local applications

After a trained model exists:

```powershell
uvicorn api.fastapi_app.main:app --host 0.0.0.0 --port 8000
```

Open the FastAPI documentation at `http://localhost:8000/docs`.

Streamlit:

```powershell
streamlit run streamlit_app/app.py --server.port 8501
```

Gradio:

```powershell
python -m gradio_app.app
```

Flask:

```powershell
python -m api.flask_app.app
```

## Testing and release validation

```powershell
python -m pytest -q
ruff check .
black --check .
isort --check-only .
python scripts/validate_release.py
```

The release validator is structural: it verifies required files, deployment references, placeholder hygiene and manifest consistency. It does not manufacture model performance.

## Docker

Build and run:

```powershell
docker build -t trustlens:latest .
docker compose up --build
```

For MLflow:

```powershell
docker compose --profile mlops up --build
```

A trained model should exist under `artifacts/model` before starting deployment services.

## Kubernetes / Kubeflow

Kubernetes manifests are under `k8s/`. Apply them only after reviewing namespace, image, storage and secret configuration for the target cluster.

The Kubeflow pipeline definition is under `kubeflow/pipeline.py`. It references the real pipeline scripts rather than embedding fabricated outputs.

## Experiment tracking

MLflow and Weights & Biases are optional tracking integrations. Keep credentials in environment variables or a secret manager; never commit real credentials.

## Historical evaluation benchmarks & reports

Historical empirical results across the fake-review (~40,420 reviews), sentiment (~170,675 reviews), and ABSA development stages are documented in:
- `reports/historical_evaluation_summary.json` (Machine-readable summary)
- `docs/historical-evaluation.md` (Detailed methodology, neutral-class analysis, and benchmark tables)

Key historical baselines:
- Fake-Review Detection: Baseline LR 93.60%, Baseline RF 90.96%, Advanced LR 95.60%, Advanced RF 91.87%.
- Sentiment Analysis: Neutral-optimized 3-class classifier with 90.47% accuracy, 90.25% weighted F1, 74.61% macro F1 (neutral F1 42.95%).
- ABSA: 34 extracted aspects across 99,672 occurrences, rule-assisted without ground-truth labels (mean confidence 0.917).

## Academic Viva & Demonstration Guide

When explaining TrustLens to faculty or evaluators:
1. **Explain the 9 Signal Layers**: Point out that TrustLens does not rely solely on text. It incorporates text stylometrics, reviewer behavior, temporal bursts, review similarity, rating consistency, aspect consistency, reviewer-product graphs, coordination clustering, and image hash signals.
2. **Probability vs Trust Score**: Clarify that fake probability ($p \in [0, 1]$) comes from the calibrated ML classifier, while the Trust Score ($100 \times (1 - p)$) is an application-level decision-support metric.
3. **Explainability**: Demonstrate that SHAP reveals global/local feature importance, LIME generates local perturbation explanations, and counterfactuals evaluate model sensitivity to hypothetical text edits.
4. **Academic Honesty on Labels**: Highlight that labels in web review corpora are often proxy or weak labels; predictions are decision-support indicators, not legal fraud determinations.

## Ethical / product language

Use:

- **predicted suspicious review**, not “definitely fake review”
- **elevated reviewer behavioural risk**, not “fake reviewer”
- **potentially coordinated review cluster**, not “fraud ring”
- **trust-adjusted rating estimate**, not “true rating”

Always distinguish model probability from the application-level TrustLens risk score, and distinguish verified labels from weak/proxy/synthetic labels.

## Final release principle

A successful software release means the code, interfaces, configuration, tests and documentation are internally consistent. It does **not** mean that model quality is proven without a real labelled evaluation dataset.

