# Operations Runbook

## Start locally

1. Create and activate `.venv`.
2. Install `requirements-dev.txt`.
3. Provide a permitted dataset if training is required.
4. Train/register a model.
5. Start FastAPI and optionally Streamlit/Gradio/Flask.

## Health checks

FastAPI health endpoint:

`GET /api/v1/health`

Docker uses this endpoint for the image health check. Kubernetes API readiness uses the same endpoint.

## Model configuration

`TRUSTLENS_MODEL_DIR` controls the model artifact location. Default:

`artifacts/model`

## Logs and request tracing

The FastAPI middleware generates/propagates `X-Request-ID` and records request timing and model version. Do not log raw sensitive review content unless there is a documented operational need.

## Monitoring

The monitoring package contains data-quality, drift and prediction-monitoring components. Monitoring results should be generated from actual observed data; never substitute invented statistics.

## Rollback

For deployment changes, retain the previously validated container image/model artifact and revert the deployment manifest/image tag rather than modifying production code in place.
