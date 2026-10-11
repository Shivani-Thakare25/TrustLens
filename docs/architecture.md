# TrustLens Architecture

## Signal layers

TrustLens combines independent evidence families rather than relying on text alone:

1. textual/content signals
2. reviewer behavioural signals
3. temporal signals
4. similarity/duplicate signals
5. rating consistency
6. aspect sentiment consistency
7. reviewer-product graph signals
8. cluster/coordination signals
9. optional image evidence

The evidence engine combines available signals into an application-level risk representation. Missing evidence must not be presented as observed evidence.

## Service boundary

`api/services/trustlens_service.py` is the service boundary consumed by FastAPI, Flask and the UI layers. Keeping inference behind this boundary prevents each interface from implementing a separate scoring pipeline.

## Explainability boundary

The explainability layer provides model/feature explanations. Explanations must be tied to the actual input and model output; they must never be hard-coded.

## Deployment

Docker packages the API and UI applications. Kubernetes separates API and Streamlit deployments and uses the configured persistent model volume. Kubeflow orchestrates the data-to-model pipeline.
