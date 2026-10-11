"""Kubeflow Pipelines v2 container components. Each runs one TrustLens script inside the TrustLens image."""

from kfp import dsl

IMAGE = "trustlens:latest"


@dsl.container_component
def download_data(url: str, raw: dsl.Output[dsl.Dataset]):
    return dsl.ContainerSpec(
        IMAGE, ["python", "scripts/download_data.py"], ["--url", url, "--out", raw.path]
    )


@dsl.container_component
def validate_data(raw: dsl.Input[dsl.Dataset], report: dsl.Output[dsl.Artifact]):
    return dsl.ContainerSpec(
        IMAGE, ["python", "scripts/validate_data.py"], [raw.path, "--out", report.path]
    )


@dsl.container_component
def preprocess(
    raw: dsl.Input[dsl.Dataset], processed: dsl.Output[dsl.Dataset], stats: dsl.Output[dsl.Artifact]
):
    return dsl.ContainerSpec(
        IMAGE,
        ["python", "scripts/preprocess.py"],
        [raw.path, "--out", processed.path, "--stats", stats.path],
    )


@dsl.container_component
def build_features(processed: dsl.Input[dsl.Dataset], features: dsl.Output[dsl.Dataset]):
    return dsl.ContainerSpec(
        IMAGE, ["python", "scripts/build_features.py"], [processed.path, "--out", features.path]
    )


@dsl.container_component
def train(processed: dsl.Input[dsl.Dataset], label_semantics: str, model: dsl.Output[dsl.Model]):
    return dsl.ContainerSpec(
        IMAGE,
        ["python", "scripts/train.py"],
        [processed.path, "--out", model.path, "--label-semantics", label_semantics],
    )


@dsl.container_component
def evaluate(model: dsl.Input[dsl.Model], report: dsl.Output[dsl.Artifact]):
    return dsl.ContainerSpec(
        IMAGE, ["python", "scripts/evaluate.py"], ["--model-dir", model.path, "--out", report.path]
    )


@dsl.container_component
def register_model(model: dsl.Input[dsl.Model], registry: dsl.Output[dsl.Artifact]):
    return dsl.ContainerSpec(
        IMAGE,
        ["python", "scripts/register_model.py"],
        ["--model-dir", model.path, "--registry", registry.path],
    )
