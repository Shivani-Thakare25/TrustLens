"""TrustLens Kubeflow pipeline: download -> validate -> preprocess -> features -> train -> evaluate -> register.

Compile: python kubeflow/pipeline.py  (writes kubeflow/trustlens_pipeline.yaml)
This is compiled and structurally checked in tests; it has NOT been executed on a Kubeflow cluster.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kfp import compiler, dsl  # noqa: E402

from kubeflow.components import steps  # noqa: E402


@dsl.pipeline(name="trustlens-training", description="TrustLens data -> model pipeline")
def trustlens_pipeline(dataset_url: str, label_semantics: str):
    raw = steps.download_data(url=dataset_url)
    steps.validate_data(raw=raw.outputs["raw"])
    prep = steps.preprocess(raw=raw.outputs["raw"])
    steps.build_features(processed=prep.outputs["processed"])
    trained = steps.train(processed=prep.outputs["processed"], label_semantics=label_semantics)
    steps.evaluate(model=trained.outputs["model"])
    steps.register_model(model=trained.outputs["model"])


def compile_pipeline(out_path: str = "kubeflow/trustlens_pipeline.yaml") -> str:
    compiler.Compiler().compile(trustlens_pipeline, out_path)
    return out_path


if __name__ == "__main__":
    print("compiled:", compile_pipeline())
