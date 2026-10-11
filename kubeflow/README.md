# Kubeflow pipeline

`pipeline.py` defines a Kubeflow Pipelines v2 pipeline whose steps run the TrustLens scripts in the
`trustlens:latest` image (build it with `docker build -t trustlens:latest .` and push it to a registry your
cluster can pull from; update `IMAGE` in `components/steps.py`).

```
pip install -r requirements-mlops.txt
python kubeflow/pipeline.py      # compiles to kubeflow/trustlens_pipeline.yaml
```

Upload the YAML in the Kubeflow UI and pass `dataset_url` (https CSV you are licensed to use) and
`label_semantics` (true | weak | proxy | synthetic).

**Status:** the pipeline compiles and its structure is checked by the test suite. It has not been run on a
Kubeflow cluster, so cluster-specific issues (image pull, resource limits, artifact storage) are untested.
Local training does not need Kubeflow: use `python scripts/run_all.py`.
