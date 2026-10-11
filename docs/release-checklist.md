# Release Checklist

## Source integrity

- [x] Integrated Phase 1â€“26 source retained.
- [x] Phase 27 documentation added.
- [x] Phase 28 structural consistency validator added.
- [x] Phase 29 manifest/completion record added.
- [x] No Pending item/Follow-up item/Pending verification markers in production source.
- [x] No fabricated benchmark/model-quality claims.

## Runtime

- [ ] Install dependencies in the target environment.
- [ ] Provide a permitted dataset.
- [ ] Train a model.
- [ ] Run the complete test suite in the target environment.
- [ ] Verify API/UI health checks.
- [ ] Build and smoke-test the Docker image.
- [ ] Validate Kubernetes manifests against the target cluster.
- [ ] Compile/test the Kubeflow pipeline in the target environment.

## Evidence

Only report model performance, latency, throughput, drift, cluster counts or similar metrics when they are generated from actual runs and retained as artifacts.

