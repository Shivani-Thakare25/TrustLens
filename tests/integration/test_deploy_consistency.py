"""Structural checks tying Docker / Compose / Kubernetes / CI / Kubeflow to the real code."""

import json
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import uvicorn
import yaml

from api.fastapi_app.main import app
from api.services.trustlens_service import reset_service

ROOT = Path(__file__).resolve().parents[2]


def _load(path):
    return yaml.safe_load((ROOT / path).read_text())


def _k8s(name):
    return _load(f"k8s/{name}.yaml")


def test_dockerfile_copies_real_paths_and_exposes_app_ports():
    text = (ROOT / "Dockerfile").read_text()
    for line in re.findall(r"^COPY (.+)$", text, re.M):
        for src in line.split()[:-1]:
            assert (ROOT / src).exists(), f"Dockerfile COPY source missing: {src}"
    assert re.search(r"EXPOSE .*8000.*8501.*7860", text)
    assert "api.fastapi_app.main:app" in text and "USER trustlens" in text
    assert "/api/v1/health" in text
    assert not re.search(r"(?i)(password|api_key|secret)\s*=", text)


def test_compose_ports_commands_and_volumes():
    c = _load("docker-compose.yml")["services"]
    assert {"api", "streamlit", "gradio", "mlflow"} <= set(c)
    assert c["api"]["ports"] == ["8000:8000"] and "api.fastapi_app.main:app" in c["api"]["command"]
    assert c["streamlit"]["ports"] == ["8501:8501"]
    assert (ROOT / c["streamlit"]["command"][2]).is_file()
    assert c["gradio"]["ports"] == ["7860:7860"] and c["gradio"]["command"] == [
        "python",
        "-m",
        "gradio_app.app",
    ]
    assert c["gradio"]["environment"]["GRADIO_PORT"] == "7860"
    assert c["mlflow"]["profiles"] == ["mlops"]
    for svc in ("api", "streamlit", "gradio"):
        assert "./artifacts:/app/artifacts:ro" in c[svc]["volumes"]
        assert c[svc]["environment"]["TRUSTLENS_MODEL_DIR"] == "/app/artifacts/model"


def test_k8s_manifests_are_mutually_consistent():
    docs = {
        n: _k8s(n)
        for n in (
            "namespace",
            "configmap",
            "secret-example",
            "model-pvc",
            "api-deployment",
            "api-service",
            "streamlit-deployment",
            "streamlit-service",
            "ingress",
        )
    }
    ns = docs["namespace"]["metadata"]["name"]
    for n, d in docs.items():
        if n != "namespace":
            assert d["metadata"]["namespace"] == ns, n
    routes = set(app.openapi()["paths"])
    for name, port, probe in (
        ("api", 8000, "/api/v1/health"),
        ("streamlit", 8501, "/_stcore/health"),
    ):
        dep, svc = docs[f"{name}-deployment"], docs[f"{name}-service"]
        ctr = dep["spec"]["template"]["spec"]["containers"][0]
        assert ctr["ports"][0]["containerPort"] == port
        assert (
            svc["spec"]["ports"][0]["targetPort"] == port
            and svc["spec"]["ports"][0]["port"] == port
        )
        assert ctr["readinessProbe"]["httpGet"] == {"path": probe, "port": port}
        assert dep["spec"]["selector"]["matchLabels"] == svc["spec"]["selector"]
        assert dep["spec"]["template"]["spec"]["securityContext"]["runAsNonRoot"] is True
        assert ctr["envFrom"][0]["configMapRef"]["name"] == docs["configmap"]["metadata"]["name"]
        assert (
            dep["spec"]["template"]["spec"]["volumes"][0]["persistentVolumeClaim"]["claimName"]
            == docs["model-pvc"]["metadata"]["name"]
        )
    assert "/api/v1/health" in routes
    backends = {
        p["backend"]["service"]["name"]: p["backend"]["service"]["port"]["number"]
        for p in docs["ingress"]["spec"]["rules"][0]["http"]["paths"]
    }
    assert backends == {"trustlens-api": 8000, "trustlens-streamlit": 8501}
    assert docs["configmap"]["data"]["TRUSTLENS_MODEL_DIR"] == "/app/artifacts/model"
    assert all(v == "" for v in docs["secret-example"]["stringData"].values())


def test_workflows_reference_real_files_and_no_secrets():
    for wf in ("tests", "lint", "build"):
        d = _load(f".github/workflows/{wf}.yml")
        text = (ROOT / f".github/workflows/{wf}.yml").read_text()
        assert "secrets." not in text
        assert d["permissions"] == {"contents": "read"}
        for req in re.findall(r"-r (requirements[\w.-]*\.txt)", text):
            assert (ROOT / req).exists(), req


def test_makefile_scripts_exist():
    mk = (ROOT / "Makefile").read_text()
    for script in set(re.findall(r"scripts/\w+\.py", mk)):
        assert (ROOT / script).is_file(), script
    for mod in re.findall(r"-m ([\w.]+)", mk):
        if mod != "pytest" and mod != "pip":
            assert (ROOT / (mod.replace(".", "/") + ".py")).is_file(), mod


def test_kubeflow_pipeline_compiles_and_references_real_scripts(tmp_path):
    pytest.importorskip("kfp")
    from kubeflow.pipeline import compile_pipeline

    out = compile_pipeline(str(tmp_path / "p.yaml"))
    spec = yaml.safe_load(Path(out).read_text())
    tasks = spec["root"]["dag"]["tasks"]
    assert set(tasks) == {
        "download-data",
        "validate-data",
        "preprocess",
        "build-features",
        "train",
        "evaluate",
        "register-model",
    }
    text = Path(out).read_text()
    for script in re.findall(r"scripts/\w+\.py", text):
        assert (ROOT / script).is_file(), script
    assert tasks["register-model"]["dependentTasks"] == ["train"]


def test_no_hardcoded_secrets_or_placeholder_markers():
    pat = re.compile(
        r"""(?i)(api[_-]?key|passwd|password|secret|token)\s*[:=]\s*["'][^"'\s]{8,}["']"""
    )
    marker = re.compile(r"\b(TODO|FIXME|TBD|implement later|omitted for brevity)\b")
    skip = {".git", "tests", ".venv", "__pycache__", "mlruns"}
    for p in ROOT.rglob("*"):
        if (
            not p.is_file()
            or set(p.relative_to(ROOT).parts) & skip
            or p.suffix not in {".py", ".yaml", ".yml", ".toml", ".md", ".txt", ".ps1"}
        ):
            continue
        text = p.read_text(errors="ignore")
        assert not pat.search(text), f"possible hardcoded secret in {p}"
        if p.name != "test_deploy_consistency.py":
            assert not marker.search(text), f"placeholder marker in {p}"
    env = (ROOT / ".env.example").read_text()
    assert re.search(r"^WANDB_API_KEY=$", env, re.M)


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_real_uvicorn_server_and_http_benchmark(model_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("TRUSTLENS_MODEL_DIR", str(model_dir))
    reset_service()
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    try:
        import urllib.request

        for _ in range(100):
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/api/v1/health", timeout=1
                ) as r:
                    assert json.loads(r.read())["model_loaded"] is True
                    break
            except OSError:
                time.sleep(0.1)
        else:
            pytest.fail("server did not start")
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/docs", timeout=5) as r:
            assert r.status == 200
        out = tmp_path / "b.json"
        r = subprocess.run(
            [
                sys.executable,
                "scripts/benchmark.py",
                "--url",
                f"http://127.0.0.1:{port}",
                "--n",
                "20",
                "--out",
                str(out),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, r.stderr[-800:]
        res = json.loads(out.read_text())
        ep = res["predict_endpoint"]
        assert ep["n"] == 20 and 0 < ep["p50_ms"] <= ep["p95_ms"] <= ep["max_ms"]
        assert res["batch"]["reviews_per_second"] > 0 and res["environment"]["cpu_count"]
    finally:
        server.should_exit = True
        t.join(timeout=10)
        reset_service()


def test_inprocess_benchmark_and_missing_model(model_dir, tmp_path):
    out = tmp_path / "b.json"
    r = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark.py",
            "--model-dir",
            str(model_dir),
            "--n",
            "15",
            "--batch-size",
            "20",
            "--out",
            str(out),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr[-800:]
    res = json.loads(out.read_text())
    s = res["single_review"]
    assert s["n"] == 15 and 0 < s["p50_ms"] <= s["p95_ms"] and res["batch"]["batch_size"] == 20
    assert res["model_version"]
    bad = subprocess.run(
        [sys.executable, "scripts/benchmark.py", "--model-dir", str(tmp_path / "none")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert bad.returncode != 0 and "No trained model" in bad.stderr
