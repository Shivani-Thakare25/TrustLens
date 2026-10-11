from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = [
    "README.md",
    "PHASE_27_29_COMPLETION.md",
    "RELEASE_MANIFEST.json",
    "requirements.txt",
    "requirements-dev.txt",
    "Dockerfile",
    "docker-compose.yml",
    "dvc.yaml",
    "pyproject.toml",
    "api/fastapi_app/main.py",
    "api/flask_app/app.py",
    "streamlit_app/app.py",
    "gradio_app/app.py",
    "kubeflow/pipeline.py",
    "k8s/api-deployment.yaml",
    "k8s/streamlit-deployment.yaml",
    ".github/workflows/tests.yml",
]

def main() -> int:
    missing = [p for p in REQUIRED if not (ROOT / p).exists()]
    if missing:
        print("Missing required files:")
        for p in missing:
            print(" -", p)
        return 1

    marker = re.compile(r"\b(Pending item|Follow-up item|Pending verification|planned implementation|not included in this document)\b")
    violations = []
    skip_parts = {".git", ".venv", "__pycache__", "tests", "mlruns"}
    extensions = {".py", ".yaml", ".yml", ".md", ".txt", ".toml", ".ps1"}

    for p in ROOT.rglob("*"):
        if not p.is_file() or set(p.relative_to(ROOT).parts) & skip_parts or p.relative_to(ROOT).as_posix() in {"PHASE_27_29_COMPLETION.md", "docs/release-checklist.md"}:
            continue
        if p.suffix not in extensions or p.name in {"test_deploy_consistency.py", "validate_release.py"}:
            continue
        try:
            content = p.read_text(errors="ignore")
        except OSError:
            continue
        if marker.search(content):
            violations.append(str(p.relative_to(ROOT)))

    env = ROOT / ".env.example"
    if env.exists() and not re.search(r"^WANDB_API_KEY=$", env.read_text(), re.M):
        violations.append(".env.example: WANDB_API_KEY must be empty")

    if violations:
        print("Release hygiene violations:")
        for p in violations:
            print(" -", p)
        return 1

    print("TrustLens release structure: PASS")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

