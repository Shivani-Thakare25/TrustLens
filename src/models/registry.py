"""Local file-based model registry: artifacts/registry/<version>/ + registry.json index."""

from __future__ import annotations

import json
import shutil
from pathlib import Path


def register_model(model_dir, registry_dir="artifacts/registry", stage: str = "candidate") -> dict:
    model_dir, reg = Path(model_dir), Path(registry_dir)
    meta_path = model_dir / "metadata.json"
    if not (model_dir / "model.joblib").exists() or not meta_path.exists():
        raise FileNotFoundError(f"No trained model artifacts in {model_dir}")
    meta = json.loads(meta_path.read_text())
    dest = reg / meta["model_version"]
    if dest.exists():
        raise FileExistsError(f"Version {meta['model_version']} is already registered")
    shutil.copytree(model_dir, dest)
    index_path = reg / "registry.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else []
    entry = {
        "model_version": meta["model_version"],
        "model_name": meta["model_name"],
        "stage": stage,
        "label_semantics": meta["label_semantics"],
        "path": str(dest),
        "training_date": meta["training_date"],
    }
    index.append(entry)
    index_path.write_text(json.dumps(index, indent=2))
    return entry
