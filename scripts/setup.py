"""Usage: python scripts/setup.py  -- creates folders, .env, and reports which optional packages are installed."""

import importlib.util
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
DIRS = ["data/raw", "data/interim", "data/processed", "data/external", "artifacts", "reports"]
CORE = ["numpy", "pandas", "sklearn", "yaml", "joblib", "fastapi", "streamlit", "shap", "lime"]
OPTIONAL = {
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "catboost": "catboost",
    "mlflow": "mlflow",
    "wandb": "wandb",
    "cv2": "OpenCV (CV)",
    "ultralytics": "YOLO (CV)",
    "autogluon": "AutoGluon",
    "h2o": "H2O",
    "autosklearn": "auto-sklearn",
    "kfp": "Kubeflow Pipelines",
    "gradio": "Gradio",
}

if __name__ == "__main__":
    if sys.version_info < (3, 10):
        sys.exit("Python 3.10+ required")
    for d in DIRS:
        (ROOT / d).mkdir(parents=True, exist_ok=True)
    env, example = ROOT / ".env", ROOT / ".env.example"
    if not env.exists() and example.exists():
        shutil.copy(example, env)
        print("created .env from .env.example (add your own values; never commit it)")
    has = lambda m: importlib.util.find_spec(m) is not None  # noqa: E731
    missing = [m for m in CORE if not has(m)]
    print(
        "core packages:",
        "all present" if not missing else f"MISSING {missing} -> pip install -r requirements.txt",
    )
    for mod, label in OPTIONAL.items():
        print(
            f"optional {label:<22} {'available' if has(mod) else 'not installed (feature disabled gracefully)'}"
        )
