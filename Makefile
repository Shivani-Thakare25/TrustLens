# Linux/macOS/WSL2. On Windows without make, use the PowerShell commands in README.md.
PY ?= python
DATA ?= data/raw/reviews.csv
LABELS ?= unknown

.PHONY: install setup test lint format pipeline train evaluate benchmark robustness api flask streamlit gradio docker-build docker-up mlflow

install:
	$(PY) -m pip install -r requirements-dev.txt
setup:
	$(PY) scripts/setup.py
test:
	$(PY) -m pytest -q
lint:
	ruff check .
	black --check .
	isort --check-only .
format:
	isort .
	black .
pipeline:
	$(PY) scripts/run_all.py $(DATA) --label-semantics $(LABELS)
train:
	$(PY) scripts/train.py data/processed/reviews.csv --label-semantics $(LABELS)
evaluate:
	$(PY) scripts/evaluate.py
benchmark:
	$(PY) scripts/benchmark.py
robustness:
	$(PY) scripts/robustness.py data/processed/reviews.csv
api:
	uvicorn api.fastapi_app.main:app --host 0.0.0.0 --port 8000
flask:
	$(PY) -m api.flask_app.app
streamlit:
	streamlit run streamlit_app/app.py --server.port 8501
gradio:
	$(PY) -m gradio_app.app
docker-build:
	docker build -t trustlens:latest .
docker-up:
	docker compose up --build
mlflow:
	mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
