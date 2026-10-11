# TrustLens image: one image, several entrypoints (API, Streamlit, Gradio). No secrets are baked in.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    TRUSTLENS_MODEL_DIR=/app/artifacts/model

WORKDIR /app

COPY requirements.txt requirements-cv.txt ./
RUN pip install -r requirements.txt
# Optional computer-vision extras: docker build --build-arg WITH_CV=1 .
ARG WITH_CV=0
RUN if [ "$WITH_CV" = "1" ]; then pip install opencv-python-headless Pillow imagehash; fi

COPY src ./src
COPY api ./api
COPY streamlit_app ./streamlit_app
COPY gradio_app ./gradio_app
COPY robustness ./robustness
COPY configs ./configs

RUN useradd --create-home --uid 10001 trustlens && chown -R trustlens /app
USER trustlens

# 8000 FastAPI | 5000 Flask | 8501 Streamlit | 7860 Gradio
EXPOSE 8000 5000 8501 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/api/v1/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "api.fastapi_app.main:app", "--host", "0.0.0.0", "--port", "8000"]
