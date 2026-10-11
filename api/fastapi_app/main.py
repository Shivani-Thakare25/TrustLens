"""FastAPI app (primary API). Run: uvicorn api.fastapi_app.main:app --port 8000. Docs: /docs"""

from __future__ import annotations

import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.fastapi_app.schemas import BatchRequest, PredictResponse, ReviewRequest
from api.services.trustlens_service import ModelNotLoadedError, get_service
from src.utils.logger import get_logger

log = get_logger("trustlens.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_service()  # load model once at startup
    yield


app = FastAPI(
    title="TrustLens API",
    version="0.1.0",
    lifespan=lifespan,
    description="Explainable review integrity analysis. Outputs are model predictions, not proof.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("TRUSTLENS_CORS_ORIGINS", "http://localhost:8501").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = rid
    t0 = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    log.info(
        "request_id=%s %s %s status=%s ms=%.1f model=%s",
        rid,
        request.method,
        request.url.path,
        response.status_code,
        (time.perf_counter() - t0) * 1000,
        get_service().p.version,
    )
    return response


def _err(request: Request, status: int, msg: str):
    return JSONResponse(
        status_code=status,
        content={"error": msg, "request_id": getattr(request.state, "request_id", "unknown")},
    )


@app.exception_handler(ModelNotLoadedError)
async def _no_model(request: Request, exc: ModelNotLoadedError):
    return _err(request, 503, str(exc))


@app.exception_handler(ValueError)
async def _bad_value(request: Request, exc: ValueError):
    return _err(request, 400, str(exc))


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):
    log.error(
        "request_id=%s unhandled %s",
        getattr(request.state, "request_id", "unknown"),
        type(exc).__name__,
    )
    return _err(request, 500, "Internal server error")


v1 = APIRouter(prefix="/api/v1")


@v1.get("/health")
def health():
    h = get_service().health()
    return JSONResponse(status_code=200 if h["model_loaded"] else 503, content=h)


@v1.post("/predict", response_model=PredictResponse)
def predict(req: ReviewRequest):
    return get_service().predict(req.model_dump())


@v1.post("/batch-predict")
def batch_predict(req: BatchRequest):
    return get_service().batch_predict(req.model_dump())


@v1.post("/aspect-sentiment")
def aspect_sentiment(req: ReviewRequest):
    return get_service().aspect_sentiment(req.model_dump())


@v1.post("/explain")
def explain(req: ReviewRequest):
    return get_service().explain(req.model_dump())


@v1.post("/counterfactual")
def counterfactual(req: ReviewRequest):
    return get_service().counterfactual(req.model_dump())


@v1.post("/similar-reviews")
def similar_reviews(req: ReviewRequest):
    return get_service().similar_reviews(req.model_dump())


@v1.get("/model-info")
def model_info():
    return get_service().model_info()


@v1.get("/features")
def features():
    return get_service().features()


@v1.get("/metrics")
def metrics():
    return get_service().metrics()


@v1.get("/monitoring/drift")
def drift():
    return get_service().drift()


app.include_router(v1)
