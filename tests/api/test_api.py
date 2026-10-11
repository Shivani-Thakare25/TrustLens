import pytest
from fastapi.testclient import TestClient

from api.fastapi_app.main import app
from api.flask_app.app import create_app
from api.services.trustlens_service import reset_service

REVIEW = {
    "review_text": "The battery is excellent and the camera quality is amazing. Best product ever! "
    "Highly recommended to everyone!!!",
    "rating": 5,
    "verified_purchase": True,
}
V1 = "/api/v1"


@pytest.fixture()
def client(model_dir, monkeypatch):
    monkeypatch.setenv("TRUSTLENS_MODEL_DIR", str(model_dir))
    reset_service()
    with TestClient(app) as c:
        yield c
    reset_service()


@pytest.fixture()
def no_model_client(tmp_path, monkeypatch):
    monkeypatch.setenv("TRUSTLENS_MODEL_DIR", str(tmp_path))
    reset_service()
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    reset_service()


def test_health_ok(client):
    r = client.get(f"{V1}/health")
    assert r.status_code == 200 and r.json()["model_loaded"] is True
    assert r.json()["model_version"] and r.headers["X-Request-ID"]


def test_health_unhealthy_without_model(no_model_client):
    r = no_model_client.get(f"{V1}/health")
    assert (
        r.status_code == 503
        and r.json()["status"] == "unhealthy"
        and r.json()["model_loaded"] is False
    )


def test_predict_contract_and_consistency(client):
    r = client.post(f"{V1}/predict", json=REVIEW)
    assert r.status_code == 200
    b = r.json()
    assert abs(b["fake_probability"] + b["genuine_probability"] - 1) < 1e-9
    assert b["risk_score"] + b["trust_score"] == 100
    assert b["classification"] in ("Predicted suspicious", "Likely genuine")
    assert {a["aspect"] for a in b["aspects"]} >= {"battery", "camera"}
    names = {s["signal_name"] for s in b["evidence"]["signals"]}
    assert {"promotional_language", "image_evidence"} <= names
    assert b["counterfactuals"]["disclaimer"]
    img = next(s for s in b["evidence"]["signals"] if s["signal_name"] == "image_evidence")
    assert img["availability"] is False and img["signal_value"] is None


def test_predict_missing_model_is_503(no_model_client):
    r = no_model_client.post(f"{V1}/predict", json=REVIEW)
    assert r.status_code == 503 and "request_id" in r.json()


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"review_text": ""},
        {"review_text": "   "},
        {"review_text": "ok", "rating": 9},
        {"review_text": "ok", "rating": "x"},
        {"review_text": "x" * 20001},
    ],
)
def test_predict_rejects_bad_input(client, body):
    assert client.post(f"{V1}/predict", json=body).status_code == 422


def test_malformed_json_rejected(client):
    r = client.post(
        f"{V1}/predict", content="{not json", headers={"Content-Type": "application/json"}
    )
    assert r.status_code == 422


def test_missing_optional_metadata_ok(client):
    r = client.post(f"{V1}/predict", json={"review_text": "Screen is bright but battery died."})
    assert r.status_code == 200


def test_batch_predict_and_limits(client):
    r = client.post(
        f"{V1}/batch-predict", json={"reviews": [REVIEW, {"review_text": "battery died fast"}]}
    )
    assert r.status_code == 200 and r.json()["count"] == 2
    assert r.json()["results"][0]["counterfactuals"] == {}
    assert client.post(f"{V1}/batch-predict", json={"reviews": []}).status_code == 422
    assert client.post(f"{V1}/batch-predict", json={"reviews": [REVIEW] * 501}).status_code == 422


def test_aspect_sentiment_works_without_model(no_model_client):
    r = no_model_client.post(
        f"{V1}/aspect-sentiment", json={"review_text": "Great camera but late delivery."}
    )
    assert r.status_code == 200
    assert {a["aspect"]: a["sentiment"] for a in r.json()["aspects"]} == {
        "camera": "Positive",
        "delivery": "Negative",
    }


def test_explain_has_shap_and_lime(client):
    b = client.post(f"{V1}/explain", json=REVIEW).json()
    assert b["explanation"]["shap"]["available"] and b["explanation"]["lime"]["available"]
    assert b["explanation"]["title"] == "WHY DID TRUSTLENS FLAG THIS?"


def test_counterfactual_endpoint(client):
    b = client.post(f"{V1}/counterfactual", json=REVIEW).json()
    assert b["title"] == "Model Counterfactual Analysis" and b["changes"]


def test_similar_reviews_uses_saved_corpus(client):
    b = client.post(
        f"{V1}/similar-reviews",
        json={"review_text": "Best product ever!!! Highly recommend, must buy now! variant 1"},
    ).json()
    assert (
        b["available"]
        and b["similar_reviews"]
        and b["similar_reviews"][0]["similarity_score"] > 0.5
    )


def test_info_features_metrics_drift(client):
    info = client.get(f"{V1}/model-info").json()
    assert info["label_semantics"] == "synthetic" and 0.05 <= info["threshold"] <= 0.95
    assert "exclaim_count" in client.get(f"{V1}/features").json()["handcrafted_features"]
    client.post(f"{V1}/predict", json=REVIEW)
    m = client.get(f"{V1}/metrics").json()
    assert m["predictions_recorded"] >= 1 and m["latency_ms"]["p95"] >= m["latency_ms"]["p50"] > 0
    assert client.get(f"{V1}/monitoring/drift").json()["status"] == "insufficient_data"


def test_drift_ok_after_enough_requests(client):
    texts = ["Best product ever!!! Highly recommend, must buy now!"] * 40
    client.post(f"{V1}/batch-predict", json={"reviews": [{"review_text": t} for t in texts]})
    d = client.get(f"{V1}/monitoring/drift").json()
    assert d["status"] == "ok" and d["n_live"] >= 30 and "exclaim_count" in d["features"]


def test_openapi_lists_all_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    for p in [
        "predict",
        "batch-predict",
        "aspect-sentiment",
        "explain",
        "counterfactual",
        "similar-reviews",
        "health",
        "model-info",
        "features",
        "metrics",
        "monitoring/drift",
    ]:
        assert f"{V1}/{p}" in paths


def test_flask_uses_same_service_and_contract(model_dir, monkeypatch):
    monkeypatch.setenv("TRUSTLENS_MODEL_DIR", str(model_dir))
    reset_service()
    c = create_app().test_client()
    assert c.get(f"{V1}/health").status_code == 200
    b = c.post(f"{V1}/predict", json=REVIEW).get_json()
    assert b["risk_score"] + b["trust_score"] == 100
    assert c.post(f"{V1}/predict", json={"review_text": ""}).status_code == 400
    assert c.post(f"{V1}/predict", data="garbage").status_code == 400
    reset_service()


def test_flask_missing_model(tmp_path, monkeypatch):
    monkeypatch.setenv("TRUSTLENS_MODEL_DIR", str(tmp_path))
    reset_service()
    c = create_app().test_client()
    assert c.get(f"{V1}/health").status_code == 503
    assert c.post(f"{V1}/predict", json=REVIEW).status_code == 503
    reset_service()
