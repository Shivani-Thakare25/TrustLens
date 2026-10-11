import base64

import cv2
import pytest
from fastapi.testclient import TestClient

from api.fastapi_app.main import app
from api.services.trustlens_service import reset_service
from src.computer_vision import image_analysis as ia
from tests.unit.test_computer_vision import make_img

V1 = "/api/v1"
TEXT = {"review_text": "Great camera but terrible battery. Best product ever!!!"}


@pytest.fixture()
def client(model_dir, monkeypatch):
    monkeypatch.setenv("TRUSTLENS_MODEL_DIR", str(model_dir))
    reset_service()
    with TestClient(app) as c:
        yield c
    reset_service()


def b64(img):
    return base64.b64encode(cv2.imencode(".png", img)[1].tobytes()).decode()


def _img_signal(body):
    return next(s for s in body["evidence"]["signals"] if s["signal_name"] == "image_evidence")


@pytest.mark.skipif(not ia.cv_available(), reason="OpenCV not installed")
def test_image_analysis_and_duplicate_evidence(client):
    first = client.post(f"{V1}/predict", json={**TEXT, "image_base64": b64(make_img(3))}).json()
    assert (
        first["image_analysis"]["available"]
        and first["image_analysis"]["duplicate_image_flag"] is False
    )
    assert _img_signal(first)["availability"] is True and _img_signal(first)["signal_value"] == 0.0
    second = client.post(f"{V1}/predict", json={**TEXT, "image_base64": b64(make_img(3))}).json()
    assert second["image_analysis"]["duplicate_image_flag"] is True
    assert (
        _img_signal(second)["signal_value"] == 1.0
        and _img_signal(second)["direction"] == "increases_risk"
    )


def test_no_image_leaves_image_evidence_unavailable(client):
    b = client.post(f"{V1}/predict", json=TEXT).json()
    assert "image_analysis" not in b or b["image_analysis"] is None
    assert _img_signal(b)["availability"] is False


@pytest.mark.skipif(not ia.cv_available(), reason="OpenCV not installed")
@pytest.mark.parametrize(
    "img",
    [
        base64.b64encode(b"GIF89a" + b"0" * 40).decode(),
        base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"junk" * 9).decode(),
        "!!not-base64!!",
    ],
)
def test_bad_images_give_400(client, img):
    r = client.post(f"{V1}/predict", json={**TEXT, "image_base64": img})
    assert r.status_code == 400 and "request_id" in r.json()


def test_oversize_image_string_rejected(client):
    assert (
        client.post(f"{V1}/predict", json={**TEXT, "image_base64": "A" * 7_000_001}).status_code
        == 422
    )


def test_batch_rejects_images(client):
    r = client.post(f"{V1}/batch-predict", json={"reviews": [{**TEXT, "image_base64": "AAAA"}]})
    assert r.status_code == 400


def test_graceful_without_opencv(client, monkeypatch):
    monkeypatch.setattr(ia, "cv_available", lambda: False)
    r = client.post(f"{V1}/predict", json={**TEXT, "image_base64": "AAAA"})
    assert r.status_code == 200
    b = r.json()
    assert b["image_analysis"]["available"] is False and "OpenCV" in b["image_analysis"]["reason"]
    assert _img_signal(b)["availability"] is False
