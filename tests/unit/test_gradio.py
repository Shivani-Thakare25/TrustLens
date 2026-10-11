import pytest

from api.services.trustlens_service import reset_service
from gradio_app.app import analyze_review


@pytest.fixture()
def with_model(model_dir, monkeypatch):
    monkeypatch.setenv("TRUSTLENS_MODEL_DIR", str(model_dir))
    reset_service()
    yield
    reset_service()


def test_gradio_function_outputs(with_model):
    cls, prob, trust, sent, rows, evidence, expl = analyze_review(
        "Great camera but the battery is terrible. Best product ever!!!", 5, "Yes"
    )
    assert cls in ("Predicted suspicious", "Likely genuine")
    assert 0 <= prob <= 1
    assert abs(trust - (100 - prob * 100)) <= 1
    assert {r[0] for r in rows} == {"camera", "battery"}
    assert "WHY DID TRUSTLENS FLAG THIS?" in expl and "image_evidence" in evidence


def test_gradio_errors_are_messages(tmp_path, monkeypatch):
    monkeypatch.setenv("TRUSTLENS_MODEL_DIR", str(tmp_path))
    reset_service()
    assert analyze_review("text", 0, "Not provided")[0].startswith("Error")
    reset_service()


def test_gradio_empty_text_message(with_model):
    assert analyze_review("  ", 0, "Not provided")[0].startswith("Error")


def test_gradio_blocks_build():
    gr = pytest.importorskip("gradio")
    assert isinstance(__import__("gradio_app.app", fromlist=["x"]).build_demo(), gr.Blocks)
