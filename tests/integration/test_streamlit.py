from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from api.services.trustlens_service import reset_service
from streamlit_app.views import PAGES

APP = str(Path(__file__).resolve().parents[2] / "streamlit_app" / "app.py")


@pytest.fixture()
def env_model(model_dir, monkeypatch):
    monkeypatch.setenv("TRUSTLENS_MODEL_DIR", str(model_dir))
    reset_service()
    yield
    reset_service()


def _run(page=None, mode=None):
    at = AppTest.from_file(APP, default_timeout=60).run()
    if page:
        at.sidebar.radio(key="page").set_value(page).run()
    if mode:
        at.sidebar.radio(key="mode").set_value(mode).run()
    return at


def test_six_pages_defined():
    assert len(PAGES) == 6
    expected = {
        "Home / Overview",
        "Single Review Analysis",
        "Aspect Sentiment",
        "Trust Analytics & Sentiment Impact",
        "Model Performance",
        "About / Methodology",
    }
    assert set(PAGES.keys()) == expected


def test_every_page_renders_with_model(env_model):
    for page in PAGES:
        at = _run(page)
        assert not at.exception, f"{page}: {at.exception}"


def test_every_page_renders_without_model(tmp_path, monkeypatch):
    monkeypatch.setenv("TRUSTLENS_MODEL_DIR", str(tmp_path))
    reset_service()
    for page in PAGES:
        at = _run(page)
        assert not at.exception, f"{page}: {at.exception}"
    reset_service()


def test_single_review_consumer_flow(env_model):
    at = _run("Single Review Analysis")
    at.button(key="single_go").click().run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert {"Trust Score", "Fake Probability", "Confidence", "Overall Sentiment"} <= set(labels)
    assert any("Classification:" in s.value for s in at.subheader)


def test_investigation_mode_shows_more(env_model):
    cons = _run("Single Review Analysis", "Consumer")
    cons.button(key="single_go").click().run()
    inv = _run("Single Review Analysis", "Investigation")
    inv.button(key="single_go").click().run()
    assert not inv.exception
    assert len(inv.dataframe) >= len(cons.dataframe)


def test_single_review_without_model_shows_error(tmp_path, monkeypatch):
    monkeypatch.setenv("TRUSTLENS_MODEL_DIR", str(tmp_path))
    reset_service()
    at = _run("Single Review Analysis")
    at.button(key="single_go").click().run()
    assert not at.exception and any("Model not loaded" in e.value for e in at.error)
    reset_service()


def test_blank_review_shows_validation_error(env_model):
    at = _run("Single Review Analysis")
    at.text_area(key="single_text").set_value("   ").run()
    at.button(key="single_go").click().run()
    assert not at.exception and any("non-empty" in e.value for e in at.error)


def test_aspect_page_works(env_model):
    at = _run("Aspect Sentiment")
    at.button(key="aspect_go").click().run()
    assert not at.exception and at.dataframe


def test_trust_analytics_sentiment_impact_renders(env_model):
    at = _run("Trust Analytics & Sentiment Impact")
    assert not at.exception
    assert any("Run Corpus Analysis" in b.label for b in at.button)
