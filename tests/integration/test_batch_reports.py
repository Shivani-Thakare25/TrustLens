import json

import pandas as pd
import pytest

from src.inference.batch import analyze_dataframe
from src.inference.predictor import TrustLensPredictor
from src.reporting.report_generator import build_report, to_csv, to_html, to_json
from src.visualization import plotly_charts as pc
from src.visualization.bokeh_plots import risk_scatter


def make_batch_df():
    """SYNTHETIC demonstration data."""
    rows = []
    for i in range(40):
        rows.append(
            {
                "review_id": f"b{i}",
                "user_id": f"u{i % 15}",
                "product_id": "p1" if i < 30 else "p2",
                "review_text": f"The hinge squeaked and delivery took {i % 9 + 2} days. Battery average.",
                "rating": 3,
                "review_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i * 3),
                "verified_purchase": "yes",
            }
        )
    for i in range(8):
        rows.append(
            {
                "review_id": f"c{i}",
                "user_id": f"x{i}",
                "product_id": "p1",
                "review_text": "Best product ever!!! Highly recommend, must buy now, five stars!!!",
                "rating": 5,
                "review_date": pd.Timestamp("2024-06-10"),
                "verified_purchase": "no",
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def predictor(model_dir):
    return TrustLensPredictor(model_dir)


def test_batch_analysis_outputs_and_context(predictor):
    out = analyze_dataframe(predictor, make_batch_df())
    res = out["results"]
    assert len(res) == 48 and out["stats"]["analyzed"] == 48
    assert set(
        ["fake_probability", "trust_score", "cluster_id", "burst_day", "top_reasons"]
    ) <= set(res.columns)
    assert (res["risk_score"] + res["trust_score"] == 100).all()
    c_rows = res[res["review_id"].str.startswith("c")]
    assert c_rows["cluster_id"].notna().all() and c_rows["cluster_id"].nunique() == 1
    assert (
        c_rows["fake_probability"].mean()
        > res[res["review_id"].str.startswith("b")]["fake_probability"].mean()
    )
    assert out["clusters"][0]["label"] == "Potentially coordinated review cluster"


def test_batch_errors(predictor):
    with pytest.raises(ValueError):
        analyze_dataframe(predictor, pd.DataFrame({"rating": [1]}))
    with pytest.raises(ValueError):
        analyze_dataframe(predictor, pd.DataFrame({"review_text": ["a", "b"]}), max_rows=1)
    with pytest.raises(ValueError):
        analyze_dataframe(predictor, pd.DataFrame({"review_text": ["", "  "]}))


def test_batch_handles_text_only_csv(predictor):
    out = analyze_dataframe(
        predictor,
        pd.DataFrame({"review_text": ["Battery died fast", "Great camera, slow delivery"]}),
    )
    assert len(out["results"]) == 2 and out["burst_days"] == {}


def test_reports_are_consistent_and_escape_html(predictor):
    review = {
        "review_text": "<script>alert(1)</script> Best product ever!!! Highly recommend!",
        "rating": 5,
    }
    a = predictor.analyze(review)
    rep = build_report(a, predictor.metadata, review)
    j = json.loads(to_json(rep))
    assert j["prediction"]["trust_score"] == a["trust_score"] and j["limitations"]
    rows = to_csv(rep).strip().splitlines()
    assert len(rows) == 1 + len(a["evidence"]["signals"])
    h = to_html(rep)
    assert "<script>alert" not in h and "&lt;script&gt;" in h
    assert "Predicted suspicious" in h or "Likely genuine" in h
    assert "review_text" not in build_report(a, None, review, include_text=False)["review"]


def test_charts_handle_empty_and_data(predictor):
    a = predictor.analyze(
        {"review_text": "Great camera but terrible battery. Best product ever!!!"}
    )
    assert pc.evidence_breakdown_chart(a["evidence"]["breakdown"]).data
    assert pc.aspect_chart(a["aspects"]).data and not pc.aspect_chart([]).data
    assert not pc.cluster_chart([]).data and not pc.rating_distribution_chart(None).data
    out = analyze_dataframe(predictor, make_batch_df())
    assert pc.cluster_chart(out["clusters"]).data
    assert risk_scatter(out["results"]).renderers
    assert pc.model_comparison_chart(predictor.metadata["all_model_results"]).data
