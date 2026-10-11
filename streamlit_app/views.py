"""Page implementations. All analysis goes through the shared service layer."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from api.services.trustlens_service import ModelNotLoadedError, get_service
from src.inference.batch import analyze_dataframe
from src.product_intelligence.product_profile import product_profile
from src.product_intelligence.reviewer_profile import reviewer_profile
from src.product_intelligence.trust_adjusted_sentiment import trust_adjusted_rating
from src.reporting.report_generator import build_report, to_csv, to_html, to_json
from src.similarity.duplicate_detector import find_exact_duplicates
from src.visualization import plotly_charts as pc

try:
    from src.visualization.bokeh_plots import risk_scatter
except ImportError:
    risk_scatter = None

EXAMPLE = (
    "The battery is excellent and the camera quality is amazing. Best product ever! "
    "Highly recommended to everyone!!!"
)
MAX_UPLOAD_MB = 10
TRAIN_HINT = "Trained model artifacts are expected in artifacts/model and models/."


def _call(fn, *args):
    try:
        return fn(*args)
    except ModelNotLoadedError:
        st.error(f"Model not loaded. {TRAIN_HINT}")
    except ValueError as e:
        st.error(str(e))
    return None


def _payload(prefix: str, default: str = "", with_image: bool = False):
    text = st.text_area("Review text", value=default, key=f"{prefix}_text", height=150)
    c1, c2 = st.columns(2)
    rating = c1.selectbox("Rating", ["Not provided", 1, 2, 3, 4, 5], key=f"{prefix}_rating")
    verified = c2.selectbox(
        "Verified purchase", ["Not provided", "Yes", "No"], key=f"{prefix}_verified"
    )
    p = {"review_text": text}
    if rating != "Not provided":
        p["rating"] = float(rating)
    if verified != "Not provided":
        p["verified_purchase"] = verified == "Yes"
    return p


def _signal_lines(signals):
    for s in signals:
        if not s["availability"]:
            continue
        icon = "⚠" if s["direction"] == "increases_risk" else "✓"
        st.write(f"{icon} {s['explanation']}")


def _render_result(r: dict, payload: dict, xai: dict | None = None):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trust Score", f"{r['trust_score']}/100")
    c2.metric("Fake Probability", f"{r['fake_probability']:.0%}")
    c3.metric("Confidence", f"{r['confidence']:.0%}")
    c4.metric("Overall Sentiment", r["overall_sentiment"])
    st.subheader(f"Classification: {r['classification']}")
    st.caption(
        f"Decision band: {r['decision_band']} (application-level band, not scientific truth). "
        "Fake probability is the model's calibrated probability; the trust score is derived from it."
    )
    st.markdown("**Main signals**")
    _signal_lines(r["evidence"]["signals"])
    unavailable = [s["signal_name"] for s in r["evidence"]["signals"] if not s["availability"]]
    if unavailable:
        st.caption("Not available for this input: " + ", ".join(unavailable))
    st.markdown("**WHY DID TRUSTLENS FLAG THIS?**")
    for i, reason in enumerate(r["explanation"]["reasons"], 1):
        st.write(f"{i}. {reason}")

    if r.get("aspects"):
        st.markdown("**Aspect sentiment**")
        st.dataframe(pd.DataFrame(r["aspects"]), use_container_width=True)

    if r.get("evidence", {}).get("breakdown"):
        st.plotly_chart(pc.evidence_breakdown_chart(r["evidence"]["breakdown"]), use_container_width=True)
    if r.get("evidence", {}).get("signals"):
        st.markdown("**All evidence signals**")
        st.dataframe(pd.DataFrame(r["evidence"]["signals"]), use_container_width=True)
    cf = r.get("counterfactuals") or {}
    if cf.get("changes"):
        st.markdown(f"**{cf['title']}**")
        st.caption(cf["disclaimer"])
        st.dataframe(pd.DataFrame(cf["changes"]), use_container_width=True)
    if xai:
        ex = xai.get("explanation", {})
        if ex.get("shap") and ex["shap"].get("available"):
            st.plotly_chart(
                pc.token_chart(
                    ex["shap"]["contributions"], "shap_value", "SHAP Token Contributions"
                ),
                use_container_width=True,
            )
        if ex.get("lime") and ex["lime"].get("available"):
            st.plotly_chart(
                pc.token_chart(ex["lime"]["contributions"], "weight", "LIME Token Weights"),
                use_container_width=True,
            )
    if r.get("similar_reviews"):
        st.markdown("**Similar reviews**")
        st.dataframe(pd.DataFrame(r["similar_reviews"]), use_container_width=True)
    st.caption(f"Model version: {r['model_version']}")

    meta = _call(get_service().model_info)
    report = build_report(r, meta, {k: v for k, v in payload.items() if k != "image_base64"})
    d1, d2, d3 = st.columns(3)
    d1.download_button(
        "Download JSON", to_json(report), "trustlens_report.json", "application/json"
    )
    d2.download_button("Download CSV", to_csv(report), "trustlens_report.csv", "text/csv")
    d3.download_button("Download HTML", to_html(report), "trustlens_report.html", "text/html")


# ==============================================================================
# 1. HOME / OVERVIEW
# ==============================================================================
def home():
    st.title("TrustLens")
    st.subheader("Explainable multi-signal review integrity analysis")
    st.write(
        "TrustLens does not just ask *is this review fake?* It shows the measurable evidence behind a "
        "prediction: text signals, aspect consistency, corpus similarity, and model sensitivities. "
        "Results are **predictions, not proof**."
    )
    h = get_service().health()
    if h["model_loaded"]:
        info = _call(get_service().model_info) or {}
        st.success(f"Fake-Review Detection Model loaded: {info.get('model_name')} (version {h['model_version']})")
        if info.get("label_semantics") in ("synthetic", "unknown", "proxy", "weak"):
            st.caption(
                f"Training label semantics: '{info.get('label_semantics')}'. Probabilistic predictions based on learned stylometric signals."
            )
    else:
        st.warning(f"No trained fake-review model found. {TRAIN_HINT}")

    svc = get_service()
    if hasattr(svc.p, "sentiment_predictor") and svc.p.sentiment_predictor and svc.p.sentiment_predictor.loaded:
        st.success(f"Sentiment Analysis Model loaded: {svc.p.sentiment_predictor.model_type} (classes: {', '.join(svc.p.sentiment_predictor.classes)})")
    else:
        st.info("Sentiment Analysis: running with lexicon baseline.")

    st.write("---")
    st.markdown("### Research & Demonstration Navigation")
    st.markdown(
        "- **Single Review Analysis**: Multi-signal review inspection, trust scoring, aspect sentiment, and integrated explainability (SHAP token attribution and model counterfactuals).\n"
        "- **Aspect Sentiment**: Aspect-level extraction and context classification using the trained sentiment model.\n"
        "- **Trust Analytics & Sentiment Impact**: Corpus-level integrity statistics, duplicate review detection, and sentiment distribution comparison before vs after filtering suspicious reviews.\n"
        "- **Model Performance**: Held-out test evaluation benchmarks for both fake review detection and sentiment analysis models.\n"
        "- **About / Methodology**: Research methodology, operational terminology, and scope limitations."
    )


# ==============================================================================
# 2. SINGLE REVIEW ANALYSIS (WITH INTEGRATED EXPLAINABILITY)
# ==============================================================================
def single_review():
    st.header("Single Review Analysis")
    st.caption("Inspect a single review for suspiciousness, sentiment, aspect polarity, and underlying signal explanations.")
    payload = _payload("single", EXAMPLE, with_image=False)

    col1, col2 = st.columns([1, 2])
    with col1:
        run_btn = st.button("Analyze Review", key="single_go")
    with col2:
        explain_opt = st.checkbox(
            "Include Deep Explainability (SHAP token attribution)",
            key="single_xai_opt",
            help="Computes SHAP feature attribution tokens. Takes a few seconds.",
        )

    if run_btn:
        r = _call(get_service().predict, payload)
        xai_res = _call(get_service().explain, payload) if (explain_opt and r) else None
        if r:
            st.session_state["single_result"] = (r, payload, xai_res)

    if "single_result" in st.session_state:
        saved = st.session_state["single_result"]
        if len(saved) == 3:
            _render_result(saved[0], saved[1], xai=saved[2])
        else:
            _render_result(saved[0], saved[1])


# ==============================================================================
# 3. ASPECT SENTIMENT
# ==============================================================================
def aspect_page():
    st.header("Aspect Sentiment")
    st.caption("Aspect-level sentiment extraction and context classification using the trained sentiment model.")
    text = st.text_area(
        "Review text",
        value="The phone has an excellent camera but the battery is disappointing and delivery was late.",
        key="aspect_text",
        height=120,
    )
    if st.button("Extract aspects", key="aspect_go"):
        r = _call(get_service().aspect_sentiment, {"review_text": text})
        if r:
            st.write(f"Overall sentiment: **{r['overall_sentiment']}**")
            st.plotly_chart(pc.aspect_chart(r["aspects"]), use_container_width=True)
            if r["aspects"]:
                st.dataframe(pd.DataFrame(r["aspects"]), use_container_width=True)
            else:
                st.info("No domain aspects were detected in this review text.")


# ==============================================================================
# 4. TRUST ANALYTICS & SENTIMENT IMPACT
# ==============================================================================
def analytics():
    st.header("Trust Analytics & Sentiment Impact")
    st.caption(
        "Corpus-level review integrity analysis, suspicious review rate estimation, "
        "and sentiment distribution comparison before and after suspicious-review filtering."
    )

    tab_src = st.radio(
        "Dataset Source",
        ["Demonstration Review Corpus (400 reviews)", "Upload Custom CSV"],
        horizontal=True,
        key="analytics_source_tab",
    )

    if tab_src == "Demonstration Review Corpus (400 reviews)":
        demo_path = Path("data/processed/demo_reviews.csv")
        if demo_path.exists():
            df_to_analyze = pd.read_csv(demo_path)
        else:
            ref_path = Path("artifacts/model/reference_corpus.csv")
            if ref_path.exists():
                df_to_analyze = pd.read_csv(ref_path)
            else:
                from scripts.train_demo import generate_demo_corpus
                df_to_analyze = generate_demo_corpus(n=400, seed=42)
        st.caption(f"Loaded built-in corpus ({len(df_to_analyze)} reviews).")
        if st.button("Run Corpus Analysis", key="demo_corpus_go"):
            out = _call(lambda: analyze_dataframe(get_service().p, df_to_analyze))
            if out:
                st.session_state["batch"] = out
    else:
        up = st.file_uploader(
            "Upload CSV (must contain column 'review_text')",
            type=["csv"],
            key="custom_csv_upload",
        )
        if up is not None:
            if up.size > MAX_UPLOAD_MB * 1024 * 1024:
                st.error(f"File exceeds {MAX_UPLOAD_MB} MB.")
            elif st.button("Run Corpus Analysis", key="custom_corpus_go"):
                try:
                    raw = pd.read_csv(up)
                    raw.columns = raw.columns.str.strip()
                    if "review_text" not in raw.columns and "text_" in raw.columns:
                        raw = raw.rename(columns={"text_": "review_text"})
                    if "review_text" not in raw.columns:
                        st.error(f"Invalid CSV: 'review_text' column is required. Columns found: {', '.join(raw.columns)}")
                        return
                    out = _call(lambda: analyze_dataframe(get_service().p, raw))
                    if out:
                        st.session_state["batch"] = out
                except Exception as e:
                    st.error(f"Could not parse CSV: {e}")
                    return

    b = st.session_state.get("batch")
    if b is None:
        st.info("Click **Run Corpus Analysis** above to compute company-level integrity and sentiment impact.")
        return

    res = b["results"]
    total_reviews = len(res)
    avg_trust = float(res["trust_score"].mean()) if total_reviews else 0.0
    suspicious_mask = res["classification"] == "Predicted suspicious"
    suspicious_count = int(suspicious_mask.sum())
    suspicious_pct = float(suspicious_mask.mean() * 100) if total_reviews else 0.0

    st.subheader("1. Company-Level Review Health Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Reviews Analyzed", f"{total_reviews:,}")
    c2.metric("Company Trust Health", f"{avg_trust:.1f} / 100")
    c3.metric("Suspicious Review Rate", f"{suspicious_pct:.1f}%")
    c4.metric("Flagged Reviews", f"{suspicious_count:,} / {total_reviews:,}")

    st.subheader("2. Sentiment Impact Before and After Suspicious-Review Filtering")
    st.caption(
        "Observed differences in corpus statistics before and after filtering suspicious reviews; "
        "does not establish causal proof."
    )

    # Trust-adjusted vs Raw Rating
    if "rating" in res and res["rating"].notna().any():
        raw_avg = float(res["rating"].mean())
        adj_avg = trust_adjusted_rating(res["rating"], res["fake_probability"])
        r1, r2, r3 = st.columns(3)
        r1.metric("Raw Average Rating", f"{raw_avg:.2f} / 5.0")
        r2.metric(
            "Trust-Adjusted Rating Estimate",
            f"{adj_avg:.2f} / 5.0" if adj_avg is not None else "Insufficient data",
        )
        if adj_avg is not None:
            r3.metric("Observed Rating Shift", f"{adj_avg - raw_avg:+.2f}")

    # Sentiment distribution before vs after filtering
    if "overall_sentiment" in res:
        before_counts = res["overall_sentiment"].value_counts()
        genuine_res = res[~suspicious_mask]
        after_counts = genuine_res["overall_sentiment"].value_counts()

        sent_classes = ["Positive", "Neutral", "Negative"]
        rows = []
        for s in sent_classes:
            b_cnt = int(before_counts.get(s, 0))
            b_pct = (b_cnt / total_reviews * 100) if total_reviews else 0.0
            a_cnt = int(after_counts.get(s, 0))
            a_pct = (a_cnt / len(genuine_res) * 100) if len(genuine_res) else 0.0
            shift = a_pct - b_pct
            rows.append({
                "Sentiment": s,
                "All Reviews (Count)": b_cnt,
                "All Reviews (%)": f"{b_pct:.1f}%",
                "After Filtering (Count)": a_cnt,
                "After Filtering (%)": f"{a_pct:.1f}%",
                "Observed Difference": f"{shift:+.1f}%",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # Charts
    st.subheader("3. Corpus Integrity Distributions")
    ch1, ch2 = st.columns(2)
    with ch1:
        st.plotly_chart(
            pc.distribution_chart(res["trust_score"], "Trust Score Distribution", "Trust score"),
            use_container_width=True,
        )
    with ch2:
        if "overall_sentiment" in res:
            st.plotly_chart(
                pc.sentiment_distribution_chart(res["overall_sentiment"].value_counts().to_dict()),
                use_container_width=True,
            )

    # Exact duplicates
    exact_dups = find_exact_duplicates(res["review_id"].tolist(), res["review_text"].tolist())
    if exact_dups:
        st.subheader(f"4. Exact Duplicate Review Groups ({len(exact_dups)} detected)")
        dup_rows = []
        for grp_idx, grp_ids in enumerate(exact_dups, 1):
            dup_text = res.loc[res["review_id"] == grp_ids[0], "review_text"].iloc[0]
            dup_rows.append(
                {
                    "Group": f"Group #{grp_idx}",
                    "Duplicates Count": len(grp_ids),
                    "Review IDs": ", ".join(grp_ids[:5]) + ("..." if len(grp_ids) > 5 else ""),
                    "Review Text Preview": dup_text[:120] + ("..." if len(dup_text) > 120 else ""),
                }
            )
        st.dataframe(pd.DataFrame(dup_rows), use_container_width=True)

    st.subheader("5. Analyzed Review Dataset")
    st.dataframe(res.head(200), use_container_width=True)
    d1, d2 = st.columns(2)
    d1.download_button(
        "Download Analyzed CSV",
        res.to_csv(index=False),
        "trustlens_corpus_analysis.csv",
        "text/csv",
    )
    d2.download_button(
        "Download Analyzed JSON",
        res.to_json(orient="records", date_format="iso"),
        "trustlens_corpus_analysis.json",
        "application/json",
    )


# ==============================================================================
# 5. MODEL PERFORMANCE
# ==============================================================================
def performance():
    st.header("Model Performance & Validation")
    st.caption(
        "Genuine evaluation metrics measured on held-out test splits without data leakage. "
        "No synthetic predictions or fabricated evaluation metrics."
    )

    svc = get_service()

    # 1. Fake Review Detection Model Evaluation
    st.subheader("1. Fake Review Detection Models")
    if svc.p.loaded:
        meta = svc.p.metadata
        st.caption(
            f"Active model: {meta.get('model_name')} | label semantics: {meta.get('label_semantics')} | "
            f"dataset: {meta.get('dataset_version')} | selection criterion: {meta.get('selection_criterion')} | "
            f"calibrated threshold: {meta.get('threshold', 0.5)}"
        )
        results = meta.get("all_model_results", {})
        rows = []
        for name, r in results.items():
            if r.get("status") != "trained":
                rows.append({"Model": name, "Status": r.get("status")})
                continue
            t = r.get("test", {})
            rows.append(
                {
                    "Model": name,
                    "Status": "trained",
                    **{
                        k.upper(): round(t[k], 4) if isinstance(t.get(k), (int, float)) else t.get(k)
                        for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier")
                        if k in t
                    },
                    "Train Time (s)": round(r.get("train_seconds", 0), 2),
                    "Latency (ms/review)": round(r.get("inference_ms_per_review", 0), 2),
                }
            )
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            st.plotly_chart(pc.model_comparison_chart(results), use_container_width=True)

    # Check for models/fake_review/evaluation.json (trained on real dataset)
    fake_eval_path = Path("models/fake_review/evaluation.json")
    if fake_eval_path.exists():
        try:
            fake_eval = json.loads(fake_eval_path.read_text())
            st.markdown("**Full Dataset Fake Review Benchmark (Held-Out Test Set)**")
            f_rows = []
            for m_name in ("logistic_regression", "random_forest"):
                if m_name in fake_eval:
                    f_rows.append({
                        "Model": m_name.replace("_", " ").title(),
                        "Accuracy": f"{fake_eval[m_name]['accuracy']:.1%}",
                        "Precision": f"{fake_eval[m_name]['precision']:.1%}",
                        "Recall": f"{fake_eval[m_name]['recall']:.1%}",
                        "F1 Score": f"{fake_eval[m_name]['f1']:.1%}",
                    })
            if f_rows:
                st.dataframe(pd.DataFrame(f_rows), use_container_width=True)
        except Exception:
            pass

    # 2. Sentiment Analysis Model Evaluation
    st.subheader("2. Sentiment Analysis Model (3-Class: Positive, Neutral, Negative)")
    st.caption("Trained on 111,696 Flipkart review corpus with leakage-safe stratified splitting.")

    # Check models/sentiment_neutral_optimized/evaluation.json
    opt_eval_path = Path("models/sentiment_neutral_optimized/evaluation.json")
    if opt_eval_path.exists():
        try:
            opt_eval = json.loads(opt_eval_path.read_text())
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Overall Accuracy", f"{opt_eval.get('accuracy', 0):.1%}")
            c2.metric("Weighted F1", f"{opt_eval.get('weighted_f1', 0):.1%}")
            c3.metric("Macro F1", f"{opt_eval.get('macro_f1', 0):.1%}")
            c4.metric("Test Set Size", f"{opt_eval.get('testing_rows', 0):,} reviews")

            # Classification report table
            cr = opt_eval.get("classification_report", {})
            cr_rows = []
            for label in ("positive", "neutral", "negative"):
                if label in cr:
                    cr_rows.append({
                        "Class": label.capitalize(),
                        "Precision": f"{cr[label].get('precision', 0):.1%}",
                        "Recall": f"{cr[label].get('recall', 0):.1%}",
                        "F1-Score": f"{cr[label].get('f1-score', 0):.1%}",
                        "Test Support": int(cr[label].get('support', 0)),
                    })
            if cr_rows:
                st.markdown("**Per-Class Held-Out Test Evaluation**")
                st.dataframe(pd.DataFrame(cr_rows), use_container_width=True)
        except Exception as e:
            st.warning(f"Could not load sentiment evaluation details: {e}")

    # Check baseline sentiment evaluation
    base_eval_path = Path("models/sentiment/evaluation.json")
    if base_eval_path.exists():
        try:
            base_eval = json.loads(base_eval_path.read_text())
            st.caption(
                f"Baseline sentiment model accuracy: {base_eval.get('accuracy', 0):.1%} "
                f"across {base_eval.get('dataset_rows', 0):,} total reviews."
            )
        except Exception:
            pass


# ==============================================================================
# 6. ABOUT / METHODOLOGY
# ==============================================================================
def about():
    st.header("About TrustLens & Research Methodology")
    st.markdown(
        "TrustLens is an explainable, multi-signal review integrity platform. Rather than treating "
        "review verification as a black-box binary decision, TrustLens combines measurable signals across "
        "multiple dimensions to provide transparent, interpretable assessments."
    )

    st.subheader("Core Methodological Signals")
    st.markdown(
        "1. **Text Stylometrics & Promotional Cues**: Measures punctuation, capitalization, lexical diversity, "
        "superlatives, urgency markers, and promotional phrases.\n"
        "2. **Aspect-Based Sentiment Consistency**: Extracts mentioned entities (battery, camera, delivery, quality) "
        "and checks whether aspect polarity aligns with overall review rating.\n"
        "3. **Corpus Similarity & Duplication**: Detects near-duplicate or exact-duplicate reviews across the product catalog.\n"
        "4. **Calibrated Trust & Risk Scoring**: Generates calibrated probabilities and maps them into clear decision bands.\n"
        "5. **Local Explainability & Counterfactuals**: Provides SHAP token-level attribution and counterfactual text sensitivities."
    )

    st.subheader("Standardized Terminology & Academic Honesty")
    st.markdown(
        "- **Predicted suspicious**: A probabilistic risk indicator based on learned patterns, not legal proof of fraud.\n"
        "- **Likely genuine**: Indicates no abnormal risk signals detected under the evaluated features.\n"
        "- **Trust-adjusted rating estimate**: An exploratory weighted heuristic, not a definitive ground-truth rating.\n"
        "- **Observed differences**: Statistical differences in sample distributions before and after filtering suspicious reviews; does not establish causal proof."
    )

    st.subheader("Limitations")
    st.markdown(
        "- Public dataset review labels often rely on synthetic or proxy annotations.\n"
        "- Behavioral metadata (user history, seller network) may be incomplete in third-party ingestion.\n"
        "- Model predictions are probabilistic and should serve as decision support for human moderators."
    )


# ==============================================================================
# PRESERVED CODE FOR UNCHANGED SUB-COMPONENTS (BACKWARD COMPATIBILITY)
# ==============================================================================
def _batch():
    b = st.session_state.get("batch")
    if b is None:
        st.info("No dataset analyzed yet. Upload a CSV or run the demo corpus on the **Trust Analytics & Sentiment Impact** page first.")
    return b


def clusters():
    st.header("Review Clusters")
    b = _batch()
    if b is None:
        return
    st.caption(
        "Groups of reviews with highly similar text. Shown as *potentially coordinated review clusters*; "
        "this is not confirmed fraud."
    )
    st.plotly_chart(pc.cluster_chart(b["clusters"]))
    for c in b["clusters"]:
        with st.expander(
            f"{c['cluster_id']} — risk {c['cluster_risk_score']} — {c['review_count']} reviews"
        ):
            for line in c["evidence"]:
                st.write("• " + line)
            st.write({k: c[k] for k in ("reviewer_count", "product_count")})


def reviewer():
    st.header("Reviewer Intelligence")
    b = _batch()
    if b is None:
        return
    res = b["results"]
    users = res["user_id"].dropna().unique().tolist()
    if not users:
        st.info("The dataset has no user_id column values; reviewer analysis is unavailable.")
        return
    uid = st.selectbox("Reviewer", users, key="rev_user")
    clustered = set(res.loc[res["cluster_id"].notna(), "review_id"])
    bursts = {d for days in b["burst_days"].values() for d in days}
    prof = reviewer_profile(res, uid, clustered_ids=clustered, burst_dates=bursts)
    st.subheader(prof["assessment"])
    st.caption("Heuristic from available signals; not a statement that the reviewer is fake.")
    st.json(prof)


def product():
    st.header("Product Intelligence")
    b = _batch()
    if b is None:
        return
    res = b["results"]
    products = res["product_id"].dropna().unique().tolist()
    if not products:
        st.info("The dataset has no product_id values; product analysis is unavailable.")
        return
    pid = st.selectbox("Product", products, key="prod_id")
    sub = res[res["product_id"] == pid]
    prof = product_profile(
        sub,
        suspicious_cluster_count=sub["cluster_id"].nunique(),
        burst_days=len(b["burst_days"].get(str(pid), [])),
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Reviews", prof["total_reviews"])
    c2.metric("Raw rating", "n/a" if prof["raw_rating"] is None else f"{prof['raw_rating']:.2f}")
    est = prof["trust_adjusted_rating_estimate"]
    c3.metric("Trust-adjusted rating estimate", "n/a" if est is None else f"{est:.2f}")
    st.caption(prof["note"])
    st.write(f"Suspicious reviews (model-predicted): {prof['suspicious_percentage']:.1f}%")
    st.plotly_chart(pc.rating_distribution_chart(prof["rating_distribution"]))


def explainable():
    st.header("Explainable AI")
    payload = _payload("xai", EXAMPLE)
    st.caption("SHAP and LIME can take several seconds.")
    if st.button("Explain", key="xai_go"):
        r = _call(get_service().explain, payload)
        if r:
            ex = r["explanation"]
            st.write(f"Model fake probability: **{r['fake_probability']:.1%}**")
            for i, reason in enumerate(ex["reasons"], 1):
                st.write(f"{i}. {reason}")
            if ex.get("shap") and ex["shap"]["available"]:
                st.plotly_chart(
                    pc.token_chart(
                        ex["shap"]["contributions"], "shap_value", "SHAP token contributions"
                    )
                )
            if ex.get("lime") and ex["lime"]["available"]:
                st.plotly_chart(
                    pc.token_chart(ex["lime"]["contributions"], "weight", "LIME token weights")
                )
            cf = _call(get_service().counterfactual, payload)
            if cf:
                st.subheader("Model Counterfactual Analysis")
                st.caption(cf["disclaimer"])
                st.dataframe(pd.DataFrame(cf["changes"]))


def similar():
    st.header("Similar Reviews")
    payload = _payload("sim", EXAMPLE)
    if st.button("Find similar reviews", key="sim_go"):
        r = _call(get_service().similar_reviews, payload)
        if r:
            if r["available"]:
                st.dataframe(pd.DataFrame(r["similar_reviews"]))
            else:
                st.info(r["reason"])


def investigation():
    single_review()


def batch():
    analytics()


def monitoring():
    st.header("Monitoring")
    svc = get_service()
    m = svc.metrics()
    if m.get("predictions_recorded", 0) == 0:
        st.info("No predictions recorded in this session yet.")
    else:
        st.json(m)
    st.subheader("Feature drift (PSI vs training baseline)")
    d = svc.drift()
    st.write(f"Status: **{d['status']}**")
    if d["status"] == "ok":
        st.write(f"Overall: {d['overall_level']} (max PSI {d['max_psi']:.3f})")
        st.dataframe(pd.DataFrame(d["features"]).T)


# ==============================================================================
# FINAL PREFERRED 6-PAGE NAVIGATION
# ==============================================================================
PAGES = {
    "Home / Overview": home,
    "Single Review Analysis": single_review,
    "Aspect Sentiment": aspect_page,
    "Trust Analytics & Sentiment Impact": analytics,
    "Model Performance": performance,
    "About / Methodology": about,
}
