"""TrustLens Streamlit UI. Run: streamlit run streamlit_app/app.py"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from streamlit_app import views  # noqa: E402

st.set_page_config(page_title="TrustLens", page_icon="🔍", layout="wide")

PAGES = views.PAGES

st.sidebar.title("TrustLens")
st.sidebar.radio("Page", list(PAGES), key="page")
PAGES[st.session_state["page"]]()
