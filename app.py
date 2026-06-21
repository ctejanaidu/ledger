"""Ledger — Streamlit app (SPEC §9).

Upload a dataset (or use the flagship lending demo), see the profile, the
leadership dashboard, the BLUF summary, and ask the agent questions live.

Run:  streamlit run app.py
"""
from __future__ import annotations

import tempfile

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from ledger.config import SETTINGS
from ledger.graph import run_analysis
from ledger.nodes.qa import answer_question
from ledger.report import render_report

load_dotenv()

st.set_page_config(page_title="Ledger — AI Data Analyst", page_icon="📊", layout="wide")
st.title("📊 Ledger — AI Data Analyst for Leadership")
st.caption("Profiles your data, models it, builds dashboards, and answers your questions — "
           "stating its confidence and limitations.")

mode = "🟢 LLM-enabled" if SETTINGS.has_api_key else "⚪ deterministic (set ANTHROPIC_API_KEY for narratives + Q&A)"
st.sidebar.markdown(f"**Mode:** {mode}")

uploaded = st.sidebar.file_uploader("Upload a CSV", type=["csv"])
question = st.sidebar.text_input("Leadership question (optional)",
                                 placeholder="e.g. what's driving our losses?")
run = st.sidebar.button("Run analysis", type="primary")

if "state" not in st.session_state:
    st.session_state.state = None

if run:
    if uploaded is not None:
        with tempfile.NamedTemporaryFile("wb", suffix=".csv", delete=False) as fh:
            fh.write(uploaded.getbuffer())
            path = fh.name
    else:
        path = SETTINGS.default_dataset
        st.sidebar.info("Using flagship lending demo dataset.")
    with st.spinner("Analyzing…"):
        st.session_state.state = run_analysis(path, question or None)

state = st.session_state.state
if state is None:
    st.info("⬅ Upload a CSV (or just hit **Run analysis** to use the lending demo).")
    st.stop()

# --- Executive summary ---
st.subheader("Executive summary")
st.write(state.executive_summary)

# --- Download executive report (M4) ---
_report_path = render_report(state)
with open(_report_path, "rb") as _fh:
    st.download_button("⬇ Download executive report (HTML · print to PDF)", _fh,
                       file_name="ledger_report.html", mime="text/html")

# --- KPI / findings ---
st.subheader("Key findings")
for f in state.findings:
    badge = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(f.confidence, "⚪")
    with st.expander(f"{badge} {f.claim}", expanded=True):
        st.caption(f"Method: {f.method} · Confidence: {f.confidence}")
        for lim in f.limitations:
            st.markdown(f"- ⚠ {lim}")

# --- Dashboard ---
st.subheader("Dashboard")
cols = st.columns(2)
for i, fig_path in enumerate(state.figures):
    with cols[i % 2]:
        st.components.v1.html(open(fig_path).read(), height=480, scrolling=False)

# --- Data profile ---
with st.expander("Data profile"):
    st.write(state.profile.quality_summary)
    st.dataframe(pd.DataFrame([c.model_dump() for c in state.profile.columns]))

# --- Q&A ---
st.subheader("Ask the analyst")
q = st.text_input("Your question", placeholder="Which model won? What does chart 2 show?")
if q:
    st.markdown(answer_question(state, q))
