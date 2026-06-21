"""Ledger — Streamlit app (SPEC §9).

Upload a dataset (or use the flagship lending demo), see the profile, the
leadership dashboard, the BLUF summary, and ask the agent questions live.

Run:  streamlit run app.py
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from ledger.config import SETTINGS
from ledger.graph import run_analysis
from ledger.llm import get_llm
from ledger.nodes.qa import answer_question
from ledger.report import render_report

load_dotenv()


def _ensure_default_dataset() -> str:
    """On a fresh (cloud) deploy the demo CSV is gitignored — generate it on demand."""
    p = Path(SETTINGS.default_dataset)
    if not p.exists():
        from data.generate_lending import generate
        p.parent.mkdir(parents=True, exist_ok=True)
        generate().to_csv(p, index=False)
    return str(p)


st.set_page_config(page_title="Ledger — AI Data Analyst", page_icon="📊", layout="wide")
st.title("📊 Ledger — AI Data Analyst for Leadership")
st.caption("Profiles your data, models it, builds dashboards, and answers your questions — "
           "stating its confidence and limitations.")

# --- Bring-your-own-key: each visitor supplies their own Anthropic key (or none) ---
user_key = st.sidebar.text_input(
    "Anthropic API key (optional)", type="password",
    help="Bring your own key to enable LLM narratives + smart Q&A. Left blank = "
         "deterministic mode (modeling, charts, report still work). Your key is used only "
         "for this session and never stored.")
if user_key:
    os.environ["ANTHROPIC_API_KEY"] = user_key
else:
    os.environ.pop("ANTHROPIC_API_KEY", None)
get_llm.cache_clear()  # pick up the current key state

mode = "🟢 LLM-enabled" if SETTINGS.has_api_key else "⚪ deterministic (add a key for narratives + Q&A)"
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
        path = _ensure_default_dataset()
        st.sidebar.info("Using flagship lending demo dataset.")
    with st.spinner("Analyzing… (training models can take ~15s)"):
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
