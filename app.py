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
from ledger.nodes.qa import converse
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

# Resolve the columns of the chosen dataset so the user can pick a target column.
try:
    if uploaded is not None:
        columns = list(pd.read_csv(uploaded, nrows=0).columns)
        uploaded.seek(0)
    else:
        columns = list(pd.read_csv(_ensure_default_dataset(), nrows=0).columns)
except Exception:
    columns = []

target_choice = st.sidebar.selectbox(
    "Target column", ["(auto-detect)"] + columns,
    help="The column to model/predict. Auto-detect handles common names "
         "(default, target, label, class, churn…). Pick explicitly for anything else — "
         "e.g. a regression target like a price or amount.")
target = None if target_choice == "(auto-detect)" else target_choice

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
        st.session_state.state = run_analysis(path, question or None, target)
    st.session_state.chat = []  # fresh conversation for a new dataset

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

# --- Conversational Q&A (multi-turn chat) ---
st.subheader("💬 Chat with the analyst")
st.caption("Have a conversation — ask follow-ups like *“why?”*, *“what about the second "
           "model?”*, *“which segment should we act on first?”*. Answers stay grounded in the "
           "analysis above.")

if "chat" not in st.session_state:
    st.session_state.chat = []

# starter suggestions (only before the conversation begins)
if not st.session_state.chat:
    cols = st.columns(3)
    starters = ["What are the top risks I should know about?",
                "Which model won and why?",
                "What should we do first?"]
    for col, s in zip(cols, starters):
        if col.button(s, use_container_width=True):
            st.session_state.chat.append({"role": "user", "content": s})
            st.rerun()

# render the conversation so far
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# answer a pending user turn (from a starter button or a previous run)
if st.session_state.chat and st.session_state.chat[-1]["role"] == "user":
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            answer = converse(state, st.session_state.chat)
        st.markdown(answer)
    st.session_state.chat.append({"role": "assistant", "content": answer})

# chat input (pinned to the bottom by Streamlit)
if prompt := st.chat_input("Ask about the data, models, or results…"):
    st.session_state.chat.append({"role": "user", "content": prompt})
    st.rerun()

if st.session_state.chat and st.sidebar.button("🗑 Clear conversation"):
    st.session_state.chat = []
    st.rerun()
