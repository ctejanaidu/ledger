"""Ledger — Streamlit app (SPEC §9).

Upload any dataset, see the profile, dashboard, and summary, and chat with the
analyst about the results.

Run:  streamlit run app.py
"""
from __future__ import annotations

import html
import os
import re
import tempfile
from pathlib import Path

import markdown as _md
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from ledger.config import SETTINGS
from ledger.graph import run_analysis
from ledger.llm import get_llm
from ledger.nodes.modeler import TECHNIQUE_CHOICES
from ledger.nodes.qa import converse
from ledger.report import render_report

load_dotenv()

# Chat avatars (easy to swap to any emoji or image URL)
USER_AVATAR = "🧑‍💻"
BOT_AVATAR = "🦉"


_LIST_RE = re.compile(r"\s*([-*]|\d+\.)\s+")


def _prep_md(text: str) -> str:
    """python-markdown needs a blank line before a list. LLMs often omit it
    (e.g. 'because:\\n- item'), so insert one (outside code fences)."""
    out, in_fence = [], False
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            out.append(line)
            continue
        if not in_fence and _LIST_RE.match(line) and out and out[-1].strip() \
                and not _LIST_RE.match(out[-1]):
            out.append("")
        out.append(line)
    return "\n".join(out)


def _render_chat(history: list[dict]) -> str:
    """Render the conversation as messaging-style bubbles: user right, analyst left."""
    rows = []
    for m in history:
        if m["role"] == "user":  # plain text, escaped (no HTML injection from input)
            body = "<p>" + html.escape(m["content"]).replace("\n", "<br>") + "</p>"
            side, avatar = "user", USER_AVATAR
        else:                    # assistant: render its markdown
            body = _md.markdown(_prep_md(m["content"]),
                                extensions=["fenced_code", "tables", "sane_lists", "nl2br"])
            side, avatar = "bot", BOT_AVATAR
        rows.append(f'<div class="lrow {side}"><div class="lav">{avatar}</div>'
                    f'<div class="lbubble">{body}</div></div>')
    return '<div class="lchat">' + "".join(rows) + "</div>"


def _ensure_default_dataset() -> str:
    """On a fresh (cloud) deploy the demo CSV is gitignored — generate it on demand."""
    p = Path(SETTINGS.default_dataset)
    if not p.exists():
        from data.generate_lending import generate
        p.parent.mkdir(parents=True, exist_ok=True)
        generate().to_csv(p, index=False)
    return str(p)


st.set_page_config(page_title="Ledger - Agentic Data Analyst Platform", page_icon="📊", layout="wide")

# --- look & feel: dark "cinematic" theme (red vignette + grid) + readable inputs ---
st.markdown("""
<style>
/* dark backdrop: red radial glow + faint grid, matching the brand look */
[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(1100px 600px at 50% -8%, rgba(255,77,77,.12), transparent 60%),
    radial-gradient(800px 600px at 100% 0%, rgba(255,77,77,.06), transparent 55%),
    linear-gradient(rgba(255,255,255,.022) 1px, transparent 1px) 0 0 / 42px 42px,
    linear-gradient(90deg, rgba(255,255,255,.022) 1px, transparent 1px) 0 0 / 42px 42px,
    #0a0e16;
}
[data-testid="stHeader"] { background: transparent; }
section[data-testid="stSidebar"] { background: #0d1320; border-right: 1px solid #1b2335; }

/* slide + fade content in */
section.main > div { animation: ledgerFade .6s ease-out; }
@keyframes ledgerFade { from { opacity:0; transform: translateY(10px); } to { opacity:1; transform:none; } }

/* white wordmark with a red shimmer + red underline (echoes the logo) */
.ledger-title { font-size: 2.35rem; font-weight: 800; margin: 0 0 .15rem 0; line-height: 1.15;
  display:inline-block; padding-bottom:.18rem; border-bottom: 3px solid #ff4d4d;
  background: linear-gradient(90deg,#ffffff,#ff8a8a,#ff4d4d,#ffffff);
  background-size: 300% auto; -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent; animation: ledgerShine 8s linear infinite; }
@keyframes ledgerShine { to { background-position: 300% center; } }
.ledger-sub { color:#93a0b4; font-size: 1rem; margin: .35rem 0 .4rem 0; }

/* cards / expanders: dark slate, red glow on hover */
div[data-testid="stExpander"] { background:#111827; border:1px solid #1f2a3d; border-radius:12px;
  transition: transform .15s ease, box-shadow .15s ease, border-color .15s ease; }
div[data-testid="stExpander"]:hover { transform: translateY(-2px);
  box-shadow: 0 8px 22px rgba(255,77,77,.10); border-color:#ff4d4d55; }

/* chat bubbles */
div[data-testid="stChatMessage"] { animation: ledgerFade .45s ease-out;
  background:#111827; border:1px solid #1f2a3d; border-radius:12px; }

/* buttons: red, with pop */
.stButton > button { transition: transform .08s ease, box-shadow .18s ease; border-radius:10px; font-weight:600; }
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 6px 16px rgba(255,77,77,.28); }
.stButton > button[kind="primary"] { background:#ff4d4d; border:0; color:#fff; }
.stButton > button[kind="primary"]:hover { background:#ff5e5e; }

/* --- readable sidebar inputs (fix small/faint placeholders) --- */
section[data-testid="stSidebar"] label p,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
  font-size: .95rem !important; font-weight: 600 !important; color:#e9edf5 !important; }
section[data-testid="stSidebar"] [data-baseweb="input"],
section[data-testid="stSidebar"] [data-baseweb="select"] > div {
  background:#161f30 !important; border:1px solid #2a3650 !important; border-radius:9px !important;
  min-height: 42px; font-size: .96rem !important; }
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] [data-baseweb="select"] div { color:#eef2f8 !important; font-size:.96rem !important; }
section[data-testid="stSidebar"] [data-baseweb="input"]:focus-within,
section[data-testid="stSidebar"] [data-baseweb="select"] > div:focus-within { border-color:#ff4d4d !important; }
input::placeholder, textarea::placeholder { color:#9aa6ba !important; opacity:1 !important; font-size:.95rem; }
/* file uploader: clearer on dark */
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
  background:#161f30 !important; border:1px dashed #2a3650 !important; }

/* --- messaging-style chat bubbles (user right / analyst left) --- */
.lchat { display:flex; flex-direction:column; gap:10px; margin:6px 0 24px 0; }
.lrow { display:flex; gap:8px; align-items:flex-end; animation: ledgerFade .35s ease-out; }
.lrow.bot { justify-content:flex-start; }
.lrow.user { justify-content:flex-end; flex-direction:row-reverse; }
.lav { font-size:1.3rem; line-height:1; flex:0 0 auto; padding-bottom:3px; }
.lbubble { max-width:74%; padding:10px 14px; border-radius:16px; font-size:.96rem; line-height:1.5;
  word-wrap:break-word; }
.lrow.bot .lbubble { background:#1b2335; color:#e9edf5; border:1px solid #283448;
  border-bottom-left-radius:5px; }
.lrow.user .lbubble { background:#ff4d4d; color:#ffffff; border-bottom-right-radius:5px; }
.lbubble p { margin:.18rem 0; } .lbubble p:first-child { margin-top:0; } .lbubble p:last-child { margin-bottom:0; }
.lbubble ul, .lbubble ol { margin:.3rem 0 .3rem 1.15rem; padding:0; }
.lbubble strong { font-weight:700; }
.lbubble code { background:#0e1626; padding:1px 5px; border-radius:5px; font-size:.9em; }
.lrow.user .lbubble code { background:rgba(255,255,255,.22); }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="ledger-title">📊 Ledger - Agentic Data Analyst Platform</div>',
            unsafe_allow_html=True)
st.markdown('<div class="ledger-sub">Drop in any dataset and Ledger does the analyst grunt '
            "work — pokes around, builds the models, draws the charts, and chats with you about "
            "what it all means (owning up to what it’s not sure about). No data-science degree "
            'required. 🪄</div>', unsafe_allow_html=True)

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

technique_choice = st.sidebar.selectbox(
    "Your ML technique (optional)", ["(let Ledger decide)"] + TECHNIQUE_CHOICES,
    help="Add a technique to the panel Ledger trains. It runs ALONGSIDE the auto-selected "
         "models. If it can't be trained (e.g. a regression technique on a classification "
         "problem, or too slow for the data), the results will tell you.")
user_technique = None if technique_choice == "(let Ledger decide)" else technique_choice

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
        st.session_state.state = run_analysis(path, None, target, user_technique)
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

# render the conversation so far as left/right bubbles
if st.session_state.chat:
    st.markdown(_render_chat(st.session_state.chat), unsafe_allow_html=True)

# answer a pending user turn (from a starter button or the chat input)
if st.session_state.chat and st.session_state.chat[-1]["role"] == "user":
    with st.spinner("Thinking…"):
        answer = converse(state, st.session_state.chat)
    st.session_state.chat.append({"role": "assistant", "content": answer})
    st.rerun()

# chat input (pinned to the bottom by Streamlit)
if prompt := st.chat_input("Ask about the data, models, or results…"):
    st.session_state.chat.append({"role": "user", "content": prompt})
    st.rerun()

if st.session_state.chat and st.sidebar.button("🗑 Clear conversation"):
    st.session_state.chat = []
    st.rerun()
