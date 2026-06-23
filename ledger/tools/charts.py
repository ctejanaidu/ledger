"""Leadership-styled charts (SPEC §9).

Every chart is titled, carries a written takeaway, and uses one consistent
executive theme — the gap between "it made a chart" and "I'd show this to a CFO".
Charts are saved as both standalone HTML (for the report) and PNG-less figure
objects returned for Streamlit.
"""
from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio

# --- dark "cinematic" theme (matches the app: near-black + coral-red accent) -----
_CARD = "#111827"      # card background (matches app cards)
_INK = "#e9edf5"       # off-white text
_ACCENT = "#ff4d4d"    # coral red (primary)
_MUTED = "#93a0b4"     # muted gray-blue
_GRID = "#212c40"      # subtle grid lines

_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        font=dict(family="Inter, Helvetica, Arial, sans-serif", color=_INK, size=14),
        # lead with red, then steel-blue / green / amber / purple / gray
        colorway=[_ACCENT, "#4a90d9", "#36d399", "#e8a13a", "#a06cd5", "#93a0b4"],
        paper_bgcolor=_CARD,
        plot_bgcolor=_CARD,
        title=dict(font=dict(size=18, color="#ffffff")),
        xaxis=dict(gridcolor=_GRID, zerolinecolor=_GRID, linecolor=_GRID,
                   tickfont=dict(color=_MUTED), title=dict(font=dict(color=_MUTED))),
        yaxis=dict(gridcolor=_GRID, zerolinecolor=_GRID, linecolor=_GRID,
                   tickfont=dict(color=_MUTED), title=dict(font=dict(color=_MUTED))),
        legend=dict(font=dict(color=_INK)),
        margin=dict(l=60, r=30, t=80, b=110),
    )
)
pio.templates["ledger"] = _TEMPLATE


def style(fig: go.Figure, title: str, takeaway: str = "") -> go.Figure:
    """Apply the executive theme: title + a takeaway annotation under the chart."""
    fig.update_layout(template="ledger", title=title)
    if takeaway:
        fig.add_annotation(
            text=f"<b>Takeaway:</b> {takeaway}",
            xref="paper", yref="paper", x=0, y=-0.28, showarrow=False,
            align="left", font=dict(size=12, color=_MUTED),
        )
    return fig


def save_html(fig: go.Figure, out_dir: str, chart_id: str) -> str:
    """Persist a chart as a standalone HTML file; return its path.

    The page body is set to the dark card color so the chart blends seamlessly
    when embedded in the (dark) Streamlit app — no white frame around the plot."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = str(Path(out_dir) / f"{chart_id}.html")
    html = fig.to_html(include_plotlyjs="cdn", full_html=True)
    html = html.replace("<head>", f"<head><style>html,body{{margin:0;background:{_CARD};}}</style>", 1)
    Path(path).write_text(html, encoding="utf-8")
    return path


def bar(x, y, *, title: str, takeaway: str = "", x_title: str = "", y_title: str = "") -> go.Figure:
    fig = go.Figure(go.Bar(x=list(x), y=list(y)))
    fig.update_layout(xaxis_title=x_title, yaxis_title=y_title)
    return style(fig, title, takeaway)


def line(x, y, *, title: str, takeaway: str = "", x_title: str = "", y_title: str = "") -> go.Figure:
    fig = go.Figure(go.Scatter(x=list(x), y=list(y), mode="lines+markers"))
    fig.update_layout(xaxis_title=x_title, yaxis_title=y_title)
    return style(fig, title, takeaway)
