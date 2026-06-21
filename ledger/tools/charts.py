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

# --- executive theme ---------------------------------------------------------
_INK = "#1f2a44"
_ACCENT = "#2f6fed"
_MUTED = "#6b7280"
_GRID = "#e8ebf0"

_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        font=dict(family="Inter, Helvetica, Arial, sans-serif", color=_INK, size=14),
        colorway=[_ACCENT, "#15a07a", "#e8833a", "#b4458f", "#6b7280"],
        paper_bgcolor="white",
        plot_bgcolor="white",
        title=dict(font=dict(size=18, color=_INK)),
        xaxis=dict(gridcolor=_GRID, zerolinecolor=_GRID),
        yaxis=dict(gridcolor=_GRID, zerolinecolor=_GRID),
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
    """Persist a chart as a standalone HTML fragment; return its path."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = str(Path(out_dir) / f"{chart_id}.html")
    fig.write_html(path, include_plotlyjs="cdn", full_html=True)
    return path


def bar(x, y, *, title: str, takeaway: str = "", x_title: str = "", y_title: str = "") -> go.Figure:
    fig = go.Figure(go.Bar(x=list(x), y=list(y)))
    fig.update_layout(xaxis_title=x_title, yaxis_title=y_title)
    return style(fig, title, takeaway)


def line(x, y, *, title: str, takeaway: str = "", x_title: str = "", y_title: str = "") -> go.Figure:
    fig = go.Figure(go.Scatter(x=list(x), y=list(y), mode="lines+markers"))
    fig.update_layout(xaxis_title=x_title, yaxis_title=y_title)
    return style(fig, title, takeaway)
