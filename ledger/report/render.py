"""Executive report — M4 (SPEC §9).

Renders a single self-contained HTML file: BLUF summary, KPI cards, key findings
(with confidence), the model leaderboard, projections, the dashboard charts, and a
limitations section. Charts are re-rendered INLINE from the artifacts in state (one
Plotly.js load), so the file is shareable and the browser's "Print → Save as PDF"
produces a high-fidelity PDF with the charts intact.

PDF note: weasyprint/kaleido can't execute Plotly's JS, so they'd drop the charts.
Browser print-to-PDF runs the JS and renders them correctly — hence that path.
"""
from __future__ import annotations

import datetime as _dt
import html
from pathlib import Path

import plotly.graph_objects as go

from ..tools import charts


# --- chart re-rendering from stored artifacts -------------------------------
def _chart_figure(spec, state) -> go.Figure | None:
    nums = spec.underlying_numbers or {}
    if spec.chart_id == "forecast" and state.projections:
        return _forecast_figure(state)
    if not nums:
        return None
    xs, ys = list(nums.keys()), list(nums.values())
    if spec.chart_type == "line":
        return charts.line(xs, ys, title=spec.title, takeaway=spec.takeaway)
    return charts.bar(xs, ys, title=spec.title, takeaway=spec.takeaway)


def _forecast_figure(state) -> go.Figure:
    p = state.projections
    x = [pr.horizon for pr in p]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=[pr.point * 100 for pr in p],
                             mode="lines+markers", name="forecast"))
    fig.add_trace(go.Scatter(
        x=x + x[::-1],
        y=[pr.upper * 100 for pr in p] + [pr.lower * 100 for pr in p][::-1],
        fill="toself", fillcolor="rgba(47,111,237,0.15)", line=dict(width=0),
        name="80% interval", hoverinfo="skip"))
    bt = p[0].backtest_error
    return charts.style(fig, f"{p[0].metric.title()} forecast",
                        f"Backtest MAE {bt.get('MAE')}; beats naive = {bt.get('beats_naive')}.")


# --- HTML pieces ------------------------------------------------------------
_CONF_COLOR = {"high": "#15a07a", "medium": "#e8833a", "low": "#d14343"}


def _kpis(state) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if state.profile:
        out.append(("Records analyzed", f"{state.profile.n_rows:,}"))
    overall = next((f for f in state.findings if "Overall" in f.claim and "rate" in f.evidence), None)
    if overall:
        label = (state.model_leaderboard.target if state.model_leaderboard else "Outcome")
        out.append((f"{label} rate", f"{overall.evidence['rate']:.2%}"))
    lb = state.model_leaderboard
    if lb:
        sel = next((r for r in lb.candidates + ([lb.ensemble] if lb.ensemble else [])
                    if r.name == lb.selected), None)
        if sel:
            out.append((f"Best model · {sel.primary_metric}", f"{sel.test_score:.3f}"))
    if state.overall_confidence:
        out.append(("Overall confidence", state.overall_confidence.title()))
    return out


def _kpi_html(state) -> str:
    cards = "".join(
        f'<div class="kpi"><div class="kpi-val">{html.escape(str(v))}</div>'
        f'<div class="kpi-label">{html.escape(k)}</div></div>'
        for k, v in _kpis(state))
    return f'<div class="kpis">{cards}</div>'


def _findings_html(state) -> str:
    rows = []
    for f in state.findings:
        c = _CONF_COLOR.get(f.confidence, "#6b7280")
        lims = "".join(f"<li>{html.escape(l)}</li>" for l in f.limitations)
        rows.append(
            f'<div class="finding"><span class="badge" style="background:{c}">'
            f'{f.confidence}</span><div class="finding-body"><p>{html.escape(f.claim)}</p>'
            f'<details><summary>method &amp; caveats</summary>'
            f'<p class="method">{html.escape(f.method)}</p><ul>{lims}</ul></details></div></div>')
    return "\n".join(rows)


def _leaderboard_html(state) -> str:
    lb = state.model_leaderboard
    if not lb:
        return ""
    allr = lb.candidates + ([lb.ensemble] if lb.ensemble else [])
    metrics = ["PR-AUC", "ROC-AUC", "F1", "Accuracy"] if "AUC" in lb.candidates[0].primary_metric \
        else ["RMSE", "MAE", "R2"]
    head = "".join(f"<th>{m}</th>" for m in metrics)
    body = ""
    for r in allr:
        sel = " class='sel'" if r.name == lb.selected else ""
        cells = "".join(f"<td>{r.all_metrics.get(m, '—')}</td>" for m in metrics)
        body += f"<tr{sel}><td>{html.escape(r.name)}</td>{cells}</tr>"
    return (f'<h2>Model leaderboard</h2><p class="note">{html.escape(lb.metric_rationale)}</p>'
            f'<table><thead><tr><th>Model</th>{head}</tr></thead><tbody>{body}</tbody></table>'
            f'<p class="note">Selected: <b>{html.escape(lb.selected)}</b> — '
            f'{html.escape(lb.selection_reason)}</p>')


def _projections_html(state) -> str:
    if not state.projections:
        return ""
    body = ""
    for p in state.projections:
        body += (f"<tr><td>{html.escape(p.horizon)}</td><td>{p.point:.1%}</td>"
                 f"<td>{p.lower:.1%} – {p.upper:.1%}</td>"
                 f"<td>{p.backtest_error.get('MAE')}</td>"
                 f"<td>{p.backtest_error.get('beats_naive')}</td></tr>")
    return ('<h2>Projections</h2><table><thead><tr><th>Horizon</th><th>Point</th>'
            '<th>80% interval</th><th>Backtest MAE</th><th>Beats naive?</th></tr></thead>'
            f'<tbody>{body}</tbody></table>')


def _charts_html(state) -> str:
    if not state.dashboard_spec or not state.dashboard_spec.charts:
        return ""
    blocks, first = [], True
    for spec in state.dashboard_spec.charts:
        fig = _chart_figure(spec, state)
        if fig is None:
            continue
        div = fig.to_html(full_html=False, include_plotlyjs=("cdn" if first else False),
                          default_height=420)
        first = False
        blocks.append(f'<div class="chart">{div}</div>')
    return '<h2>Dashboard</h2><div class="charts">' + "\n".join(blocks) + "</div>"


def _limitations_html(state) -> str:
    lims = state.limitations or []
    if not lims:
        return ""
    items = "".join(f"<li>{html.escape(l)}</li>" for l in lims)
    return f'<h2>Limitations &amp; assumptions</h2><ul class="lims">{items}</ul>'


_CSS = """
:root{--bg:#0a0e16;--card:#111827;--ink:#e9edf5;--muted:#93a0b4;--line:#1f2a3d;--accent:#ff4d4d;}
*{box-sizing:border-box}
body{font-family:Inter,Helvetica,Arial,sans-serif;color:var(--ink);max-width:980px;margin:0 auto;
padding:32px;line-height:1.5;
background:radial-gradient(1000px 520px at 50% -8%, rgba(255,77,77,.10), transparent 60%), var(--bg);}
header{border-bottom:3px solid var(--accent);padding-bottom:12px;margin-bottom:20px}
h1{margin:0;font-size:26px;color:#fff}.sub{color:var(--muted);font-size:13px;margin-top:4px}
h2{font-size:18px;margin:28px 0 10px;border-bottom:1px solid var(--line);padding-bottom:6px;color:#fff}
.bluf{background:#141d2e;border-left:4px solid var(--accent);padding:14px 16px;border-radius:8px}
.kpis{display:flex;gap:14px;flex-wrap:wrap;margin:18px 0}
.kpi{flex:1;min-width:160px;background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:14px 16px}
.kpi-val{font-size:24px;font-weight:700;color:#fff}.kpi-label{color:var(--muted);font-size:12px;margin-top:2px}
.finding{display:flex;gap:10px;margin:10px 0;align-items:flex-start}
.badge{color:#0a0e16;border-radius:20px;padding:2px 10px;font-size:11px;text-transform:uppercase;
font-weight:700;white-space:nowrap;margin-top:2px}
.finding-body p{margin:0}.method{color:var(--muted);font-size:12px}
details summary{cursor:pointer;color:var(--accent);font-size:12px}
table{width:100%;border-collapse:collapse;font-size:13px;margin:8px 0}
th,td{border:1px solid var(--line);padding:6px 10px;text-align:left}
th{background:#141d2e}tr.sel{background:#1c2a22;font-weight:600}
.note{color:var(--muted);font-size:12px}.lims li{color:var(--muted);font-size:13px}
.charts{display:flex;flex-direction:column;gap:8px}
footer{margin-top:28px;border-top:1px solid var(--line);padding-top:10px;color:var(--muted);font-size:12px}
@media print{body{padding:0;background:var(--bg)}.chart{break-inside:avoid}.finding{break-inside:avoid}
details{display:none}}
"""


def render_report(state, out_path: str = "outputs/report.html",
                  title: str = "Ledger — Executive Data Report") -> str:
    ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    dataset = Path(state.dataset_path).name
    summary = html.escape(state.executive_summary or "No summary generated.")
    body = f"""<!doctype html><html><head><meta charset="utf-8">
<title>{html.escape(title)}</title><style>{_CSS}</style></head><body>
<header><h1>{html.escape(title)}</h1>
<div class="sub">Dataset: {html.escape(dataset)} &middot; Generated {ts}
&middot; Overall confidence: {html.escape(str(state.overall_confidence or 'n/a'))}</div></header>
<h2>Executive summary</h2><div class="bluf">{summary}</div>
{_kpi_html(state)}
<h2>Key findings</h2>{_findings_html(state)}
{_leaderboard_html(state)}
{_projections_html(state)}
{_charts_html(state)}
{_limitations_html(state)}
<footer>Generated by Ledger — an agentic AI data analyst. Findings are grounded in
computed results; review limitations before acting. To save as PDF: open in a browser
and use Print &rarr; Save as PDF.</footer>
</body></html>"""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")
    return str(out)
