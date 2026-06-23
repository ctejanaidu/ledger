"""Forecaster node — projections with uncertainty (SPEC §5, §8).

Projects the outcome rate forward, but only with EARNED credibility:
  - a walk-forward backtest measures real out-of-sample error and compares the
    model against a naive last-value baseline (a projection that can't beat naive
    is reported as such),
  - every projection carries an 80% PREDICTION INTERVAL, not just a point.

Deterministic (statsmodels OLS trend). Skips when there's no usable time series.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm

from ..dataio import load_csv
from ..state import ChartSpec, Projection
from ..targeting import resolve_target
from ..tools import charts

OUTPUT_DIR = "outputs/charts"
HORIZON = 3
MIN_POINTS = 8


def _ols_forecast(y: np.ndarray, steps: int, alpha: float = 0.20):
    t = np.arange(len(y))
    model = sm.OLS(y, sm.add_constant(t)).fit()
    tf = np.arange(len(y), len(y) + steps)
    Xf = sm.add_constant(tf, has_constant="add")
    sf = model.get_prediction(Xf).summary_frame(alpha=alpha)
    return sf  # columns incl. mean, obs_ci_lower, obs_ci_upper


def forecaster(state) -> dict:
    profile = state.profile
    cols = [c.name for c in profile.columns] if profile else []
    target = resolve_target(state.target, profile.target_candidates if profile else [], cols)
    tcol = profile.time_column if profile else None
    if not target or not tcol:
        return {"log": state.log + ["forecaster: no time+target -> skipped"]}

    df = load_csv(state.dataset_path)
    df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
    ts = (df.dropna(subset=[tcol, target]).set_index(tcol)[target]
            .resample("MS").mean().dropna())
    if len(ts) < MIN_POINTS:
        return {"log": state.log + [f"forecaster: only {len(ts)} points -> skipped"]}

    y = ts.to_numpy(dtype=float)

    # --- walk-forward backtest vs naive last-value ---
    n_bt = min(6, len(y) // 3)
    model_err, naive_err = [], []
    for i in range(len(y) - n_bt, len(y)):
        sf = _ols_forecast(y[:i], 1)
        pred = float(sf["mean"].iloc[0])
        model_err.append(abs(pred - y[i]))
        naive_err.append(abs(y[i - 1] - y[i]))
    mae = float(np.mean(model_err))
    naive_mae = float(np.mean(naive_err))
    denom = np.clip(np.abs(y[-n_bt:]), 1e-9, None)
    mape = float(np.mean(np.array(model_err) / denom))
    beats_naive = mae <= naive_mae

    # --- fit on full series, forecast HORIZON months ---
    sf = _ols_forecast(y, HORIZON)
    last = ts.index[-1]
    future = pd.date_range(last + pd.offsets.MonthBegin(1), periods=HORIZON, freq="MS")
    projections: list[Projection] = []
    for k in range(HORIZON):
        point = float(np.clip(sf["mean"].iloc[k], 0, 1))
        lo = float(np.clip(sf["obs_ci_lower"].iloc[k], 0, 1))
        hi = float(np.clip(sf["obs_ci_upper"].iloc[k], 0, 1))
        projections.append(Projection(
            metric=f"{target} rate",
            horizon=f"{future[k]:%b %Y}",
            point=round(point, 4), lower=round(lo, 4), upper=round(hi, 4),
            backtest_error={"MAE": round(mae, 4), "MAPE": round(mape, 4),
                            "naive_MAE": round(naive_mae, 4), "beats_naive": beats_naive},
            method="OLS linear trend; 80% prediction interval; walk-forward backtest",
            caveats=[
                "Linear-trend baseline: no seasonality or regime-change modeling.",
                f"Short history ({len(y)} months) -> intervals are wide and uncertain.",
                "Recent months may be right-censored (immature outcomes), biasing the trend.",
            ] + ([] if beats_naive else ["Model does NOT beat a naive last-value forecast — treat as low-value."]),
        ))

    # --- chart: history + forecast with PI band ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts.index, y=ts.values * 100, mode="lines+markers", name="history"))
    fig.add_trace(go.Scatter(x=future, y=[p.point * 100 for p in projections],
                             mode="lines+markers", name="forecast", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(
        x=list(future) + list(future[::-1]),
        y=[p.upper * 100 for p in projections] + [p.lower * 100 for p in projections][::-1],
        fill="toself", fillcolor="rgba(47,111,237,0.15)", line=dict(width=0),
        name="80% interval", hoverinfo="skip"))
    verdict = "beats naive baseline" if beats_naive else "does NOT beat naive — low confidence"
    charts.style(fig, f"{target.title()} rate — {HORIZON}-month forecast",
                 f"Backtest MAE {mae:.3f} ({verdict}); 80% interval shown.")
    fig.update_layout(xaxis_title="month", yaxis_title=f"{target} rate (%)")
    path = charts.save_html(fig, OUTPUT_DIR, "forecast")

    chart = ChartSpec(
        chart_id="forecast", title=f"{target.title()} rate forecast ({HORIZON}m)",
        chart_type="line", encoded_fields={"x": "month", "y": f"{target}_rate"},
        underlying_numbers={p.horizon: p.point for p in projections},
        takeaway=f"Projected {projections[-1].horizon}: {projections[-1].point:.1%} "
                 f"[{projections[-1].lower:.1%}, {projections[-1].upper:.1%}]; backtest MAE {mae:.3f}.",
        figure_path=path,
    )
    dash = state.dashboard_spec
    if dash is not None:
        dash.charts = dash.charts + [chart]

    return {
        "projections": projections,
        "dashboard_spec": dash,
        "figures": state.figures + [path],
        "log": state.log + [f"forecaster: {HORIZON}m forecast, MAE {mae:.3f}, "
                            f"{'beats' if beats_naive else 'loses to'} naive"],
    }
