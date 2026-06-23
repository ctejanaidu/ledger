"""Analyst node (SPEC §5) — compute real findings and build leadership charts.

M1 scope: deterministic, grounded analysis (no hallucinated numbers). It finds
the binary target, computes the outcome rate, identifies the strongest
categorical driver, and (if a time column exists) the trend over time. Each
result becomes a Finding (with confidence + limitations) and a ChartSpec (with
the underlying numbers stored, so Part-3 Q&A can answer precisely).

The LLM-driven, code-writing analyst loop (write->run->inspect->refine via the
sandbox) is layered on in M2; this M1 version is the trustworthy backbone.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..dataio import load_csv
from ..state import ChartSpec, DashboardSpec, Finding
from ..targeting import to_binary01
from ..tools import charts

OUTPUT_DIR = "outputs/charts"
_MIN_GROUP = 30  # below this, we down-rate confidence


def _confidence(n: int) -> str:
    return "high" if n >= 500 else "medium" if n >= _MIN_GROUP else "low"


def _is_binary(s: pd.Series) -> bool:
    return s.dropna().nunique() == 2


def _pick_target(df: pd.DataFrame, profile, explicit: str | None = None) -> str | None:
    # The baseline analyst does binary rate/segment analysis, so it only adopts a
    # BINARY target (of ANY dtype — 0/1, 1/2, True/False, Yes/No). A non-binary
    # explicit target (regression/multiclass) falls through to None and is handled
    # by the modeler instead.
    if explicit:
        return explicit if (explicit in df.columns and _is_binary(df[explicit])) else None
    for cand in profile.target_candidates:
        if cand in df.columns and _is_binary(df[cand]):
            return cand
    return None


def _categorical_drivers(df: pd.DataFrame, target: str) -> list[str]:
    cats = []
    for c in df.columns:
        if c == target:
            continue
        if (df[c].dtype == "object" or df[c].nunique(dropna=True) <= 8) and df[c].nunique() > 1:
            if "id" not in c.lower():
                cats.append(c)
    return cats


def _numeric_drivers(df: pd.DataFrame, target: str, exclude: set[str]) -> list[tuple[str, float]]:
    """Rank numeric features by point-biserial correlation with the binary target."""
    rows: list[tuple[str, float]] = []
    for c in df.select_dtypes(include="number").columns:
        if c == target or c in exclude or df[c].nunique(dropna=True) <= 1:
            continue
        corr = df[c].corr(df[target])
        if pd.notna(corr):
            rows.append((c, float(corr)))
    rows.sort(key=lambda r: abs(r[1]), reverse=True)
    return rows


def analyst(state) -> dict:
    df = load_csv(state.dataset_path)
    profile = state.profile
    findings: list[Finding] = []
    chart_specs: list[ChartSpec] = []
    figures: list[str] = []

    target = _pick_target(df, profile, state.target)
    if target is None:
        if state.target:  # user chose a non-binary target -> the modeler handles it
            msg = (f"Target '{state.target}' isn't binary, so the baseline rate/segment "
                   "analysis doesn't apply — see the model results below for the regression "
                   "analysis and its drivers.")
            lim = ["Baseline rate analysis is for binary targets; the modeler handles this one."]
        else:
            msg = ("No target column was auto-detected, so I ran descriptive profiling only. "
                   "To unlock modeling, driver analysis, and forecasting, pick the column you "
                   "want to predict in the sidebar 'Target column' dropdown.")
            lim = ["Modeling and driver analysis need a target — choose one in the sidebar "
                   "'Target column' dropdown."]
        return {
            "findings": [Finding(claim=msg, confidence="low", method="target detection",
                                 limitations=lim)],
            "log": state.log + ["analyst: no binary target"],
        }

    # Encode a (possibly text / boolean / 1-2) binary target to 0/1 for analysis.
    y01, pos_label = to_binary01(df[target])
    df = df.assign(**{target: y01})

    # --- Finding 1: overall outcome rate ---
    rate = float(df[target].mean())
    pos_txt = "" if str(pos_label) in ("1", "True") else f" (positive class = {pos_label})"
    rate_limits = ["Pooled rate hides segment variation (see driver breakdown)."]
    if rate < 0.05 or rate > 0.95:
        rate_limits.append(
            f"Severe class imbalance ({rate:.2%} positive) — raw accuracy will be "
            "misleading; modeling must use ROC-AUC / PR-AUC and class-imbalance handling.")
    findings.append(Finding(
        claim=f"Overall {target} rate is {rate:.2%}{pos_txt} across {len(df):,} records "
              f"({int(df[target].sum()):,} positive cases).",
        evidence={"rate": round(rate, 5), "n": len(df), "positives": int(df[target].sum()),
                  "positive_class": str(pos_label)},
        confidence=_confidence(len(df)),
        method="mean of binary target",
        limitations=rate_limits,
    ))

    # --- Finding 2: strongest categorical driver ---
    drivers = _categorical_drivers(df, target)
    best = None
    for c in drivers:
        grp = df.groupby(c)[target].agg(["mean", "count"])
        grp = grp[grp["count"] >= _MIN_GROUP]
        if len(grp) < 2:
            continue
        spread = grp["mean"].max() - grp["mean"].min()
        if best is None or spread > best[1]:
            best = (c, spread, grp.sort_values("mean", ascending=False))

    if best:
        col, spread, grp = best
        top_cat = grp.index[0]
        findings.append(Finding(
            claim=(f"'{col}' is the strongest categorical driver of {target}: "
                   f"rate ranges {grp['mean'].min():.1%}-{grp['mean'].max():.1%} "
                   f"(highest for {col}={top_cat} at {grp['mean'].iloc[0]:.1%})."),
            evidence={"by_" + col: grp["mean"].round(4).to_dict(),
                      "counts": grp["count"].to_dict()},
            confidence=_confidence(int(grp["count"].min())),
            method="group-by mean of target with min group size filter",
            limitations=["Association, not proven causation.",
                         "Confounding with other variables not yet controlled."],
        ))
        fig = charts.bar(
            grp.index.astype(str), (grp["mean"] * 100).round(1),
            title=f"{target.title()} rate by {col}",
            takeaway=f"{col}={top_cat} carries the highest {target} rate "
                     f"({grp['mean'].iloc[0]:.0%}).",
            x_title=col, y_title=f"{target} rate (%)",
        )
        path = charts.save_html(fig, OUTPUT_DIR, f"driver_{col}")
        figures.append(path)
        chart_specs.append(ChartSpec(
            chart_id=f"driver_{col}", title=f"{target.title()} rate by {col}",
            chart_type="bar", encoded_fields={"x": col, "y": f"{target}_rate"},
            underlying_numbers={k: round(v, 4) for k, v in grp["mean"].to_dict().items()},
            takeaway=f"{col}={top_cat} has the highest {target} rate.",
            figure_path=path,
        ))

    # --- Finding 3: strongest numeric drivers (for all-numeric / feature datasets) ---
    exclude = {c.name for c in profile.columns if "possible_id" in c.quality_flags}
    if profile.time_column:
        exclude.add(profile.time_column)
    num_drivers = [r for r in _numeric_drivers(df, target, exclude) if abs(r[1]) >= 0.02]
    if num_drivers:
        top = num_drivers[:6]
        name0, corr0 = top[0]
        lims = ["Linear association only; non-linear effects need a model (Part 1).",
                "Correlation, not causation."]
        if any(n.lower().startswith("v") and n[1:].isdigit() for n, _ in top):
            lims.append("V1..V28 are PCA-anonymized, so individual features aren't directly interpretable.")
        findings.append(Finding(
            claim=(f"Among numeric features, '{name0}' is most associated with {target} "
                   f"(r={corr0:+.2f}). Top drivers: "
                   + ", ".join(f"{n} ({c:+.2f})" for n, c in top[:3]) + "."),
            evidence={"correlations": {n: round(c, 4) for n, c in top}},
            confidence=_confidence(len(df)),
            method="point-biserial correlation (feature vs binary target)",
            limitations=lims,
        ))
        fig = charts.bar(
            [n for n, _ in top], [round(abs(c), 3) for _, c in top],
            title=f"Features most associated with {target}",
            takeaway=f"{name0} shows the strongest (|r|={abs(corr0):.2f}) association with {target}.",
            x_title="feature", y_title=f"|correlation| with {target}",
        )
        path = charts.save_html(fig, OUTPUT_DIR, "numeric_drivers")
        figures.append(path)
        chart_specs.append(ChartSpec(
            chart_id="numeric_drivers", title=f"Features most associated with {target}",
            chart_type="bar", encoded_fields={"x": "feature", "y": f"abs_corr_with_{target}"},
            underlying_numbers={n: round(c, 4) for n, c in top},
            takeaway=f"{name0} is the strongest numeric correlate of {target}.",
            figure_path=path,
        ))

    # --- Finding 4: trend over time (if applicable) ---
    tcol = profile.time_column
    if tcol:
        d = df.copy()
        d[tcol] = pd.to_datetime(d[tcol], errors="coerce")
        ts = d.dropna(subset=[tcol]).set_index(tcol)[target].resample("MS").mean().dropna()
        if len(ts) >= 3:
            direction = "rising" if ts.iloc[-1] > ts.iloc[0] else "falling"
            findings.append(Finding(
                claim=(f"{target.title()} rate is {direction} over time: "
                       f"{ts.iloc[0]:.1%} -> {ts.iloc[-1]:.1%} "
                       f"from {ts.index[0]:%b %Y} to {ts.index[-1]:%b %Y}."),
                evidence={"first": round(float(ts.iloc[0]), 4),
                          "last": round(float(ts.iloc[-1]), 4), "points": len(ts)},
                confidence="medium",
                method="monthly resample of target mean",
                limitations=["Trend is descriptive; no seasonality adjustment yet.",
                             "Recent months may have immature outcomes (right-censoring)."],
            ))
            fig = charts.line(
                ts.index, (ts * 100).round(2),
                title=f"{target.title()} rate over time",
                takeaway=f"Rate has been {direction} across the observed window.",
                x_title="Month", y_title=f"{target} rate (%)",
            )
            path = charts.save_html(fig, OUTPUT_DIR, "trend_over_time")
            figures.append(path)
            chart_specs.append(ChartSpec(
                chart_id="trend_over_time", title=f"{target.title()} rate over time",
                chart_type="line", encoded_fields={"x": tcol, "y": f"{target}_rate"},
                underlying_numbers={str(k.date()): round(float(v), 4) for k, v in ts.items()},
                takeaway=f"{target.title()} rate is {direction}.",
                figure_path=path,
            ))

    dashboard = DashboardSpec(
        charts=chart_specs,
        layout_rationale="Lead with the outcome rate, then the strongest driver, "
                         "then the time trend — the order a leadership briefing follows.",
    )
    return {
        "findings": findings,
        "dashboard_spec": dashboard,
        "figures": figures,
        "log": state.log + [f"analyst: {len(findings)} findings, {len(figures)} charts"],
    }
