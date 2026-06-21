"""Diagnostician node — root-cause analysis (SPEC §5).

Answers "what's driving the change?" rigorously and deterministically. When a time
column and a binary target exist, it runs a RATE vs MIX decomposition: did the
outcome rate rise because rates worsened WITHIN segments, or because the portfolio
MIX shifted toward riskier segments? This is the disciplined version of the story
the agentic loop finds informally — and it explicitly frames results as attribution,
not proven causation.

Skips gracefully when there is no usable time dimension or segmentation.
"""
from __future__ import annotations

import pandas as pd

from ..state import Finding

_MIN_GROUP = 30


def _segment_col(df: pd.DataFrame, target: str, time_col: str) -> str | None:
    """Pick a low-cardinality categorical segmentation with real rate spread."""
    best, best_spread = None, 0.0
    for c in df.columns:
        if c in (target, time_col) or "id" in c.lower():
            continue
        if df[c].dtype == "object" or df[c].nunique(dropna=True) <= 8:
            if df[c].nunique(dropna=True) < 2:
                continue
            grp = df.groupby(c)[target].agg(["mean", "count"])
            grp = grp[grp["count"] >= _MIN_GROUP]
            if len(grp) >= 2:
                spread = grp["mean"].max() - grp["mean"].min()
                if spread > best_spread:
                    best, best_spread = c, spread
    return best


def diagnostician(state) -> dict:
    profile = state.profile
    target = next((c for c in (profile.target_candidates or [])
                   if profile and c.lower() in {"default", "target", "label", "class", "fraud", "churn"}),
                  None)
    tcol = profile.time_column if profile else None
    if not target or not tcol:
        return {"log": state.log + ["diagnostician: no time+target -> skipped"]}

    df = pd.read_csv(state.dataset_path)
    if target not in df or df[target].nunique() != 2:
        return {"log": state.log + ["diagnostician: target not binary -> skipped"]}
    df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
    df = df.dropna(subset=[tcol, target])

    seg = _segment_col(df, target, tcol)
    if not seg:
        return {"log": state.log + ["diagnostician: no segmentation -> skipped"]}

    # split into historical vs recent halves by time
    cutoff = df[tcol].quantile(0.5)
    a = df[df[tcol] <= cutoff]   # historical
    b = df[df[tcol] > cutoff]    # recent
    if len(a) < _MIN_GROUP or len(b) < _MIN_GROUP:
        return {"log": state.log + ["diagnostician: insufficient per-period data -> skipped"]}

    rate_a, rate_b = a[target].mean(), b[target].mean()
    delta = rate_b - rate_a

    # rate/mix decomposition across segments
    wa = a[seg].value_counts(normalize=True)
    wb = b[seg].value_counts(normalize=True)
    ra = a.groupby(seg)[target].mean()
    rb = b.groupby(seg)[target].mean()
    segs = sorted(set(wa.index) & set(wb.index) & set(ra.index) & set(rb.index))

    rate_effect = sum(wa.get(g, 0) * (rb.get(g, 0) - ra.get(g, 0)) for g in segs)
    mix_effect = sum((wb.get(g, 0) - wa.get(g, 0)) * ra.get(g, 0) for g in segs)
    interaction = delta - rate_effect - mix_effect

    dominant = "within-segment rate deterioration" if abs(rate_effect) >= abs(mix_effect) \
        else "a shift in portfolio mix toward riskier segments"

    # which segment contributed most to the change
    contrib = {g: wa.get(g, 0) * (rb.get(g, 0) - ra.get(g, 0))
                  + (wb.get(g, 0) - wa.get(g, 0)) * ra.get(g, 0) for g in segs}
    top_seg = max(contrib, key=lambda g: abs(contrib[g])) if contrib else None

    finding = Finding(
        claim=(f"{target.title()} rate moved {delta:+.1%} ({rate_a:.1%}→{rate_b:.1%}) from the "
               f"earlier to the recent period. Decomposing by '{seg}': within-segment rate change "
               f"contributes {rate_effect:+.1%} and mix shift {mix_effect:+.1%} "
               f"(interaction {interaction:+.1%}) — i.e. the move is driven mainly by {dominant}"
               + (f", concentrated in {seg}={top_seg}." if top_seg is not None else ".")),
        evidence={"delta": round(float(delta), 4), "rate_effect": round(float(rate_effect), 4),
                  "mix_effect": round(float(mix_effect), 4),
                  "interaction": round(float(interaction), 4),
                  "segment_contributions": {str(g): round(float(v), 4) for g, v in contrib.items()}},
        confidence="medium",
        method=f"rate/mix decomposition by '{seg}', historical vs recent split at the time median",
        limitations=[
            "Attribution, NOT proven causation — confounders are uncontrolled.",
            "Recent-period outcomes may be immature (right-censoring), biasing the change.",
            "Two-period split is coarse; sensitive to the cut point.",
        ],
    )
    return {"findings": state.findings + [finding],
            "log": state.log + [f"diagnostician: {dominant} (rate {rate_effect:+.1%} / mix {mix_effect:+.1%})"]}
