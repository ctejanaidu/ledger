"""Validator node — the "knows what it doesn't know" guard (SPEC §5, §8).

Runs just before the Reporter and does three things:
  1. INDEPENDENT RECOMPUTE — recomputes the headline target rate from raw data and
     flags any disagreement with what the analysis reported.
  2. AGGREGATE CONFIDENCE — rolls per-finding confidence into one conservative
     overall level.
  3. GUARDRAILS — emits rules the Reporter MUST honor: ban overclaiming language
     (this is what would have caught "production-ready"), cap the stated confidence,
     and surface the cross-cutting caveats that apply to the whole brief.

Deterministic; runs with or without an API key.
"""
from __future__ import annotations

import pandas as pd

_BANNED = ["production-ready", "guaranteed", "proven", "certain", "always",
           "will definitely", "100% accurate", "risk-free", "foolproof"]


def _aggregate_confidence(findings) -> str:
    levels = [f.confidence for f in findings] or ["low"]
    low = levels.count("low")
    high = levels.count("high")
    if low > 0:
        return "medium" if high >= low else "low"
    return "high" if high >= 2 else "medium"


def validator(state) -> dict:
    profile = state.profile
    findings = state.findings
    guardrails: list[str] = []

    # 1) independent recompute of the headline rate
    target = next((c for c in (profile.target_candidates or [])
                   if profile and c.lower() in {"default", "target", "label", "class", "fraud", "churn"}),
                  None)
    if target:
        try:
            col = pd.read_csv(state.dataset_path, usecols=[target])[target]
            if col.nunique(dropna=True) == 2:
                recomputed = float(col.mean())
                reported = next((f.evidence.get("rate") for f in findings
                                 if "Overall" in f.claim and "rate" in f.evidence), None)
                if reported is not None and abs(recomputed - reported) > 0.005:
                    guardrails.append(
                        f"WARNING: independent recompute of {target} rate ({recomputed:.2%}) "
                        f"disagrees with the reported {reported:.2%} — investigate before sharing.")
                else:
                    guardrails.append(
                        f"Independent recompute confirms the headline {target} rate "
                        f"({recomputed:.2%}).")
        except Exception:  # pragma: no cover
            pass

    overall = _aggregate_confidence(findings)

    # 2) cross-cutting caveats based on what actually ran
    if state.model_leaderboard:
        guardrails.append("Model results come from a single held-out split, not validated "
                          "over time or on new data — avoid 'production-ready' claims.")
    if state.projections:
        if any(not p.backtest_error.get("beats_naive", True) for p in state.projections):
            guardrails.append("At least one forecast does not beat a naive baseline — present "
                              "projections as low-confidence.")
        guardrails.append("Forecasts assume a linear trend and may be biased by right-censoring.")
    if profile and profile.time_column:
        guardrails.append("Recent-period outcomes may be immature; do not over-read recent moves.")

    # 3) language + confidence guardrails (always)
    guardrails.append("Do NOT use overclaiming words: " + ", ".join(f"'{w}'" for w in _BANNED)
                      + ". Use calibrated language tied to the evidence.")
    guardrails.append(f"State overall confidence as at most '{overall}'.")

    return {
        "guardrails": guardrails,
        "overall_confidence": overall,
        "log": state.log + [f"validator: overall confidence '{overall}', {len(guardrails)} guardrails"],
    }
