"""Reporter node (SPEC §5, §9) — BLUF executive summary + limitations.

Bottom-line-up-front: the headline finding first, then evidence, then an honest
limitations list. Uses Claude for executive phrasing when available; otherwise a
clean templated summary built from the (already grounded) findings.
"""
from __future__ import annotations

from ..llm import complete


def reporter(state) -> dict:
    findings = state.findings
    lines = [f"- {f.claim} (confidence: {f.confidence})" for f in findings]
    bullet_block = "\n".join(lines)

    # Validator guardrails (M3b) become hard constraints on the summary, plus part
    # of the surfaced limitations.
    guardrails = state.guardrails or []
    guard_block = "\n".join(f"- {g}" for g in guardrails)
    limitations = sorted({lim for f in findings for lim in f.limitations}) + guardrails

    summary = complete(
        "You are briefing a leadership team. Write a concise BLUF executive summary "
        "(3-5 sentences): lead with the single most important takeaway, then supporting "
        "points, then a one-line confidence statement.\n\n"
        "You MUST obey these guardrails from the validator:\n"
        f"{guard_block}\n\n"
        "Use ONLY these grounded findings (do not invent numbers):\n"
        f"{bullet_block}",
        fast=False,
        fallback=(
            "Executive summary (BLUF): "
            + (findings[0].claim if findings else "No findings generated.")
            + " " + " ".join(f.claim for f in findings[1:])
            + (f" Overall confidence: {state.overall_confidence}." if state.overall_confidence else "")
        ),
    )
    return {
        "executive_summary": summary,
        "limitations": limitations,
        "log": state.log + ["reporter: done"],
    }
