"""Planner node (SPEC §5) — turn the question + profile into analysis steps.

M1: a lightweight planner. When an API key is set, Claude proposes the analysis
plan from the question and schema; otherwise a sensible default plan is used.
"""
from __future__ import annotations

from ..llm import complete

_DEFAULT_PLAN = [
    "Summarize dataset size, target balance, and data quality.",
    "Compute the key outcome rate and break it down by the main categorical drivers.",
    "If a time column exists, show the outcome trend over time.",
    "Surface the top insight for leadership with a confidence level.",
]


def planner(state) -> dict:
    profile = state.profile
    question = state.question or "Give leadership an overview of this dataset and its key drivers."

    schema = ", ".join(c.name for c in profile.columns) if profile else ""
    plan_txt = complete(
        "You are a senior data analyst planning an analysis for leadership.\n"
        f"Question: {question}\nColumns: {schema}\n"
        f"Target candidate(s): {profile.target_candidates if profile else []}\n\n"
        "Return ONLY a bulleted list of 3-5 concrete analysis steps. Each line MUST "
        "start with '- '. No headings, no preamble, no commentary, no markdown other "
        "than the bullets. Keep each step to one short sentence.",
        fast=True, fallback="",
    )
    plan = _parse_bullets(plan_txt) or _DEFAULT_PLAN
    return {"plan": plan[:5], "log": state.log + ["planner: done"]}


def _parse_bullets(text: str) -> list[str]:
    """Keep only genuine list items; drop headers, prose, and blank lines."""
    items: list[str] = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        if s[0] in "-*•":
            item = s.lstrip("-*• ").strip()
        elif s[0].isdigit() and len(s) > 2 and s[1] in ").":
            item = s[2:].strip()
        else:
            continue  # not a bullet -> skip headers/prose
        item = item.strip("*").strip()  # drop stray markdown bold
        if item and not item.endswith(":"):  # ':' usually marks a header
            items.append(item)
    return items
