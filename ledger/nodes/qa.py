"""Q&A — PART 3: interrogate the analyst (SPEC §8b).

The rule: answer ONLY from the artifacts the agent already produced (profile,
findings, model leaderboard, dashboard spec) — never re-guess. This module builds
a grounded context block and asks Claude to answer from it; without an API key it
falls back to keyword routing over the same artifacts so it still gives real,
grounded answers.
"""
from __future__ import annotations

from ..llm import chat, complete


def _grounded_context(state) -> str:
    parts: list[str] = []
    if state.profile:
        p = state.profile
        parts.append("## DATASET\n" + p.quality_summary)
        parts.append("Columns: " + ", ".join(
            f"{c.name} [{c.dtype}]" + (f" = {c.inferred_meaning}" if c.inferred_meaning else "")
            for c in p.columns))
    if state.findings:
        parts.append("## FINDINGS\n" + "\n".join(
            f"- {f.claim} (confidence: {f.confidence}; method: {f.method})"
            for f in state.findings))
    if state.model_leaderboard:
        lb = state.model_leaderboard
        rows = "\n".join(f"  - {m.name}: {m.primary_metric}={m.test_score:.4f} (cv {m.cv_score:.4f})"
                         for m in lb.candidates)
        note = f"\n  User-requested technique: {lb.user_technique_note}" if lb.user_technique_note else ""
        parts.append(f"## MODELS (target={lb.target}, metric rationale: {lb.metric_rationale})\n"
                     f"{rows}\n  Selected: {lb.selected} — {lb.selection_reason}{note}")
    if state.dashboard_spec:
        parts.append("## DASHBOARD\n" + "\n".join(
            f"- {c.title} ({c.chart_type}): takeaway = {c.takeaway}; numbers = {c.underlying_numbers}"
            for c in state.dashboard_spec.charts))
    if state.projections:
        parts.append("## PROJECTIONS\n" + "\n".join(
            f"- {pr.metric} ({pr.horizon}): {pr.point} [{pr.lower}, {pr.upper}]"
            for pr in state.projections))
    return "\n\n".join(parts)


def _fallback_answer(question: str, context: str, state) -> str:
    """No-LLM grounded answer: route by keyword to the relevant artifact."""
    q = question.lower()
    if any(w in q for w in ("model", "accuracy", "auc", "best", "won", "algorithm")):
        if state.model_leaderboard:
            lb = state.model_leaderboard
            return (f"Selected model: {lb.selected} ({lb.selection_reason}). "
                    f"Metric used: {lb.metric_rationale}. "
                    + "; ".join(f"{m.name} {m.primary_metric}={m.test_score:.3f}"
                                for m in lb.candidates))
        return "No model has been trained yet (Part 1 / Modeler runs in a later milestone)."
    if any(w in q for w in ("chart", "dashboard", "visual", "graph", "plot")):
        if state.dashboard_spec:
            return "Dashboard charts:\n" + "\n".join(
                f"- {c.title}: {c.takeaway}" for c in state.dashboard_spec.charts)
        return "No dashboard has been built yet."
    if any(w in q for w in ("driver", "strongest", "cause", "why", "factor", "loss")):
        drivers = [f for f in state.findings if "driver" in f.claim.lower()]
        if drivers:
            d = drivers[0]
            return f"{d.claim} (confidence: {d.confidence}). Caveat: {'; '.join(d.limitations)}"
    if any(w in q for w in ("trend", "over time", "rising", "falling")):
        trend = [f for f in state.findings if "over time" in f.claim.lower()]
        if trend:
            return f"{trend[0].claim} (confidence: {trend[0].confidence})"
    # default: lead finding
    head = state.findings[0].claim if state.findings else "No findings yet."
    return f"Based on the analysis: {head} (confidence: "\
           f"{state.findings[0].confidence if state.findings else 'n/a'})"


def answer_question(state, question: str) -> str:
    context = _grounded_context(state)
    return complete(
        "You are a data analyst answering a leadership question. Answer ONLY using the "
        "grounded context below — do not invent numbers. If the context lacks the answer, "
        "say so. Always end with a one-line confidence note.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {question}",
        fast=False,
        fallback=_fallback_answer(question, context, state),
    )


_CHAT_SYSTEM = (
    "You are a data analyst in a live conversation with a leadership team about a dataset "
    "you just analyzed. Ground EVERY answer in the context below — never invent numbers. "
    "Use the conversation so far to resolve follow-ups (e.g. 'why?', 'what about the second "
    "model?', 'and the riskiest segment?'). Keep answers concise and executive; when you "
    "state a result, end with a short confidence note. If the context can't answer, say so "
    "and suggest what analysis would.\n\nGROUNDED CONTEXT:\n{context}"
)


def converse(state, history: list[dict]) -> str:
    """Multi-turn chat. `history` is a list of {'role','content'} (last item = newest
    user turn). Grounded in the agent's artifacts; falls back to keyword routing
    on the latest user message when no API key is set."""
    context = _grounded_context(state)
    messages: list[tuple[str, str]] = [("system", _CHAT_SYSTEM.format(context=context))]
    for m in history[-12:]:  # cap context window to recent turns
        messages.append((m["role"], m["content"]))
    last_user = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
    return chat(messages, fast=False, fallback=_fallback_answer(last_user, context, state))
