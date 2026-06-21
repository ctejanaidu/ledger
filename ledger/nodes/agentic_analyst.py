"""Agentic analyst loop — M2 (SPEC §5, the Analyst ⇄ CodeExec cycle).

This is what makes Ledger an *agent* and not a fixed pipeline: for each step in
the plan, the LLM WRITES pandas code, we RUN it in the sandbox, the agent INSPECTS
the real output, SELF-CORRECTS on error, then converts the verified result into a
grounded Finding. Every number in an agentic finding comes from code that actually
executed — never the model's imagination.

Layered on top of the deterministic baseline analyst (which always runs and
guarantees the core findings). With no API key this node is a no-op, so the
deterministic pipeline and the eval harness are unaffected.
"""
from __future__ import annotations

import json
import re

from ..llm import complete, get_llm
from ..state import Finding
from ..tools.sandbox import run_python

MAX_STEPS = 3   # how many plan steps the loop executes (cost/latency bound)
MAX_FIX = 1     # self-correction retries per step


def _extract_code(text: str) -> str:
    """Pull a python code block out of an LLM response."""
    if not text:
        return ""
    fence = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    code = fence.group(1) if fence else text
    return code.strip()


def _extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of an LLM response."""
    if not text:
        return None
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def _gen_code(step: str, schema: str, question: str) -> str:
    return complete(
        "You are a data analyst writing Python to answer a leadership question.\n"
        f"Leadership question: {question}\n"
        f"Analysis step: {step}\n"
        f"The pandas DataFrame `df` is ALREADY loaded (pd and np available).\n"
        f"Columns: {schema}\n\n"
        "Write a SHORT block of Python that computes the answer. Assign the key answer "
        "to a variable named `result` (a number, dict, or small DataFrame) and print the "
        "supporting numbers. Do not read files, make plots, or import os/sys/network "
        "modules. Return ONLY python code.",
        fast=False, fallback="",
    )


def _fix_code(step: str, code: str, error: str) -> str:
    return complete(
        f"This Python errored.\nStep: {step}\nError: {error}\n\nCode:\n{code}\n\n"
        "Return a corrected version. The DataFrame `df` is already loaded. "
        "Return ONLY python code.",
        fast=False, fallback="",
    )


def _interpret(step: str, code: str, stdout: str, result_repr: str) -> Finding | None:
    raw = complete(
        "Convert this executed analysis into ONE finding for leadership.\n"
        f"Step: {step}\nCode:\n{code}\nstdout:\n{stdout}\nresult:\n{result_repr}\n\n"
        "Return ONLY JSON with keys: claim (one executive sentence that QUOTES the actual "
        "numbers from the output), confidence (one of high|medium|low), method (short string), "
        "limitations (list of 1-3 short strings). Use ONLY numbers present in the output.",
        fast=False, fallback="",
    )
    data = _extract_json(raw)
    if not data or "claim" not in data:
        return None
    conf = str(data.get("confidence", "medium")).lower()
    if conf not in {"high", "medium", "low"}:
        conf = "medium"
    return Finding(
        claim=str(data["claim"]),
        evidence={"code": code, "output": (stdout or result_repr)[:1000]},
        confidence=conf,
        method=f"agentic analysis (LLM-written code, sandbox-verified) — {data.get('method', '')}".strip(" -"),
        limitations=[str(x) for x in data.get("limitations", [])][:3],
    )


def agentic_analyst(state) -> dict:
    if get_llm() is None:  # no API key -> rely on the deterministic baseline
        return {"log": state.log + ["agentic_analyst: skipped (no API key)"]}

    schema = ", ".join(c.name for c in state.profile.columns) if state.profile else ""
    question = state.question or "Give leadership the key insights from this dataset."
    new_findings: list[Finding] = []
    executed = 0

    for step in state.plan[:MAX_STEPS]:
        code = _extract_code(_gen_code(step, schema, question))
        if not code:
            continue
        res = run_python(code, state.dataset_path)
        fixes = 0
        while not res.ok and fixes < MAX_FIX:
            code = _extract_code(_fix_code(step, code, res.error or ""))
            if not code:
                break
            res = run_python(code, state.dataset_path)
            fixes += 1
        if not res.ok:
            continue  # do NOT fabricate a finding for a step we couldn't compute
        finding = _interpret(step, code, res.stdout, res.result_repr or "")
        if finding:
            new_findings.append(finding)
            executed += 1

    return {
        "findings": state.findings + new_findings,
        "log": state.log + [f"agentic_analyst: {len(new_findings)} grounded findings "
                            f"from {executed}/{min(len(state.plan), MAX_STEPS)} steps"],
    }
