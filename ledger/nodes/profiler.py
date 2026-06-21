"""Profiler node (SPEC §5) — understand the data before analyzing it.

Deterministic at the core (dtypes, missingness, cardinality, outliers, leakage
hints, target/time detection). When an API key is present, Claude adds plain-
English column meanings and a quality narrative.
"""
from __future__ import annotations

import pandas as pd

from ..llm import complete
from ..state import ColumnProfile, DatasetProfile


def _quality_flags(s: pd.Series, n_rows: int) -> list[str]:
    flags: list[str] = []
    miss = s.isna().mean()
    if miss > 0.20:
        flags.append("high_missing")
    if s.nunique(dropna=True) == n_rows and n_rows > 0:
        flags.append("possible_id")
    if s.nunique(dropna=True) <= 1:
        flags.append("constant")
    if pd.api.types.is_numeric_dtype(s) and s.dropna().size > 10:
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            outliers = ((s < q1 - 3 * iqr) | (s > q3 + 3 * iqr)).mean()
            if outliers > 0.01:
                flags.append("outliers")
    return flags


def _detect_target_candidates(df: pd.DataFrame) -> list[str]:
    named = [c for c in df.columns if c.lower() in {"default", "target", "label", "y", "churn", "fraud"}]
    binary = [c for c in df.columns
              if df[c].nunique(dropna=True) == 2 and c not in named]
    return named + binary


def _detect_time_column(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if "date" in c.lower() or "time" in c.lower():
            try:
                conv = pd.to_datetime(df[c], errors="raise")
            except Exception:
                continue
            # Reject implausible ranges — e.g. numeric "elapsed seconds" columns
            # epoch-misparse to 1970. Only accept genuine calendar dates.
            yrs = conv.dt.year
            if 1990 <= int(yrs.min()) and int(yrs.max()) <= 2100:
                return c
    return None


def profiler(state) -> dict:
    df = pd.read_csv(state.dataset_path)
    n_rows = len(df)

    cols = []
    for name in df.columns:
        s = df[name]
        cols.append(ColumnProfile(
            name=name,
            dtype=str(s.dtype),
            missing_pct=round(float(s.isna().mean()) * 100, 2),
            n_unique=int(s.nunique(dropna=True)),
            quality_flags=_quality_flags(s, n_rows),
        ))

    time_col = _detect_time_column(df)
    targets = _detect_target_candidates(df)

    # optional: LLM-inferred business meaning per column
    schema_txt = "\n".join(f"- {c.name} ({c.dtype}, {c.n_unique} unique, "
                           f"{c.missing_pct}% missing)" for c in cols)
    meanings = complete(
        "You are a data analyst. For this dataset schema, give a one-line plain-English "
        "business meaning for EACH column, formatted as 'column: meaning' on its own line.\n\n"
        f"{schema_txt}",
        fast=True, fallback="",
    )
    if meanings:
        lookup = {}
        for ln in meanings.splitlines():
            if ":" in ln:
                k, _, v = ln.partition(":")
                lookup[k.strip().lstrip("- ").lower()] = v.strip()
        for c in cols:
            c.inferred_meaning = lookup.get(c.name.lower(), "")

    flagged = [f"{c.name}: {', '.join(c.quality_flags)}" for c in cols if c.quality_flags]
    quality_summary = (
        f"{n_rows:,} rows x {len(cols)} columns. "
        + (f"Quality flags -> {'; '.join(flagged)}. " if flagged else "No major quality flags. ")
        + (f"Detected time column: {time_col}. " if time_col else "")
        + (f"Likely target(s): {', '.join(targets)}." if targets else "")
    )

    profile = DatasetProfile(
        n_rows=n_rows, n_cols=len(cols), columns=cols,
        time_column=time_col, target_candidates=targets,
        quality_summary=quality_summary,
    )
    return {"profile": profile, "log": state.log + ["profiler: done"]}
