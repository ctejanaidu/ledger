"""Target-column resolution, shared by every node that needs the target.

Precedence: an explicit user-chosen target wins (this is the "pick your target
column" feature); otherwise fall back to auto-detection — a recognizably-named
column first, else the first detected candidate.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

NAMED = {"default", "target", "label", "class", "fraud", "churn", "y", "outcome"}

# Tokens that indicate the "positive"/event class when a binary target is text-coded.
_POS_TOKENS = {"yes", "true", "y", "t", "1", "positive", "pos", "default", "fraud",
               "churn", "disease", "diabetic", "spam", "attrition", "bad", "success", "win"}


def to_binary01(s: "pd.Series") -> tuple["pd.Series", object]:
    """Map a 2-class column to 0/1. Returns (encoded_int_series, positive_label).

    Works for numeric 0/1, integers like 1/2, booleans, and text labels (Yes/No,
    churned/active, …). The positive class (1) is a recognizable token if present,
    otherwise the minority class — i.e. the event of interest."""
    vals = list(pd.Series(s.dropna().unique()))
    if set(o if not isinstance(o, bool) else int(o) for o in vals) <= {0, 1}:
        return s.fillna(0).astype(int), 1
    pos = next((v for v in vals if str(v).strip().lower() in _POS_TOKENS), None)
    if pos is None:
        pos = s.value_counts().idxmin()  # minority class = event of interest
    return (s == pos).astype(int), pos


def resolve_target(explicit: Optional[str], candidates: list[str],
                   columns: list[str]) -> Optional[str]:
    if explicit and explicit in columns:   # user override wins
        return explicit
    for c in candidates or []:              # else a recognizably-named target
        if c.lower() in NAMED:
            return c
    return candidates[0] if candidates else None  # else first auto-detected candidate
