"""Target-column resolution, shared by every node that needs the target.

Precedence: an explicit user-chosen target wins (this is the "pick your target
column" feature); otherwise fall back to auto-detection — a recognizably-named
column first, else the first detected candidate.
"""
from __future__ import annotations

from typing import Optional

NAMED = {"default", "target", "label", "class", "fraud", "churn", "y", "outcome"}


def resolve_target(explicit: Optional[str], candidates: list[str],
                   columns: list[str]) -> Optional[str]:
    if explicit and explicit in columns:   # user override wins
        return explicit
    for c in candidates or []:              # else a recognizably-named target
        if c.lower() in NAMED:
            return c
    return candidates[0] if candidates else None  # else first auto-detected candidate
