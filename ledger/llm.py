"""Claude access for Ledger.

Design choice: every LLM call is OPTIONAL. The deterministic data-science work
(profiling, stats, charts, modeling) runs with no API key. Claude adds the
natural-language layer (column-meaning inference, narratives, leadership Q&A).
This keeps the M1 demo runnable out of the box and makes the DS core testable.
"""
from __future__ import annotations

from functools import lru_cache

from .config import SETTINGS


@lru_cache(maxsize=2)
def get_llm(fast: bool = False):
    """Return a configured ChatAnthropic client, or None if no API key is set."""
    if not SETTINGS.has_api_key:
        return None
    from langchain_anthropic import ChatAnthropic

    model = SETTINGS.fast_model if fast else SETTINGS.reasoning_model
    kwargs = dict(model=model, max_tokens=SETTINGS.max_tokens)
    # Opus 4.8 deprecates `temperature`; only the fast (Haiku) model accepts it.
    if fast:
        kwargs["temperature"] = SETTINGS.temperature
    return ChatAnthropic(**kwargs)


def complete(prompt: str, *, fast: bool = False, fallback: str = "") -> str:
    """Single-shot completion. Returns `fallback` when no LLM is available."""
    llm = get_llm(fast=fast)
    if llm is None:
        return fallback
    try:
        return llm.invoke(prompt).content
    except Exception as exc:  # pragma: no cover - network/credential issues
        # Surface the problem to stderr so failures aren't invisible, but still
        # degrade gracefully to the deterministic fallback.
        import sys
        print(f"[ledger] LLM call failed ({'fast' if fast else 'reasoning'}): {exc}",
              file=sys.stderr)
        return fallback or f"[LLM call failed: {exc}]"


_ROLE_MAP = {"user": "human", "assistant": "ai", "system": "system",
             "human": "human", "ai": "ai"}


def chat(messages: list[tuple[str, str]], *, fast: bool = False, fallback: str = "") -> str:
    """Multi-turn completion. `messages` is a list of (role, content) tuples where
    role is system/user/assistant. Returns `fallback` when no LLM is available."""
    llm = get_llm(fast=fast)
    if llm is None:
        return fallback
    lc_msgs = [(_ROLE_MAP.get(r, "human"), c) for r, c in messages]
    try:
        return llm.invoke(lc_msgs).content
    except Exception as exc:  # pragma: no cover - network/credential issues
        import sys
        print(f"[ledger] chat call failed: {exc}", file=sys.stderr)
        return fallback or f"[chat failed: {exc}]"
