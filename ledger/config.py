"""Central configuration for Ledger."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # Claude model ids (see SPEC §12). Opus for analyst reasoning, Haiku for cheap steps.
    reasoning_model: str = "claude-opus-4-8"
    fast_model: str = "claude-haiku-4-5-20251001"
    temperature: float = 0.0
    max_tokens: int = 4000

    # Default flagship dataset.
    default_dataset: str = "data/sample_lending.csv"
    target_hint: str = "default"  # known target for the lending demo

    @property
    def has_api_key(self) -> bool:
        return bool(os.getenv("ANTHROPIC_API_KEY"))


SETTINGS = Settings()
