"""Host settings. Every field has a default; most hosts set none.

Values a typical host would not change live in :mod:`elastimem.tiers`.
"""

from __future__ import annotations

import enum
import os
import re
from dataclasses import dataclass


class Tier(enum.IntEnum):
    """How much background work the machine can afford. Ordered: LITE < FULL."""

    LITE = 0
    STANDARD = 1
    FULL = 2

    @classmethod
    def from_env(cls) -> "Tier | None":
        """The ``ELASTIMEM_TIER`` override (``lite|standard|full``), if set."""
        return cls.__members__.get(os.environ.get("ELASTIMEM_TIER", "").strip().upper())


def normalize_key(key: str) -> str:
    """Fact keys are casefolded snake_case: ``"Favorite Color"`` -> ``favorite_color``."""
    return re.sub(r"\W+", "_", str(key).strip().casefold()).strip("_")


@dataclass
class ElastimemConfig:
    """Host-tunable settings.

    ``context_tokens`` is the chat model's context window. ``reserved_tokens``
    is what the host needs for itself: its own prompt, tool output and the
    reply. Elastimem budgets the rest.
    """

    context_tokens: int = 4096
    reserved_tokens: int = 1112
    tier: Tier | None = None          # fixed tier; None = decided by the governor
    pinned: frozenset[str] = frozenset()   # fact keys always shown first
    blocked: frozenset[str] = frozenset()  # fact keys owned by the host, never stored
    turn_budget_ms: float = 50.0      # memory work per turn before the tier steps down
    half_life_days: float = 30.0      # how fast unused memories fade
    max_lessons: int = 15

    def __post_init__(self) -> None:
        if self.context_tokens < 1024:
            raise ValueError("context_tokens must be >= 1024")
        if not 0 <= self.reserved_tokens <= self.context_tokens - 256:
            raise ValueError("reserved_tokens must leave at least 256 tokens for memory")
        if self.half_life_days <= 0 or self.turn_budget_ms <= 0 or self.max_lessons < 1:
            raise ValueError("half_life_days, turn_budget_ms and max_lessons must be positive")
        if self.tier is None:
            self.tier = Tier.from_env()
        self.pinned = frozenset(normalize_key(k) for k in self.pinned)
        self.blocked = frozenset(normalize_key(k) for k in self.blocked)
