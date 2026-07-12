"""Context assembly: turning stored memory into budgeted prompt sections.

`build_context()` produces a :class:`ContextPlan` — named sections, each
greedily filled to the governor's token budget for that section. The host
renders sections into its own prompt format (or calls :meth:`ContextPlan.render`
for a sensible plain-text default) and applies the working-window guidance to
its own message list. Engram plans; the host applies.
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass, field
from typing import Callable

from . import semantic
from .config import EngramConfig, MemoryProfile
from .semantic import _days_since

TokenizerFn = Callable[[str], int]

# Section keys, in canonical injection order.
SECTION_FACTS = "user_facts"
SECTION_EPISODIC = "relevant_past_moments"
SECTION_SESSIONS = "previous_sessions"
SECTION_LESSONS = "lessons"

_SECTION_TITLES = {
    SECTION_FACTS: "WHAT YOU KNOW ABOUT THE USER",
    SECTION_EPISODIC: "RELEVANT PAST MOMENTS (from earlier conversations)",
    SECTION_SESSIONS: "PREVIOUS SESSIONS",
    SECTION_LESSONS: "LESSONS FROM PAST MISTAKES",
}


def estimate_tokens(text: str, tokenizer_fn: TokenizerFn | None = None) -> int:
    """Token count, via the host's tokenizer when given, else the chars/4 proxy."""
    if tokenizer_fn is not None:
        return tokenizer_fn(text)
    return max(1, len(text) // 4)


@dataclass(frozen=True)
class ContextPlan:
    """What the host should put in front of the model this turn."""

    sections: dict[str, str]          # section key -> rendered block ('' = omit)
    rolling_summary: str | None       # condensed evicted turns, or None
    keep_last_n_turns: int            # working-window guidance for the host
    profile: MemoryProfile
    fact_ids: tuple[int, ...] = field(default=())   # injected facts (for touch())

    def render(self) -> str:
        """Default plain-text rendering of all non-empty sections."""
        blocks = [
            f"{_SECTION_TITLES[key]}\n{text}"
            for key, text in self.sections.items()
            if text
        ]
        return "\n\n".join(blocks)


def fit_lines(lines: list[str], budget_tokens: int,
              tokenizer_fn: TokenizerFn | None = None) -> list[str]:
    """Greedy prefix of ``lines`` that fits the budget (order = priority)."""
    kept: list[str] = []
    used = 0
    for line in lines:
        cost = estimate_tokens(line, tokenizer_fn)
        if used + cost > budget_tokens:
            break
        kept.append(line)
        used += cost
    return kept


def build_facts_section(
    conn: sqlite3.Connection,
    config: EngramConfig,
    profile: MemoryProfile,
    relevance: dict[int, float] | None = None,
    tokenizer_fn: TokenizerFn | None = None,
) -> tuple[str, list[int]]:
    """Facts block: profile facts always lead; notes ranked by
    ``relevance × importance × recency``. Returns (text, injected fact ids).

    ``relevance`` maps fact id → FTS relevance (retrieval supplies it when a
    query is available); unmatched facts get a 0.3 floor so important
    off-topic facts still surface.
    """
    facts = semantic.current_facts(conn)
    if not facts:
        return "", []
    relevance = relevance or {}

    profile_facts = [f for f in facts if f.category == "profile"]
    notes = [f for f in facts if f.category != "profile"]
    notes.sort(
        key=lambda f: (
            relevance.get(f.id, 0.3)
            * f.importance
            * math.exp(-_days_since(f.valid_from) / config.fact_recency_half_life_days)
        ),
        reverse=True,
    )

    ordered = profile_facts + notes
    lines = [f"- {f.key}: {f.value}" for f in ordered]
    kept = fit_lines(lines, profile.budgets.facts, tokenizer_fn)
    return "\n".join(kept), [f.id for f in ordered[: len(kept)]]


def build_lessons_section(
    conn: sqlite3.Connection,
    config: EngramConfig,
    profile: MemoryProfile,
    tokenizer_fn: TokenizerFn | None = None,
) -> str:
    from . import procedural

    lessons = procedural.load_lessons(conn, config.lessons_in_prompt)
    kept = fit_lines([f"- {l}" for l in lessons], profile.budgets.lessons, tokenizer_fn)
    return "\n".join(kept)


def estimate_window_turns(budget_tokens: int, avg_turn_tokens: int = 120,
                          min_turns: int = 2) -> int:
    """How many verbatim turns the working budget affords (guidance only)."""
    return max(min_turns, budget_tokens // max(1, avg_turn_tokens) // 2)
