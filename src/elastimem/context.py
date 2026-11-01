"""The per-turn context: memory sections fitted to their token budgets.

The host puts ``str(context)`` in its prompt and keeps the last
``context.keep_turns`` turns of its own message list; older turns are
already folded into ``context.summary``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from . import episodes, facts, graph, lessons, search, tiers
from .db import days_since
from .episodes import fit_lines
from .governor import Status

if TYPE_CHECKING:
    from .store import Elastimem

TITLES = {
    "facts": "What you know about the user",
    "topics": "Related topics",
    "moments": "Relevant past moments",
    "sessions": "Previous sessions",
    "lessons": "Lessons from past mistakes",
}


@dataclass(frozen=True)
class Context:
    sections: dict[str, str]    # section name -> text ('' when empty)
    summary: str | None         # this session's turns that left the window
    keep_turns: int
    status: Status

    def __str__(self) -> str:
        blocks = [f"## Earlier in this conversation\n{self.summary}"] if self.summary else []
        blocks += [f"## {TITLES[name]}\n{text}" for name, text in self.sections.items() if text]
        return "\n\n".join(blocks)


def build(store: "Elastimem", query: str) -> tuple[Context, list[int]]:
    """The context for this turn, plus the ids of the facts it shows."""
    conn, status, tok = store._conn, store.status, store._tokenizer
    b = status.budgets
    q = search.analyse(store, query) if query.strip() else None

    facts_text, fact_ids = _facts(store, q)
    moments = search.moments(store, q, tiers.MOMENTS_K, store._session) if q else []
    moment_lines = [f"- [{r['created_at'][:10]}] {_condense(r['text'])}" for _, r in moments]
    sections = {
        "facts": facts_text,
        "topics": "\n".join(fit_lines(_topics(store, q), b.topics, tok)),
        "moments": "\n".join(fit_lines(moment_lines, b.moments, tok)),
        "sessions": "\n".join(fit_lines(
            [f"- {s}" for s in episodes.summaries(conn, tiers.SESSIONS_SHOWN)], b.sessions, tok)),
        "lessons": "\n".join(fit_lines(
            [f"- {s}" for s in lessons.recent(conn, tiers.LESSONS_SHOWN)], b.lessons, tok)),
    }
    summary = episodes.rolling_summary(conn, store._session)
    return Context(sections, summary, status.keep_turns, status), fact_ids


def _facts(store: "Elastimem", q: search.Query | None) -> tuple[str, list[int]]:
    """Pinned keys, then explicit facts, then facts matching the query, then
    the rest by importance and age."""
    rows = facts.current(store._conn)
    rank = {s.id: i for i, (s, _) in enumerate(search.facts(store, q))} if q else {}
    pinned = store._config.pinned
    half_life = store._config.half_life_days

    def order(r):
        group = 0 if r["key"] in pinned else 1 if r["source"] == "explicit" else 2
        return (group, rank.get(r["id"], math.inf),
                -r["importance"] * math.exp(-days_since(r["since"]) / half_life))

    rows.sort(key=order)
    lines = fit_lines([f"- {r['key']}: {r['value']}" for r in rows], store.status.budgets.facts,
                      store._tokenizer)
    return "\n".join(lines), [r["id"] for r in rows[: len(lines)]]


def _topics(store: "Elastimem", q: search.Query | None) -> list[str]:
    if not q or not q.reach:
        return []
    grouped: dict[str | None, list[str]] = {}
    for r in graph.entities(store._conn, list(q.reach)):
        grouped.setdefault(r["topic"], []).append(r["name"])
    return [f"- {topic}: {', '.join(sorted(names))}" if topic else f"- {', '.join(sorted(names))}"
            for topic, names in sorted(grouped.items(), key=lambda kv: kv[0] or "")]


def _condense(text: str, limit: int = 240) -> str:
    one_line = " ".join(text.split())
    return one_line[:limit] + ("..." if len(one_line) > limit else "")
