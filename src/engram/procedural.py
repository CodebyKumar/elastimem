"""Procedural memory: operational lessons the agent learns about its own
behavior ("the weather tool needs a city name, not coordinates"). Deduplicated
by exact text; capped by archiving the oldest beyond ``max_lessons`` — never
deleted, so a recurring mistake can resurrect its lesson."""

from __future__ import annotations

import sqlite3

from .config import EngramConfig
from .db import utcnow


def add_lesson(
    conn: sqlite3.Connection, config: EngramConfig, text: str, tag: str | None = None
) -> bool:
    text = text.strip()
    if not text:
        return False
    with conn:
        cur = conn.execute(
            "INSERT INTO lessons(text, tag, created_at) VALUES (?,?,?)"
            " ON CONFLICT(text) DO UPDATE SET use_count=use_count+1, archived=0",
            (text, tag, utcnow()),
        )
        # Cap: archive oldest beyond max_lessons.
        conn.execute(
            "UPDATE lessons SET archived=1 WHERE archived=0 AND id NOT IN"
            " (SELECT id FROM lessons WHERE archived=0 ORDER BY id DESC LIMIT ?)",
            (config.max_lessons,),
        )
    return cur.rowcount > 0


def load_lessons(conn: sqlite3.Connection, n: int) -> list[str]:
    """The n most recent unarchived lessons, oldest first."""
    rows = conn.execute(
        "SELECT text FROM lessons WHERE archived=0 ORDER BY id DESC LIMIT ?", (n,)
    ).fetchall()
    return [r["text"] for r in reversed(rows)]
