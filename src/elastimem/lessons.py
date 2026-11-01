"""Lessons: notes the agent keeps about its own behavior ("the weather tool
needs a city name"). Exact duplicates are merged; beyond ``max_lessons`` the
oldest are archived, and a repeated lesson comes back."""

from __future__ import annotations

import sqlite3

from .db import utcnow


def learn(conn: sqlite3.Connection, text: str, max_lessons: int) -> bool:
    text = text.strip()
    if not text:
        return False
    with conn:
        conn.execute(
            "INSERT INTO lessons(text, created_at) VALUES (?,?)"
            " ON CONFLICT(text) DO UPDATE SET uses=uses+1, archived=0",
            (text, utcnow()),
        )
        conn.execute(
            "UPDATE lessons SET archived=1 WHERE archived=0 AND id NOT IN"
            " (SELECT id FROM lessons WHERE archived=0 ORDER BY id DESC LIMIT ?)",
            (max_lessons,),
        )
    return True


def recent(conn: sqlite3.Connection, n: int) -> list[str]:
    """The ``n`` newest active lessons, oldest first."""
    rows = conn.execute(
        "SELECT text FROM lessons WHERE archived=0 ORDER BY id DESC LIMIT ?", (n,)
    ).fetchall()
    return [r["text"] for r in reversed(rows)]
