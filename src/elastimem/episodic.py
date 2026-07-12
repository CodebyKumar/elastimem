"""Episodic memory: full transcripts, chunked for recall, session lifecycle.

Every exchange is persisted twice: verbatim in ``messages`` (the permanent
record, re-indexable later with better models) and condensed into ``chunks``
(the retrieval unit — one per exchange, split at sentence boundaries when an
exchange outgrows ``chunk_target_tokens``).

Sessions are cheap rows; a crash leaves ``ended_at`` NULL and the next
startup closes such orphans with a title-only summary (degradation floor:
no LLM needed to keep the record coherent).
"""

from __future__ import annotations

import re
import sqlite3

from .assembly import estimate_tokens
from .config import ElastimemConfig
from .db import utcnow


def begin_session(conn: sqlite3.Connection, host_tag: str | None = None) -> int:
    with conn:
        cur = conn.execute(
            "INSERT INTO sessions(started_at, host_tag) VALUES (?,?)",
            (utcnow(), host_tag),
        )
    return cur.lastrowid


def close_orphan_sessions(conn: sqlite3.Connection) -> int:
    """Close sessions a crash left open, titling them from their first message."""
    orphans = conn.execute(
        "SELECT id FROM sessions WHERE ended_at IS NULL"
    ).fetchall()
    now = utcnow()
    with conn:
        for row in orphans:
            first = conn.execute(
                "SELECT content FROM messages WHERE session_id=? AND role='user'"
                " ORDER BY id LIMIT 1", (row["id"],)
            ).fetchone()
            title = (first["content"][:80] if first else None)
            conn.execute(
                "UPDATE sessions SET ended_at=?, title=COALESCE(title, ?)"
                " WHERE id=?", (now, title, row["id"]),
            )
    return len(orphans)


def end_session(
    conn: sqlite3.Connection, session_id: int, summary: str | None = None
) -> None:
    first = conn.execute(
        "SELECT content FROM messages WHERE session_id=? AND role='user'"
        " ORDER BY id LIMIT 1", (session_id,)
    ).fetchone()
    title = first["content"][:80] if first else None
    with conn:
        conn.execute(
            "UPDATE sessions SET ended_at=?, title=COALESCE(title,?),"
            " summary=COALESCE(?, summary) WHERE id=?",
            (utcnow(), title, summary, session_id),
        )


def record_turn(
    conn: sqlite3.Connection,
    config: ElastimemConfig,
    session_id: int,
    turn: int,
    user_text: str,
    assistant_text: str,
) -> list[int]:
    """Persist one exchange. Returns the ids of the chunks created (for the
    embedding queue)."""
    now = utcnow()
    with conn:
        cur = conn.execute(
            "INSERT INTO messages(session_id, turn, role, content, created_at,"
            " token_estimate) VALUES (?,?,?,?,?,?)",
            (session_id, turn, "user", user_text, now, estimate_tokens(user_text)),
        )
        first_id = cur.lastrowid
        cur = conn.execute(
            "INSERT INTO messages(session_id, turn, role, content, created_at,"
            " token_estimate) VALUES (?,?,?,?,?,?)",
            (session_id, turn, "assistant", assistant_text, now,
             estimate_tokens(assistant_text)),
        )
        last_id = cur.lastrowid
        conn.execute(
            "UPDATE sessions SET message_count = message_count + 2 WHERE id=?",
            (session_id,),
        )

        chunk_ids = []
        for text in _chunk_exchange(user_text, assistant_text,
                                    config.chunk_target_tokens):
            cur = conn.execute(
                "INSERT INTO chunks(session_id, first_msg_id, last_msg_id, text,"
                " created_at) VALUES (?,?,?,?,?)",
                (session_id, first_id, last_id, text, now),
            )
            chunk_ids.append(cur.lastrowid)
    return chunk_ids


def _chunk_exchange(user_text: str, assistant_text: str,
                    target_tokens: int) -> list[str]:
    """One retrieval chunk per exchange; oversized exchanges split at
    sentence boundaries so no chunk grossly exceeds the target."""
    combined = f"User: {user_text}\nAssistant: {assistant_text}"
    if estimate_tokens(combined) <= target_tokens:
        return [combined]

    sentences = re.split(r"(?<=[.!?])\s+", combined)
    chunks: list[str] = []
    buf = ""
    for sentence in sentences:
        candidate = f"{buf} {sentence}".strip()
        if buf and estimate_tokens(candidate) > target_tokens:
            chunks.append(buf)
            buf = sentence
        else:
            buf = candidate
    if buf:
        chunks.append(buf)
    return chunks


def list_sessions(conn: sqlite3.Connection, n: int = 20) -> list[dict]:
    rows = conn.execute(
        "SELECT id, started_at, ended_at, title, summary, message_count"
        " FROM sessions ORDER BY id DESC LIMIT ?", (n,)
    ).fetchall()
    return [dict(r) for r in rows]


def session_tail(
    conn: sqlite3.Connection, session_id: int, budget_tokens: int
) -> tuple[str | None, list[dict]]:
    """For resume: the session's rolling summary plus as many trailing
    messages as fit the budget, oldest first."""
    row = conn.execute(
        "SELECT rolling_summary FROM sessions WHERE id=?", (session_id,)
    ).fetchone()
    if row is None:
        return None, []
    messages: list[dict] = []
    used = 0
    for m in conn.execute(
        "SELECT role, content, token_estimate FROM messages WHERE session_id=?"
        " AND role IN ('user','assistant') ORDER BY id DESC", (session_id,)
    ):
        if used + m["token_estimate"] > budget_tokens and messages:
            break
        messages.append({"role": m["role"], "content": m["content"]})
        used += m["token_estimate"]
    messages.reverse()
    return row["rolling_summary"], messages
