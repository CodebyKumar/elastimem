"""Episodes: what was said. Messages, their search chunks, and sessions.

Every exchange is stored verbatim in ``messages`` and as one or more
``chunks`` (the search unit, split at sentence ends when an exchange is
long). A crash leaves a session open; the next start closes it.

Turns that fall out of the host's window are folded into the session's
``rolling_summary``. ``summary_turn`` records how far folding has got, so it
survives restarts.
"""

from __future__ import annotations

import re
import sqlite3
from typing import Callable

from . import tiers
from .db import utcnow

TokenizerFn = Callable[[str], int]


def estimate_tokens(text: str, tokenizer: TokenizerFn | None = None) -> int:
    """The host's tokenizer when given, else about four characters per token."""
    return tokenizer(text) if tokenizer else max(1, len(text) // 4)


def fit_lines(lines: list[str], budget: int, tokenizer: TokenizerFn | None = None) -> list[str]:
    """The longest prefix of ``lines`` that fits ``budget`` tokens."""
    kept, used = [], 0
    for line in lines:
        used += estimate_tokens(line, tokenizer)
        if used > budget:
            break
        kept.append(line)
    return kept


# --- sessions ----------------------------------------------------------------
def begin(conn: sqlite3.Connection, tag: str | None = None) -> int:
    with conn:
        return conn.execute(
            "INSERT INTO sessions(started_at, tag) VALUES (?,?)", (utcnow(), tag)
        ).lastrowid


def end(conn: sqlite3.Connection, session_id: int, summary: str | None = None) -> None:
    """Close a session, titled by its first user message."""
    with conn:
        conn.execute(
            "UPDATE sessions SET ended_at=?, summary=COALESCE(?, summary),"
            " title=COALESCE(title, (SELECT substr(content, 1, 80) FROM messages"
            "   WHERE session_id=sessions.id AND role='user' ORDER BY id LIMIT 1))"
            " WHERE id=?",
            (utcnow(), summary, session_id),
        )


def close_orphans(conn: sqlite3.Connection) -> None:
    """Close sessions a crash left open."""
    for row in conn.execute("SELECT id FROM sessions WHERE ended_at IS NULL").fetchall():
        end(conn, row["id"])


def list_sessions(conn: sqlite3.Connection, n: int) -> list[dict]:
    rows = conn.execute(
        "SELECT id, started_at, ended_at, title, summary, message_count"
        " FROM sessions ORDER BY id DESC LIMIT ?", (n,)
    )
    return [dict(r) for r in rows]


def summaries(conn: sqlite3.Connection, n: int) -> list[str]:
    """The last ``n`` session summaries, oldest first."""
    rows = conn.execute(
        "SELECT summary FROM sessions WHERE summary IS NOT NULL AND summary != ''"
        " ORDER BY id DESC LIMIT ?", (n,)
    ).fetchall()
    return [r["summary"] for r in reversed(rows)]


def user_turns(conn: sqlite3.Connection, session_id: int) -> list[str]:
    return [r["content"] for r in conn.execute(
        "SELECT content FROM messages WHERE session_id=? AND role='user' ORDER BY id",
        (session_id,),
    )]


def tail(
    conn: sqlite3.Connection, session_id: int, budget: int
) -> tuple[str | None, list[dict]]:
    """A session's rolling summary plus the trailing messages that fit ``budget``."""
    row = conn.execute("SELECT rolling_summary FROM sessions WHERE id=?", (session_id,)).fetchone()
    if row is None:
        return None, []
    messages, used = [], 0
    for m in conn.execute(
        "SELECT role, content, tokens FROM messages WHERE session_id=?"
        " AND role IN ('user','assistant') ORDER BY id DESC", (session_id,)
    ):
        if messages and used + m["tokens"] > budget:
            break
        messages.append({"role": m["role"], "content": m["content"]})
        used += m["tokens"]
    messages.reverse()
    return row["rolling_summary"], messages


# --- turns -------------------------------------------------------------------
def record(
    conn: sqlite3.Connection,
    session_id: int,
    turn: int,
    user: str,
    reply: str,
    tokenizer: TokenizerFn | None = None,
) -> list[int]:
    """Store one exchange. Returns the new chunk ids."""
    now = utcnow()
    with conn:
        ids = [
            conn.execute(
                "INSERT INTO messages(session_id, turn, role, content, created_at, tokens)"
                " VALUES (?,?,?,?,?,?)",
                (session_id, turn, role, text, now, estimate_tokens(text, tokenizer)),
            ).lastrowid
            for role, text in (("user", user), ("assistant", reply))
        ]
        conn.execute(
            "UPDATE sessions SET message_count = message_count + 2 WHERE id=?", (session_id,)
        )
        return [
            conn.execute(
                "INSERT INTO chunks(session_id, first_msg_id, last_msg_id, text, created_at)"
                " VALUES (?,?,?,?,?)",
                (session_id, ids[0], ids[1], text, now),
            ).lastrowid
            for text in _split(f"User: {user}\nAssistant: {reply}", tokenizer)
        ]


def _split(text: str, tokenizer: TokenizerFn | None) -> list[str]:
    """One chunk per exchange; long ones split at sentence ends."""
    if estimate_tokens(text, tokenizer) <= tiers.CHUNK_TOKENS:
        return [text]
    chunks, buf = [], ""
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        candidate = f"{buf} {sentence}".strip()
        if buf and estimate_tokens(candidate, tokenizer) > tiers.CHUNK_TOKENS:
            chunks.append(buf)
            buf = sentence
        else:
            buf = candidate
    return chunks + [buf] if buf else chunks


def mark_important(conn: sqlite3.Connection, chunk_ids: list[int], importance: float = 0.8) -> None:
    """A turn that produced a fact is likelier to matter later."""
    if chunk_ids:
        with conn:
            conn.execute(
                f"UPDATE chunks SET importance = MAX(importance, ?)"
                f" WHERE id IN ({','.join('?' * len(chunk_ids))})",
                (importance, *chunk_ids),
            )


def unextracted(
    conn: sqlite3.Connection, limit: int, session_id: int | None = None
) -> list[dict]:
    """Turns not yet run through LLM extraction, oldest first:
    ``{"chunk_ids", "user", "reply"}``."""
    rows = conn.execute(
        "SELECT c.first_msg_id, c.last_msg_id, group_concat(c.id) AS ids,"
        "  u.content AS user, a.content AS reply"
        " FROM chunks c JOIN messages u ON u.id = c.first_msg_id"
        " JOIN messages a ON a.id = c.last_msg_id"
        " WHERE c.extracted_at IS NULL AND (? IS NULL OR c.session_id = ?)"
        " GROUP BY c.first_msg_id ORDER BY c.first_msg_id LIMIT ?",
        (session_id, session_id, limit),
    ).fetchall()
    return [
        {"chunk_ids": [int(i) for i in r["ids"].split(",")], "user": r["user"], "reply": r["reply"]}
        for r in rows
    ]


def count_unextracted(conn: sqlite3.Connection) -> int:
    return conn.execute(
        "SELECT count(DISTINCT first_msg_id) FROM chunks WHERE extracted_at IS NULL"
    ).fetchone()[0]


def mark_extracted(conn: sqlite3.Connection, chunk_ids: list[int]) -> None:
    with conn:
        conn.execute(
            f"UPDATE chunks SET extracted_at=? WHERE id IN ({','.join('?' * len(chunk_ids))})",
            (utcnow(), *chunk_ids),
        )


# --- rolling summary -----------------------------------------------------------
def evicted(
    conn: sqlite3.Connection, session_id: int, keep_turns: int
) -> tuple[int, str | None, list[tuple[str, str]]]:
    """Turns that left the window and are not folded yet.

    Returns ``(up_to_turn, current_summary, [(user, reply), ...])``.
    """
    row = conn.execute(
        "SELECT rolling_summary, summary_turn, (SELECT max(turn) FROM messages"
        " WHERE session_id=sessions.id) AS last FROM sessions WHERE id=?", (session_id,)
    ).fetchone()
    if row is None or row["last"] is None:
        return 0, None, []
    up_to = row["last"] - keep_turns
    if up_to <= row["summary_turn"]:
        return 0, row["rolling_summary"], []
    msgs = conn.execute(
        "SELECT turn, role, content FROM messages WHERE session_id=? AND turn > ? AND turn <= ?"
        " ORDER BY id", (session_id, row["summary_turn"], up_to)
    ).fetchall()
    pairs: dict[int, list[str]] = {}
    for m in msgs:
        pairs.setdefault(m["turn"], ["", ""])[m["role"] == "assistant"] = m["content"]
    return up_to, row["rolling_summary"], [tuple(p) for p in pairs.values()]


def save_summary(conn: sqlite3.Connection, session_id: int, summary: str, up_to: int) -> None:
    with conn:
        conn.execute(
            "UPDATE sessions SET rolling_summary=?, summary_turn=? WHERE id=?",
            (summary, up_to, session_id),
        )


def rolling_summary(conn: sqlite3.Connection, session_id: int | None) -> str | None:
    if session_id is None:
        return None
    row = conn.execute("SELECT rolling_summary FROM sessions WHERE id=?", (session_id,)).fetchone()
    return row["rolling_summary"] if row else None


def extractive_summary(
    previous: str | None,
    turns: list[tuple[str, str]],
    budget: int,
    tokenizer: TokenizerFn | None = None,
) -> str:
    """Fold turns into the summary without a model: the first sentence of
    each user turn. Under a tight budget the oldest lines go first."""
    lines = previous.splitlines() if previous else []
    for user, _reply in turns:
        clause = _first_sentence(user)
        if clause and f"- {clause}" not in lines:
            lines.append(f"- {clause}")
    kept = fit_lines(list(reversed(lines)), budget, tokenizer)
    return "\n".join(reversed(kept))


def _first_sentence(text: str, max_chars: int = 120) -> str:
    text = " ".join(text.split())
    match = re.search(r"[.!?](\s|$)", text[:max_chars])
    return text[: match.end()].strip() if match else text[:max_chars].strip()
