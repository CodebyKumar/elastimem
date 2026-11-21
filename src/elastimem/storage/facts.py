"""Facts: versioned key-value statements about the user.

Never destructive. Changing a fact closes the old version (``until``,
``replaced_by``) and inserts a new one, so :func:`history` keeps the full
chain. Forgetting closes the current version; nothing is deleted.

Automatic facts (from LLM extraction) must be **grounded**: the value has to
share a word with what the user said, and must not just copy the message.
That one structural check stands in for word lists of placeholders, echoed
transcripts and statements about the assistant, and works in any language.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

from .. import tiers
from ..config import normalize_key
from .db import utcnow

IMPORTANCE = {"explicit": 1.0, "import": 0.8, "auto": 0.5}


@dataclass(frozen=True)
class Fact:
    key: str
    value: str
    source: str     # explicit | auto | import
    since: str
    until: str | None


def _words(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.casefold()))


def ungrounded(value: str, user_text: str) -> str | None:
    """Why an automatic fact should be rejected, or None if it is grounded."""
    value_words, user_words = _words(value), _words(user_text)
    if not value_words & user_words:
        return "not grounded in the user's message"
    if len(user_words) >= 6 and len(value_words) >= 0.8 * len(user_words):
        return "copies the message instead of stating a fact"
    return None


def store(
    conn: sqlite3.Connection,
    key: str,
    value: str,
    source: str = "explicit",
    *,
    blocked: frozenset[str] = frozenset(),
    user_text: str = "",
) -> tuple[bool, str]:
    """Validate and store one fact version. Returns ``(changed, reason)``.

    ``user_text`` is required for automatic facts (the grounding check).
    Rejected automatic facts are kept in ``rejected`` for inspection.
    """
    key, value = normalize_key(key), str(value).strip()
    reason = None
    if not key or not value:
        reason = "empty key or value"
    elif key in blocked or any(key.endswith(f"_{b}") for b in blocked):
        reason = "blocked key (owned by the host)"
    elif source == "auto":
        reason = ungrounded(value, user_text)
    if reason:
        if source == "auto":
            reject(conn, key, value, reason, source)
        return False, reason

    now = utcnow()
    current = conn.execute(
        "SELECT id, value FROM facts WHERE key=? AND until IS NULL", (key,)
    ).fetchone()
    with conn:
        if current is not None and current["value"] == value:
            conn.execute("UPDATE facts SET last_used=?, archived=0 WHERE id=?", (now, current["id"]))
            return False, "already stored"
        # Close before inserting: idx_facts_current allows one open version per key.
        if current is not None:
            conn.execute("UPDATE facts SET until=? WHERE id=?", (now, current["id"]))
        new_id = conn.execute(
            "INSERT INTO facts(key, value, source, importance, created_at, since)"
            " VALUES (?,?,?,?,?,?)",
            (key, value, source, IMPORTANCE.get(source, 0.5), now, now),
        ).lastrowid
        if current is not None:
            conn.execute("UPDATE facts SET replaced_by=? WHERE id=?", (new_id, current["id"]))
    return True, "stored"


def current(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Open, unarchived fact versions."""
    return conn.execute(
        "SELECT id, key, value, source, importance, since FROM facts"
        " WHERE until IS NULL AND archived=0"
    ).fetchall()


def as_dict(conn: sqlite3.Connection) -> dict[str, str]:
    return {r["key"]: r["value"] for r in current(conn)}


def history(conn: sqlite3.Connection, key: str) -> list[Fact]:
    """Every version of ``key``, oldest first."""
    rows = conn.execute(
        "SELECT key, value, source, since, until FROM facts WHERE key=? ORDER BY since, id",
        (normalize_key(key),),
    )
    return [Fact(r["key"], r["value"], r["source"], r["since"], r["until"]) for r in rows]


def forget(conn: sqlite3.Connection, key: str) -> bool:
    with conn:
        cur = conn.execute(
            "UPDATE facts SET until=? WHERE key=? AND until IS NULL", (utcnow(), normalize_key(key))
        )
    return cur.rowcount > 0


def touch(conn: sqlite3.Connection, ids: list[int]) -> None:
    """Mark facts as shown to the model; used facts never fade."""
    if ids:
        now = utcnow()
        with conn:
            conn.executemany(
                "UPDATE facts SET last_used=?, uses=uses+1 WHERE id=?", [(now, i) for i in ids]
            )


def archive_stale(conn: sqlite3.Connection, half_life_days: float) -> int:
    """Archive automatic facts that were never used and have faded."""
    with conn:
        return conn.execute(
            "UPDATE facts SET archived=1 WHERE until IS NULL AND archived=0"
            " AND source='auto' AND uses=0"
            " AND decay(importance, COALESCE(last_used, since), ?) < ?",
            (half_life_days, tiers.ARCHIVE_BELOW),
        ).rowcount


def recent_changes(conn: sqlite3.Connection, limit: int = 5) -> list[sqlite3.Row]:
    """Keys whose value changed within the merge window: ``key, old, new, new_id``."""
    return conn.execute(
        "SELECT old.key, old.value AS old, new.value AS new, new.id AS new_id"
        " FROM facts old JOIN facts new ON new.id = old.replaced_by"
        " WHERE old.until >= datetime('now', ?) AND new.until IS NULL LIMIT ?",
        (f"-{tiers.MERGE_WINDOW_DAYS} days", limit),
    ).fetchall()


def set_value(conn: sqlite3.Connection, fact_id: int, value: str) -> None:
    with conn:
        conn.execute(
            "UPDATE facts SET value=?, last_used=?, embedding=NULL WHERE id=?",
            (value, utcnow(), fact_id),
        )


def reject(conn: sqlite3.Connection, key: str, value: str, reason: str, source: str) -> None:
    """Keep a rejected automatic fact for inspection (never shown to the model)."""
    with conn:
        conn.execute(
            "INSERT INTO rejected(ts, key, value, reason, source) VALUES (?,?,?,?,?)",
            (utcnow(), str(key)[:200], str(value)[:500], reason, source),
        )
        trim_rejected(conn)


def trim_rejected(conn: sqlite3.Connection) -> None:
    conn.execute(
        "DELETE FROM rejected WHERE id NOT IN (SELECT id FROM rejected ORDER BY id DESC LIMIT ?)",
        (tiers.REJECTED_CAP,),
    )


def list_rejected(conn: sqlite3.Connection, n: int) -> list[dict]:
    rows = conn.execute(
        "SELECT ts, key, value, reason, source FROM rejected ORDER BY id DESC LIMIT ?", (n,)
    ).fetchall()
    return [dict(r) for r in reversed(rows)]
