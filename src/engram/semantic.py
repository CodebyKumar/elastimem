"""Semantic memory: temporally-versioned facts about the user.

The core rule is **never destructive**. Updating a fact invalidates the old
version row (``invalidated_at``, ``invalidated_by``) and inserts a new one,
so "I quit my design job, I'm a writer now" resolves the contradiction while
keeping the full audit chain (:func:`fact_history`). Forgetting archives;
it does not delete.

Importance model: explicit facts (the user asked) weigh 1.0, rule-captured
0.7, auto-extracted 0.5, imported 0.8. At consolidation time facts decay as
``importance * exp(-days_since_access / half_life)``; only never-accessed
``auto`` facts may be archived by decay.
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

from . import guards
from .config import EngramConfig
from .db import utcnow

SOURCE_IMPORTANCE = {"explicit": 1.0, "rule": 0.7, "auto": 0.5, "import": 0.8}


@dataclass(frozen=True)
class Fact:
    id: int
    key: str
    value: str
    category: str
    source: str
    importance: float
    valid_from: str
    invalidated_at: str | None
    archived: bool


def _row_to_fact(row: sqlite3.Row) -> Fact:
    return Fact(
        id=row["id"], key=row["key"], value=row["value"],
        category=row["category"], source=row["source"],
        importance=row["importance"], valid_from=row["valid_from"],
        invalidated_at=row["invalidated_at"], archived=bool(row["archived"]),
    )


def store_fact(
    conn: sqlite3.Connection,
    config: EngramConfig,
    key: str,
    value: str,
    source: str = "explicit",
) -> tuple[bool, str]:
    """Validate and upsert one fact. Returns ``(changed, reason)``.

    Rejected automatic extractions are quarantined, not silently dropped.
    Same-value writes only refresh ``last_accessed_at``.
    """
    normalized, reason = guards.validate_fact(
        key, value, source, reserved_keys=config.reserved_keys
    )
    if normalized is None:
        if source in ("auto", "rule"):
            guards.quarantine(conn, key, value, reason, source, config.quarantine_cap)
        return False, reason
    value = str(value).strip()

    now = utcnow()
    current = conn.execute(
        "SELECT * FROM facts WHERE key=? AND invalidated_at IS NULL", (normalized,)
    ).fetchone()

    if current is not None:
        if current["value"] == value:
            with conn:
                conn.execute(
                    "UPDATE facts SET last_accessed_at=?, archived=0 WHERE id=?",
                    (now, current["id"]),
                )
            return False, "already stored"
        if source == "auto" and len(value) < 2:
            # A suspiciously short auto-extracted value must not clobber a
            # real one (don't let a guess overwrite a name).
            guards.quarantine(
                conn, key, value,
                f"too short to override existing '{current['value']}'",
                source, config.quarantine_cap,
            )
            return False, "kept existing value"

    category = "profile" if normalized in config.profile_keys else "note"
    importance = SOURCE_IMPORTANCE.get(source, 0.5)
    with conn:
        # Invalidate before inserting: the unique index on current versions
        # (idx_facts_current) forbids two live rows for the same key.
        if current is not None:
            conn.execute(
                "UPDATE facts SET invalidated_at=? WHERE id=?", (now, current["id"])
            )
        cur = conn.execute(
            "INSERT INTO facts(key, value, category, source, importance,"
            " created_at, valid_from) VALUES (?,?,?,?,?,?,?)",
            (normalized, value, category, source, importance, now, now),
        )
        if current is not None:
            conn.execute(
                "UPDATE facts SET invalidated_by=? WHERE id=?",
                (cur.lastrowid, current["id"]),
            )
    return True, "stored"


def current_facts(
    conn: sqlite3.Connection, include_archived: bool = False
) -> list[Fact]:
    """All current (non-invalidated) fact versions, profile first."""
    sql = "SELECT * FROM facts WHERE invalidated_at IS NULL"
    if not include_archived:
        sql += " AND archived=0"
    sql += " ORDER BY category='profile' DESC, importance DESC, valid_from DESC"
    return [_row_to_fact(r) for r in conn.execute(sql)]


def facts_as_dict(conn: sqlite3.Connection) -> dict[str, str]:
    """Merged {key: value} view of current facts (compat with simple hosts)."""
    return {f.key: f.value for f in current_facts(conn)}


def fact_history(conn: sqlite3.Connection, key: str) -> list[Fact]:
    """Every version ever stored for ``key``, oldest first — the audit chain."""
    rows = conn.execute(
        "SELECT * FROM facts WHERE key=? ORDER BY valid_from, id",
        (guards.normalize_key(key),),
    )
    return [_row_to_fact(r) for r in rows]


def forget(conn: sqlite3.Connection, key: str) -> bool:
    """Invalidate the current version of ``key`` (tombstone, non-destructive)."""
    normalized = guards.normalize_key(key)
    with conn:
        cur = conn.execute(
            "UPDATE facts SET invalidated_at=? WHERE key=? AND invalidated_at IS NULL",
            (utcnow(), normalized),
        )
    return cur.rowcount > 0


def touch(conn: sqlite3.Connection, fact_ids: list[int]) -> None:
    """Mark facts as accessed (injected into a prompt) — feeds decay math."""
    if not fact_ids:
        return
    now = utcnow()
    with conn:
        conn.executemany(
            "UPDATE facts SET last_accessed_at=?, access_count=access_count+1 WHERE id=?",
            [(now, fid) for fid in fact_ids],
        )


def effective_importance(fact_row: sqlite3.Row, config: EngramConfig) -> float:
    """Decayed importance: ``importance * exp(-days_since_access/half_life)``."""
    anchor = fact_row["last_accessed_at"] or fact_row["valid_from"]
    days = _days_since(anchor)
    return fact_row["importance"] * math.exp(-days / config.fact_decay_half_life_days)


def apply_decay(conn: sqlite3.Connection, config: EngramConfig) -> int:
    """Archive decayed auto-facts. Explicit/rule/import facts are never
    auto-archived. Returns the number archived."""
    archived = 0
    rows = conn.execute(
        "SELECT * FROM facts WHERE invalidated_at IS NULL AND archived=0"
        " AND source='auto' AND access_count=0"
    ).fetchall()
    with conn:
        for row in rows:
            if effective_importance(row, config) < config.fact_archive_threshold:
                conn.execute("UPDATE facts SET archived=1 WHERE id=?", (row["id"],))
                archived += 1
    return archived


def _days_since(iso_ts: str) -> float:
    try:
        then = datetime.fromisoformat(iso_ts)
        if then.tzinfo is None:
            then = then.replace(tzinfo=timezone.utc)
    except ValueError:
        return 0.0
    return max(0.0, (datetime.now(timezone.utc) - then).total_seconds() / 86400.0)
