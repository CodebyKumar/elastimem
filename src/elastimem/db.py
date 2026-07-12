"""SQLite layer: schema, migrations, connections, corruption recovery.

One Elastimem store is one SQLite file in WAL mode. Concurrency model:

* WAL allows one writer and many readers at once.
* Each thread gets its own connection (:func:`connect` is cheap; callers cache
  per-thread). All writes are serialized through ``Elastimem._write_lock`` at the
  store level, so ``check_same_thread=False`` is never needed.
* FTS5 is feature-detected once per store; when absent, retrieval falls back
  to ``LIKE`` matching (see the degradation matrix in docs/governor.md).
"""

from __future__ import annotations

import logging
import os
import sqlite3
import time

log = logging.getLogger("elastimem")

SCHEMA_VERSION = 1

_PRAGMAS = (
    "PRAGMA journal_mode=WAL",
    "PRAGMA synchronous=NORMAL",
    "PRAGMA foreign_keys=ON",
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
  id              INTEGER PRIMARY KEY,
  started_at      TEXT NOT NULL,
  ended_at        TEXT,
  title           TEXT,
  summary         TEXT,
  rolling_summary TEXT,
  message_count   INTEGER NOT NULL DEFAULT 0,
  host_tag        TEXT
);

CREATE TABLE IF NOT EXISTS messages (
  id             INTEGER PRIMARY KEY,
  session_id     INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  turn           INTEGER NOT NULL,
  role           TEXT NOT NULL CHECK (role IN ('user','assistant','system')),
  content        TEXT NOT NULL,
  created_at     TEXT NOT NULL,
  token_estimate INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, turn);

CREATE TABLE IF NOT EXISTS chunks (
  id              INTEGER PRIMARY KEY,
  session_id      INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  first_msg_id    INTEGER NOT NULL,
  last_msg_id     INTEGER NOT NULL,
  text            TEXT NOT NULL,
  created_at      TEXT NOT NULL,
  embedding       BLOB,
  embedding_model TEXT,
  importance      REAL NOT NULL DEFAULT 0.5
);
CREATE INDEX IF NOT EXISTS idx_chunks_session ON chunks(session_id);
CREATE INDEX IF NOT EXISTS idx_chunks_pending_embed ON chunks(id) WHERE embedding IS NULL;

CREATE TABLE IF NOT EXISTS facts (
  id               INTEGER PRIMARY KEY,
  key              TEXT NOT NULL,
  value            TEXT NOT NULL,
  category         TEXT NOT NULL CHECK (category IN ('profile','note')),
  source           TEXT NOT NULL CHECK (source IN ('explicit','auto','rule','import')),
  importance       REAL NOT NULL,
  created_at       TEXT NOT NULL,
  valid_from       TEXT NOT NULL,
  invalidated_at   TEXT,
  invalidated_by   INTEGER REFERENCES facts(id),
  archived         INTEGER NOT NULL DEFAULT 0,
  last_accessed_at TEXT,
  access_count     INTEGER NOT NULL DEFAULT 0
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_facts_current
  ON facts(key) WHERE invalidated_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_facts_key_history ON facts(key, valid_from);

CREATE TABLE IF NOT EXISTS lessons (
  id         INTEGER PRIMARY KEY,
  text       TEXT NOT NULL UNIQUE,
  tag        TEXT,
  created_at TEXT NOT NULL,
  use_count  INTEGER NOT NULL DEFAULT 0,
  archived   INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS quarantine (
  id     INTEGER PRIMARY KEY,
  ts     TEXT NOT NULL,
  key    TEXT,
  value  TEXT,
  reason TEXT NOT NULL,
  source TEXT
);
"""

_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  text, content='chunks', content_rowid='id', tokenize='porter unicode61');

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
  INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE OF text ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
  INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
  key, value, content='facts', content_rowid='id', tokenize='porter unicode61');

CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
  INSERT INTO facts_fts(rowid, key, value) VALUES (new.id, new.key, new.value);
END;
CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
  INSERT INTO facts_fts(facts_fts, rowid, key, value)
    VALUES ('delete', old.id, old.key, old.value);
END;
CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE OF key, value ON facts BEGIN
  INSERT INTO facts_fts(facts_fts, rowid, key, value)
    VALUES ('delete', old.id, old.key, old.value);
  INSERT INTO facts_fts(rowid, key, value) VALUES (new.id, new.key, new.value);
END;
"""


def _fts5_available(conn: sqlite3.Connection) -> bool:
    try:
        conn.execute("CREATE VIRTUAL TABLE temp._fts_probe USING fts5(x)")
        conn.execute("DROP TABLE temp._fts_probe")
        return True
    except sqlite3.OperationalError:
        return False


def connect(path: str) -> sqlite3.Connection:
    """Open a connection to an existing store. One per thread; no schema work."""
    conn = sqlite3.connect(path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    for pragma in _PRAGMAS:
        conn.execute(pragma)
    return conn


def open_store(path: str) -> tuple[sqlite3.Connection, bool]:
    """Open (creating/migrating if needed) the store at ``path``.

    Returns ``(connection, fts_enabled)``. If the existing file is corrupt it
    is renamed to ``<path>.corrupt-<ts>`` and a fresh store is created — the
    store must never take the host down (degradation floor).
    """
    if path != ":memory:":
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    try:
        conn = connect(path)
        conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
    except sqlite3.DatabaseError:
        quarantined = f"{path}.corrupt-{int(time.time())}"
        log.warning("elastimem: store at %s is corrupt; moving to %s and recreating",
                    path, quarantined)
        os.replace(path, quarantined)
        conn = connect(path)

    fts = _fts5_available(conn)
    with conn:
        conn.executescript(_SCHEMA)
        if fts:
            conn.executescript(_FTS_SCHEMA)
        _migrate(conn)
    return conn, fts


def _migrate(conn: sqlite3.Connection) -> None:
    """Versioned migrations. v1 is the base schema; future versions append here."""
    row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO meta(key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        return
    current = int(row["value"])
    if current > SCHEMA_VERSION:
        raise RuntimeError(
            f"store schema v{current} is newer than this elastimem (v{SCHEMA_VERSION}); "
            "upgrade the elastimem package"
        )
    # future: stepwise `if current < 2: ...` blocks, then bump the meta row.


def utcnow() -> str:
    """ISO-8601 UTC timestamp, the single time format used across the store."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")
