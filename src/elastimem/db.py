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

SCHEMA_VERSION = 3

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

_GRAPH_SCHEMA = """
CREATE TABLE IF NOT EXISTS graph_nodes (
  id              INTEGER PRIMARY KEY,
  type            TEXT NOT NULL DEFAULT 'entity',
  canonical_name  TEXT NOT NULL,
  aliases         TEXT NOT NULL DEFAULT '[]',
  created_at      TEXT NOT NULL,
  updated_at      TEXT NOT NULL,
  importance      REAL NOT NULL DEFAULT 0.5,
  confidence      REAL NOT NULL DEFAULT 0.5,
  mention_count   INTEGER NOT NULL DEFAULT 1,
  cluster_id      INTEGER,
  cluster_label   TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_graph_nodes_canonical
  ON graph_nodes(type, canonical_name);

CREATE TABLE IF NOT EXISTS graph_edges (
  id               INTEGER PRIMARY KEY,
  source_node      INTEGER NOT NULL REFERENCES graph_nodes(id) ON DELETE CASCADE,
  target_node      INTEGER NOT NULL REFERENCES graph_nodes(id) ON DELETE CASCADE,
  relationship     TEXT NOT NULL,
  confidence       REAL NOT NULL DEFAULT 0.5,
  importance       REAL NOT NULL DEFAULT 0.5,
  weight           REAL NOT NULL DEFAULT 1.0,
  created_at       TEXT NOT NULL,
  last_seen        TEXT NOT NULL,
  seen_count       INTEGER NOT NULL DEFAULT 1,
  source_chunk_id  INTEGER REFERENCES chunks(id) ON DELETE SET NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_graph_edges_dedup
  ON graph_edges(source_node, target_node, relationship);
CREATE INDEX IF NOT EXISTS idx_graph_edges_source ON graph_edges(source_node);
CREATE INDEX IF NOT EXISTS idx_graph_edges_target ON graph_edges(target_node);
"""

# Applied separately from _GRAPH_SCHEMA (not folded into the CREATE TABLE
# block above): the cluster_id/cluster_label columns are new as of v3, so
# this index can only be created after a v2 store's ALTER TABLE has run
# (see _migrate's `current < 3` step). A fresh store gets both the columns
# (baked into _GRAPH_SCHEMA's CREATE TABLE) and this index applied here,
# in open_store(), right after _GRAPH_SCHEMA - safe either way since
# CREATE INDEX IF NOT EXISTS is idempotent and the column always exists by
# this point on a fresh store.
_GRAPH_CLUSTER_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_graph_nodes_cluster ON graph_nodes(cluster_id)"
)

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
    """Open a connection to an existing store. One per thread; no schema work.

    ``:memory:`` is the one exception to "one per thread": sqlite3 gives
    each new connection to ``:memory:`` its own separate, empty database,
    so a ":memory:" store must share a single connection across every
    thread that touches it (see ``store.py``'s ``_memory_conn``) rather
    than opening a fresh one per thread the way file-backed stores do.
    That shared connection is therefore genuinely used from multiple
    threads, so it needs ``check_same_thread=False`` — safe here because
    every write already goes through ``Elastimem._write_lock``, and reads
    are short synchronous calls, the same safety property WAL mode
    provides for file-backed stores' actually-separate per-thread
    connections.
    """
    conn = sqlite3.connect(
        path, timeout=10.0, check_same_thread=(path != ":memory:")
    )
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
        conn.executescript(_GRAPH_SCHEMA)
        if fts:
            conn.executescript(_FTS_SCHEMA)
        _migrate(conn)
        # Only safe to run after _migrate(): on a store upgrading from v2,
        # cluster_id doesn't exist as a column until _migrate's `current <
        # 3` step adds it. On a fresh store the column already exists (see
        # _GRAPH_SCHEMA above), so this is a harmless no-op there.
        conn.execute(_GRAPH_CLUSTER_INDEX)
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
    if current < 2:
        conn.executescript(_GRAPH_SCHEMA)
        current = 2
    if current < 3:
        # cluster_id/cluster_label are new columns on an existing table for
        # any v2 store — CREATE TABLE IF NOT EXISTS (already applied above/
        # unconditionally in open_store) is a no-op here, unlike the v1->v2
        # step where graph_nodes didn't exist at all yet.
        existing_cols = {
            r["name"] for r in conn.execute("PRAGMA table_info(graph_nodes)")
        }
        if "cluster_id" not in existing_cols:
            conn.execute("ALTER TABLE graph_nodes ADD COLUMN cluster_id INTEGER")
        if "cluster_label" not in existing_cols:
            conn.execute("ALTER TABLE graph_nodes ADD COLUMN cluster_label TEXT")
        current = 3
    if current != int(row["value"]):
        conn.execute(
            "UPDATE meta SET value=? WHERE key='schema_version'", (str(current),)
        )
    # future: stepwise `if current < 3: ...` blocks, then bump the meta row.


def utcnow() -> str:
    """ISO-8601 UTC timestamp, the single time format used across the store."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")
