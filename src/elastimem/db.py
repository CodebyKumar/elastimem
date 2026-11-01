"""SQLite storage: schema, migrations, connections.

One store is one SQLite file in WAL mode. Each thread opens its own
connection and writes are serialized by the store's write lock. A
``:memory:`` store is the exception: every new ``:memory:`` connection is a
separate empty database, so all threads share the first one.

``messages`` is the source of truth. Every other table is derived from it,
and pending work is whatever derived data is still missing (a chunk with no
embedding, a turn with no ``extracted_at``), so nothing is lost on a crash.
"""

from __future__ import annotations

import logging
import math
import os
import sqlite3
import time
from datetime import datetime, timezone

log = logging.getLogger("elastimem")

SCHEMA_VERSION = 4

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS sessions (
  id              INTEGER PRIMARY KEY,
  started_at      TEXT NOT NULL,
  ended_at        TEXT,
  title           TEXT,
  summary         TEXT,
  rolling_summary TEXT,
  summary_turn    INTEGER NOT NULL DEFAULT 0,
  message_count   INTEGER NOT NULL DEFAULT 0,
  tag             TEXT
);

CREATE TABLE IF NOT EXISTS messages (
  id         INTEGER PRIMARY KEY,
  session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  turn       INTEGER NOT NULL,
  role       TEXT NOT NULL CHECK (role IN ('user','assistant','system')),
  content    TEXT NOT NULL,
  created_at TEXT NOT NULL,
  tokens     INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, turn);

CREATE TABLE IF NOT EXISTS chunks (
  id              INTEGER PRIMARY KEY,
  session_id      INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  first_msg_id    INTEGER NOT NULL,
  last_msg_id     INTEGER NOT NULL,
  text            TEXT NOT NULL,
  created_at      TEXT NOT NULL,
  importance      REAL NOT NULL DEFAULT 0.5,
  embedding       BLOB,
  embedding_model TEXT,
  extracted_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunks_session ON chunks(session_id);
CREATE INDEX IF NOT EXISTS idx_chunks_unembedded ON chunks(id) WHERE embedding IS NULL;
CREATE INDEX IF NOT EXISTS idx_chunks_unextracted ON chunks(id) WHERE extracted_at IS NULL;

CREATE TABLE IF NOT EXISTS facts (
  id              INTEGER PRIMARY KEY,
  key             TEXT NOT NULL,
  value           TEXT NOT NULL,
  source          TEXT NOT NULL CHECK (source IN ('explicit','auto','import')),
  importance      REAL NOT NULL,
  created_at      TEXT NOT NULL,
  since           TEXT NOT NULL,
  until           TEXT,
  replaced_by     INTEGER,
  archived        INTEGER NOT NULL DEFAULT 0,
  last_used       TEXT,
  uses            INTEGER NOT NULL DEFAULT 0,
  embedding       BLOB,
  embedding_model TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_facts_current ON facts(key) WHERE until IS NULL;
CREATE INDEX IF NOT EXISTS idx_facts_history ON facts(key, since);
CREATE INDEX IF NOT EXISTS idx_facts_unembedded ON facts(id) WHERE embedding IS NULL;

CREATE TABLE IF NOT EXISTS lessons (
  id         INTEGER PRIMARY KEY,
  text       TEXT NOT NULL UNIQUE,
  created_at TEXT NOT NULL,
  uses       INTEGER NOT NULL DEFAULT 0,
  archived   INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS rejected (
  id     INTEGER PRIMARY KEY,
  ts     TEXT NOT NULL,
  key    TEXT,
  value  TEXT,
  reason TEXT NOT NULL,
  source TEXT
);

CREATE TABLE IF NOT EXISTS entities (
  id         INTEGER PRIMARY KEY,
  type       TEXT NOT NULL DEFAULT 'entity',
  name       TEXT NOT NULL,
  aliases    TEXT NOT NULL DEFAULT '[]',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  confidence REAL NOT NULL DEFAULT 0.5,
  seen       INTEGER NOT NULL DEFAULT 1,
  topic_id   INTEGER,
  topic      TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_entities_name ON entities(type, name);
CREATE INDEX IF NOT EXISTS idx_entities_topic ON entities(topic_id);

CREATE TABLE IF NOT EXISTS relations (
  id         INTEGER PRIMARY KEY,
  source     INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
  target     INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
  relation   TEXT NOT NULL,
  confidence REAL NOT NULL DEFAULT 0.5,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  seen       INTEGER NOT NULL DEFAULT 1,
  chunk_id   INTEGER REFERENCES chunks(id) ON DELETE SET NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_relations_identity ON relations(source, target, relation);
CREATE INDEX IF NOT EXISTS idx_relations_target ON relations(target);

CREATE TABLE IF NOT EXISTS mentions (
  entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
  chunk_id  INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
  PRIMARY KEY (entity_id, chunk_id)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_mentions_chunk ON mentions(chunk_id);
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

CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
  key, value, content='facts', content_rowid='id', tokenize='porter unicode61');
CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
  INSERT INTO facts_fts(rowid, key, value) VALUES (new.id, new.key, new.value);
END;
CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
  INSERT INTO facts_fts(facts_fts, rowid, key, value) VALUES ('delete', old.id, old.key, old.value);
END;
CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE OF key, value ON facts BEGIN
  INSERT INTO facts_fts(facts_fts, rowid, key, value) VALUES ('delete', old.id, old.key, old.value);
  INSERT INTO facts_fts(rowid, key, value) VALUES (new.id, new.key, new.value);
END;
"""

# v3 -> v4. Tables whose constraints change are rebuilt; the rest are
# renamed in place. Turns recorded before v4 count as already extracted, so
# opening an old store does not queue its whole history for the LLM.
_MIGRATE_V3 = """
ALTER TABLE sessions ADD COLUMN summary_turn INTEGER NOT NULL DEFAULT 0;
ALTER TABLE sessions RENAME COLUMN host_tag TO tag;
ALTER TABLE messages RENAME COLUMN token_estimate TO tokens;
ALTER TABLE chunks ADD COLUMN extracted_at TEXT;
UPDATE chunks SET extracted_at = created_at;
ALTER TABLE quarantine RENAME TO rejected;

DROP INDEX IF EXISTS idx_chunks_pending_embed;
DROP TRIGGER IF EXISTS chunks_au;
DROP TRIGGER IF EXISTS facts_ai;
DROP TRIGGER IF EXISTS facts_ad;
DROP TRIGGER IF EXISTS facts_au;
DROP TABLE IF EXISTS facts_fts;
DROP INDEX IF EXISTS idx_facts_current;
DROP INDEX IF EXISTS idx_facts_key_history;
ALTER TABLE facts RENAME TO facts_v3;
ALTER TABLE lessons RENAME TO lessons_v3;
"""

_COPY_V3 = """
INSERT INTO facts(id, key, value, source, importance, created_at, since, until,
                  replaced_by, archived, last_used, uses)
  SELECT id, key, value, CASE source WHEN 'rule' THEN 'auto' ELSE source END,
         importance, created_at, valid_from, invalidated_at, invalidated_by,
         archived, last_accessed_at, access_count
  FROM facts_v3;
INSERT INTO lessons(id, text, created_at, uses, archived)
  SELECT id, text, created_at, use_count, archived FROM lessons_v3;
INSERT INTO entities(id, type, name, aliases, created_at, updated_at, confidence,
                     seen, topic_id, topic)
  SELECT id, type, canonical_name, aliases, created_at, updated_at, confidence,
         mention_count, cluster_id, cluster_label
  FROM graph_nodes;
INSERT INTO relations(id, source, target, relation, confidence, created_at,
                      updated_at, seen, chunk_id)
  SELECT id, source_node, target_node, relationship, confidence, created_at,
         last_seen, seen_count, source_chunk_id
  FROM graph_edges;
INSERT OR IGNORE INTO mentions(entity_id, chunk_id)
  SELECT e.id, c.id FROM entities e JOIN chunks c
  ON lower(c.text) LIKE '%' || e.name || '%' WHERE e.name != 'user';
DROP TABLE facts_v3;
DROP TABLE lessons_v3;
DROP TABLE graph_edges;
DROP TABLE graph_nodes;
"""


def utcnow() -> str:
    """ISO-8601 UTC timestamp, the one time format used in the store."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def days_since(iso_ts: str | None) -> float:
    """Days elapsed since an ``utcnow()`` timestamp (0 for missing/bad input)."""
    if not iso_ts:
        return 0.0
    try:
        then = datetime.fromisoformat(iso_ts)
    except ValueError:
        return 0.0
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - then).total_seconds() / 86400.0)


def decay(value: float, iso_ts: str | None, half_life_days: float) -> float:
    """``value * exp(-days_since(ts) / half_life)``, also registered in SQL."""
    return value * math.exp(-days_since(iso_ts) / half_life_days)


def connect(path: str) -> sqlite3.Connection:
    """Open a connection to an existing store. No schema work."""
    # Each thread uses its own connection, but close() runs on the host's
    # thread and must close the worker's too, so the thread check is off.
    conn = sqlite3.connect(path, timeout=10.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.create_function("decay", 3, decay, deterministic=True)
    return conn


def fts5_available(conn: sqlite3.Connection) -> bool:
    try:
        conn.execute("CREATE VIRTUAL TABLE temp._fts_probe USING fts5(x)")
        conn.execute("DROP TABLE temp._fts_probe")
        return True
    except sqlite3.OperationalError:
        return False


def open_store(path: str) -> tuple[sqlite3.Connection, bool]:
    """Open, creating or migrating if needed. Returns ``(conn, fts_enabled)``.

    A corrupt file is moved to ``<path>.corrupt-<ts>`` and a fresh store is
    created: memory must never take the host down.
    """
    if path != ":memory:":
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    try:
        conn = connect(path)
        conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
    except sqlite3.DatabaseError:
        moved = f"{path}.corrupt-{int(time.time())}"
        log.warning("elastimem: store at %s is corrupt; moved to %s", path, moved)
        os.replace(path, moved)
        conn = connect(path)

    fts = fts5_available(conn)
    version = _version(conn)
    if version is not None and version > SCHEMA_VERSION:
        raise RuntimeError(
            f"store schema v{version} is newer than this elastimem (v{SCHEMA_VERSION}); "
            "upgrade the elastimem package"
        )
    if version is not None and version < 3:
        raise RuntimeError(
            f"store schema v{version} is too old to migrate; open it once with "
            "elastimem 0.2 to upgrade it to v3"
        )
    migrating = version == 3
    script = "".join([
        _MIGRATE_V3 if migrating else "",
        _SCHEMA,
        _COPY_V3 if migrating else "",
        _FTS_SCHEMA if fts else "",
        "INSERT INTO facts_fts(facts_fts) VALUES ('rebuild');" if fts and migrating else "",
        f"INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', '{SCHEMA_VERSION}');",
    ])
    # One transaction: a failed migration leaves the v3 store untouched.
    # Foreign keys are off while tables are rebuilt; the pragma is a no-op
    # inside a transaction, so it is set around it.
    conn.execute("PRAGMA foreign_keys=OFF")
    try:
        conn.executescript(f"BEGIN;{script}COMMIT;")
    except sqlite3.Error:
        conn.rollback()
        raise
    finally:
        conn.execute("PRAGMA foreign_keys=ON")
    return conn, fts


def _version(conn: sqlite3.Connection) -> int | None:
    try:
        row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    except sqlite3.OperationalError:
        return None
    return int(row["value"]) if row else None
