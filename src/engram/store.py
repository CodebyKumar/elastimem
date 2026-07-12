"""The Engram store: the one class hosts interact with.

Thread model: public methods may be called from the host's main thread; the
background worker (phase 4) owns deferred writes. All writes acquire the
store-level ``_write_lock``; reads use per-thread connections (WAL-safe).
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from typing import Callable

from . import procedural, semantic
from .config import EngramConfig
from .db import open_store, connect

log = logging.getLogger("engram")

CompleteFn = Callable[..., str]
EmbedFn = Callable[[list[str]], list[list[float]]]


class Engram:
    """One persistent memory store backed by a single SQLite file.

    ``complete_fn`` and ``embed_fn`` are optional host-injected capabilities;
    Engram degrades gracefully around whatever is missing (see
    docs/governor.md for the full degradation matrix).

    ``complete_fn(prompt: str, *, max_tokens: int, temperature: float) -> str``
    ``embed_fn(texts: list[str]) -> list[list[float]]``
    """

    def __init__(
        self,
        path: str,
        *,
        complete_fn: CompleteFn | None = None,
        embed_fn: EmbedFn | None = None,
        config: EngramConfig | None = None,
    ) -> None:
        self.path = os.path.expanduser(path) if path != ":memory:" else path
        self.config = config or EngramConfig()
        self.complete_fn = complete_fn
        self.embed_fn = embed_fn
        self._write_lock = threading.RLock()
        self._local = threading.local()

        conn, self.fts_enabled = open_store(self.path)
        self._local.conn = conn

    # ------------------------------------------------------------------ #
    # connections
    # ------------------------------------------------------------------ #
    @property
    def _conn(self) -> sqlite3.Connection:
        """Per-thread connection (:memory: stores share the creating thread's)."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = connect(self.path)
            self._local.conn = conn
        return conn

    # ------------------------------------------------------------------ #
    # semantic memory (facts)
    # ------------------------------------------------------------------ #
    def remember(self, key: str, value: str, source: str = "explicit") -> tuple[bool, str]:
        """Durably store one fact, synchronously. Returns ``(changed, reason)``."""
        with self._write_lock:
            return semantic.store_fact(self._conn, self.config, key, value, source)

    def facts(self) -> dict[str, str]:
        """Current facts as a merged ``{key: value}`` dict."""
        return semantic.facts_as_dict(self._conn)

    def fact_history(self, key: str) -> list[semantic.Fact]:
        """Every version ever stored for ``key`` — the audit chain."""
        return semantic.fact_history(self._conn, key)

    def forget(self, key: str) -> bool:
        """Invalidate the current version of ``key`` (non-destructive tombstone)."""
        with self._write_lock:
            return semantic.forget(self._conn, key)

    # ------------------------------------------------------------------ #
    # procedural memory (lessons)
    # ------------------------------------------------------------------ #
    def add_lesson(self, text: str, tag: str | None = None) -> bool:
        with self._write_lock:
            return procedural.add_lesson(self._conn, self.config, text, tag)

    def lessons(self, n: int | None = None) -> list[str]:
        return procedural.load_lessons(self._conn, n or self.config.lessons_in_prompt)

    # ------------------------------------------------------------------ #
    # inspection / maintenance
    # ------------------------------------------------------------------ #
    def quarantine_entries(self, n: int = 20) -> list[dict]:
        """Recently rejected automatic extractions (debugging; never prompted)."""
        rows = self._conn.execute(
            "SELECT ts, key, value, reason, source FROM quarantine"
            " ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def stats(self) -> dict:
        counts = {
            table: self._conn.execute(f"SELECT count(*) c FROM {table}").fetchone()["c"]
            for table in ("facts", "sessions", "messages", "chunks", "lessons", "quarantine")
        }
        size = os.path.getsize(self.path) if self.path != ":memory:" else 0
        return {"path": self.path, "db_bytes": size, "fts_enabled": self.fts_enabled,
                **counts}

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None
