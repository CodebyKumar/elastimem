"""Vectors: finding text by meaning, for chunks and facts.

Vectors are stored as float32 BLOBs next to their rows and kept in an
in-RAM index per table (about 1.5 KB per row at 384 dimensions), so a
search is one matrix product instead of a scan of the database. numpy is
used when importable (the ``embed`` extra already requires it), otherwise a
pure-Python dot product.

The embedder is the host's function, the built-in model, or none:

* a host function ``(list[str]) -> list[list[float]]`` is already in the
  host's memory, so using it costs no extra RAM;
* the built-in model (BAAI/bge-small-en-v1.5 via fastembed, ``pip install
  elastimem[embed]``) is downloaded on first use and adds ~360 MB of RAM
  (measured), which is why LITE may use it but never load it.

A failing embedder turns the vector term off for the rest of the process.
"""

from __future__ import annotations

import importlib.util
import logging
import math
import operator
import sqlite3
import threading
from array import array
from typing import Callable

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised by the pure-Python path test
    np = None

log = logging.getLogger("elastimem")

EmbedFn = Callable[[list[str]], list[list[float]]]

BUILTIN_LABEL = "elastimem:bge-small-en-v1.5"   # same label 0.2 wrote, so its vectors stay valid
_BUILTIN_MODEL = "BAAI/bge-small-en-v1.5"
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_TEXT_SQL = {
    "chunks": "SELECT id, text FROM chunks WHERE embedding IS NULL LIMIT ?",
    "facts": "SELECT id, replace(key, '_', ' ') || ': ' || value AS text FROM facts"
             " WHERE embedding IS NULL AND until IS NULL AND archived=0 LIMIT ?",
}

# --- built-in model (one per process) -------------------------------------------
_model = None
_model_error: str | None = None
_model_lock = threading.Lock()


def builtin_installed() -> bool:
    return importlib.util.find_spec("fastembed") is not None


def builtin_loaded() -> bool:
    return _model is not None


def _builtin():
    global _model, _model_error
    with _model_lock:
        if _model is None and _model_error is None:
            try:
                from fastembed import TextEmbedding
                _model = TextEmbedding(model_name=_BUILTIN_MODEL)
            except Exception as exc:
                _model_error = f"built-in embedder failed to load: {exc}"
                log.warning("elastimem: %s; using keyword search only", _model_error)
        if _model is None:
            raise RuntimeError(_model_error)
        return _model


def _normalize(vector):
    """Unit-length float32 vector: a numpy array when numpy is present."""
    if np is not None:
        v = np.asarray(vector, dtype=np.float32)
        return v / (float(np.linalg.norm(v)) or 1.0)
    v = array("f", vector)
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return array("f", (x / norm for x in v))


def _decode(blob: bytes):
    if np is not None:
        return np.frombuffer(blob, dtype=np.float32)
    v = array("f")
    v.frombytes(blob)
    return v


class _Index:
    """Row id -> normalized vector, searchable by dot product."""

    def __init__(self) -> None:
        self.pos: dict[int, int] = {}
        self.ids: list[int] = []
        self.rows: list[array] = []
        self._matrix = None

    def add(self, row_id: int, vector) -> None:
        if row_id in self.pos:
            self.rows[self.pos[row_id]] = vector
        else:
            self.pos[row_id] = len(self.ids)
            self.ids.append(row_id)
            self.rows.append(vector)
        self._matrix = None

    def search(self, query, limit: int, exclude: set[int]) -> list[tuple[int, float]]:
        dim = len(query)
        if np is not None and self.rows:
            if self._matrix is None:
                # shortcut: rebuilt after each batch of adds, ~ms at 10k rows; grow in place if it shows up in profiles
                zero = np.zeros(dim, dtype=np.float32)
                self._matrix = np.vstack([r if len(r) == dim else zero for r in self.rows])
            scores = zip(self.ids, (self._matrix @ query).tolist())
        else:
            scores = ((i, sum(map(operator.mul, r, query)) if len(r) == dim else 0.0)
                      for i, r in zip(self.ids, self.rows))
        hits = [(i, s) for i, s in scores if s > 0 and i not in exclude]
        hits.sort(key=lambda pair: -pair[1])
        return hits[:limit]


class Vectors:
    """Owns the embedder and the per-table indexes."""

    def __init__(self, embedder: EmbedFn | str | None = "auto") -> None:
        if embedder == "auto":
            embedder = "builtin" if builtin_installed() else None
        self.builtin = embedder == "builtin"
        self.host_fn = embedder if callable(embedder) else None
        self.label = BUILTIN_LABEL if self.builtin else "host"
        self.failed: str | None = None
        self._indexes: dict[str, _Index] = {}
        self._lock = threading.Lock()

    def unavailable(self, may_load: bool) -> str | None:
        """Why the vector term cannot run right now, or None if it can."""
        if not self.builtin and self.host_fn is None:
            return "no embedder (install elastimem[embed] for the built-in one)"
        if self.failed:
            return self.failed
        if self.builtin and not builtin_loaded() and not may_load:
            return "built-in model not loaded, and this tier may not load it"
        return None

    def _embed(self, texts: list[str], query: bool = False) -> list | None:
        try:
            if self.builtin:
                prefix = _QUERY_PREFIX if query else ""
                raw = list(_builtin().embed([prefix + t for t in texts]))
            else:
                raw = self.host_fn(texts)
            return [_normalize(v) for v in raw]
        except Exception as exc:
            self.failed = f"embedder failed: {exc}"
            log.exception("elastimem: embedder failed; vector search off for this process")
            return None

    def forget_other_models(self, conn: sqlite3.Connection, write_lock) -> None:
        """Clear vectors written by a different embedder, so they are redone.
        shortcut: every host function shares the label "host", so switching
        between two host embedders is not detected; label by model name if hosts need that."""
        if self.unavailable(may_load=True) is None:
            with write_lock, conn:
                for table in _TEXT_SQL:
                    conn.execute(
                        f"UPDATE {table} SET embedding=NULL, embedding_model=NULL"
                        " WHERE embedding IS NOT NULL AND embedding_model IS NOT ?", (self.label,),
                    )

    def _index(self, conn: sqlite3.Connection, table: str) -> _Index:
        index = self._indexes.get(table)
        if index is None:
            index = self._indexes[table] = _Index()
            for row in conn.execute(f"SELECT id, embedding FROM {table} WHERE embedding IS NOT NULL"):
                index.add(row["id"], _normalize(_decode(row["embedding"])))
        return index

    def embed_pending(self, conn: sqlite3.Connection, write_lock, table: str, limit: int) -> int:
        """Embed up to ``limit`` rows of ``table`` that have no vector yet."""
        rows = conn.execute(_TEXT_SQL[table], (limit,)).fetchall()
        if not rows:
            return 0
        vectors = self._embed([r["text"] for r in rows])
        if vectors is None:
            return 0
        with write_lock, conn:
            conn.executemany(
                f"UPDATE {table} SET embedding=?, embedding_model=? WHERE id=?",
                [(v.tobytes(), self.label, r["id"]) for r, v in zip(rows, vectors)],
            )
        with self._lock:
            index = self._index(conn, table)
            for r, v in zip(rows, vectors):
                index.add(r["id"], v)
        return len(rows)

    def query(self, text: str):
        vectors = self._embed([text], query=True)
        return vectors[0] if vectors else None

    def similar(
        self, conn: sqlite3.Connection, table: str, query, limit: int,
        exclude: set[int] = frozenset(),
    ) -> list[tuple[int, float]]:
        """``(row_id, similarity)`` pairs, best first, similarity > 0."""
        with self._lock:
            return self._index(conn, table).search(query, limit, exclude)
