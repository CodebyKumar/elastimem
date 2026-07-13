"""Embedding storage and similarity search.

Vectors are stored as little-endian float32 BLOBs (``array('f')``) alongside
their chunk rows — no separate index. At personal-memory scale (thousands of
chunks, not millions) brute-force cosine in pure Python answers in tens of
milliseconds; ``benchmarks/bench_recall.py`` keeps that claim honest. Cosine
scoring here is unconditional and brute-force — every embedded chunk is
scanned per query. The ``elastimem[vec]`` extra is reserved for a future
``sqlite-vec``-backed index for larger stores; it is declared in
``pyproject.toml`` but not yet wired into this module, so installing it
today has no effect.

Elastimem never embeds on its own: the host's ``embed_fn`` does the work, the
governor decides whether it may be called at all, and a failing ``embed_fn``
disables the vector leg for the rest of the session (degradation floor:
FTS5-only retrieval).
"""

from __future__ import annotations

import logging
import math
from array import array
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .store import Elastimem

log = logging.getLogger("elastimem")


def encode(vector: list[float]) -> bytes:
    return array("f", vector).tobytes()


def decode(blob: bytes) -> array:
    a = array("f")
    a.frombytes(blob)
    return a


def cosine(a, b) -> float:
    dot = num_a = num_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        num_a += x * x
        num_b += y * y
    if num_a == 0.0 or num_b == 0.0:
        return 0.0
    return dot / math.sqrt(num_a * num_b)


def embed_chunks(store: "Elastimem", chunk_ids: list[int]) -> int:
    """Embed and persist vectors for the given chunks. Returns count done.
    A failing ``embed_fn`` disables embeddings for the session (logged once)."""
    if store.embed_fn is None or getattr(store, "_embed_failed", False):
        return 0
    conn = store._conn
    placeholders = ",".join("?" * len(chunk_ids))
    rows = conn.execute(
        f"SELECT id, text FROM chunks WHERE id IN ({placeholders})"
        " AND embedding IS NULL",
        chunk_ids,
    ).fetchall()
    if not rows:
        return 0
    try:
        vectors = store.embed_fn([r["text"] for r in rows])
    except Exception:
        store._embed_failed = True
        log.exception(
            "elastimem: embed_fn failed; vector search disabled for this session"
        )
        return 0
    # Distinguishes the built-in embedder from a host-supplied one in the
    # embedding_model label, so a future re-index (or a host switching from
    # the built-in default to its own embedder) can tell which vectors need
    # re-computing rather than assuming every row came from the same model.
    model_label = "elastimem:bge-small-en-v1.5" if store.embed_query_fn is not None else "host"
    with store._write_lock:
        conn.executemany(
            "UPDATE chunks SET embedding=?, embedding_model=? WHERE id=?",
            [
                (encode(v), model_label, r["id"])
                for r, v in zip(rows, vectors)
            ],
        )
        conn.commit()
    return len(rows)


def embed_pending(store: "Elastimem", limit: int = 200) -> int:
    """Backfill: embed chunks that predate the embedder (or were missed)."""
    rows = store._conn.execute(
        "SELECT id FROM chunks WHERE embedding IS NULL LIMIT ?", (limit,)
    ).fetchall()
    return embed_chunks(store, [r["id"] for r in rows]) if rows else 0


def similar_chunks(store: "Elastimem", query: str, limit: int = 20) -> list[tuple[int, float]]:
    """(chunk_id, cosine_similarity) pairs most similar to the query, best
    first. The raw score (not just rank) matters to the caller: retrieval
    fusion uses it directly rather than discarding it into rank-only RRF,
    since RRF alone can't distinguish "genuinely relevant" from "merely the
    least-irrelevant of a small candidate set" — see search_chunks.

    Raises when embeddings are unavailable — the caller (retrieval fusion)
    treats that as 'no vector leg' and carries on with FTS5 alone.
    """
    if store.embed_fn is None or getattr(store, "_embed_failed", False):
        raise RuntimeError("no embedder available")
    if not store.profile.embeddings_enabled:
        raise RuntimeError("embeddings disabled by governor tier")

    try:
        # Some embedding models (e.g. the built-in bge-small default) are
        # trained asymmetrically: queries and passages need different
        # encoding. embed_query_fn, when set, is that model's query-side
        # encoder; a host-supplied symmetric embed_fn (the common case —
        # most embedding APIs are symmetric) leaves it unset and this falls
        # back to the same embed_fn used for passages, exactly as before.
        if store.embed_query_fn is not None:
            query_vec = store.embed_query_fn(query)
        else:
            query_vec = store.embed_fn([query])[0]
    except Exception:
        store._embed_failed = True
        log.exception("elastimem: query embedding failed; disabling vector leg")
        raise

    scored: list[tuple[float, int]] = []
    for row in store._conn.execute(
        "SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL"
    ):
        scored.append((cosine(query_vec, decode(row["embedding"])), row["id"]))
    scored.sort(reverse=True)
    return [(chunk_id, score) for score, chunk_id in scored[:limit]]
