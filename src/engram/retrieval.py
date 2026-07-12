"""Retrieval: hybrid search over facts and episodic chunks.

Ranking model, used everywhere: ``relevance × importance × recency``.

* relevance — FTS5 BM25 rank and/or vector cosine, fused with Reciprocal
  Rank Fusion (RRF, ``1/(60+rank)``). Facts that match nothing keep a 0.3
  floor so important off-topic facts still surface.
* importance — source-derived weight (explicit 1.0 … auto 0.5).
* recency — ``exp(-age_days / half_life)``.

Degradation: no embeddings → FTS5 only; no FTS5 → ``LIKE`` term matching.
Nothing in this module raises to the caller; failures return empty results.
"""

from __future__ import annotations

import logging
import math
import re
import sqlite3
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .assembly import fit_lines
from .config import MemoryProfile
from .semantic import _days_since

if TYPE_CHECKING:
    from .store import Engram

log = logging.getLogger("engram")

_RRF_K = 60

# Stopwords are stripped from queries (not from the index): in an OR query,
# BM25's length normalization can let a short document win on "about"/"what"
# alone, burying the document that matched the actual content words.
_STOPWORDS = frozenset(
    "a an and are as at be but by did do does for from had has have he her him "
    "his how i if in is it its me my of on or our she so that the their them "
    "they this to unclear us was we were what when where which who will with "
    "you your about again also can could should would tell remind know talk "
    "talked discuss discussed say said".split()
)


def _query_terms(text: str) -> list[str]:
    terms = re.findall(r"[a-zA-Z0-9]{2,}", text.lower())
    kept = [t for t in dict.fromkeys(terms) if t not in _STOPWORDS]
    # An all-stopword query ("what did we do") falls back to the raw terms —
    # a weak query beats no query.
    return kept or list(dict.fromkeys(terms))


def _fts_query(text: str) -> str:
    """Turn free text into a safe FTS5 OR-query of quoted content terms."""
    return " OR ".join(f'"{t}"' for t in _query_terms(text))


def _rrf(rank: int) -> float:
    return 1.0 / (_RRF_K + rank)


# --------------------------------------------------------------------------- #
# facts
# --------------------------------------------------------------------------- #
def fact_relevance(
    conn: sqlite3.Connection, query: str, *, fts: bool
) -> dict[int, float]:
    """Map of fact id → relevance for the current facts matching ``query``.

    Scores are RRF-scaled then normalized so the best match is 1.0 (the 0.3
    floor for non-matches is applied by the assembly layer).
    """
    q = _fts_query(query)
    if not q:
        return {}
    try:
        if fts:
            rows = conn.execute(
                "SELECT f.id FROM facts_fts JOIN facts f ON f.id = facts_fts.rowid"
                " WHERE facts_fts MATCH ? AND f.invalidated_at IS NULL"
                " ORDER BY facts_fts.rank LIMIT 20",
                (q,),
            ).fetchall()
        else:
            rows = _like_match(conn, query, table="facts",
                               where="invalidated_at IS NULL", columns=("key", "value"))
        if not rows:
            return {}
        raw = {row["id"]: _rrf(i) for i, row in enumerate(rows)}
        top = max(raw.values())
        return {fid: score / top for fid, score in raw.items()}
    except sqlite3.Error:
        log.exception("engram: fact relevance query failed")
        return {}


# --------------------------------------------------------------------------- #
# episodic chunks
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Hit:
    """One retrieval result, ready to render."""

    kind: str          # 'chunk' | 'fact'
    text: str
    date: str          # YYYY-MM-DD
    score: float
    session_id: int | None = None


def search_chunks(
    store: "Engram",
    query: str,
    k: int = 8,
    exclude_session: int | None = None,
) -> list[Hit]:
    """Hybrid search over episodic chunks. FTS5 + vectors when available."""
    conn = store._conn
    fused: dict[int, float] = {}

    q = _fts_query(query)
    if q:
        try:
            if store.fts_enabled:
                rows = conn.execute(
                    "SELECT c.id FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid"
                    " WHERE chunks_fts MATCH ? ORDER BY chunks_fts.rank LIMIT 20",
                    (q,),
                ).fetchall()
            else:
                rows = _like_match(conn, query, table="chunks", columns=("text",))
            for i, row in enumerate(rows):
                fused[row["id"]] = fused.get(row["id"], 0.0) + _rrf(i)
        except sqlite3.Error:
            log.exception("engram: chunk FTS query failed")

    # Vector leg (phase 5): only when the store has an embedder wired and the
    # governor allows it.
    try:
        from . import embeddings

        vec_rows = embeddings.similar_chunks(store, query, limit=20)
        for i, chunk_id in enumerate(vec_rows):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + _rrf(i)
    except Exception:
        pass  # no embeddings available — FTS leg stands alone

    if not fused:
        return []

    placeholders = ",".join("?" * len(fused))
    rows = conn.execute(
        f"SELECT id, session_id, text, created_at, importance FROM chunks"
        f" WHERE id IN ({placeholders})",
        list(fused),
    ).fetchall()

    hits = []
    for row in rows:
        if exclude_session is not None and row["session_id"] == exclude_session:
            continue
        recency = math.exp(
            -_days_since(row["created_at"])
            / store.config.episodic_recency_half_life_days
        )
        hits.append(
            Hit(
                kind="chunk",
                text=row["text"],
                date=row["created_at"][:10],
                score=fused[row["id"]] * row["importance"] * recency,
                session_id=row["session_id"],
            )
        )
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:k]


def search_all(store: "Engram", query: str, k: int = 5) -> list[Hit]:
    """Chunks + facts combined — backs ``Engram.recall()`` and search tools."""
    hits = search_chunks(store, query, k=k)
    conn = store._conn
    for fact_id, rel in fact_relevance(conn, query, fts=store.fts_enabled).items():
        row = conn.execute(
            "SELECT key, value, importance, valid_from FROM facts WHERE id=?",
            (fact_id,),
        ).fetchone()
        if row is None:
            continue
        recency = math.exp(
            -_days_since(row["valid_from"]) / store.config.fact_recency_half_life_days
        )
        hits.append(
            Hit(kind="fact", text=f"{row['key']}: {row['value']}",
                date=row["valid_from"][:10],
                score=rel * row["importance"] * recency)
        )
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:k]


def episodic_section(
    store: "Engram",
    query: str,
    profile: MemoryProfile,
    tokenizer_fn=None,
) -> str:
    """The RELEVANT PAST MOMENTS block, fitted to the episodic budget."""
    exclude = getattr(store, "session_id", None)
    hits = search_chunks(store, query, k=profile.episodic_top_k,
                         exclude_session=exclude)
    lines = [f"- [{h.date}] {_condense(h.text)}" for h in hits]
    return "\n".join(fit_lines(lines, profile.budgets.episodic, tokenizer_fn))


def _condense(chunk_text: str, limit: int = 240) -> str:
    one_line = " ".join(chunk_text.split())
    return one_line[:limit] + ("…" if len(one_line) > limit else "")


def _like_match(
    conn: sqlite3.Connection,
    query: str,
    *,
    table: str,
    columns: tuple[str, ...],
    where: str = "1=1",
) -> list[sqlite3.Row]:
    """Degradation floor when FTS5 is unavailable: rank by LIKE term count."""
    terms = _query_terms(query)[:8]
    if not terms:
        return []
    score_expr = " + ".join(
        f"(lower({col}) LIKE ?)" for col in columns for _ in terms
    )
    params = [f"%{t}%" for _ in columns for t in terms]
    return conn.execute(
        f"SELECT id, ({score_expr}) AS s FROM {table} WHERE {where}"
        f" AND ({score_expr.replace(' + ', ' OR ')}) ORDER BY s DESC LIMIT 20",
        params + params,
    ).fetchall()
