"""Retrieval: hybrid search over facts and episodic chunks.

Ranking model, used everywhere: ``relevance × importance × recency``.

* relevance — FTS5 BM25 rank and/or vector cosine, fused with Reciprocal
  Rank Fusion (RRF, ``1/(60+rank)``). Facts that match nothing keep a 0.3
  floor so important off-topic facts still surface.
* importance — for facts, source-derived weight (explicit 1.0 … auto 0.5,
  see semantic.SOURCE_IMPORTANCE). For chunks, a neutral 0.5 baseline
  bumped to 0.8 if the exchange yielded a stored fact (see
  episodic.bump_importance) — the only per-chunk signal currently derived;
  everything else scores as an average exchange.
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
    from .store import Elastimem

log = logging.getLogger("elastimem")

_RRF_K = 60

# Stopwords are stripped from queries (not from the index): in an OR query,
# BM25's length normalization can let a short document win on "about"/"what"
# alone, burying the document that matched the actual content words.
_STOPWORDS = frozenset(
    "a an and are as at be but by did do does for from had has have he her him "
    "his how i if in is it its me my of on or our she so that the their them "
    "they this to unclear us was we were what when where which who will with "
    "you your about again also can could should would tell remind know talk "
    "talked discuss discussed say said "
    # Every episodic chunk is stored as "User: ...\nAssistant: ..."
    # (episodic.py's record_turn) — these two words are a structural
    # formatting artifact present in EVERY chunk, not a content word. Left
    # unfiltered, "user"/"assistant" match every row equally in the FTS5
    # leg and inject noise into RRF fusion strong enough to outrank a
    # genuinely relevant vector-leg match (e.g. "where does the user "
    # "currently live" matching on the literal word 'user' in every chunk).
    "user assistant".split()
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
        log.exception("elastimem: fact relevance query failed")
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
    store: "Elastimem",
    query: str,
    k: int = 8,
    exclude_session: int | None = None,
) -> list[Hit]:
    """Hybrid search over episodic chunks. FTS5 + vectors when available.

    The two legs are fused differently from a textbook RRF because BM25
    rank and cosine similarity carry very different amounts of information:
    BM25's own score isn't cheaply comparable across queries, so the FTS leg
    stays rank-based (RRF). Cosine similarity IS directly meaningful and
    comparable (a 0.60 match really is stronger than a 0.15 match, same
    query) — discarding it into rank-only RRF was the actual bug here: with
    a small candidate set, RRF assigns every retrieved chunk a score that's
    only marginally lower than the top match regardless of whether it's
    genuinely relevant or just "the 5th-least-irrelevant of 5 chunks", which
    let importance/recency multipliers trivially flip the ranking (a
    fact-bearing but topically unrelated chunk beating the actually-relevant
    one). The vector leg's raw cosine score is used directly instead,
    min-max normalized against the best score in this result set so it's
    comparable in scale to the FTS leg's RRF score.
    """
    conn = store._conn
    fts_scores: dict[int, float] = {}
    vec_scores: dict[int, float] = {}

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
                fts_scores[row["id"]] = fts_scores.get(row["id"], 0.0) + _rrf(i)
        except sqlite3.Error:
            log.exception("elastimem: chunk FTS query failed")

    # Vector leg: only when the store has an embedder wired and the governor
    # allows it. Raw cosine scores, not just rank (see docstring above).
    try:
        from . import embeddings

        vec_hits = embeddings.similar_chunks(store, query, limit=20)
        if vec_hits:
            best_cosine = vec_hits[0][1]
            if best_cosine > 0:
                for chunk_id, cosine_score in vec_hits:
                    # Negative/zero cosine carries no positive relevance
                    # signal - clamp rather than let it drag the fused
                    # score negative.
                    vec_scores[chunk_id] = max(0.0, cosine_score) / best_cosine
    except Exception:
        pass  # no embeddings available — FTS leg stands alone

    # Fuse: whichever leg found a chunk contributes its own normalized
    # relevance; a chunk both legs agree on gets the stronger of the two
    # signals rather than double-counted, since FTS's RRF scale (~0.016 max)
    # and the vector leg's 0-1 cosine scale aren't the same unit - summing
    # them would let FTS silently dominate every fused score.
    fused: dict[int, float] = {}
    for chunk_id in set(fts_scores) | set(vec_scores):
        fts_component = fts_scores.get(chunk_id, 0.0) / _rrf(0)  # normalize RRF to ~0-1 too
        vec_component = vec_scores.get(chunk_id, 0.0)
        fused[chunk_id] = max(fts_component, vec_component)

    if not fused:
        return []

    # importance/recency are intentionally gentle secondary factors: each
    # leg's contribution to `fused` is already normalized to its own 0-1
    # scale (RRF-vs-RRF-max for the FTS leg, cosine-vs-cosine-max for the
    # vector leg), so they break ties among comparably-relevant hits instead
    # of being able to override relevance itself (see the docstring above
    # for the bug this replaced).
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
        relevance = fused[row["id"]]
        # importance/recency are tie-breaker NUDGES, not multipliers, now
        # that relevance is a real 0-1 signal from actual cosine similarity
        # / BM25 rank. A multiplier (importance in [0.5, 0.8], a 1.6x swing)
        # was strong enough to let a fact-bearing-but-topically-unrelated
        # chunk (bumped by episodic.bump_importance) outrank a chunk that
        # scored meaningfully higher on actual query relevance - e.g. "where
        # does the user live" losing to an unrelated favorite-food chunk
        # purely because the food chunk happened to yield a stored fact
        # earlier. Recency gets the same treatment for the same reason (an
        # old, highly-relevant chunk shouldn't lose to a new, irrelevant
        # one). Both are scaled to a small additive nudge instead.
        importance_nudge = (row["importance"] - 0.5) * 0.15   # +/- 0.045 max
        recency_nudge = recency * 0.1                          # 0 to +0.1
        hits.append(
            Hit(
                kind="chunk",
                text=row["text"],
                date=row["created_at"][:10],
                score=relevance + importance_nudge + recency_nudge,
                session_id=row["session_id"],
            )
        )
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:k]


def search_all(store: "Elastimem", query: str, k: int = 5) -> list[Hit]:
    """Chunks + facts combined — backs ``Elastimem.recall()`` and search tools."""
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
    store: "Elastimem",
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
