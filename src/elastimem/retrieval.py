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
# knowledge graph
# --------------------------------------------------------------------------- #
def graph_relevance(
    store: "Elastimem", query: str, *, hops: int
) -> tuple[dict[int, float], set[str]]:
    """(chunk_id -> weighted graph-match strength in [0,1], expanded entity
    canonical_names). Empty when ``hops`` is 0 (LITE tier) or the graph has
    no match for this query. Never raises.

    Each expanded entity's contribution is weighted by the node's own
    extraction-time confidence (a running average built up across repeated
    extractions — see graph.upsert_node) scaled by match specificity (a
    longer canonical-name/alias match is stronger evidence than a short
    one). This keeps a one-off, low-confidence, short entity match (e.g. an
    entity literally named "May" colliding with the month) from carrying
    the same weight as a well-corroborated, specific one — weight is
    self-relative, normalized against the strongest match in this query's
    own expanded set, mirroring how fact_relevance() normalizes its own
    RRF scores to the query's top match.
    """
    if hops <= 0:
        return {}, set()
    conn = store._conn
    try:
        from . import graph as graph_mod

        seeds = graph_mod.detect_seed_nodes(conn, query)
        if not seeds:
            return {}, set()
        expanded_ids = graph_mod.expand(conn, [s[0] for s in seeds], hops)
        rows = graph_mod.nodes_for_ids(conn, expanded_ids)
        if not rows:
            return {}, set()

        raw = {
            r["canonical_name"]: r["confidence"] * min(1.0, len(r["canonical_name"]) / 12)
            for r in rows
        }
        top = max(raw.values()) or 1.0
        weights = {name: w / top for name, w in raw.items()}
        names = set(weights)

        scored: dict[int, float] = {}
        for name, weight in weights.items():
            for row in conn.execute(
                "SELECT id FROM chunks WHERE lower(text) LIKE ? LIMIT 50",
                (f"%{name}%",),
            ):
                scored[row["id"]] = max(scored.get(row["id"], 0.0), weight)
        return scored, names
    except Exception:
        log.exception("elastimem: graph relevance failed")
        return {}, set()


# --------------------------------------------------------------------------- #
# episodic chunks
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ChunkScoreBreakdown:
    """Per-chunk score components behind one ``search_chunks`` hit — the
    same numbers that produce ``Hit.score``, kept separate so
    :func:`explain` can show its work instead of just a final number."""

    chunk_id: int
    fts: float
    vector: float
    fused_relevance: float
    importance_nudge: float
    recency_nudge: float
    graph_nudge: float
    graph_matched_names: tuple[str, ...]
    total: float


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
    Thin wrapper over :func:`_search_chunks_scored` — see its docstring for
    the fusion/nudge model. This function only exists to keep the public
    ``Hit`` shape unchanged for existing callers; :func:`explain` uses the
    scored variant directly to show its work.
    """
    return [
        breakdown_hit[1]
        for breakdown_hit in _search_chunks_scored(
            store, query, k=k, exclude_session=exclude_session
        )
    ]


def _search_chunks_scored(
    store: "Elastimem",
    query: str,
    k: int = 8,
    exclude_session: int | None = None,
) -> list[tuple[ChunkScoreBreakdown, "Hit"]]:
    """Same result as ``search_chunks``, paired with the score breakdown
    that produced each hit. Hybrid search over episodic chunks. FTS5 +
    vectors when available.

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

    # Graph leg: a chunk mentioning an entity reachable from the query is
    # corroborating evidence, not independent relevance evidence of the
    # same strength as FTS/vector — so it's an additive nudge like
    # importance/recency below, not a fourth `max()` leg. The nudge ceiling
    # is a fraction of THIS query's own top fused relevance (not a fixed
    # constant), so a graph match can break a near-tie but can never
    # manufacture a top result out of an otherwise weak query.
    graph_scores, graph_names = graph_relevance(store, query, hops=store.profile.graph_hops)
    graph_cap = 0.15 * max(fused.values(), default=0.0)

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

    scored: list[tuple[ChunkScoreBreakdown, Hit]] = []
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
        chunk_graph_score = graph_scores.get(row["id"], 0.0)
        graph_nudge = graph_cap * chunk_graph_score
        total = relevance + importance_nudge + recency_nudge + graph_nudge
        matched_names = tuple(
            n for n in graph_names if n in row["text"].lower()
        ) if chunk_graph_score > 0 else ()
        breakdown = ChunkScoreBreakdown(
            chunk_id=row["id"],
            fts=fts_scores.get(row["id"], 0.0) / _rrf(0),
            vector=vec_scores.get(row["id"], 0.0),
            fused_relevance=relevance,
            importance_nudge=importance_nudge,
            recency_nudge=recency_nudge,
            graph_nudge=graph_nudge,
            graph_matched_names=matched_names,
            total=total,
        )
        scored.append((
            breakdown,
            Hit(
                kind="chunk",
                text=row["text"],
                date=row["created_at"][:10],
                score=total,
                session_id=row["session_id"],
            ),
        ))
    scored.sort(key=lambda pair: pair[1].score, reverse=True)
    return scored[:k]


def search_all(store: "Elastimem", query: str, k: int = 5) -> list[Hit]:
    """Chunks + facts combined — backs ``Elastimem.recall()`` and search tools."""
    hits = search_chunks(store, query, k=k)
    conn = store._conn
    fact_scores = fact_relevance(conn, query, fts=store.fts_enabled)
    top_fact_relevance = max(fact_scores.values(), default=0.0)
    fact_graph_cap = 0.15 * top_fact_relevance
    _, graph_names = graph_relevance(store, query, hops=store.profile.graph_hops)
    for fact_id, rel in fact_scores.items():
        row = conn.execute(
            "SELECT key, value, importance, valid_from FROM facts WHERE id=?",
            (fact_id,),
        ).fetchone()
        if row is None:
            continue
        recency = math.exp(
            -_days_since(row["valid_from"]) / store.config.fact_recency_half_life_days
        )
        # Additive on top of the (pre-existing, unchanged) multiplicative
        # rel*importance*recency fact score — deliberately not folded into
        # the multiplication, consistent with the "nudge not multiplier"
        # convention for this new signal specifically.
        graph_nudge = fact_graph_cap if any(
            n in row["value"].lower() for n in graph_names
        ) else 0.0
        hits.append(
            Hit(kind="fact", text=f"{row['key']}: {row['value']}",
                date=row["valid_from"][:10],
                score=rel * row["importance"] * recency + graph_nudge)
        )
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:k]


# --------------------------------------------------------------------------- #
# explain — retrieval transparency
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FactScoreBreakdown:
    """Per-fact score components, the fact-side analogue of
    :class:`ChunkScoreBreakdown`."""

    fact_id: int
    relevance: float
    importance: float
    recency_nudge: float
    graph_nudge: float
    graph_matched_names: tuple[str, ...]
    total: float


@dataclass(frozen=True)
class GraphTraversalStep:
    """One node reached during query-time graph expansion."""

    canonical_name: str
    hop_distance: int
    is_seed: bool


@dataclass(frozen=True)
class ExplainResult:
    """Full retrieval breakdown for one query — what :func:`explain`
    returns. Mirrors ``search_all``'s hits (chunks + facts, same ranking)
    but keeps every signal that produced each score, plus the graph
    traversal that ran underneath it. Never raises to the caller; a
    failure in any one leg degrades that leg to empty, same as the rest of
    this module.
    """

    query: str
    graph_hops: int
    chunk_breakdowns: tuple[ChunkScoreBreakdown, ...]
    fact_breakdowns: tuple[FactScoreBreakdown, ...]
    graph_traversal: tuple[GraphTraversalStep, ...]
    hits: tuple["Hit", ...]   # chunks + facts, ranked — same ordering as search_all


def explain(store: "Elastimem", query: str, k: int = 5) -> ExplainResult:
    """Retrieval transparency: run the same search ``recall()`` runs, but
    keep every per-leg score and the graph traversal path instead of
    collapsing them into a final number. Intended for debugging retrieval
    quality and building user-facing "why was this retrieved" views — not
    on the hot path of every turn, so it recomputes rather than caching
    anything ``recall()`` already computed. Never raises.
    """
    try:
        conn = store._conn
        hops = store.profile.graph_hops

        chunk_scored = _search_chunks_scored(store, query, k=k)
        chunk_breakdowns = tuple(breakdown for breakdown, _ in chunk_scored)

        fact_scores = fact_relevance(conn, query, fts=store.fts_enabled)
        top_fact_relevance = max(fact_scores.values(), default=0.0)
        fact_graph_cap = 0.15 * top_fact_relevance
        _, graph_names = graph_relevance(store, query, hops=hops)

        fact_breakdowns: list[FactScoreBreakdown] = []
        fact_hits: list[Hit] = []
        for fact_id, rel in fact_scores.items():
            row = conn.execute(
                "SELECT key, value, importance, valid_from FROM facts WHERE id=?",
                (fact_id,),
            ).fetchone()
            if row is None:
                continue
            recency = math.exp(
                -_days_since(row["valid_from"]) / store.config.fact_recency_half_life_days
            )
            matched_names = tuple(
                n for n in graph_names if n in row["value"].lower()
            )
            graph_nudge = fact_graph_cap if matched_names else 0.0
            total = rel * row["importance"] * recency + graph_nudge
            fact_breakdowns.append(FactScoreBreakdown(
                fact_id=fact_id, relevance=rel, importance=row["importance"],
                recency_nudge=recency, graph_nudge=graph_nudge,
                graph_matched_names=matched_names, total=total,
            ))
            fact_hits.append(Hit(
                kind="fact", text=f"{row['key']}: {row['value']}",
                date=row["valid_from"][:10], score=total,
            ))

        traversal: tuple[GraphTraversalStep, ...] = ()
        if hops > 0:
            try:
                from . import graph as graph_mod

                seeds = graph_mod.detect_seed_nodes(conn, query)
                seed_ids = {s[0] for s in seeds}
                if seed_ids:
                    reached = graph_mod.expand_with_depth(conn, list(seed_ids), hops)
                    nodes = {r["id"]: r["canonical_name"] for r in graph_mod.nodes_for_ids(
                        conn, [nid for nid, _ in reached]
                    )}
                    steps = [
                        GraphTraversalStep(
                            canonical_name=nodes[nid], hop_distance=depth,
                            is_seed=nid in seed_ids,
                        )
                        for nid, depth in reached if nid in nodes
                    ]
                    steps.sort(key=lambda s: (s.hop_distance, s.canonical_name))
                    traversal = tuple(steps)
            except Exception:
                log.exception("elastimem: explain() graph traversal failed")

        chunk_hits = [hit for _, hit in chunk_scored]
        all_hits = sorted(chunk_hits + fact_hits, key=lambda h: h.score, reverse=True)[:k]

        return ExplainResult(
            query=query, graph_hops=hops,
            chunk_breakdowns=chunk_breakdowns,
            fact_breakdowns=tuple(fact_breakdowns),
            graph_traversal=traversal,
            hits=tuple(all_hits),
        )
    except Exception:
        log.exception("elastimem: explain() failed")
        return ExplainResult(
            query=query, graph_hops=0, chunk_breakdowns=(), fact_breakdowns=(),
            graph_traversal=(), hits=(),
        )


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
