"""Search: turn a query into the signals the ranking equation needs.

The query is analysed once (words, query vector, graph reach) and the
result is shared by every prompt section and by ``search()``.

Signals, each scaled so the best candidate scores 1.0:

* keyword - FTS5 BM25 score over the query's distinctive words. Words that
  appear in more than half of a table's rows say nothing about which row is
  relevant, so they are dropped (data-driven, no stopword list). Without
  FTS5, ``LIKE`` matching counts matched words instead.
* vector - similarity between the query vector and stored vectors, unscaled
  (already 0-1 and comparable between moments and facts).
* graph - the item mentions an entity reachable from one named in the
  query; closer and more confident entities count more.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .. import tiers
from ..storage import graph
from ..storage.db import days_since
from . import scoring

if TYPE_CHECKING:
    from ..store import Elastimem

log = logging.getLogger("elastimem")


@dataclass(frozen=True)
class Hit:
    """One search result."""

    kind: str               # 'moment' | 'fact'
    text: str
    date: str               # YYYY-MM-DD
    score: float
    session: int | None = None


@dataclass
class Query:
    text: str
    words: list[str]
    vector: object = None
    reach: dict[int, int] = field(default_factory=dict)   # entity id -> hops from the query
    missing: dict[str, str] = field(default_factory=dict)  # term -> why it cannot contribute


def analyse(store: "Elastimem", text: str) -> Query:
    status = store.status
    words = list(dict.fromkeys(re.findall(r"\w{2,}", text.casefold())))
    q = Query(text, words)
    if not words:
        q.missing["keyword"] = "no searchable words"

    reason = store._vectors.unavailable(status.spec.load_embedder)
    if reason is None:
        q.vector = store._vectors.query(text)
        reason = None if q.vector is not None else store._vectors.failed
    if reason:
        q.missing["vector"] = reason

    if status.graph_hops:
        try:
            q.reach = graph.expand(store._conn, graph.seeds(store._conn, text), status.graph_hops)
        except sqlite3.Error:
            log.exception("elastimem: graph lookup failed")
    if not q.reach:
        q.missing["graph"] = "no known entities in the query"
    return q


# --- signals -----------------------------------------------------------------
def _distinctive(conn: sqlite3.Connection, table: str, words: list[str], rows: int) -> list[str]:
    """Drop words found in most rows; if all are common, keep the rarest."""
    if rows < tiers.COMMON_TERM_MIN_ROWS:
        return words
    counts = {
        w: conn.execute(f"SELECT count(*) FROM {table}_fts WHERE {table}_fts MATCH ?",
                        (f'"{w}"',)).fetchone()[0]
        for w in words
    }
    kept = [w for w in words if counts[w] / rows <= tiers.COMMON_TERM_SHARE]
    return kept or [min(words, key=counts.get)]


def _keyword(store: "Elastimem", table: str, words: list[str], where: str) -> dict[int, float]:
    if not words:
        return {}
    conn = store._conn
    try:
        if store._fts:
            rows = conn.execute(f"SELECT count(*) FROM {table} t WHERE {where}").fetchone()[0]
            match = " OR ".join(f'"{w}"' for w in _distinctive(conn, table, words, rows))
            # bm25() is negative with more negative = better; negated, the best is largest.
            found = conn.execute(
                f"SELECT t.id, -bm25({table}_fts) AS s FROM {table}_fts"
                f" JOIN {table} t ON t.id = {table}_fts.rowid"
                f" WHERE {table}_fts MATCH ? AND {where} ORDER BY s DESC LIMIT ?",
                (match, tiers.CANDIDATES),
            ).fetchall()
            return scoring.scaled({r["id"]: r["s"] for r in found})
        column = "text" if table == "chunks" else "key || ' ' || value"
        score = " + ".join([f"(lower({column}) LIKE ?)"] * len(words[:8]))
        found = conn.execute(
            f"SELECT t.id, ({score}) AS s FROM {table} t WHERE {where} AND ({score}) > 0"
            f" ORDER BY s DESC LIMIT ?",
            [f"%{w}%" for w in words[:8]] * 2 + [tiers.CANDIDATES],
        ).fetchall()
        return scoring.scaled({r["id"]: float(r["s"]) for r in found})
    except sqlite3.Error:
        log.exception("elastimem: keyword search failed")
        return {}


def _entity_weights(store: "Elastimem", q: Query) -> dict[int, float]:
    return {
        r["id"]: r["confidence"] * 0.5 ** q.reach[r["id"]]
        for r in graph.entities(store._conn, list(q.reach))
    }


def _signals(store: "Elastimem", q: Query, table: str, where: str, graph_scores) -> dict:
    signals = {"keyword": _keyword(store, table, q.words, where)}
    if q.vector is not None:
        # Closed fact versions are few; chunks of the current session are
        # dropped by moments() instead, since a long session has many.
        exclude = {r["id"] for r in store._conn.execute(
            "SELECT id FROM facts WHERE NOT (until IS NULL AND archived=0) AND embedding IS NOT NULL"
        )} if table == "facts" else frozenset()
        hits = store._vectors.similar(store._conn, table, q.vector, tiers.CANDIDATES, exclude)
        # Unscaled: similarity of unit vectors is already a 0-1 score that
        # compares across tables and queries. Scaling by the best match made
        # the only stored fact score 1.0 however unrelated it was.
        signals["vector"] = dict(hits)
    if q.reach:
        signals["graph"] = scoring.scaled(graph_scores)
    return signals


# --- ranked results -------------------------------------------------------------
def moments(
    store: "Elastimem", q: Query, k: int, exclude_session: int | None = None
) -> list[tuple[scoring.Scored, sqlite3.Row]]:
    """Past exchanges, best first."""
    conn = store._conn
    where = "1=1" if exclude_session is None else f"t.session_id != {int(exclude_session)}"
    graph_scores: dict[int, float] = {}
    if q.reach:
        weights = _entity_weights(store, q)
        for m in graph.chunk_mentions(conn, list(weights)):
            graph_scores[m["chunk_id"]] = max(graph_scores.get(m["chunk_id"], 0.0), weights[m["entity_id"]])
    signals = _signals(store, q, "chunks", where, graph_scores)
    ids = set().union(*signals.values())
    rows = {r["id"]: r for r in conn.execute(
        f"SELECT id, session_id, text, created_at, importance FROM chunks t"
        f" WHERE id IN ({','.join('?' * len(ids))}) AND {where}", list(ids),
    )} if ids else {}
    signals = {t: {i: v for i, v in sig.items() if i in rows} for t, sig in signals.items()}
    ranked = scoring.rank(
        signals, store.status.spec.weights,
        {i: r["importance"] for i, r in rows.items()},
        {i: days_since(r["created_at"]) for i, r in rows.items()},
        store._config.half_life_days,
    )
    out, seen = [], set()
    for s in ranked:
        text = rows[s.id]["text"]
        if text not in seen:            # an exchange repeated word for word adds nothing
            seen.add(text)
            out.append((s, rows[s.id]))
            if len(out) == k:
                break
    return out


def facts(store: "Elastimem", q: Query) -> list[tuple[scoring.Scored, sqlite3.Row]]:
    """Current facts that match the query, best first."""
    conn = store._conn
    rows = {r["id"]: r for r in conn.execute(
        "SELECT id, key, value, source, importance, since FROM facts WHERE until IS NULL AND archived=0"
    )}
    graph_scores: dict[int, float] = {}
    if q.reach:
        weights = _entity_weights(store, q)
        names = {r["id"]: f" {r['name']} " for r in graph.entities(conn, list(weights))}
        for fact_id, r in rows.items():
            text = graph.padded_words(f"{r['key'].replace('_', ' ')} {r['value']}")
            graph_scores[fact_id] = max((w for e, w in weights.items() if names[e] in text), default=0.0)
    ranked = scoring.rank(
        _signals(store, q, "facts", "t.until IS NULL AND t.archived=0", graph_scores),
        store.status.spec.weights,
        {i: r["importance"] for i, r in rows.items()},
        {i: days_since(r["since"]) for i, r in rows.items()},
        store._config.half_life_days,
    )
    return [(s, rows[s.id]) for s in ranked if s.id in rows]


def search(store: "Elastimem", text: str, k: int) -> list[Hit]:
    """Moments and facts together, best first."""
    q = analyse(store, text)
    hits = [
        Hit("moment", r["text"], r["created_at"][:10], s.score, r["session_id"])
        for s, r in moments(store, q, k)
    ] + [
        Hit("fact", f"{r['key']}: {r['value']}", r["since"][:10], s.score)
        for s, r in facts(store, q)
    ]
    hits.sort(key=lambda h: -h.score)
    return hits[:k]
