"""Embedded semantic knowledge graph: entities and relationships extracted
from conversation, stored as plain tables in the same SQLite file.

The graph is not a separate product — it is one more retrieval signal
(see :mod:`elastimem.retrieval`), governed by the same Memory Governor tier
gating as everything else (:class:`elastimem.config.MemoryProfile.graph_hops`).
No NER library, no graph library: entities/relationships come from the same
LLM completion :mod:`elastimem.extraction` already makes for facts, and
multi-hop traversal is a plain SQLite ``WITH RECURSIVE`` query.

Write-time dedup (unique indexes on node identity and edge identity) means
repeated mentions of the same entity/relationship update an existing row
instead of creating a new one. On top of that, :func:`_enforce_caps` bounds
growth from genuinely distinct entities/relationships, mirroring
``guards.quarantine``'s trim-on-write pattern.

Nothing in this module raises to the caller; failures degrade to "no graph
signal" (empty results / stored-seeds-only), matching the rest of the
extraction and retrieval pipeline.
"""

from __future__ import annotations

import json
import logging
import sqlite3

from .db import utcnow

log = logging.getLogger("elastimem")

_ARTICLES = ("the ", "a ", "an ")
_MAX_ALIASES = 8
_VALID_ENTITY_TYPES = frozenset({"person", "place", "org", "thing", "entity"})


def _canonicalize(name: str) -> str:
    """Lowercase, collapse whitespace, strip a leading article. No fuzzy
    matching — this is deliberately simple, matching the project's
    no-new-dependency stance on entity resolution."""
    normalized = " ".join(name.strip().lower().split())
    for article in _ARTICLES:
        if normalized.startswith(article):
            normalized = normalized[len(article):]
            break
    return normalized


def upsert_node(
    conn: sqlite3.Connection, node_type: str, name: str, *, confidence: float = 0.5
) -> int | None:
    """Insert or update one entity by canonical identity. Returns the node
    id, or None if ``name`` canonicalizes to nothing (empty/whitespace)."""
    raw = str(name).strip()
    canonical = _canonicalize(raw)
    if not canonical:
        return None
    node_type = node_type if node_type in _VALID_ENTITY_TYPES else "entity"
    now = utcnow()
    row = conn.execute(
        "INSERT INTO graph_nodes(type, canonical_name, aliases, created_at,"
        " updated_at, importance, confidence, mention_count)"
        " VALUES (?, ?, '[]', ?, ?, 0.5, ?, 1)"
        " ON CONFLICT(type, canonical_name) DO UPDATE SET"
        "   updated_at = excluded.updated_at,"
        "   mention_count = mention_count + 1,"
        "   confidence = (confidence * mention_count + excluded.confidence)"
        "                / (mention_count + 1)"
        " RETURNING id, aliases, mention_count",
        (node_type, canonical, now, now, confidence),
    ).fetchone()
    node_id = row["id"]

    if raw != canonical:
        try:
            aliases = json.loads(row["aliases"]) or []
        except (TypeError, ValueError):
            aliases = []
        if raw not in aliases:
            aliases = ([raw] + aliases)[:_MAX_ALIASES]
            conn.execute(
                "UPDATE graph_nodes SET aliases=? WHERE id=?",
                (json.dumps(aliases), node_id),
            )
    return node_id


def upsert_edge(
    conn: sqlite3.Connection,
    source_id: int,
    target_id: int,
    relationship: str,
    *,
    confidence: float = 0.5,
    source_chunk_id: int | None = None,
) -> int | None:
    """Insert or update one directed relationship. Returns the edge id, or
    None if ``relationship`` is empty or the two endpoints are identical
    (a self-loop carries no traversal value here)."""
    relation = " ".join(str(relationship).strip().lower().split())
    if not relation or source_id == target_id:
        return None
    now = utcnow()
    row = conn.execute(
        "INSERT INTO graph_edges(source_node, target_node, relationship,"
        " confidence, importance, weight, created_at, last_seen, seen_count,"
        " source_chunk_id)"
        " VALUES (?, ?, ?, ?, 0.5, 1.0, ?, ?, 1, ?)"
        " ON CONFLICT(source_node, target_node, relationship) DO UPDATE SET"
        "   last_seen = excluded.last_seen,"
        "   seen_count = seen_count + 1,"
        "   confidence = (confidence * seen_count + excluded.confidence)"
        "                / (seen_count + 1)"
        " RETURNING id",
        (source_id, target_id, relation, confidence, now, now, source_chunk_id),
    ).fetchone()
    return row["id"]


def _enforce_caps(conn: sqlite3.Connection, node_cap: int, edge_cap: int) -> None:
    """Trim graph_nodes/graph_edges to their configured caps, evicting the
    least important / least recently seen rows first. ON DELETE CASCADE on
    graph_edges' foreign keys cleans up edges orphaned by a trimmed node."""
    conn.execute(
        "DELETE FROM graph_nodes WHERE id NOT IN ("
        " SELECT id FROM graph_nodes"
        " ORDER BY importance DESC, mention_count DESC, updated_at DESC LIMIT ?)",
        (node_cap,),
    )
    conn.execute(
        "DELETE FROM graph_edges WHERE id NOT IN ("
        " SELECT id FROM graph_edges"
        " ORDER BY importance DESC, seen_count DESC, last_seen DESC LIMIT ?)",
        (edge_cap,),
    )


def store_extraction(
    conn: sqlite3.Connection,
    entities: object,
    relationships: object,
    *,
    node_cap: int = 2000,
    edge_cap: int = 5000,
    source_chunk_id: int | None = None,
) -> dict[str, int]:
    """Single defensive entry point for one LLM extraction's graph payload.

    ``entities``/``relationships`` are whatever the model returned under
    those JSON keys — untyped and untrusted. Malformed entries are skipped
    individually rather than aborting the whole batch; this function never
    raises. Returns ``{"nodes": n, "edges": m}`` counts actually stored.
    """
    with conn:
        node_ids: dict[str, int] = {}
        node_count = 0
        if isinstance(entities, list):
            for item in entities:
                if not isinstance(item, dict):
                    continue
                name = item.get("name")
                if not isinstance(name, str) or not name.strip():
                    continue
                etype = item.get("type")
                etype = etype if isinstance(etype, str) else "entity"
                node_id = upsert_node(conn, etype, name)
                if node_id is not None:
                    node_ids[_canonicalize(name)] = node_id
                    node_count += 1

        edge_count = 0
        if isinstance(relationships, list):
            for item in relationships:
                if not isinstance(item, dict):
                    continue
                source = item.get("source")
                relation = item.get("relation")
                target = item.get("target")
                if not all(isinstance(v, str) and v.strip() for v in (source, relation, target)):
                    continue
                # Endpoints may reference an entity from `entities` above or
                # be a bare mention not otherwise extracted (e.g. "user") —
                # upsert either way so the edge always has valid endpoints.
                source_id = node_ids.get(_canonicalize(source)) or upsert_node(
                    conn, "entity", source
                )
                target_id = node_ids.get(_canonicalize(target)) or upsert_node(
                    conn, "entity", target
                )
                if source_id is None or target_id is None:
                    continue
                edge_id = upsert_edge(
                    conn, source_id, target_id, relation,
                    source_chunk_id=source_chunk_id,
                )
                if edge_id is not None:
                    edge_count += 1

        if node_count or edge_count:
            _enforce_caps(conn, node_cap, edge_cap)

    return {"nodes": node_count, "edges": edge_count}


def detect_seed_nodes(conn: sqlite3.Connection, query: str, limit: int = 5) -> list[tuple[int, int]]:
    """[(node_id, match_length), ...] for nodes whose canonical_name or an
    alias appears in ``query`` (case-insensitive substring match). No NER —
    deliberately simple, an O(nodes) Python scan. Longer/more specific
    matches are ranked first. Never raises."""
    terms = query.lower()
    try:
        rows = conn.execute("SELECT id, canonical_name, aliases FROM graph_nodes").fetchall()
    except sqlite3.Error:
        log.exception("elastimem: seed node scan failed")
        return []
    matched: list[tuple[int, int]] = []
    for row in rows:
        names = [row["canonical_name"]]
        try:
            names.extend(json.loads(row["aliases"]) or [])
        except (TypeError, ValueError):
            pass
        best = 0
        for name in names:
            if name and name.lower() in terms:
                best = max(best, len(name))
        if best:
            matched.append((row["id"], best))
    matched.sort(key=lambda pair: -pair[1])
    return matched[:limit]


def expand(conn: sqlite3.Connection, seed_ids: list[int], hops: int) -> list[int]:
    """Node ids reachable from any seed within ``hops`` edges (traversal is
    undirected for retrieval purposes, even though edges are stored
    directed — a query matching the object of a relationship must still
    expand to the subject and vice versa). Includes the seeds themselves.
    Degrades to the seed list on any SQL error, never raises."""
    if not seed_ids:
        return []
    if hops <= 0:
        return list(dict.fromkeys(seed_ids))
    placeholders = ",".join("?" * len(seed_ids))
    try:
        rows = conn.execute(
            f"""
            WITH RECURSIVE reachable(id, depth) AS (
                SELECT id, 0 FROM graph_nodes WHERE id IN ({placeholders})
                UNION
                SELECT e.target_node, r.depth + 1
                FROM graph_edges e JOIN reachable r ON e.source_node = r.id
                WHERE r.depth < ?
                UNION
                SELECT e.source_node, r.depth + 1
                FROM graph_edges e JOIN reachable r ON e.target_node = r.id
                WHERE r.depth < ?
            )
            SELECT DISTINCT id FROM reachable
            """,
            [*seed_ids, hops, hops],
        ).fetchall()
        # Union with the input seeds themselves: a seed id that no longer
        # has a row in graph_nodes (e.g. evicted by cap enforcement between
        # detection and expansion) must still come back rather than
        # silently vanishing — "includes the seeds" is an unconditional
        # guarantee, not contingent on the row still existing.
        return list(dict.fromkeys([*seed_ids, *(row["id"] for row in rows)]))
    except sqlite3.Error:
        log.exception("elastimem: graph expansion query failed")
        return list(dict.fromkeys(seed_ids))


def expand_with_depth(
    conn: sqlite3.Connection, seed_ids: list[int], hops: int
) -> list[tuple[int, int]]:
    """Like :func:`expand`, but returns ``(node_id, min_hop_distance)``
    pairs instead of a bare id list — used by ``retrieval.explain()`` to
    show the traversal path a query took, not just its end result.
    Degrades to seeds-at-depth-0 on any SQL error, never raises."""
    if not seed_ids:
        return []
    if hops <= 0:
        return [(sid, 0) for sid in dict.fromkeys(seed_ids)]
    placeholders = ",".join("?" * len(seed_ids))
    try:
        rows = conn.execute(
            f"""
            WITH RECURSIVE reachable(id, depth) AS (
                SELECT id, 0 FROM graph_nodes WHERE id IN ({placeholders})
                UNION
                SELECT e.target_node, r.depth + 1
                FROM graph_edges e JOIN reachable r ON e.source_node = r.id
                WHERE r.depth < ?
                UNION
                SELECT e.source_node, r.depth + 1
                FROM graph_edges e JOIN reachable r ON e.target_node = r.id
                WHERE r.depth < ?
            )
            SELECT id, MIN(depth) AS depth FROM reachable GROUP BY id
            """,
            [*seed_ids, hops, hops],
        ).fetchall()
        found = {row["id"]: row["depth"] for row in rows}
        for sid in seed_ids:
            found.setdefault(sid, 0)
        return list(found.items())
    except sqlite3.Error:
        log.exception("elastimem: graph expansion-with-depth query failed")
        return [(sid, 0) for sid in dict.fromkeys(seed_ids)]


def nodes_for_ids(conn: sqlite3.Connection, node_ids: list[int]) -> list[sqlite3.Row]:
    """canonical_name/confidence rows for a set of node ids. Never raises."""
    if not node_ids:
        return []
    placeholders = ",".join("?" * len(node_ids))
    try:
        return conn.execute(
            f"SELECT id, canonical_name, confidence FROM graph_nodes"
            f" WHERE id IN ({placeholders})",
            node_ids,
        ).fetchall()
    except sqlite3.Error:
        log.exception("elastimem: graph node lookup failed")
        return []
