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
import math
import sqlite3
from datetime import datetime, timezone
from typing import Callable

from .db import utcnow

log = logging.getLogger("elastimem")

CompleteFn = Callable[..., str]

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


# --------------------------------------------------------------------------- #
# maintenance: decay/archival, duplicate merging
# --------------------------------------------------------------------------- #
def _days_since(iso_ts: str) -> float:
    try:
        then = datetime.fromisoformat(iso_ts)
        if then.tzinfo is None:
            then = then.replace(tzinfo=timezone.utc)
    except ValueError:
        return 0.0
    return max(0.0, (datetime.now(timezone.utc) - then).total_seconds() / 86400.0)


def _decayed(confidence: float, anchor_ts: str, half_life_days: float) -> float:
    """confidence * exp(-days_since_anchor / half_life) — same shape as
    semantic.effective_importance, applied to graph confidence instead of
    fact importance."""
    return confidence * math.exp(-_days_since(anchor_ts) / half_life_days)


def apply_decay(
    conn: sqlite3.Connection, *, half_life_days: float, archive_threshold: float
) -> dict[str, int]:
    """Hard-delete nodes/edges whose decayed confidence has fallen below
    ``archive_threshold``. Unlike facts (which soft-archive to preserve an
    audit trail), the graph is a derived retrieval index with no audit
    requirement, so a decayed row is simply removed — consistent with how
    cap enforcement already hard-deletes on overflow. Reinforcement
    (re-extraction bumping mention_count/seen_count and refreshing
    updated_at/last_seen) resets the decay clock, so an entity that keeps
    coming up in conversation never decays away.

    Edges are checked first: a node with no edges left after edge decay is
    a good deletion candidate on its own decay pass (ON DELETE CASCADE
    handles the reverse — deleting a decayed node removes its edges).
    Returns ``{"nodes": n, "edges": m}`` counts removed. Never raises.
    """
    removed = {"nodes": 0, "edges": 0}
    try:
        with conn:
            edge_rows = conn.execute(
                "SELECT id, confidence, last_seen FROM graph_edges"
            ).fetchall()
            stale_edges = [
                row["id"] for row in edge_rows
                if _decayed(row["confidence"], row["last_seen"], half_life_days)
                < archive_threshold
            ]
            if stale_edges:
                placeholders = ",".join("?" * len(stale_edges))
                conn.execute(
                    f"DELETE FROM graph_edges WHERE id IN ({placeholders})",
                    stale_edges,
                )
                removed["edges"] = len(stale_edges)

            node_rows = conn.execute(
                "SELECT id, confidence, updated_at FROM graph_nodes"
            ).fetchall()
            stale_nodes = [
                row["id"] for row in node_rows
                if _decayed(row["confidence"], row["updated_at"], half_life_days)
                < archive_threshold
            ]
            if stale_nodes:
                placeholders = ",".join("?" * len(stale_nodes))
                conn.execute(
                    f"DELETE FROM graph_nodes WHERE id IN ({placeholders})",
                    stale_nodes,
                )
                removed["nodes"] = len(stale_nodes)
    except sqlite3.Error:
        log.exception("elastimem: graph decay sweep failed")
    return removed


_MERGE_PROMPT = (
    "Are these two labels the same real-world entity, just written "
    "differently (e.g. abbreviation, typo, or rephrasing)? Reply with "
    "ONLY \"yes\" or \"no\"."
)


def merge_duplicates(
    conn: sqlite3.Connection,
    complete_fn: CompleteFn,
    *,
    review_window_days: float,
    max_tokens: int = 16,
) -> int:
    """FULL-tier-only maintenance: ask the LLM whether pairs of same-type
    entities created within ``review_window_days`` of each other and
    sharing a token (cheap pre-filter — no fuzzy-matching library, see
    graph._canonicalize's docstring) are actually the same entity under
    different names. On "yes", the newer node's edges are repointed to the
    older node and the newer node is deleted.

    This is the one piece of graph maintenance that costs an LLM call, so
    it is capped at a handful of pairs per sweep (mirrors
    extraction.consolidate's fact-merge review, which caps at 5). Never
    raises; a malformed/absent response is treated as "no merge".
    """
    merged = 0
    try:
        candidates = conn.execute(
            "SELECT id, type, canonical_name, created_at FROM graph_nodes"
            " WHERE created_at >= datetime('now', ?) ORDER BY created_at",
            (f"-{review_window_days} days",),
        ).fetchall()
    except sqlite3.Error:
        log.exception("elastimem: graph duplicate-merge candidate scan failed")
        return 0

    checked = 0
    for i, a in enumerate(candidates):
        if checked >= 5:
            break
        a_tokens = set(a["canonical_name"].split())
        for b in candidates[i + 1:]:
            if checked >= 5:
                break
            if b["type"] != a["type"]:
                continue
            b_tokens = set(b["canonical_name"].split())
            if not (a_tokens & b_tokens):
                continue
            checked += 1
            try:
                answer = complete_fn(
                    f"{_MERGE_PROMPT}\n\nA: {a['canonical_name']}\nB: {b['canonical_name']}",
                    max_tokens=max_tokens, temperature=0.0,
                ) or ""
            except Exception:
                log.exception("elastimem: graph duplicate-merge completion failed")
                continue
            if not answer.strip().lower().startswith("y"):
                continue
            try:
                with conn:
                    conn.execute(
                        "UPDATE OR IGNORE graph_edges SET source_node=? WHERE source_node=?",
                        (a["id"], b["id"]),
                    )
                    conn.execute(
                        "UPDATE OR IGNORE graph_edges SET target_node=? WHERE target_node=?",
                        (a["id"], b["id"]),
                    )
                    conn.execute("DELETE FROM graph_nodes WHERE id=?", (b["id"],))
                merged += 1
            except sqlite3.Error:
                log.exception("elastimem: graph duplicate-merge apply failed")
    return merged


# --------------------------------------------------------------------------- #
# semantic clusters: topic discovery via connected components
# --------------------------------------------------------------------------- #
_CLUSTER_LABEL_PROMPT = (
    "These are labels for related entities that came up in conversation. "
    "Reply with ONLY a short (1-4 word) topic name covering all of them, "
    "e.g. \"Local AI\" or \"Home Renovation\". No punctuation, no explanation."
)


def _find(parent: dict[int, int], x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]   # path halving
        x = parent[x]
    return x


def _union(parent: dict[int, int], a: int, b: int) -> None:
    ra, rb = _find(parent, a), _find(parent, b)
    if ra != rb:
        parent[ra] = rb


def compute_clusters(conn: sqlite3.Connection, *, min_size: int = 2) -> dict[int, list[int]]:
    """Group entities into topics via connected components over
    graph_edges — no clustering library, same union-find approach any
    graph toolkit uses internally. A cluster is "all entities reachable
    from each other," full stop; no distance/similarity threshold since
    edges themselves are the only relationship signal available.

    Returns ``{root_node_id: [member_node_id, ...]}`` for components with
    at least ``min_size`` members — singletons (a node with no edges, or
    only edges to nodes that were themselves filtered) aren't a "topic,"
    just an isolated fact, so they're excluded by default. Never raises;
    returns ``{}`` on any failure or an edgeless graph.
    """
    try:
        node_ids = [r["id"] for r in conn.execute("SELECT id FROM graph_nodes")]
        if not node_ids:
            return {}
        parent = {nid: nid for nid in node_ids}
        for row in conn.execute("SELECT source_node, target_node FROM graph_edges"):
            if row["source_node"] in parent and row["target_node"] in parent:
                _union(parent, row["source_node"], row["target_node"])

        components: dict[int, list[int]] = {}
        for nid in node_ids:
            root = _find(parent, nid)
            components.setdefault(root, []).append(nid)
        return {root: members for root, members in components.items()
                if len(members) >= min_size}
    except sqlite3.Error:
        log.exception("elastimem: cluster computation failed")
        return {}


def store_clusters(conn: sqlite3.Connection, clusters: dict[int, list[int]]) -> None:
    """Stamp ``cluster_id`` (the component's root node id — stable across
    sweeps as long as the component doesn't split/merge) onto every
    clustered node, and clear it from nodes no longer in any cluster.
    ``cluster_label`` is left untouched here — see :func:`label_clusters`,
    a separate step so unlabeled clustering never blocks on an LLM call.
    Never raises.
    """
    try:
        with conn:
            clustered_ids = {nid for members in clusters.values() for nid in members}
            conn.execute(
                "UPDATE graph_nodes SET cluster_id=NULL, cluster_label=NULL"
                " WHERE cluster_id IS NOT NULL"
                + (" AND id NOT IN ({})".format(",".join("?" * len(clustered_ids)))
                   if clustered_ids else ""),
                list(clustered_ids),
            )
            for root, members in clusters.items():
                placeholders = ",".join("?" * len(members))
                conn.execute(
                    f"UPDATE graph_nodes SET cluster_id=? WHERE id IN ({placeholders})",
                    [root, *members],
                )
    except sqlite3.Error:
        log.exception("elastimem: storing clusters failed")


def label_clusters(
    conn: sqlite3.Connection, complete_fn: CompleteFn, *, max_tokens: int = 16
) -> int:
    """FULL-tier-only: ask the LLM for a short topic name for each
    unlabeled cluster (``cluster_label IS NULL``), one completion per
    cluster. Costs an LLM call per new cluster, so — like
    :func:`merge_duplicates` — this is a separate, tier-gated step from
    :func:`store_clusters`, not folded into it. A cluster works fine as a
    retrieval grouping with no label at all; the label is presentation
    sugar for a host that wants to show "Local AI" instead of a bare list
    of entity names. Returns the number of clusters labeled. Never raises;
    a failed/malformed response leaves that cluster unlabeled rather than
    stamping a bad label.
    """
    labeled = 0
    try:
        rows = conn.execute(
            "SELECT DISTINCT cluster_id FROM graph_nodes"
            " WHERE cluster_id IS NOT NULL AND cluster_label IS NULL"
        ).fetchall()
    except sqlite3.Error:
        log.exception("elastimem: cluster label candidate scan failed")
        return 0

    for row in rows:
        cluster_id = row["cluster_id"]
        try:
            members = conn.execute(
                "SELECT canonical_name FROM graph_nodes WHERE cluster_id=? LIMIT 12",
                (cluster_id,),
            ).fetchall()
            names = ", ".join(m["canonical_name"] for m in members)
            label = complete_fn(
                f"{_CLUSTER_LABEL_PROMPT}\n\nEntities: {names}",
                max_tokens=max_tokens, temperature=0.0,
            ) or ""
            label = label.strip().strip('"').strip("'")
            if not label or len(label) > 60:
                continue
            with conn:
                conn.execute(
                    "UPDATE graph_nodes SET cluster_label=? WHERE cluster_id=?",
                    (label, cluster_id),
                )
            labeled += 1
        except Exception:
            log.exception("elastimem: cluster labeling failed for cluster %s", cluster_id)
    return labeled


def list_clusters(conn: sqlite3.Connection) -> list[dict]:
    """Read-only view of all current clusters: ``[{"cluster_id": int,
    "label": str | None, "members": [canonical_name, ...]}, ...]``, largest
    first. Never raises."""
    try:
        rows = conn.execute(
            "SELECT cluster_id, cluster_label, canonical_name FROM graph_nodes"
            " WHERE cluster_id IS NOT NULL ORDER BY cluster_id"
        ).fetchall()
    except sqlite3.Error:
        log.exception("elastimem: cluster listing failed")
        return []
    grouped: dict[int, dict] = {}
    for row in rows:
        entry = grouped.setdefault(
            row["cluster_id"],
            {"cluster_id": row["cluster_id"], "label": row["cluster_label"], "members": []},
        )
        entry["members"].append(row["canonical_name"])
        if row["cluster_label"]:
            entry["label"] = row["cluster_label"]
    return sorted(grouped.values(), key=lambda c: -len(c["members"]))
