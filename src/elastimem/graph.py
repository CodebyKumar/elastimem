"""Graph: entities and relations extracted from conversation, plus
``mentions`` linking entities to the chunks they came from.

Entities and relations come from the same LLM call that extracts facts.
Traversal is a ``WITH RECURSIVE`` query. Repeated mentions update the
existing row (unique indexes on identity); caps bound growth from distinct
entities. Topics are connected components, optionally named by the LLM.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3

from . import tiers
from .db import utcnow

log = logging.getLogger("elastimem")

# The extraction prompt names the user "user". That node links to nearly
# every entity, so traversal never passes through it.
USER = "user"
_TYPES = frozenset({"person", "place", "org", "thing", "entity"})
_MAX_ALIASES = 8


def _name(raw: str) -> str:
    return " ".join(str(raw).casefold().split())


def padded_words(text: str) -> str:
    """Space-padded word string, for word-boundary name matching."""
    return " " + " ".join(re.findall(r"\w+", text.casefold())) + " "


# --- writing -----------------------------------------------------------------
def upsert_entity(
    conn: sqlite3.Connection, kind: str, raw: str, confidence: float = 0.5
) -> int | None:
    name = _name(raw)
    if not name:
        return None
    now = utcnow()
    row = conn.execute(
        "INSERT INTO entities(type, name, created_at, updated_at, confidence)"
        " VALUES (?,?,?,?,?)"
        " ON CONFLICT(type, name) DO UPDATE SET updated_at=excluded.updated_at,"
        "   seen=seen+1, confidence=(confidence*seen + excluded.confidence)/(seen+1)"
        " RETURNING id, aliases",
        (kind if kind in _TYPES else "entity", name, now, now, confidence),
    ).fetchone()
    raw = str(raw).strip()
    aliases = json.loads(row["aliases"] or "[]")
    if raw != name and raw not in aliases:
        conn.execute(
            "UPDATE entities SET aliases=? WHERE id=?",
            (json.dumps(([raw] + aliases)[:_MAX_ALIASES]), row["id"]),
        )
    return row["id"]


def store_extraction(
    conn: sqlite3.Connection, entities: object, relations: object, chunk_ids: list[int]
) -> int:
    """Store one LLM extraction's untrusted ``entities``/``relations`` lists.
    Malformed items are skipped. Returns the number of entities stored."""
    ids: dict[str, int] = {}
    with conn:
        for item in entities if isinstance(entities, list) else []:
            if isinstance(item, dict) and isinstance(item.get("name"), str):
                kind = item.get("type") if isinstance(item.get("type"), str) else "entity"
                entity_id = upsert_entity(conn, kind, item["name"])
                if entity_id:
                    ids[_name(item["name"])] = entity_id
        now = utcnow()
        for item in relations if isinstance(relations, list) else []:
            if not isinstance(item, dict):
                continue
            parts = [item.get(k) for k in ("source", "relation", "target")]
            if not all(isinstance(p, str) and p.strip() for p in parts):
                continue
            source, relation, target = parts
            ends = [ids.get(_name(n)) or upsert_entity(conn, "entity", n) for n in (source, target)]
            relation = "_".join(relation.casefold().split())
            if None in ends or ends[0] == ends[1]:
                continue
            conn.execute(
                "INSERT INTO relations(source, target, relation, created_at, updated_at, chunk_id)"
                " VALUES (?,?,?,?,?,?)"
                " ON CONFLICT(source, target, relation) DO UPDATE SET"
                "   updated_at=excluded.updated_at, seen=seen+1",
                (*ends, relation, now, now, chunk_ids[0] if chunk_ids else None),
            )
        conn.executemany(
            "INSERT OR IGNORE INTO mentions(entity_id, chunk_id) VALUES (?,?)",
            [(e, c) for name, e in ids.items() if name != USER for c in chunk_ids],
        )
        if ids:
            _enforce_caps(conn)
    return len(ids)


def _enforce_caps(conn: sqlite3.Connection) -> None:
    conn.execute(
        "DELETE FROM entities WHERE id NOT IN (SELECT id FROM entities"
        " ORDER BY seen DESC, updated_at DESC LIMIT ?)", (tiers.ENTITY_CAP,),
    )
    conn.execute(
        "DELETE FROM relations WHERE id NOT IN (SELECT id FROM relations"
        " ORDER BY seen DESC, updated_at DESC LIMIT ?)", (tiers.RELATION_CAP,),
    )


# --- reading -----------------------------------------------------------------
def seeds(conn: sqlite3.Connection, query: str, limit: int = 5) -> list[int]:
    """Entities named in ``query`` (name or alias, whole words), longest first."""
    words = padded_words(query)
    found = []
    for row in conn.execute("SELECT id, name, aliases FROM entities WHERE name != ?", (USER,)):
        names = [row["name"], *json.loads(row["aliases"] or "[]")]
        best = max((len(n) for n in names if padded_words(n).strip() and padded_words(n) in words), default=0)
        if best:
            found.append((best, row["id"]))
    return [entity_id for _, entity_id in sorted(found, reverse=True)[:limit]]


def expand(conn: sqlite3.Connection, seed_ids: list[int], hops: int) -> dict[int, int]:
    """``{entity_id: hop_distance}`` for everything within ``hops`` of a seed,
    seeds included. Relations count in both directions."""
    if not seed_ids:
        return {}
    marks = ",".join("?" * len(seed_ids))
    rows = conn.execute(
        f"""WITH RECURSIVE reach(id, depth) AS (
              SELECT id, 0 FROM entities WHERE id IN ({marks})
              UNION
              SELECT CASE WHEN r.source = reach.id THEN r.target ELSE r.source END, depth + 1
              FROM reach JOIN relations r ON reach.id IN (r.source, r.target)
              JOIN entities e ON e.id = reach.id
              WHERE depth < ? AND (depth = 0 OR e.name != ?)
            )
            SELECT reach.id, min(depth) AS depth FROM reach JOIN entities e ON e.id = reach.id
            WHERE e.name != ? GROUP BY reach.id""",
        (*seed_ids, hops, USER, USER),
    ).fetchall()
    return {r["id"]: r["depth"] for r in rows}


def entities(conn: sqlite3.Connection, ids: list[int]) -> list[sqlite3.Row]:
    if not ids:
        return []
    return conn.execute(
        f"SELECT id, name, confidence, topic FROM entities WHERE id IN ({','.join('?' * len(ids))})",
        ids,
    ).fetchall()


def chunk_mentions(conn: sqlite3.Connection, ids: list[int]) -> list[sqlite3.Row]:
    """``(entity_id, chunk_id)`` rows for the given entities."""
    if not ids:
        return []
    return conn.execute(
        f"SELECT entity_id, chunk_id FROM mentions WHERE entity_id IN ({','.join('?' * len(ids))})",
        ids,
    ).fetchall()


# --- maintenance ---------------------------------------------------------------
def retire_stale(conn: sqlite3.Connection, half_life_days: float) -> int:
    """Delete relations, then entities, whose decayed confidence is too low.
    The graph is a derived index, so retired rows are deleted, not archived."""
    with conn:
        removed = conn.execute(
            "DELETE FROM relations WHERE decay(confidence, updated_at, ?) < ?",
            (half_life_days, tiers.ARCHIVE_BELOW),
        ).rowcount
        removed += conn.execute(
            "DELETE FROM entities WHERE decay(confidence, updated_at, ?) < ?",
            (half_life_days, tiers.ARCHIVE_BELOW),
        ).rowcount
    return removed


def merge_candidates(conn: sqlite3.Connection, limit: int = 5) -> list[tuple[sqlite3.Row, sqlite3.Row]]:
    """Recent same-type entity pairs sharing a word: possible duplicates."""
    rows = conn.execute(
        "SELECT id, type, name FROM entities WHERE created_at >= datetime('now', ?)"
        " AND name != ? ORDER BY created_at", (f"-{tiers.MERGE_WINDOW_DAYS} days", USER),
    ).fetchall()
    pairs = []
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            if a["type"] == b["type"] and set(a["name"].split()) & set(b["name"].split()):
                pairs.append((a, b))
                if len(pairs) == limit:
                    return pairs
    return pairs


def merge(conn: sqlite3.Connection, keep_id: int, drop_id: int) -> None:
    """Fold entity ``drop_id`` into ``keep_id``."""
    with conn:
        for column in ("source", "target"):
            conn.execute(f"UPDATE OR IGNORE relations SET {column}=? WHERE {column}=?", (keep_id, drop_id))
        conn.execute("UPDATE OR IGNORE mentions SET entity_id=? WHERE entity_id=?", (keep_id, drop_id))
        conn.execute("DELETE FROM entities WHERE id=?", (drop_id,))


def recompute_topics(conn: sqlite3.Connection) -> int:
    """Topics are connected components of 2+ entities (union-find over
    relations, ignoring the user node). Returns the number of topics."""
    ids = [r["id"] for r in conn.execute("SELECT id FROM entities WHERE name != ?", (USER,))]
    parent = {i: i for i in ids}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for r in conn.execute("SELECT source, target FROM relations"):
        if r["source"] in parent and r["target"] in parent:
            parent[find(r["source"])] = find(r["target"])
    groups: dict[int, list[int]] = {}
    for i in ids:
        groups.setdefault(find(i), []).append(i)
    topics = {root: members for root, members in groups.items() if len(members) > 1}
    with conn:
        # A topic keeps its id (its root entity), and so its name, while it lasts.
        conn.execute(
            "UPDATE entities SET topic_id=NULL, topic=NULL WHERE topic_id IS NOT NULL"
            f" AND topic_id NOT IN ({','.join('?' * len(topics))})", list(topics),
        )
        for root, members in topics.items():
            conn.execute(
                "UPDATE entities SET topic_id=?,"
                " topic=(SELECT max(topic) FROM entities WHERE topic_id=?)"
                f" WHERE id IN ({','.join('?' * len(members))})",
                (root, root, *members),
            )
    return len(topics)


def unnamed_topics(conn: sqlite3.Connection) -> dict[int, list[str]]:
    rows = conn.execute(
        "SELECT topic_id, name FROM entities WHERE topic_id IS NOT NULL AND topic IS NULL"
    ).fetchall()
    out: dict[int, list[str]] = {}
    for r in rows:
        out.setdefault(r["topic_id"], []).append(r["name"])
    return out


def name_topic(conn: sqlite3.Connection, topic_id: int, label: str) -> None:
    with conn:
        conn.execute("UPDATE entities SET topic=? WHERE topic_id=?", (label, topic_id))


def list_topics(conn: sqlite3.Connection) -> list[dict]:
    """``[{"id", "name", "members"}]``, largest first."""
    out: dict[int, dict] = {}
    for r in conn.execute(
        "SELECT topic_id, topic, name FROM entities WHERE topic_id IS NOT NULL ORDER BY name"
    ):
        entry = out.setdefault(r["topic_id"], {"id": r["topic_id"], "name": r["topic"], "members": []})
        entry["members"].append(r["name"])
    return sorted(out.values(), key=lambda t: -len(t["members"]))
