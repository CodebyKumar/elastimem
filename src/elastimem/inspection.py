"""Looking inside memory: why a result ranked where it did, and how a fact
changed over time."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from . import facts, search
from .config import Tier, normalize_key
from .search import Hit
from .tiers import Weights

if TYPE_CHECKING:
    from .store import Elastimem


@dataclass(frozen=True)
class Explained:
    hit: Hit
    parts: dict[str, float]     # weighted contribution of each term; they sum to hit.score


@dataclass(frozen=True)
class Explanation:
    query: str
    tier: Tier
    weights: Weights            # the tier's equation, before missing terms are reweighted
    missing: dict[str, str]     # term -> why it did not contribute
    items: list[Explained]


def explain(store: "Elastimem", query: str, k: int = 5) -> Explanation:
    q = search.analyse(store, query)
    items = [
        Explained(Hit("moment", r["text"], r["created_at"][:10], s.score, r["session_id"]), s.parts)
        for s, r in search.moments(store, q, k)
    ] + [
        Explained(Hit("fact", f"{r['key']}: {r['value']}", r["since"][:10], s.score), s.parts)
        for s, r in search.facts(store, q)
    ]
    items.sort(key=lambda e: -e.hit.score)
    status = store.status
    return Explanation(query, status.tier, status.spec.weights, dict(q.missing), items[:k])


def history(store: "Elastimem", key_or_question: str) -> list[facts.Fact]:
    """Every version of a fact, oldest first. Takes a key, or a question
    that is matched against current facts."""
    versions = facts.history(store._conn, normalize_key(key_or_question))
    if versions:
        return versions
    ranked = search.facts(store, search.analyse(store, key_or_question))
    return facts.history(store._conn, ranked[0][1]["key"]) if ranked else []
