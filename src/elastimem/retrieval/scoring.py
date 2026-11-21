"""The ranking equation, used for every kind of memory::

    relevance = vector*w_vector + keyword*w_keyword + graph*w_graph
    score     = relevance + importance*w_importance + recency*w_recency

Each signal is in [0, 1]. The weights are fixed per tier (see
:mod:`elastimem.tiers`). A relevance term that found nothing for a query
hands its weight to the others in proportion, so the scale stays the same.
Only items with relevance > 0 are ranked: importance and recency break
ties, they never make an unrelated memory appear.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..tiers import Weights

RELEVANCE = ("vector", "keyword", "graph")


@dataclass(frozen=True)
class Scored:
    id: int
    score: float
    parts: dict[str, float]     # each term's weighted contribution; they sum to score


def effective_weights(weights: Weights, missing: set[str]) -> dict[str, float]:
    """Relevance weights after moving missing terms' weight to the rest."""
    base = {t: getattr(weights, t) for t in RELEVANCE}
    present = {t: w for t, w in base.items() if t not in missing}
    total = sum(present.values())
    if not total:
        return {}
    scale = sum(base.values()) / total
    return {t: w * scale for t, w in present.items()}


def rank(
    signals: dict[str, dict[int, float]],
    weights: Weights,
    importance: dict[int, float],
    age_days: dict[int, float],
    half_life_days: float,
) -> list[Scored]:
    """Score every item any signal found, best first."""
    relevance_weights = effective_weights(weights, {t for t in RELEVANCE if not signals.get(t)})
    ids = set().union(*(signals.get(t, {}) for t in relevance_weights))
    out = []
    for i in ids:
        parts = {t: w * signals[t].get(i, 0.0) for t, w in relevance_weights.items()}
        if sum(parts.values()) <= 0:
            continue
        parts["importance"] = weights.importance * importance.get(i, 0.5)
        parts["recency"] = weights.recency * math.exp(-age_days.get(i, 0.0) / half_life_days)
        out.append(Scored(i, sum(parts.values()), parts))
    out.sort(key=lambda s: -s.score)
    return out


def scaled(raw: dict[int, float]) -> dict[int, float]:
    """Divide by the best value so the top item scores 1.0."""
    top = max(raw.values(), default=0.0)
    return {i: v / top for i, v in raw.items() if v > 0} if top > 0 else {}
