"""What each tier allows and how it ranks memories: the one place to tune.

A tier is chosen by the governor from what the machine can afford. Each tier
has a fixed ranking equation (see :mod:`elastimem.scoring`)::

    relevance = vector*w_vector + keyword*w_keyword + graph*w_graph
    score     = relevance + importance*w_importance + recency*w_recency

and a fixed set of capabilities. Nothing here is computed at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Tier


@dataclass(frozen=True)
class Weights:
    vector: float
    keyword: float
    graph: float
    importance: float = 0.05
    recency: float = 0.05


@dataclass(frozen=True)
class TierSpec:
    weights: Weights
    graph_hops: int
    load_embedder: bool      # may load the built-in model (~360 MB measured)
    extraction: str          # LLM fact extraction cadence: per_turn | batched | session_end
    llm_summary: bool        # rolling summary by the LLM (otherwise extractive)
    llm_maintenance: bool    # LLM fact merges, entity merges, topic labels
    idle_consolidation: bool


# One equation for every tier, tuned on benchmarks/eval_recall.py (vector 0.8 /
# keyword 0.1 gave the best MRR; the 0.1 graph weight is untested there, as
# the eval has no LLM). Tiers differ in which terms are available, not in
# how much a term is worth: that is handled by missing-term reweighting.
_EQUATION = Weights(vector=0.80, keyword=0.10, graph=0.10)

TIERS: dict[Tier, TierSpec] = {
    Tier.FULL: TierSpec(_EQUATION, graph_hops=2, load_embedder=True,
                        extraction="per_turn", llm_summary=True, llm_maintenance=True,
                        idle_consolidation=True),
    Tier.STANDARD: TierSpec(_EQUATION, graph_hops=1, load_embedder=True,
                            extraction="batched", llm_summary=True, llm_maintenance=False,
                            idle_consolidation=False),
    Tier.LITE: TierSpec(_EQUATION, graph_hops=1, load_embedder=False,
                        extraction="session_end", llm_summary=False, llm_maintenance=False,
                        idle_consolidation=False),
}

# --- governor --------------------------------------------------------------
FULL_RAM_GIB = 16          # total RAM for FULL
STANDARD_RAM_GIB = 8       # total RAM for STANDARD
WARN_FREE_SHARE = 0.20     # Linux/psutil: free share below this caps at STANDARD
CRITICAL_FREE_SHARE = 0.10  # ... and below this drops to LITE
PROBE_SECONDS = 5.0        # hardware is re-read at most this often
SLOW_TURNS = 3             # turns over turn_budget_ms before stepping down
UPGRADE_TURNS = 10         # healthy turns before stepping back up
PRESSURE_COOLDOWN_SECONDS = 30.0

# --- context budgets (same on every tier) ------------------------------------
WORKING_SHARE = 0.55       # of the free tokens, kept for the recent conversation
SPLIT = {"facts": 0.40, "moments": 0.30, "sessions": 0.15, "lessons": 0.15}
TOPICS_SHARE = 0.25        # carved out of the moments budget
TURN_TOKENS = 120          # average tokens per turn, for keep_turns
MIN_KEEP_TURNS = 2
MOMENTS_K = 5
SESSIONS_SHOWN = 5
LESSONS_SHOWN = 5

# --- search ----------------------------------------------------------------
CANDIDATES = 20            # per search leg
COMMON_TERM_SHARE = 0.5    # a query term in more than this share of rows is ignored
COMMON_TERM_MIN_ROWS = 20  # ... once the table has at least this many rows

# --- background work -------------------------------------------------------
CHUNK_TOKENS = 400         # longer exchanges are split at sentence ends
BATCH_TURNS = 2            # STANDARD extracts once this many turns are pending
EMBED_BATCH = 64
LLM_MAX_TOKENS = 96        # cap on every background LLM call
IDLE_SECONDS = 90.0
MERGE_WINDOW_DAYS = 7.0    # how far back LLM merges look

# --- retention -------------------------------------------------------------
ARCHIVE_BELOW = 0.15       # decayed importance/confidence below this is retired
REJECTED_CAP = 200
ENTITY_CAP = 2000
RELATION_CAP = 5000
