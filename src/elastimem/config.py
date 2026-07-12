"""Configuration and shared value types for Elastimem.

Three objects travel through the whole framework:

* :class:`ElastimemConfig` — host-supplied knobs, all optional.
* :class:`Tier` — the governor's coarse resource class (LITE/STANDARD/FULL).
* :class:`MemoryProfile` — the governor's *output*: a frozen snapshot of every
  capability decision and token budget, consumed by retrieval assembly and the
  background worker. Hosts read it; they never build it.
"""

from __future__ import annotations

import enum
import os
from dataclasses import dataclass, field


class Tier(enum.IntEnum):
    """Resource class of the host machine. Ordering is meaningful (LITE < FULL)."""

    LITE = 0
    STANDARD = 1
    FULL = 2

    @classmethod
    def from_env(cls) -> "Tier | None":
        """Read the ELASTIMEM_TIER override (``lite|standard|full``), if set."""
        raw = os.environ.get("ELASTIMEM_TIER", "").strip().lower()
        return {"lite": cls.LITE, "standard": cls.STANDARD, "full": cls.FULL}.get(raw)


class Cadence(str, enum.Enum):
    """When background LLM extraction runs."""

    PER_TURN = "per_turn"
    BATCHED = "batched"          # coalesced, every N turns
    SESSION_END = "session_end"
    OFF = "off"


class ConsolidationLevel(str, enum.Enum):
    """How aggressive idle/exit consolidation is."""

    FULL = "full"                # dedupe + decay + LLM contradiction-merge + backfill
    DEDUPE_ONLY = "dedupe_only"  # dedupe + decay math, no LLM
    OFF = "off"                  # exact dedupe at exit only


@dataclass(frozen=True)
class Budgets:
    """Per-turn token budgets for each injected prompt section."""

    working: int
    facts: int
    episodic: int
    sessions: int
    lessons: int

    @property
    def memory_total(self) -> int:
        return self.facts + self.episodic + self.sessions + self.lessons


@dataclass(frozen=True)
class MemoryProfile:
    """The governor's decision snapshot. Recreated on every tier change."""

    tier: Tier
    budgets: Budgets
    embeddings_enabled: bool
    llm_extraction_enabled: bool
    extraction_cadence: Cadence
    rolling_summary_enabled: bool
    consolidation_level: ConsolidationLevel
    episodic_top_k: int
    window_min_turns: int = 2


# Default allowlist of keys that belong to the stable user profile (always
# injected into the prompt). Everything else lands in the 'note' category.
DEFAULT_PROFILE_KEYS = frozenset(
    {"name", "email", "location", "age", "occupation", "pronouns"}
)


@dataclass
class ElastimemConfig:
    """Host-tunable configuration. Every field has a sensible default.

    ``context_tokens`` should be the context window of the host's chat model
    (llama.cpp ``n_ctx``, or the API model's limit). ``static_prompt_tokens``
    is the host-measured size of its own fixed prompt (persona, tool specs)
    so the governor can budget only what is actually free.
    """

    # --- budget inputs -----------------------------------------------------
    context_tokens: int = 4096
    output_reserve: int = 512        # generation headroom
    tool_reserve: int = 600          # mid-turn tool observations (ReAct etc.)
    static_prompt_tokens: int = 0    # host's fixed prompt, measured by the host

    # --- fact hygiene ------------------------------------------------------
    profile_keys: frozenset[str] = DEFAULT_PROFILE_KEYS
    reserved_keys: frozenset[str] = frozenset()   # host-owned keys Elastimem must reject
    quarantine_cap: int = 200
    # How far back consolidate() looks for a key whose value changed, to
    # offer the LLM a chance to merge OLD/NEW into one value instead of a
    # blunt overwrite ("moving to Austin in May" + "lives in Austin"). A
    # change older than this is assumed settled and reviewed only if it
    # changes again. FULL tier only consolidates on session end / idle
    # ticks, not continuously, so this should stay well above the gap
    # between two such ticks in normal use, or genuinely-recent changes can
    # age out of the window before consolidation ever sees them.
    fact_merge_review_window_days: float = 7.0

    # --- procedural memory -------------------------------------------------
    max_lessons: int = 15            # older lessons archived beyond this
    lessons_in_prompt: int = 5

    # --- episodic memory ---------------------------------------------------
    chunk_target_tokens: int = 400   # split exchanges bigger than this
    min_query_words: int = 4         # skip retrieval/extraction below this

    # --- forgetting --------------------------------------------------------
    fact_decay_half_life_days: float = 60.0
    fact_archive_threshold: float = 0.15
    episodic_recency_half_life_days: float = 30.0
    fact_recency_half_life_days: float = 90.0

    # --- governor ----------------------------------------------------------
    tier_override: Tier | None = field(default_factory=Tier.from_env)
    upgrade_healthy_ticks: int = 10  # consecutive healthy ticks before tier upgrade
    idle_consolidate_seconds: float = 90.0

    # --- background worker -------------------------------------------------
    worker_max_tokens: int = 96      # cap on every background LLM call
    batched_every_n_turns: int = 3

    def __post_init__(self) -> None:
        if self.context_tokens < 1024:
            raise ValueError("context_tokens must be >= 1024")
        self.profile_keys = frozenset(k.lower() for k in self.profile_keys)
        self.reserved_keys = frozenset(k.lower() for k in self.reserved_keys)
