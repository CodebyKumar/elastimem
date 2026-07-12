"""Elastimem — elastic, resource-adaptive memory for local-first AI agents.

Quickstart::

    from elastimem import Elastimem

    mem = Elastimem("~/.myagent/memory.db")
    mem.remember("favorite_color", "blue")
    mem.facts()   # {'favorite_color': 'blue'}
"""

from .config import (
    Budgets,
    Cadence,
    ConsolidationLevel,
    ElastimemConfig,
    MemoryProfile,
    Tier,
)
from .semantic import Fact
from .store import Elastimem

__version__ = "0.1.0"

__all__ = [
    "Elastimem",
    "ElastimemConfig",
    "MemoryProfile",
    "Budgets",
    "Tier",
    "Cadence",
    "ConsolidationLevel",
    "Fact",
    "__version__",
]
