"""Engram — elastic, resource-adaptive memory for local-first AI agents.

Quickstart::

    from engram import Engram

    mem = Engram("~/.myagent/memory.db")
    mem.remember("favorite_color", "blue")
    mem.facts()   # {'favorite_color': 'blue'}
"""

from .config import (
    Budgets,
    Cadence,
    ConsolidationLevel,
    EngramConfig,
    MemoryProfile,
    Tier,
)
from .semantic import Fact
from .store import Engram

__version__ = "0.1.0"

__all__ = [
    "Engram",
    "EngramConfig",
    "MemoryProfile",
    "Budgets",
    "Tier",
    "Cadence",
    "ConsolidationLevel",
    "Fact",
    "__version__",
]
