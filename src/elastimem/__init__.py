"""Elastimem — elastic, resource-adaptive memory for local-first AI agents.

Quickstart::

    import elastimem

    mem = elastimem.open("~/.myagent/memory.db")
    mem.remember("favorite_color", "blue")
    mem.facts()   # {'favorite_color': 'blue'}

With an LLM and embedder (both optional — everything degrades gracefully)::

    mem = elastimem.open(
        "~/.myagent/memory.db",
        llm=my_complete_fn,       # (prompt, *, max_tokens, temperature) -> str
        embedder=my_embed_fn,     # (list[str]) -> list[list[float]]
        context_tokens=4096,      # any ElastimemConfig field, passed inline
    )
"""

from importlib.metadata import PackageNotFoundError, version

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

try:
    __version__ = version("elastimem")
except PackageNotFoundError:
    # Editable/dev checkout with no installed distribution metadata.
    __version__ = "0.0.0+dev"


def open(path: str, **kwargs) -> Elastimem:
    """Open (creating if needed) a memory store at ``path``.

    The friendly front door — identical to ``Elastimem(path, **kwargs)``.
    Accepts ``llm=``, ``embedder=``, ``config=ElastimemConfig(...)``, and any
    individual :class:`ElastimemConfig` field as a keyword argument.
    """
    return Elastimem(path, **kwargs)


__all__ = [
    "open",
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
