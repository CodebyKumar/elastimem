"""Elastimem: long-term memory for local AI agents, in one SQLite file.

::

    import elastimem

    with elastimem.open("~/.myagent/memory.db", llm=my_llm) as mem:
        context = mem.context(user)
        reply = my_llm(f"{context}\\n\\n{user}")
        mem.record(user, reply)
"""

from importlib.metadata import PackageNotFoundError, version

from .config import ElastimemConfig, Tier
from .retrieval.context import Context
from .retrieval.inspection import Explanation
from .retrieval.search import Hit
from .runtime.governor import Budgets, Status
from .storage.facts import Fact
from .store import Elastimem

try:
    __version__ = version("elastimem")
except PackageNotFoundError:
    __version__ = "0.0.0+dev"


def open(path: str, **kwargs) -> Elastimem:
    """Open (creating if needed) the store at ``path``. Same as ``Elastimem(path, ...)``."""
    return Elastimem(path, **kwargs)


__all__ = [
    "open", "Elastimem", "ElastimemConfig", "Tier", "Status", "Budgets",
    "Context", "Fact", "Hit", "Explanation", "__version__",
]
