"""Elastimem's built-in embedder: activates automatically when a host does
not supply its own ``embedder=``/``embed_fn=``, so semantic recall works out
of the box with no setup on the host's part — no model to pick, no API key,
no manual wiring. Elastimem's core stays dependency-free; this module is the
one place that reaches for an optional extra (``pip install
elastimem[embed]``), and it is only ever imported lazily, on first actual
embedding call, never at package import time.

Model: BAAI/bge-small-en-v1.5 via fastembed (ONNX Runtime, no torch) — a
real, MTEB-benchmarked retrieval model, not a hashing trick. ~130MB, fetched
from Hugging Face on first use and cached under the host's platform cache
dir (fastembed's own default), then reused across process runs.

Failure is always non-fatal: if the ``embed`` extra isn't installed, or the
first download fails (no network, disk full, ...), ``get_default_embedder()``
returns ``None`` and the store falls back to FTS5/keyword retrieval exactly
as it did before this module existed — nothing here can turn a working
degraded mode into a crash.
"""

from __future__ import annotations

import logging
import threading

log = logging.getLogger("elastimem")

_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# bge-small's own recommended asymmetric prefixes: queries and passages are
# embedded differently on purpose (this is how the model was trained/
# benchmarked) — using the wrong one measurably hurts retrieval accuracy, so
# encode_chunks/encode_query are NOT interchangeable despite both returning
# the same vector shape.
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_lock = threading.Lock()
_model = None          # lazily constructed fastembed.TextEmbedding
_unavailable = False    # sticky: True once we've confirmed the extra/model can't load


def _load_model():
    """Imports fastembed and constructs the model. Only ever called once,
    under _lock. Raises on any failure — callers translate that into the
    'unavailable, fall back to FTS' path."""
    from fastembed import TextEmbedding  # optional extra: elastimem[embed]
    return TextEmbedding(model_name=_MODEL_NAME)


def _get_model():
    global _model, _unavailable
    if _model is not None:
        return _model
    if _unavailable:
        return None
    with _lock:
        if _model is not None:
            return _model
        if _unavailable:
            return None
        try:
            _model = _load_model()
        except Exception:
            _unavailable = True
            log.warning(
                "elastimem: built-in embedder unavailable (install the "
                "'embed' extra: pip install elastimem[embed]) — falling "
                "back to FTS5/keyword retrieval.",
                exc_info=True,
            )
            return None
        return _model


def is_available() -> bool:
    """True once the model has successfully loaded. Does NOT trigger a load
    itself — use to check state without paying for a first-use download."""
    return _model is not None


def embed_passages(texts: list[str]) -> list[list[float]] | None:
    """Embeds chunk/fact text for storage. Returns None (never raises) if
    the built-in embedder isn't available, so callers can fall back."""
    model = _get_model()
    if model is None:
        return None
    # bge's passage side takes raw text, no prefix.
    return [list(v) for v in model.embed(texts)]


def embed_query(text: str) -> list[float] | None:
    """Embeds a single search query. Uses bge's query prefix — required for
    the model's trained retrieval behavior, distinct from embed_passages."""
    model = _get_model()
    if model is None:
        return None
    vectors = list(model.embed([_QUERY_PREFIX + text]))
    return list(vectors[0]) if vectors else None


def default_embed_fn(texts: list[str]) -> list[list[float]]:
    """The EmbedFn-shaped callable Store wires in when the host doesn't
    supply one. Matches elastimem's EmbedFn contract: list[str] ->
    list[list[float]], one vector per input, in order. Passage-side
    encoding (chunk storage) — query-side goes through embed_query()
    directly inside retrieval, since EmbedFn's single-signature contract
    can't distinguish the two call sites bge needs distinguished.

    Raises (rather than returning a degraded result) on failure, matching
    every other EmbedFn's contract (embeddings.py already treats a raising
    embed_fn as 'disable the vector leg for this session' — see
    embeddings.embed_chunks/similar_chunks)."""
    vectors = embed_passages(texts)
    if vectors is None:
        raise RuntimeError(
            "elastimem's built-in embedder is unavailable (fastembed not "
            "installed, or the model failed to download) — install with "
            "pip install elastimem[embed]"
        )
    return vectors
