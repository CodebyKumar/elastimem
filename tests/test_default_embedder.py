"""Integration tests for elastimem's built-in default embedder
(default_embedder.py) — the real fastembed/ONNX model, not a fake.

Marked slow: downloads ~130MB on first run (cached after), so these are
excluded from the default test run (see pyproject.toml's addopts) and must
be run explicitly with `pytest -m slow`. Skipped automatically if the
optional 'embed' extra (fastembed) isn't installed, matching how the
feature itself degrades in production."""

import math

import pytest

fastembed = pytest.importorskip("fastembed", reason="optional 'embed' extra not installed")

from elastimem import default_embedder


pytestmark = pytest.mark.slow


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def test_embed_passages_returns_one_vector_per_input():
    vectors = default_embedder.embed_passages(["hello world", "goodbye world"])
    assert vectors is not None
    assert len(vectors) == 2
    assert len(vectors[0]) == len(vectors[1]) > 0


def test_embed_query_returns_a_single_vector():
    vec = default_embedder.embed_query("a search query")
    assert vec is not None
    assert len(vec) > 0


def test_paraphrase_ranks_above_unrelated_text():
    """The actual point of wiring in a real embedder: paraphrases should
    rank close even with zero shared vocabulary."""
    passages = default_embedder.embed_passages([
        "I recently moved to Austin, Texas for a new job.",
        "The weather has been really hot lately.",
    ])
    query = default_embedder.embed_query("where does the user currently live")

    sim_relevant = cosine(query, passages[0])
    sim_unrelated = cosine(query, passages[1])
    assert sim_relevant > sim_unrelated


def test_default_embed_fn_matches_embed_passages():
    a = default_embedder.embed_passages(["consistency check"])
    b = default_embedder.default_embed_fn(["consistency check"])
    assert a == b
