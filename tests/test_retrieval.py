"""Hybrid retrieval: vector leg, RRF fusion, semantic paraphrase recall."""

import hashlib
import math

from elastimem import Elastimem, ElastimemConfig
from elastimem.governor import GIB


def toy_embed(texts):
    """Deterministic 'semantic' embedding: bag of word-stems hashed into 64
    dims. Paraphrases sharing words land close; good enough to test fusion."""
    out = []
    for text in texts:
        vec = [0.0] * 64
        for word in text.lower().split():
            stem = word.strip(".,!?')(\"")[:5]
            if len(stem) < 3:
                continue
            idx = int(hashlib.md5(stem.encode()).hexdigest(), 16) % 64
            vec[idx] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        out.append([x / norm for x in vec])
    return out


def make_store(path, embed=toy_embed, **cfg):
    # embed=None means "this test wants no embedder at all" - Elastimem now
    # auto-activates its own built-in embedder whenever embed_fn is None
    # (see store.py), so expressing "really, no embedder" requires the
    # dedicated opt-out flag instead of relying on None falling through.
    if embed is None:
        cfg.setdefault("disable_builtin_embedder", True)
    return Elastimem(str(path), embed_fn=embed, config=ElastimemConfig(**cfg),
                  probe_fn=lambda: (32 * GIB, 20 * GIB))


def seed(store):
    store.record_turn("my car needs new brake pads, the fronts squeal",
                      "Worn pads squeal — get the fronts checked soon.")
    store.record_turn("thinking about repainting the kitchen walls",
                      "Nice — what color are you considering?")
    store.record_turn("my sourdough starter keeps dying after a week",
                      "Feed it twice daily and keep it warmer.")
    store.end_session()


def test_chunks_get_embedded_in_background(tmp_path):
    s = make_store(tmp_path / "e.db")
    seed(s)
    s.drain(timeout=5)
    n = s._conn.execute(
        "SELECT count(*) c FROM chunks WHERE embedding IS NOT NULL").fetchone()["c"]
    assert n == 3
    s.close()


def test_hybrid_beats_keyword_on_shared_vocabulary(tmp_path):
    s = make_store(tmp_path / "h.db")
    seed(s)
    s.drain(timeout=5)
    hits = s.recall("brake pads squealing on my car")
    assert hits and "brake" in hits[0].text
    # vector leg agrees with FTS -> fused score of the right chunk dominates
    assert all(h.score <= hits[0].score for h in hits)
    s.close()


def test_embed_failure_degrades_to_fts_only(tmp_path):
    calls = {"n": 0}

    def broken_embed(texts):
        calls["n"] += 1
        raise RuntimeError("model exploded")

    s = make_store(tmp_path / "b.db", embed=broken_embed)
    seed(s)
    s.drain(timeout=5)
    first_calls = calls["n"]
    assert first_calls >= 1          # it tried once...
    hits = s.recall("brake pads for my car")
    assert hits and "brake" in hits[0].text   # ...FTS still answers
    s.recall("sourdough starter advice")
    assert calls["n"] == first_calls  # ...and never tried again this session
    s.close()


def test_embed_backfill(tmp_path):
    from elastimem import embeddings

    path = tmp_path / "bf.db"
    s = make_store(path, embed=None)   # no embedder: chunks stay unembedded
    seed(s)
    s.close()

    s2 = make_store(path)              # embedder arrives later
    done = embeddings.embed_pending(s2)
    assert done == 3
    s2.close()


def test_no_fts_falls_back_to_like(tmp_path):
    s = make_store(tmp_path / "nofts.db", embed=None)
    seed(s)
    s.fts_enabled = False              # simulate a sqlite built without FTS5
    hits = s.recall("brake pads car")
    assert hits and "brake" in hits[0].text
    s.close()


def test_importance_does_not_override_relevance(tmp_path):
    """Regression test: a topically-unrelated chunk whose importance was
    bumped by rules.capture (see episodic.bump_importance) must never
    outrank a chunk that's genuinely relevant to the query. Before this fix,
    `score = relevance * importance * recency` let importance's 0.5->0.8
    multiplier (1.6x) swamp RRF's deliberately narrow rank-based spread,
    so a fact-bearing-but-unrelated chunk could beat the actually-relevant
    one on almost any query."""
    from elastimem import episodic

    s = make_store(tmp_path / "imp.db")
    s.record_turn("I recently moved to Austin for a new job",
                  "That's exciting! How do you like it so far?")
    s.record_turn("my favorite food is pasta", "Great choice!")
    s.drain(timeout=5)

    # Simulate the fact-bearing bump the pasta turn would get from
    # rules.capture matching a favorite_food pattern, WITHOUT the pasta
    # chunk having any real relevance to a location query.
    pasta_chunk_id = s._conn.execute(
        "SELECT id FROM chunks WHERE text LIKE '%pasta%'"
    ).fetchone()["id"]
    episodic.bump_importance(s._conn, [pasta_chunk_id])

    # toy_embed is a bag-of-word-stem-hash fake, not a real semantic model —
    # it has no paraphrase understanding, so the query must share actual
    # terms with the relevant chunk (a real embedder wouldn't need this;
    # see the built-in-embedder integration test for that case).
    hits = s.recall("moved to Austin for work")
    assert hits, "expected at least one hit"
    assert "austin" in hits[0].text.lower(), (
        f"importance bump on an unrelated chunk incorrectly outranked the "
        f"relevant one: top hit was {hits[0].text!r}"
    )
    s.close()


def test_vector_leg_uses_real_similarity_not_just_rank(tmp_path):
    """Regression test: similar_chunks must return real cosine scores, not
    just an ordering — search_chunks needs the magnitude to tell 'strongly
    relevant' apart from 'merely least-irrelevant of a small candidate set'
    (RRF alone can't, since it only ever sees rank)."""
    from elastimem import embeddings

    s = make_store(tmp_path / "cos.db")
    seed(s)
    s.drain(timeout=5)

    hits = embeddings.similar_chunks(s, "brake pads squealing on my car", limit=10)
    assert hits, "expected vector hits"
    assert all(isinstance(h, tuple) and len(h) == 2 for h in hits), (
        "similar_chunks must return (chunk_id, cosine_score) pairs"
    )
    scores = [score for _, score in hits]
    assert scores == sorted(scores, reverse=True), "hits must be sorted best-first"
    s.close()
