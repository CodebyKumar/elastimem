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
