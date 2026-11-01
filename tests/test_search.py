import pytest

import elastimem
from elastimem import Tier, db, scoring
from elastimem.tiers import Weights

from conftest import toy_embed


def _fill(mem, n=30):
    for i in range(n):
        mem.record(f"Random chat number {i} about the weather.", "Sure.")
    mem.new_session()


def test_keyword_finds_a_past_moment(mem):
    _fill(mem)
    mem.record("My dentist appointment is on Friday", "Good luck.")
    mem.new_session()
    hits = mem.search("dentist appointment")
    assert hits[0].kind == "moment" and "dentist" in hits[0].text


def test_common_words_are_ignored(mem):
    _fill(mem)
    mem.record("The cabin wifi password is pinecone42", "Saved.")
    mem.new_session()
    # "user"/"assistant"/"weather" appear in most chunks and must not decide the ranking.
    assert "pinecone42" in mem.search("user assistant weather pinecone42")[0].text


def test_vectors_find_by_meaning(path):
    with elastimem.open(path, embedder=toy_embed, tier=Tier.FULL) as mem:
        _fill(mem)
        mem.record("I adopted a puppy named Bruno", "Lovely.")
        mem.new_session()
        mem.flush()
        hit = mem.search("the puppy I adopted")[0]
        assert "Bruno" in hit.text
        parts = mem.explain("the puppy I adopted").items[0].parts
        assert parts["vector"] > 0


def test_current_session_is_not_a_past_moment(mem):
    mem.record("The secret word is marmalade", "ok")
    assert mem.context("marmalade").sections["moments"] == ""
    mem.new_session()
    assert "marmalade" in mem.context("marmalade").sections["moments"]


def test_repeated_text_appears_once(mem):
    for _ in range(5):
        mem.record("I like tea", "ok")
        mem.new_session()
    assert len([h for h in mem.search("tea", k=5) if h.kind == "moment"]) == 1


def test_facts_are_searchable(mem):
    mem.remember("favorite_color", "teal")
    hits = mem.search("favorite color")
    assert hits[0].kind == "fact" and hits[0].text == "favorite_color: teal"


def test_like_fallback_without_fts5(path, monkeypatch):
    monkeypatch.setattr(db, "fts5_available", lambda conn: False)
    with elastimem.open(path, embedder=None, tier=Tier.FULL) as mem:
        assert not mem.stats()["fts"]
        mem.record("I collect vintage stamps", "Nice.")
        mem.new_session()
        assert "stamps" in mem.search("stamps")[0].text


def test_explain_reports_missing_terms(mem):
    mem.record("hello", "hi")
    ex = mem.explain("hello")
    assert ex.tier is Tier.FULL
    assert set(ex.missing) == {"vector", "graph"}
    for item in ex.items:
        assert sum(item.parts.values()) == pytest.approx(item.hit.score)


# --- the equation ----------------------------------------------------------------
W = Weights(vector=0.8, keyword=0.1, graph=0.1)


def test_missing_terms_hand_over_their_weight():
    assert scoring.effective_weights(W, set()) == {"vector": 0.8, "keyword": 0.1, "graph": 0.1}
    w = scoring.effective_weights(W, {"vector"})
    assert w == pytest.approx({"keyword": 0.5, "graph": 0.5})
    assert scoring.effective_weights(W, {"vector", "keyword", "graph"}) == {}


def test_importance_and_recency_never_add_an_unrelated_item():
    ranked = scoring.rank({"keyword": {1: 1.0}}, W, {1: 0.5, 2: 1.0}, {1: 0, 2: 0}, 30)
    assert [s.id for s in ranked] == [1]


def test_relevance_beats_importance():
    ranked = scoring.rank({"keyword": {1: 1.0, 2: 0.6}}, W, {1: 0.5, 2: 1.0}, {1: 0, 2: 0}, 30)
    assert [s.id for s in ranked] == [1, 2]


def test_pure_python_vectors_match_numpy(path, monkeypatch):
    from elastimem import vectors

    with elastimem.open(path, embedder=toy_embed, tier=Tier.FULL) as mem:
        _fill(mem, 5)
        mem.record("I adopted a puppy named Bruno", "Lovely.")
        mem.new_session()
        mem.flush()
        with_numpy = [h.text for h in mem.search("the puppy I adopted")]
    monkeypatch.setattr(vectors, "np", None)
    with elastimem.open(path, embedder=toy_embed, tier=Tier.FULL) as mem:
        assert [h.text for h in mem.search("the puppy I adopted")] == with_numpy


def test_an_unrelated_fact_does_not_outrank_a_relevant_moment(path):
    # Regression: vector scores were scaled per table, so the only fact scored 1.0.
    with elastimem.open(path, embedder=toy_embed, tier=Tier.FULL) as mem:
        mem.remember("favorite_color", "blue")
        mem.record("what is the capital of Japan?", "Tokyo is the capital of Japan.")
        mem.new_session()
        mem.flush()
        assert "Tokyo" in mem.search("capital of Japan")[0].text
