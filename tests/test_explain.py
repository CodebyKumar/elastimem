"""explain(): retrieval transparency — per-leg score breakdowns and graph
traversal path, without changing recall()'s actual ranking."""

from elastimem import Elastimem, ElastimemConfig
from elastimem.governor import GIB, Tier


def make_store(path, **cfg):
    cfg.setdefault("disable_builtin_embedder", True)
    return Elastimem(str(path), embed_fn=None, config=ElastimemConfig(**cfg),
                  probe_fn=lambda: (32 * GIB, 20 * GIB))


def test_explain_matches_recall_ranking(tmp_path):
    import pytest

    s = make_store(tmp_path / "e.db")
    s.record_turn("my car needs new brake pads, the fronts squeal",
                  "Worn pads squeal — get the fronts checked soon.")
    s.record_turn("thinking about repainting the kitchen walls",
                  "Nice — what color are you considering?")
    s.drain(timeout=5)

    recall_hits = s.recall("brake pads squealing on my car")
    explain_result = s.explain("brake pads squealing on my car")
    assert [h.text for h in recall_hits] == [h.text for h in explain_result.hits]
    for r, e in zip(recall_hits, explain_result.hits):
        assert r.score == pytest.approx(e.score, abs=1e-4)
    s.close()


def test_explain_breakdown_components_sum_to_total(tmp_path):
    s = make_store(tmp_path / "sum.db")
    s.record_turn("my car needs new brake pads", "Noted.")
    s.drain(timeout=5)

    result = s.explain("brake pads for my car")
    assert result.chunk_breakdowns
    b = result.chunk_breakdowns[0]
    assert abs(
        (b.fused_relevance + b.importance_nudge + b.recency_nudge + b.graph_nudge)
        - b.total
    ) < 1e-9


def test_explain_graph_matched_names_reflects_nudge(tmp_path):
    """The graph nudge only ever augments a chunk that already has real
    FTS/vector relevance to the query (see graph_relevance's docstring) —
    it cannot manufacture a hit from zero. So the query must share
    vocabulary with the target chunk to reach the graph leg at all."""
    from elastimem import graph

    s = make_store(tmp_path / "gn.db", tier_override=Tier.FULL)
    s.record_turn("Tuffy is a device that needs a new battery pack", "Got it!")
    s.drain(timeout=5)

    conn = s._conn
    jetson_id = graph.upsert_node(conn, "thing", "Jetson", confidence=1.0)
    tuffy_id = graph.upsert_node(conn, "thing", "Tuffy", confidence=1.0)
    graph.upsert_edge(conn, tuffy_id, jetson_id, "runs_on", confidence=1.0)

    result = s.explain("tell me about the device and my Jetson")
    assert result.chunk_breakdowns
    b = result.chunk_breakdowns[0]
    assert b.graph_nudge > 0
    assert "tuffy" in b.graph_matched_names
    s.close()


def test_explain_graph_traversal_shows_hop_path(tmp_path):
    from elastimem import graph

    s = make_store(tmp_path / "tp.db", tier_override=Tier.FULL)
    conn = s._conn
    jetson_id = graph.upsert_node(conn, "thing", "Jetson", confidence=1.0)
    tuffy_id = graph.upsert_node(conn, "thing", "Tuffy", confidence=1.0)
    graph.upsert_edge(conn, tuffy_id, jetson_id, "runs_on", confidence=1.0)

    result = s.explain("tell me about my Jetson only")
    names_by_depth = {step.canonical_name: step.hop_distance for step in result.graph_traversal}
    assert names_by_depth.get("jetson") == 0
    assert names_by_depth.get("tuffy") == 1
    seeds = {step.canonical_name for step in result.graph_traversal if step.is_seed}
    assert seeds == {"jetson"}
    s.close()


def test_explain_graph_hops_zero_empty_traversal(tmp_path):
    from elastimem import graph

    s = make_store(tmp_path / "hz.db", tier_override=Tier.LITE)
    graph.upsert_node(s._conn, "thing", "Jetson")
    result = s.explain("tell me about my Jetson")
    assert result.graph_hops == 0
    assert result.graph_traversal == ()
    s.close()


def test_explain_fact_breakdown(tmp_path):
    s = make_store(tmp_path / "fb.db")
    s.remember("favorite_color", "blue")
    result = s.explain("what is my favorite color")
    assert result.fact_breakdowns
    fb = result.fact_breakdowns[0]
    assert abs((fb.relevance * fb.importance * fb.recency_nudge + fb.graph_nudge) - fb.total) < 1e-9


def test_explain_empty_store_never_raises(tmp_path):
    s = make_store(tmp_path / "empty.db")
    result = s.explain("anything at all")
    assert result.hits == ()
    assert result.chunk_breakdowns == ()
    assert result.fact_breakdowns == ()
    s.close()


def test_explain_graph_error_degrades_silently(tmp_path, monkeypatch):
    from elastimem import graph

    s = make_store(tmp_path / "err.db", tier_override=Tier.FULL)
    s.record_turn("Tuffy needs a battery", "Got it!")
    s.drain(timeout=5)
    conn = s._conn
    graph.upsert_node(conn, "thing", "Tuffy", confidence=1.0)

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(graph, "detect_seed_nodes", _boom)
    result = s.explain("tell me about Tuffy")
    assert result.graph_traversal == ()
    assert result.hits  # chunk retrieval itself still works
    s.close()


def test_search_chunks_unaffected_by_refactor(tmp_path):
    """Regression: search_chunks()'s public return shape/behavior must be
    byte-for-byte unchanged after the breakdown refactor."""
    from elastimem.retrieval import search_chunks, Hit

    s = make_store(tmp_path / "rc.db")
    s.record_turn("my car needs new brake pads", "Noted.")
    s.drain(timeout=5)
    hits = search_chunks(s, "brake pads")
    assert hits and all(isinstance(h, Hit) for h in hits)
    s.close()
