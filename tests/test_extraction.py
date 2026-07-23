"""extraction.extract_facts: combined facts+graph parsing, defensiveness,
backward compatibility with the pre-graph flat-JSON shape."""

import json

from elastimem import ElastimemConfig


def _store_fn(store):
    return lambda k, v, s: store.remember(k, v, source=s)


class _CompleteFn:
    def __init__(self, response: str):
        self.response = response

    def __call__(self, prompt, *, max_tokens, temperature):
        return self.response


def test_extract_facts_stores_entities_and_relationships(store):
    from elastimem import extraction

    payload = {
        "facts": {"favorite_color": "blue"},
        "entities": [{"name": "Tuffy", "type": "thing"}],
        "relationships": [{"source": "user", "relation": "builds", "target": "Tuffy"}],
    }
    stored = extraction.extract_facts(
        store._conn, store._config, _CompleteFn(json.dumps(payload)),
        "user text", "assistant text", _store_fn(store), graph_hops=2,
    )
    assert stored == {"favorite_color": "blue"}
    row = store._conn.execute("SELECT count(*) c FROM graph_edges").fetchone()
    assert row["c"] == 1


def test_extract_facts_graph_hops_zero_skips_graph_storage(store):
    from elastimem import extraction

    payload = {
        "facts": {},
        "entities": [{"name": "Tuffy", "type": "thing"}],
        "relationships": [{"source": "user", "relation": "builds", "target": "Tuffy"}],
    }
    extraction.extract_facts(
        store._conn, store._config, _CompleteFn(json.dumps(payload)),
        "user text", "assistant text", _store_fn(store), graph_hops=0,
    )
    row = store._conn.execute("SELECT count(*) c FROM graph_nodes").fetchone()
    assert row["c"] == 0


def test_extract_facts_malformed_entities_list_ignored(store):
    from elastimem import extraction

    payload = {"facts": {"name": "Alex"}, "entities": "not a list", "relationships": 42}
    stored = extraction.extract_facts(
        store._conn, store._config, _CompleteFn(json.dumps(payload)),
        "user text", "assistant text", _store_fn(store), graph_hops=2,
    )
    assert stored == {"name": "Alex"}
    row = store._conn.execute("SELECT count(*) c FROM graph_nodes").fetchone()
    assert row["c"] == 0


def test_extract_facts_entity_missing_name_skipped(store):
    from elastimem import extraction

    payload = {"facts": {}, "entities": [{"type": "thing"}], "relationships": []}
    extraction.extract_facts(
        store._conn, store._config, _CompleteFn(json.dumps(payload)),
        "user text", "assistant text", _store_fn(store), graph_hops=2,
    )
    row = store._conn.execute("SELECT count(*) c FROM graph_nodes").fetchone()
    assert row["c"] == 0


def test_extract_facts_backward_compat_flat_json(store):
    from elastimem import extraction

    # Old shape: no "facts"/"entities" wrapper at all.
    payload = {"favorite_color": "blue", "occupation": "designer"}
    stored = extraction.extract_facts(
        store._conn, store._config, _CompleteFn(json.dumps(payload)),
        "user text", "assistant text", _store_fn(store), graph_hops=2,
    )
    assert stored == {"favorite_color": "blue", "occupation": "designer"}


def test_extract_facts_none_response_unchanged(store):
    from elastimem import extraction

    stored = extraction.extract_facts(
        store._conn, store._config, _CompleteFn("NONE"),
        "user text", "assistant text", _store_fn(store), graph_hops=2,
    )
    assert stored == {}


def test_extract_facts_malformed_json_never_raises(store):
    from elastimem import extraction

    stored = extraction.extract_facts(
        store._conn, store._config, _CompleteFn("not json at all { garbage"),
        "user text", "assistant text", _store_fn(store), graph_hops=2,
    )
    assert stored == {}


def test_extract_facts_graph_storage_failure_does_not_break_facts(store, monkeypatch):
    from elastimem import extraction, graph

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated graph failure")

    monkeypatch.setattr(graph, "store_extraction", _boom)
    payload = {
        "facts": {"favorite_color": "blue"},
        "entities": [{"name": "Tuffy", "type": "thing"}],
        "relationships": [],
    }
    stored = extraction.extract_facts(
        store._conn, store._config, _CompleteFn(json.dumps(payload)),
        "user text", "assistant text", _store_fn(store), graph_hops=2,
    )
    assert stored == {"favorite_color": "blue"}


def test_extract_facts_source_chunk_id_threaded_through(store):
    from elastimem import extraction

    store.record_turn("I am building a robot called Tuffy", "Cool!")
    chunk_id = store._conn.execute("SELECT id FROM chunks LIMIT 1").fetchone()["id"]

    payload = {
        "facts": {},
        "entities": [{"name": "Tuffy", "type": "thing"}],
        "relationships": [{"source": "user", "relation": "builds", "target": "Tuffy"}],
    }
    extraction.extract_facts(
        store._conn, store._config, _CompleteFn(json.dumps(payload)),
        "user text", "assistant text", _store_fn(store), graph_hops=2,
        source_chunk_id=chunk_id,
    )
    row = store._conn.execute("SELECT source_chunk_id FROM graph_edges").fetchone()
    assert row["source_chunk_id"] == chunk_id


def test_consolidate_runs_graph_decay(store):
    from elastimem import extraction, graph

    conn = store._conn
    node_id = graph.upsert_node(conn, "thing", "OldGadget", confidence=0.1)
    with conn:
        conn.execute(
            "UPDATE graph_nodes SET updated_at='2020-01-01T00:00:00+00:00' WHERE id=?",
            (node_id,),
        )
    stats = extraction.consolidate(conn, store._config, None, llm_merge=False)
    assert stats["graph_nodes_archived"] == 1
    assert conn.execute("SELECT count(*) c FROM graph_nodes").fetchone()["c"] == 0


def test_consolidate_llm_merge_gates_graph_duplicate_merge(store):
    from elastimem import extraction, graph

    conn = store._conn
    graph.upsert_node(conn, "org", "acme corp")
    graph.upsert_node(conn, "org", "acme corporation")

    stats = extraction.consolidate(conn, store._config, None, llm_merge=False)
    assert "graph_merged" not in stats  # no complete_fn, no llm_merge -> not attempted
    assert conn.execute("SELECT count(*) c FROM graph_nodes").fetchone()["c"] == 2

    stats = extraction.consolidate(
        conn, store._config, _CompleteFn("yes"), llm_merge=True,
    )
    assert stats["graph_merged"] == 1
    assert conn.execute("SELECT count(*) c FROM graph_nodes").fetchone()["c"] == 1


def test_consolidate_never_raises_on_graph_maintenance_failure(store, monkeypatch):
    from elastimem import extraction, graph

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(graph, "apply_decay", _boom)
    stats = extraction.consolidate(store._conn, store._config, None, llm_merge=False)
    assert "archived" in stats  # fact decay still ran and returned normally


def test_consolidate_computes_clusters(store):
    from elastimem import extraction, graph

    conn = store._conn
    j = graph.upsert_node(conn, "thing", "Jetson")
    t = graph.upsert_node(conn, "thing", "Tuffy")
    graph.upsert_edge(conn, t, j, "runs_on")

    stats = extraction.consolidate(conn, store._config, None, llm_merge=False)
    assert stats["clusters"] == 1
    assert "clusters_labeled" not in stats  # no llm_merge -> labeling not attempted
    rows = conn.execute("SELECT cluster_id FROM graph_nodes").fetchall()
    assert all(r["cluster_id"] is not None for r in rows)


def test_consolidate_labels_clusters_under_llm_merge(store):
    from elastimem import extraction, graph

    conn = store._conn
    j = graph.upsert_node(conn, "thing", "Jetson")
    t = graph.upsert_node(conn, "thing", "Tuffy")
    graph.upsert_edge(conn, t, j, "runs_on")

    def llm(prompt, *, max_tokens, temperature):
        return "Local AI"

    stats = extraction.consolidate(conn, store._config, llm, llm_merge=True)
    assert stats["clusters_labeled"] == 1
    row = conn.execute(
        "SELECT DISTINCT cluster_label FROM graph_nodes WHERE cluster_id IS NOT NULL"
    ).fetchone()
    assert row["cluster_label"] == "Local AI"
