"""Knowledge graph: canonicalization, dedup, caps, traversal, seed detection."""

from elastimem import graph


def _count(conn, table):
    return conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


# --------------------------------------------------------------------------- #
# canonicalization / dedup
# --------------------------------------------------------------------------- #
def test_upsert_node_dedupes_case_and_whitespace(store):
    conn = store._conn
    id1 = graph.upsert_node(conn, "org", "Acme Corp")
    id2 = graph.upsert_node(conn, "org", "acme corp")
    id3 = graph.upsert_node(conn, "org", "  ACME   CORP  ")
    assert id1 == id2 == id3
    row = conn.execute("SELECT mention_count FROM graph_nodes WHERE id=?", (id1,)).fetchone()
    assert row["mention_count"] == 3


def test_upsert_node_tracks_aliases(store):
    conn = store._conn
    node_id = graph.upsert_node(conn, "org", "Acme Corp")
    graph.upsert_node(conn, "org", "ACME CORP")
    row = conn.execute("SELECT aliases FROM graph_nodes WHERE id=?", (node_id,)).fetchone()
    import json
    aliases = json.loads(row["aliases"])
    assert "ACME CORP" in aliases


def test_upsert_node_empty_name_returns_none(store):
    assert graph.upsert_node(store._conn, "org", "   ") is None


def test_upsert_edge_dedupes_repeated_relationship(store):
    conn = store._conn
    a = graph.upsert_node(conn, "person", "user")
    b = graph.upsert_node(conn, "org", "Acme Corp")
    graph.upsert_edge(conn, a, b, "works_at")
    graph.upsert_edge(conn, a, b, "works_at")
    graph.upsert_edge(conn, a, b, "WORKS_AT")  # case-insensitive relation too
    assert _count(conn, "graph_edges") == 1
    row = conn.execute("SELECT seen_count FROM graph_edges").fetchone()
    assert row["seen_count"] == 3


def test_upsert_edge_self_loop_rejected(store):
    conn = store._conn
    a = graph.upsert_node(conn, "person", "user")
    assert graph.upsert_edge(conn, a, a, "knows") is None


def test_upsert_edge_empty_relationship_rejected(store):
    conn = store._conn
    a = graph.upsert_node(conn, "person", "user")
    b = graph.upsert_node(conn, "org", "Acme")
    assert graph.upsert_edge(conn, a, b, "   ") is None


# --------------------------------------------------------------------------- #
# store_extraction — defensive batch entry point
# --------------------------------------------------------------------------- #
def test_store_extraction_happy_path(store):
    conn = store._conn
    result = graph.store_extraction(
        conn,
        entities=[{"name": "Tuffy", "type": "thing"}, {"name": "Jetson", "type": "thing"}],
        relationships=[{"source": "user", "relation": "builds", "target": "Tuffy"}],
    )
    assert result["nodes"] == 2
    assert result["edges"] == 1
    assert _count(conn, "graph_nodes") == 3  # Tuffy, Jetson, user (edge endpoint)


def test_store_extraction_malformed_entities_ignored(store):
    conn = store._conn
    result = graph.store_extraction(conn, entities="not a list", relationships=None)
    assert result == {"nodes": 0, "edges": 0}
    assert _count(conn, "graph_nodes") == 0


def test_store_extraction_skips_bad_items(store):
    conn = store._conn
    result = graph.store_extraction(
        conn,
        entities=[{"name": "Tuffy"}, {"no_name": "x"}, "not a dict", {"name": ""}],
        relationships=[
            {"source": "user", "relation": "builds", "target": "Tuffy"},
            {"source": "user", "relation": "builds"},  # missing target
            {"source": "", "relation": "x", "target": "y"},  # blank source
        ],
    )
    assert result["nodes"] == 1
    assert result["edges"] == 1


def test_store_extraction_enforces_caps(store):
    conn = store._conn
    entities = [{"name": f"entity_{i}", "type": "thing"} for i in range(10)]
    graph.store_extraction(conn, entities=entities, relationships=None, node_cap=5, edge_cap=5)
    assert _count(conn, "graph_nodes") <= 5


def test_store_extraction_source_chunk_id_recorded(store):
    conn = store._conn
    store.record_turn("I am building a robot called Tuffy", "Cool!")
    chunk_id = conn.execute("SELECT id FROM chunks LIMIT 1").fetchone()["id"]
    graph.store_extraction(
        conn,
        entities=[{"name": "Tuffy", "type": "thing"}],
        relationships=[{"source": "user", "relation": "builds", "target": "Tuffy"}],
        source_chunk_id=chunk_id,
    )
    row = conn.execute("SELECT source_chunk_id FROM graph_edges").fetchone()
    assert row["source_chunk_id"] == chunk_id


# --------------------------------------------------------------------------- #
# traversal
# --------------------------------------------------------------------------- #
def _chain(conn):
    """A -> B -> C -> D"""
    a = graph.upsert_node(conn, "thing", "A")
    b = graph.upsert_node(conn, "thing", "B")
    c = graph.upsert_node(conn, "thing", "C")
    d = graph.upsert_node(conn, "thing", "D")
    graph.upsert_edge(conn, a, b, "rel")
    graph.upsert_edge(conn, b, c, "rel")
    graph.upsert_edge(conn, c, d, "rel")
    return a, b, c, d


def test_expand_zero_hops_returns_seeds_only(store):
    conn = store._conn
    a, b, c, d = _chain(conn)
    assert set(graph.expand(conn, [a], 0)) == {a}


def test_expand_one_hop(store):
    conn = store._conn
    a, b, c, d = _chain(conn)
    assert set(graph.expand(conn, [a], 1)) == {a, b}


def test_expand_two_hops(store):
    conn = store._conn
    a, b, c, d = _chain(conn)
    assert set(graph.expand(conn, [a], 2)) == {a, b, c}


def test_expand_bidirectional(store):
    conn = store._conn
    a, b, c, d = _chain(conn)
    # d is only ever a target_node; expansion from d must still reach c.
    assert c in graph.expand(conn, [d], 1)


def test_expand_handles_cycles(store):
    conn = store._conn
    a = graph.upsert_node(conn, "thing", "A")
    b = graph.upsert_node(conn, "thing", "B")
    graph.upsert_edge(conn, a, b, "rel")
    graph.upsert_edge(conn, b, a, "rel")
    result = graph.expand(conn, [a], 5)
    assert set(result) == {a, b}


def test_expand_empty_graph_returns_seeds(store):
    conn = store._conn
    assert graph.expand(conn, [999], 2) == [999]


def test_expand_no_seeds_returns_empty(store):
    assert graph.expand(store._conn, [], 2) == []


def test_expand_sql_error_degrades_to_seeds(store):
    conn = store._conn
    a = graph.upsert_node(conn, "thing", "A")
    conn.execute("DROP TABLE graph_edges")  # forces the recursive query to error
    assert graph.expand(conn, [a], 2) == [a]


# --------------------------------------------------------------------------- #
# query-time seed detection
# --------------------------------------------------------------------------- #
def test_detect_seed_nodes_matches_canonical_name(store):
    conn = store._conn
    node_id = graph.upsert_node(conn, "thing", "Jetson")
    matches = graph.detect_seed_nodes(conn, "tell me about my Jetson")
    assert node_id in [m[0] for m in matches]


def test_detect_seed_nodes_matches_alias(store):
    conn = store._conn
    node_id = graph.upsert_node(conn, "org", "Acme Corp")
    graph.upsert_node(conn, "org", "ACME CORP")  # records alias
    matches = graph.detect_seed_nodes(conn, "I work at ACME CORP now")
    assert node_id in [m[0] for m in matches]


def test_detect_seed_nodes_no_match_returns_empty(store):
    graph.upsert_node(store._conn, "thing", "Jetson")
    assert graph.detect_seed_nodes(store._conn, "totally unrelated query") == []


def test_detect_seed_nodes_prefers_longer_match(store):
    conn = store._conn
    short_id = graph.upsert_node(conn, "org", "Acme")
    long_id = graph.upsert_node(conn, "org", "Acme Corp International")
    matches = graph.detect_seed_nodes(conn, "I work at Acme Corp International")
    assert matches[0][0] == long_id


# --------------------------------------------------------------------------- #
# maintenance: decay/archival
# --------------------------------------------------------------------------- #
def test_apply_decay_removes_stale_low_confidence_node(store):
    conn = store._conn
    node_id = graph.upsert_node(conn, "thing", "OldGadget", confidence=0.2)
    with conn:
        conn.execute(
            "UPDATE graph_nodes SET updated_at='2020-01-01T00:00:00+00:00' WHERE id=?",
            (node_id,),
        )
    removed = graph.apply_decay(conn, half_life_days=30.0, archive_threshold=0.15)
    assert removed["nodes"] == 1
    assert _count(conn, "graph_nodes") == 0


def test_apply_decay_keeps_recently_reinforced_node(store):
    conn = store._conn
    node_id = graph.upsert_node(conn, "thing", "Jetson", confidence=0.9)
    removed = graph.apply_decay(conn, half_life_days=30.0, archive_threshold=0.15)
    assert removed["nodes"] == 0
    assert _count(conn, "graph_nodes") == 1


def test_apply_decay_removes_stale_edge_but_keeps_reinforced_node(store):
    conn = store._conn
    a = graph.upsert_node(conn, "thing", "A", confidence=0.9)
    b = graph.upsert_node(conn, "thing", "B", confidence=0.9)
    edge_id = graph.upsert_edge(conn, a, b, "rel", confidence=0.2)
    with conn:
        conn.execute(
            "UPDATE graph_edges SET last_seen='2020-01-01T00:00:00+00:00' WHERE id=?",
            (edge_id,),
        )
    removed = graph.apply_decay(conn, half_life_days=30.0, archive_threshold=0.15)
    assert removed["edges"] == 1
    assert _count(conn, "graph_edges") == 0
    # Both nodes have high confidence and recent updated_at - survive.
    assert _count(conn, "graph_nodes") == 2


def test_apply_decay_cascade_deletes_orphaned_node_edges(store):
    conn = store._conn
    a = graph.upsert_node(conn, "thing", "A", confidence=0.2)
    b = graph.upsert_node(conn, "thing", "B", confidence=0.9)
    graph.upsert_edge(conn, a, b, "rel", confidence=0.9)
    with conn:
        conn.execute(
            "UPDATE graph_nodes SET updated_at='2020-01-01T00:00:00+00:00' WHERE id=?",
            (a,),
        )
    graph.apply_decay(conn, half_life_days=30.0, archive_threshold=0.15)
    assert _count(conn, "graph_nodes") == 1
    assert _count(conn, "graph_edges") == 0  # cascaded


def test_apply_decay_empty_graph_noop(store):
    removed = graph.apply_decay(store._conn, half_life_days=30.0, archive_threshold=0.15)
    assert removed == {"nodes": 0, "edges": 0}


# --------------------------------------------------------------------------- #
# maintenance: LLM-assisted duplicate merging
# --------------------------------------------------------------------------- #
def test_merge_duplicates_merges_on_yes(store):
    conn = store._conn
    a = graph.upsert_node(conn, "org", "acme corp")
    b = graph.upsert_node(conn, "org", "acme corporation")
    other = graph.upsert_node(conn, "thing", "unrelated")
    graph.upsert_edge(conn, b, other, "rel")

    calls = []

    def llm(prompt, *, max_tokens, temperature):
        calls.append(prompt)
        return "yes"

    merged = graph.merge_duplicates(conn, llm, review_window_days=30.0)
    assert merged == 1
    assert calls  # the LLM was actually consulted
    assert _count(conn, "graph_nodes") == 2  # a + other; b merged away
    # b's edge repointed to a, not deleted
    row = conn.execute("SELECT source_node FROM graph_edges").fetchone()
    assert row["source_node"] == a


def test_merge_duplicates_no_merge_on_no(store):
    conn = store._conn
    graph.upsert_node(conn, "org", "acme corp")
    graph.upsert_node(conn, "org", "acme industries")

    def llm(prompt, *, max_tokens, temperature):
        return "no"

    merged = graph.merge_duplicates(conn, llm, review_window_days=30.0)
    assert merged == 0
    assert _count(conn, "graph_nodes") == 2


def test_merge_duplicates_skips_pairs_with_no_shared_token(store):
    conn = store._conn
    graph.upsert_node(conn, "org", "acme corp")
    graph.upsert_node(conn, "org", "widgets inc")

    calls = []

    def llm(prompt, *, max_tokens, temperature):
        calls.append(prompt)
        return "yes"

    graph.merge_duplicates(conn, llm, review_window_days=30.0)
    assert calls == []  # pre-filter rejected the pair, no LLM call made


def test_merge_duplicates_skips_different_types(store):
    conn = store._conn
    graph.upsert_node(conn, "org", "acme corp")
    graph.upsert_node(conn, "place", "acme corp")

    calls = []

    def llm(prompt, *, max_tokens, temperature):
        calls.append(prompt)
        return "yes"

    graph.merge_duplicates(conn, llm, review_window_days=30.0)
    assert calls == []


def test_merge_duplicates_llm_failure_never_raises(store):
    conn = store._conn
    graph.upsert_node(conn, "org", "acme corp")
    graph.upsert_node(conn, "org", "acme corporation")

    def broken_llm(prompt, *, max_tokens, temperature):
        raise RuntimeError("simulated failure")

    merged = graph.merge_duplicates(conn, broken_llm, review_window_days=30.0)
    assert merged == 0
    assert _count(conn, "graph_nodes") == 2


# --------------------------------------------------------------------------- #
# semantic clusters
# --------------------------------------------------------------------------- #
def _chain4(conn):
    """Jetson - Tuffy - Whisper - CUDA, all one connected component."""
    j = graph.upsert_node(conn, "thing", "Jetson")
    t = graph.upsert_node(conn, "thing", "Tuffy")
    w = graph.upsert_node(conn, "thing", "Whisper")
    c = graph.upsert_node(conn, "thing", "CUDA")
    graph.upsert_edge(conn, t, j, "runs_on")
    graph.upsert_edge(conn, j, c, "supports")
    graph.upsert_edge(conn, t, w, "uses")
    return j, t, w, c


def test_compute_clusters_groups_connected_entities(store):
    conn = store._conn
    j, t, w, c = _chain4(conn)
    clusters = graph.compute_clusters(conn)
    assert len(clusters) == 1
    members = next(iter(clusters.values()))
    assert set(members) == {j, t, w, c}


def test_compute_clusters_excludes_singletons(store):
    conn = store._conn
    _chain4(conn)
    graph.upsert_node(conn, "org", "Unrelated Corp")   # no edges at all
    clusters = graph.compute_clusters(conn)
    all_members = {m for members in clusters.values() for m in members}
    unrelated_id = store._conn.execute(
        "SELECT id FROM graph_nodes WHERE canonical_name='unrelated corp'"
    ).fetchone()["id"]
    assert unrelated_id not in all_members


def test_compute_clusters_separates_disjoint_components(store):
    conn = store._conn
    a = graph.upsert_node(conn, "thing", "A")
    b = graph.upsert_node(conn, "thing", "B")
    graph.upsert_edge(conn, a, b, "rel")
    c = graph.upsert_node(conn, "thing", "C")
    d = graph.upsert_node(conn, "thing", "D")
    graph.upsert_edge(conn, c, d, "rel")

    clusters = graph.compute_clusters(conn)
    assert len(clusters) == 2
    member_sets = [set(m) for m in clusters.values()]
    assert {a, b} in member_sets
    assert {c, d} in member_sets


def test_compute_clusters_empty_graph_returns_empty(store):
    assert graph.compute_clusters(store._conn) == {}


def test_store_clusters_stamps_cluster_id(store):
    conn = store._conn
    j, t, w, c = _chain4(conn)
    clusters = graph.compute_clusters(conn)
    graph.store_clusters(conn, clusters)
    rows = conn.execute("SELECT id, cluster_id FROM graph_nodes").fetchall()
    cluster_ids = {row["cluster_id"] for row in rows if row["id"] in (j, t, w, c)}
    assert len(cluster_ids) == 1 and None not in cluster_ids


def test_store_clusters_clears_stale_membership(store):
    conn = store._conn
    j, t, w, c = _chain4(conn)
    graph.store_clusters(conn, graph.compute_clusters(conn))

    # Now the graph changes: delete the edges so nothing is connected.
    conn.execute("DELETE FROM graph_edges")
    graph.store_clusters(conn, graph.compute_clusters(conn))

    rows = conn.execute("SELECT cluster_id FROM graph_nodes").fetchall()
    assert all(row["cluster_id"] is None for row in rows)


def test_label_clusters_assigns_llm_label(store):
    conn = store._conn
    _chain4(conn)
    graph.store_clusters(conn, graph.compute_clusters(conn))

    def llm(prompt, *, max_tokens, temperature):
        return "Local AI"

    labeled = graph.label_clusters(conn, llm)
    assert labeled == 1
    row = conn.execute(
        "SELECT DISTINCT cluster_label FROM graph_nodes WHERE cluster_id IS NOT NULL"
    ).fetchone()
    assert row["cluster_label"] == "Local AI"


def test_label_clusters_skips_already_labeled(store):
    conn = store._conn
    _chain4(conn)
    graph.store_clusters(conn, graph.compute_clusters(conn))

    calls = []

    def llm(prompt, *, max_tokens, temperature):
        calls.append(prompt)
        return "Local AI"

    graph.label_clusters(conn, llm)
    graph.label_clusters(conn, llm)   # second sweep: nothing left to label
    assert len(calls) == 1


def test_label_clusters_llm_failure_never_raises(store):
    conn = store._conn
    _chain4(conn)
    graph.store_clusters(conn, graph.compute_clusters(conn))

    def broken_llm(prompt, *, max_tokens, temperature):
        raise RuntimeError("simulated failure")

    labeled = graph.label_clusters(conn, broken_llm)
    assert labeled == 0


def test_label_clusters_rejects_overlong_label(store):
    conn = store._conn
    _chain4(conn)
    graph.store_clusters(conn, graph.compute_clusters(conn))

    def llm(prompt, *, max_tokens, temperature):
        return "x" * 200

    labeled = graph.label_clusters(conn, llm)
    assert labeled == 0


def test_list_clusters_returns_members_and_label(store):
    conn = store._conn
    _chain4(conn)
    graph.store_clusters(conn, graph.compute_clusters(conn))

    def llm(prompt, *, max_tokens, temperature):
        return "Local AI"

    graph.label_clusters(conn, llm)
    clusters = graph.list_clusters(conn)
    assert len(clusters) == 1
    assert clusters[0]["label"] == "Local AI"
    assert set(clusters[0]["members"]) == {"jetson", "tuffy", "whisper", "cuda"}


def test_list_clusters_empty_graph_returns_empty(store):
    assert graph.list_clusters(store._conn) == []


def test_elastimem_clusters_end_to_end(store):
    """Elastimem.clusters() through the public API, no direct graph.py use."""
    conn = store._conn
    _chain4(conn)
    graph.store_clusters(conn, graph.compute_clusters(conn))
    result = store.clusters()
    assert len(result) == 1
    assert set(result[0]["members"]) == {"jetson", "tuffy", "whisper", "cuda"}
    assert result[0]["label"] is None  # no LLM involved in this test


def test_elastimem_clusters_empty_store_never_raises(store):
    assert store.clusters() == []
