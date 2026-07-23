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
