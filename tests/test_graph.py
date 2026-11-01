from elastimem import graph


def _extract(conn, chunk_ids, entities, relations=()):
    return graph.store_extraction(
        conn, [{"name": n, "type": t} for n, t in entities],
        [{"source": s, "relation": r, "target": t} for s, r, t in relations], chunk_ids,
    )


def test_extraction_stores_entities_relations_and_mentions(mem):
    mem.record("My sister Priya lives in Lisbon", "Nice.")
    conn = mem._conn
    _extract(conn, [1], [("Priya", "person"), ("Lisbon", "place")],
             [("Priya", "lives_in", "Lisbon"), ("user", "sister_of", "Priya")])
    assert {r["name"] for r in conn.execute("SELECT name FROM entities")} == {"priya", "lisbon", "user"}
    assert conn.execute("SELECT count(*) FROM relations").fetchone()[0] == 2
    assert {r["chunk_id"] for r in conn.execute("SELECT chunk_id FROM mentions")} == {1}


def test_repeated_mentions_update_one_row(mem):
    conn = mem._conn
    for _ in range(3):
        _extract(conn, [], [("Acme Corp", "org")])
    row = conn.execute("SELECT seen, aliases FROM entities").fetchone()
    assert row["seen"] == 3 and row["aliases"] == '["Acme Corp"]'


def test_malformed_items_are_skipped(mem):
    assert graph.store_extraction(mem._conn, ["x", {"name": 5}, {"name": "ok"}], "junk", []) == 1


def test_seeds_match_whole_words_only(mem):
    conn = mem._conn
    _extract(conn, [], [("Art", "thing"), ("New York", "place")])
    names = lambda q: {r["name"] for r in graph.entities(conn, graph.seeds(conn, q))}
    assert names("let's start over") == set()
    assert names("I love art in new york") == {"art", "new york"}


def test_traversal_does_not_pass_through_the_user(mem):
    conn = mem._conn
    _extract(conn, [], [("Bruno", "thing"), ("Acme", "org")],
             [("user", "owns", "Bruno"), ("user", "works_at", "Acme")])
    reach = graph.expand(conn, graph.seeds(conn, "Bruno"), hops=2)
    assert {r["name"] for r in graph.entities(conn, list(reach))} == {"bruno"}


def test_topics_and_merge(mem):
    conn = mem._conn
    _extract(conn, [], [("llama", "thing"), ("gguf", "thing"), ("llama cpp", "thing")],
             [("llama", "runs", "gguf"), ("llama cpp", "loads", "gguf")])
    assert graph.recompute_topics(conn) == 1
    graph.name_topic(conn, graph.list_topics(conn)[0]["id"], "Local AI")
    a, b = graph.merge_candidates(conn)[0]
    graph.merge(conn, a["id"], b["id"])
    graph.recompute_topics(conn)
    topic = mem.topics()[0]
    assert topic["name"] == "Local AI" and len(topic["members"]) == 2


def test_graph_signal_and_topics_section(mem):
    mem.record("I adopted Bruno from the shelter", "Lovely.")
    _extract(mem._conn, [1], [("Bruno", "thing"), ("Shelter", "place")], [("Bruno", "from", "Shelter")])
    mem.new_session()
    ctx = mem.context("How is Bruno?")
    assert "bruno" in ctx.sections["topics"] and "shelter" in ctx.sections["topics"]
    assert "Bruno" in ctx.sections["moments"]


def test_stale_entities_are_retired(mem):
    conn = mem._conn
    _extract(conn, [], [("Old thing", "thing")])
    conn.execute("UPDATE entities SET updated_at='2000-01-01T00:00:00+00:00'")
    assert graph.retire_stale(conn, 30.0) == 1
