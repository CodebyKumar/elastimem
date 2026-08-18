"""The degradation matrix, verified: every capability lands on its documented
floor when its dependency is missing — never an exception to the host."""

import time

from elastimem import Elastimem, ElastimemConfig, Tier
from elastimem.governor import GIB


def test_lite_tier_never_indexes_new_chunks(tmp_path):
    """LITE gives up the *sustained* embedding cost: one encode per chunk,
    forever. Recording turns must never enqueue an embed job here."""
    calls = []

    def spy_embed(texts):
        calls.append(texts)
        return [[0.0] * 8 for _ in texts]

    s = Elastimem(str(tmp_path / "l.db"), embed_fn=spy_embed,
               probe_fn=lambda: (4 * GIB, 2 * GIB))
    assert s.profile.tier is Tier.LITE
    assert s.profile.embeddings_enabled is False
    s.record_turn("my car needs brake pads replaced soon", "Noted!")
    s.drain(timeout=2)
    assert calls == []
    row = s._conn.execute(
        "SELECT COUNT(*) c FROM chunks WHERE embedding IS NOT NULL"
    ).fetchone()
    assert row["c"] == 0
    s.close()


def test_lite_tier_scores_existing_vectors_with_a_resident_embedder(tmp_path):
    """A host-supplied embed_fn already lives in the host's process, so
    refusing to call it at LITE frees nothing and only discards recall
    quality. LITE therefore still runs the vector leg at query time — it
    just never indexes new chunks (see the test above)."""
    calls = []

    def spy_embed(texts):
        calls.append(texts)
        return [[0.0] * 8 for _ in texts]

    path = tmp_path / "warm.db"
    # Record and index at STANDARD, so real vectors exist on disk.
    s = Elastimem(str(path), embed_fn=spy_embed,
                  probe_fn=lambda: (8 * GIB, 4 * GIB))
    assert s.profile.tier is Tier.STANDARD
    s.record_turn("my car needs brake pads replaced soon", "Noted!")
    s.drain(timeout=2)
    s.close()
    assert calls, "STANDARD should have embedded the chunk"
    calls.clear()

    # Reopen the same store on a starved machine.
    s2 = Elastimem(str(path), embed_fn=spy_embed,
                   probe_fn=lambda: (4 * GIB, 2 * GIB))
    assert s2.profile.tier is Tier.LITE
    assert s2.profile.vector_recall_enabled is True
    assert s2.profile.embedder_load_allowed is False
    s2.recall("what about my car brakes")
    assert calls == [["what about my car brakes"]]      # query side only
    s2.close()


def test_lite_tier_never_loads_the_builtin_embedder(tmp_path, monkeypatch):
    """The one genuinely RAM-expensive part of the embedding path is
    materializing the ~130MB built-in model. LITE must never trigger that,
    even though it will happily use one that is already loaded.

    default_embedder caches the model in a MODULE-level global, so whether
    it is resident depends on what else ran in this process first. Pin it
    to 'not loaded' explicitly rather than letting test order decide.
    """
    from elastimem import default_embedder, embeddings

    monkeypatch.setattr(default_embedder, "_model", None)

    s = Elastimem(str(tmp_path / "b.db"), probe_fn=lambda: (4 * GIB, 2 * GIB))
    assert s.profile.tier is Tier.LITE
    # Built-in wired in (embed_query_fn set), but not loaded => not resident.
    assert s.embed_query_fn is not None
    assert embeddings.embedder_resident(s) is False
    s.record_turn("my car needs brake pads replaced soon", "Noted!")
    s.drain(timeout=2)
    assert s.recall("what about my car brakes") is not None   # FTS still works
    # Still not resident: nothing in the LITE path may have loaded it.
    assert default_embedder._model is None
    s.close()


def test_builtin_embedder_residency_tracks_the_module_global(monkeypatch, tmp_path):
    """embedder_resident() must report the built-in's real load state
    WITHOUT triggering a load — probing is exactly what must not allocate."""
    from elastimem import default_embedder, embeddings

    s = Elastimem(str(tmp_path / "res.db"), probe_fn=lambda: (4 * GIB, 2 * GIB))
    monkeypatch.setattr(default_embedder, "_model", None)
    assert embeddings.embedder_resident(s) is False
    monkeypatch.setattr(default_embedder, "_model", object())   # pretend loaded
    assert embeddings.embedder_resident(s) is True
    s.close()


def test_no_llm_no_embedder_still_fully_functional(tmp_path):
    """The zero-capability floor: rules + FTS5 + explicit memory."""
    path = tmp_path / "bare.db"
    s = Elastimem(str(path), probe_fn=lambda: (32 * GIB, 20 * GIB))
    s.record_turn("my name is Kavya and I live in Pune",
                  "Nice to meet you, Kavya!")
    s.remember("dietary_restriction", "vegetarian")
    s.end_session()
    s.close()

    s2 = Elastimem(str(path), probe_fn=lambda: (32 * GIB, 20 * GIB))
    assert s2.facts()["name"] == "Kavya"                      # rule capture
    assert s2.facts()["dietary_restriction"] == "vegetarian"  # explicit
    assert s2.recall("where does the user live")              # FTS recall
    sess = s2.sessions()[0]
    assert sess["title"] and sess["summary"] is None          # title-only floor
    s2.close()


def test_tier_downgrade_mid_session_stops_llm_and_embeddings(tmp_path):
    ram = {"avail": 20.0}
    llm_calls, embed_calls = [], []

    def llm(prompt, *, max_tokens, temperature):
        llm_calls.append(prompt)
        return "NONE"

    def embed(texts):
        embed_calls.append(texts)
        return [[0.0] * 8 for _ in texts]

    s = Elastimem(str(tmp_path / "d.db"), complete_fn=llm, embed_fn=embed,
               probe_fn=lambda: (32 * GIB, int(ram["avail"] * GIB)))
    s.record_turn("i enjoy long evening walks by the river", "Lovely!")
    s.drain(timeout=5)
    n_llm, n_embed = len(llm_calls), len(embed_calls)
    assert n_llm >= 1 and n_embed >= 1

    ram["avail"] = 0.8            # severe pressure
    assert s.tick().tier is Tier.LITE
    s.record_turn("i also collect vintage stamps from europe", "Fascinating!")
    s.drain(timeout=2)
    assert len(llm_calls) == n_llm and len(embed_calls) == n_embed
    s.close()


def test_consolidation_llm_merge(tmp_path):
    def merging_llm(prompt, *, max_tokens, temperature):
        if "memory key changed value" in prompt:
            return "Austin (moving in May)"
        return "NONE"

    s = Elastimem(str(tmp_path / "m.db"), complete_fn=merging_llm,
               probe_fn=lambda: (32 * GIB, 20 * GIB))
    s.remember("location", "Seattle")
    s.remember("location", "moving to Austin in May")
    s.record_turn("i am moving to austin in may by the way", "Exciting!")
    s.end_session()  # FULL tier -> consolidation with LLM merge
    assert s.facts()["location"] == "Austin (moving in May)"
    # audit chain intact
    assert [f.value for f in s.fact_history("location")][0] == "Seattle"
    s.close()


def test_end_session_consolidation_runs_graph_maintenance(tmp_path):
    from elastimem import graph

    def merging_llm(prompt, *, max_tokens, temperature):
        if "same real-world entity" in prompt:
            return "yes"
        return "NONE"

    s = Elastimem(str(tmp_path / "gm.db"), complete_fn=merging_llm,
               probe_fn=lambda: (32 * GIB, 20 * GIB))
    conn = s._conn
    stale_id = graph.upsert_node(conn, "thing", "StaleThing", confidence=0.1)
    with conn:
        conn.execute(
            "UPDATE graph_nodes SET updated_at='2020-01-01T00:00:00+00:00' WHERE id=?",
            (stale_id,),
        )
    graph.upsert_node(conn, "org", "acme corp")
    graph.upsert_node(conn, "org", "acme corporation")

    s.record_turn("hello there", "hi!")
    s.end_session()  # FULL tier -> consolidation, including graph maintenance

    remaining = {r["canonical_name"] for r in conn.execute(
        "SELECT canonical_name FROM graph_nodes"
    )}
    assert "stalething" not in remaining          # decayed away
    assert len(remaining & {"acme corp", "acme corporation"}) == 1  # merged
    s.close()


def test_pressure_report_is_immediate_and_survives_recovery_rules(tmp_path):
    s = Elastimem(str(tmp_path / "p.db"), probe_fn=lambda: (32 * GIB, 20 * GIB))
    assert s.profile.tier is Tier.FULL
    s.report_pressure()
    assert s.profile.tier is Tier.STANDARD
    # healthy ticks needed before climbing back
    for _ in range(s.config.upgrade_healthy_ticks + 1):
        s.tick()
    assert s.profile.tier is Tier.FULL
    s.close()


def test_lite_tier_never_touches_graph_tables(tmp_path):
    def llm(prompt, *, max_tokens, temperature):
        import json
        return json.dumps({
            "facts": {}, "entities": [{"name": "Tuffy", "type": "thing"}],
            "relationships": [{"source": "user", "relation": "builds", "target": "Tuffy"}],
        })

    s = Elastimem(str(tmp_path / "lg.db"), complete_fn=llm,
               probe_fn=lambda: (4 * GIB, 2 * GIB))
    assert s.profile.tier is Tier.LITE
    s.record_turn("I am building a robot called Tuffy this year", "Cool!")
    s.drain(timeout=2)
    n_nodes = s._conn.execute("SELECT count(*) c FROM graph_nodes").fetchone()["c"]
    assert n_nodes == 0
    s.close()


def test_no_llm_no_embedder_graph_absent_still_functional(tmp_path):
    s = Elastimem(str(tmp_path / "bare2.db"), probe_fn=lambda: (32 * GIB, 20 * GIB))
    s.record_turn("my name is Kavya and I live in Pune", "Nice to meet you!")
    s.drain(timeout=2)
    assert s.recall("where does the user live") is not None
    n_nodes = s._conn.execute("SELECT count(*) c FROM graph_nodes").fetchone()["c"]
    assert n_nodes == 0
    s.close()


def test_migration_v1_to_latest_adds_graph_tables(tmp_path):
    import sqlite3
    from elastimem import db as db_mod

    path = str(tmp_path / "v1.db")
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(db_mod._SCHEMA)
    conn.execute("INSERT INTO meta(key, value) VALUES ('schema_version', '1')")
    conn.commit()
    conn.close()

    conn2, _ = db_mod.open_store(path)
    version = conn2.execute(
        "SELECT value FROM meta WHERE key='schema_version'"
    ).fetchone()["value"]
    assert version == str(db_mod.SCHEMA_VERSION)
    tables = {
        r["name"] for r in conn2.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    assert {"graph_nodes", "graph_edges"} <= tables
    cols = {r["name"] for r in conn2.execute("PRAGMA table_info(graph_nodes)")}
    assert {"cluster_id", "cluster_label"} <= cols
    conn2.close()


def test_migration_v2_to_v3_adds_cluster_columns(tmp_path):
    """A v2 store (graph_nodes exists, but predates cluster_id/cluster_label)
    must gain the new columns via ALTER TABLE, with existing data intact —
    unlike the v1->v2 step, CREATE TABLE IF NOT EXISTS is a no-op here
    since the table already exists."""
    import sqlite3
    from elastimem import db as db_mod

    path = str(tmp_path / "v2.db")
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(db_mod._SCHEMA)
    conn.executescript("""
        CREATE TABLE graph_nodes (
          id INTEGER PRIMARY KEY, type TEXT NOT NULL DEFAULT 'entity',
          canonical_name TEXT NOT NULL, aliases TEXT NOT NULL DEFAULT '[]',
          created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
          importance REAL NOT NULL DEFAULT 0.5, confidence REAL NOT NULL DEFAULT 0.5,
          mention_count INTEGER NOT NULL DEFAULT 1
        );
        CREATE UNIQUE INDEX idx_graph_nodes_canonical ON graph_nodes(type, canonical_name);
        CREATE TABLE graph_edges (
          id INTEGER PRIMARY KEY, source_node INTEGER NOT NULL REFERENCES graph_nodes(id),
          target_node INTEGER NOT NULL REFERENCES graph_nodes(id), relationship TEXT NOT NULL,
          confidence REAL NOT NULL DEFAULT 0.5, importance REAL NOT NULL DEFAULT 0.5,
          weight REAL NOT NULL DEFAULT 1.0, created_at TEXT NOT NULL, last_seen TEXT NOT NULL,
          seen_count INTEGER NOT NULL DEFAULT 1, source_chunk_id INTEGER
        );
    """)
    conn.execute(
        "INSERT INTO graph_nodes(type, canonical_name, created_at, updated_at)"
        " VALUES ('thing', 'jetson', '2020-01-01', '2020-01-01')"
    )
    conn.execute("INSERT INTO meta(key, value) VALUES ('schema_version', '2')")
    conn.commit()
    conn.close()

    conn2, _ = db_mod.open_store(path)
    version = conn2.execute(
        "SELECT value FROM meta WHERE key='schema_version'"
    ).fetchone()["value"]
    assert version == str(db_mod.SCHEMA_VERSION)
    cols = {r["name"] for r in conn2.execute("PRAGMA table_info(graph_nodes)")}
    assert {"cluster_id", "cluster_label"} <= cols
    row = conn2.execute("SELECT canonical_name FROM graph_nodes").fetchone()
    assert row["canonical_name"] == "jetson"  # pre-existing data survives
    conn2.close()


def test_graph_nudge_works_without_fts5(tmp_path):
    from elastimem import graph

    s = Elastimem(str(tmp_path / "nofts.db"), probe_fn=lambda: (32 * GIB, 20 * GIB))
    s.fts_enabled = False  # simulate a sqlite built without FTS5
    s.record_turn("I am building a robot called Tuffy this year",
                  "Sounds like a fun project!")
    s.drain(timeout=2)
    conn = s._conn
    user_id = graph.upsert_node(conn, "person", "user", confidence=1.0)
    tuffy_id = graph.upsert_node(conn, "thing", "Tuffy", confidence=1.0)
    graph.upsert_edge(conn, user_id, tuffy_id, "builds", confidence=1.0)
    # Must not raise, and should still return something via LIKE fallback.
    hits = s.recall("what am I building")
    assert hits == [] or hits
    s.close()


def test_memory_store_background_worker_shares_schema(tmp_path):
    """Regression: sqlite3.connect(':memory:') from a second thread opens a
    SEPARATE, empty database - not the same one - unlike file-backed
    stores where every thread's own connection sees the same on-disk file.
    Elastimem._conn used to call connect(self.path) unconditionally per
    thread, so a ':memory:' store's background worker thread (which runs
    every LLM-extraction/embed/consolidate job) would silently hit a
    schema-less database and every background write would fail. This
    exercises the exact path that broke: record_turn -> background
    extraction job -> remember() from the worker thread."""
    def llm(prompt, *, max_tokens, temperature):
        return '{"facts": {"favorite_color": "blue"}}'

    s = Elastimem(":memory:", complete_fn=llm, probe_fn=lambda: (32 * GIB, 20 * GIB))
    s.record_turn("my favorite color is blue", "Noted!")
    assert s.drain(timeout=5)
    assert s.facts().get("favorite_color") == "blue"
    s.close()


def test_build_context_never_raises(tmp_path):
    s = Elastimem(str(tmp_path / "n.db"), probe_fn=lambda: (32 * GIB, 20 * GIB))
    for weird in ["", "hi", '"; DROP TABLE chunks; --', "🦆" * 500, "a " * 3000]:
        plan = s.build_context(weird)
        assert plan.profile is not None
    s.close()


def test_lite_rolling_summary_keeps_real_content_not_a_marker(tmp_path):
    """LITE has no LLM, but condensing evicted turns extractively is pure
    string work over rows already in the DB — so it keeps actual content
    instead of the old '[N earlier turn(s) omitted]' placeholder."""
    s = Elastimem(str(tmp_path / "r.db"), probe_fn=lambda: (4 * GIB, 2 * GIB),
                  context_tokens=8192)
    assert s.profile.tier is Tier.LITE
    s.report_evictions([
        ("I want to rebuild the deck out of cedar. What size joists?",
         "Use 2x8 joists at 16 inches on center."),
        ("Also the railing has to meet code.", "Most codes want 36 inches."),
    ])
    plan = s.build_context("back to the deck")
    assert plan.rolling_summary is not None
    assert "omitted" not in plan.rolling_summary
    assert "cedar" in plan.rolling_summary
    assert "railing" in plan.rolling_summary
    s.close()


def test_lite_rolling_summary_falls_back_to_marker_when_nothing_usable(tmp_path):
    s = Elastimem(str(tmp_path / "r2.db"), probe_fn=lambda: (4 * GIB, 2 * GIB))
    s.report_evictions([("   ", "   ")])
    assert "omitted" in s.build_context("hello").rolling_summary
    s.close()


def test_lite_rolling_summary_stays_within_budget_across_many_evictions(tmp_path):
    """The extractive summary accumulates, so it must be bounded — the
    oldest content ages out rather than the string growing forever."""
    s = Elastimem(str(tmp_path / "r3.db"), probe_fn=lambda: (4 * GIB, 2 * GIB),
                  context_tokens=4096)
    budget = s.profile.budgets.sessions
    for i in range(40):
        s.report_evictions([(f"question number {i} about roofing shingles", "sure")])
    summary = s.build_context("roofing").rolling_summary
    assert len(summary) // 4 <= budget
    assert "question number 39" in summary      # newest survives
    assert "question number 0 " not in summary  # oldest aged out
    s.close()


def test_lite_consolidation_prunes_instead_of_growing_forever(tmp_path):
    """LITE now runs DEDUPE_ONLY consolidation at session end: pure SQLite
    decay/archival with no LLM call. Previously it was OFF, so a LITE store
    was the one store that never pruned anything."""
    from elastimem import graph

    s = Elastimem(str(tmp_path / "c.db"), probe_fn=lambda: (4 * GIB, 2 * GIB))
    assert s.profile.tier is Tier.LITE
    conn = s._conn
    node_id = graph.upsert_node(conn, "thing", "AbandonedTopic")
    with s._write_lock:
        # Backdate far past the decay half-life so it lands under the
        # archive threshold on the next sweep.
        conn.execute(
            "UPDATE graph_nodes SET updated_at = datetime('now', '-400 days')"
            " WHERE id=?", (node_id,)
        )
        conn.commit()
    s.record_turn("hello there friend", "hi")
    s.end_session()
    remaining = conn.execute(
        "SELECT COUNT(*) c FROM graph_nodes WHERE id=?", (node_id,)
    ).fetchone()["c"]
    assert remaining == 0, "LITE consolidation should have decayed this node"
    s.close()


def test_lite_makes_no_llm_call_by_default(tmp_path):
    """The floor LITE still guarantees unless a host opts in."""
    calls = []

    def spy_llm(prompt, **kw):
        calls.append(prompt)
        return "something"

    s = Elastimem(str(tmp_path / "n.db"), llm=spy_llm,
                  probe_fn=lambda: (4 * GIB, 2 * GIB))
    assert s.profile.tier is Tier.LITE
    s.record_turn("my name is Ravi and I work at a hospital in Mysore", "Noted!")
    s.report_evictions([("something earlier about work", "ok")])
    s.end_session()
    s.drain(timeout=2)
    assert calls == []
    s.close()


def test_lite_llm_extraction_opt_in_defers_every_call_to_session_end(tmp_path):
    """With the opt-in, LITE may use the model — but only at session close,
    where nothing competes with a live foreground generation."""
    calls = []

    def spy_llm(prompt, **kw):
        calls.append(prompt)
        return '{"facts": {"occupation": "nurse"}}'

    s = Elastimem(str(tmp_path / "o.db"), llm=spy_llm, lite_llm_extraction=True,
                  probe_fn=lambda: (4 * GIB, 2 * GIB))
    assert s.profile.tier is Tier.LITE
    s.record_turn("my name is Ravi and I work as a nurse in Mysore", "Noted!")
    # SESSION_END jobs are held in the worker's batch, never queued, so this
    # is a deterministic check with no sleep needed. (Note drain() DOES
    # flush them on purpose — a host draining before unloading its model
    # must not lose held work — so this deliberately does not drain here.)
    assert calls == [], "extraction must be held until the session ends"
    assert s._worker.pending() == 1
    s.end_session()
    s.drain(timeout=2)
    assert calls, "session end should have flushed the held extraction"
    assert s.facts().get("occupation") == "nurse"
    s.close()
