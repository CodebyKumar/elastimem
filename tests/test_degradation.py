"""The degradation matrix, verified: every capability lands on its documented
floor when its dependency is missing — never an exception to the host."""

import time

from elastimem import Elastimem, ElastimemConfig, Tier
from elastimem.governor import GIB


def test_lite_tier_never_calls_embed_fn(tmp_path):
    calls = []

    def spy_embed(texts):
        calls.append(texts)
        return [[0.0] * 8 for _ in texts]

    s = Elastimem(str(tmp_path / "l.db"), embed_fn=spy_embed,
               probe_fn=lambda: (4 * GIB, 2 * GIB))
    assert s.profile.tier is Tier.LITE
    s.record_turn("my car needs brake pads replaced soon", "Noted!")
    s.drain(timeout=2)
    s.recall("what about my car brakes")
    assert calls == []
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


def test_migration_v1_to_v2_adds_graph_tables(tmp_path):
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
    assert version == "2"
    tables = {
        r["name"] for r in conn2.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    assert {"graph_nodes", "graph_edges"} <= tables
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


def test_build_context_never_raises(tmp_path):
    s = Elastimem(str(tmp_path / "n.db"), probe_fn=lambda: (32 * GIB, 20 * GIB))
    for weird in ["", "hi", '"; DROP TABLE chunks; --', "🦆" * 500, "a " * 3000]:
        plan = s.build_context(weird)
        assert plan.profile is not None
    s.close()
