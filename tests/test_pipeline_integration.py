"""End-to-end integration: the full knowledge-graph stack working together
in one place — extraction, graph storage, retrieval nudging, maintenance
(decay/dedup/clustering), explain(), timeline(), and the graph-context
section in build_context(). Individual modules already have thorough unit
coverage elsewhere; this file exists to catch integration bugs a
module-by-module test suite can't (wrong data threaded between stages,
wrong call order, a working component that breaks once real data flows
through the ones around it — see the :memory: worker-thread bug this style
of test caught during development).
"""

import json

from elastimem import Elastimem, ElastimemConfig
from elastimem.governor import GIB, Tier


def _fake_llm(prompt, *, max_tokens, temperature):
    # Order matters: check the most specific prompts first, since several
    # share incidental substrings (e.g. every extraction-shaped prompt
    # mentions "facts").
    if "same real-world entity" in prompt:
        return "no"
    if "topic name" in prompt:
        return "Local AI"
    if "memory key changed value" in prompt:
        # Decline to merge: echo the NEW value back unchanged (the real
        # contract is "return the value to keep", not yes/no — see
        # extraction._MERGE_PROMPT). Returning the new value verbatim
        # means consolidate()'s `merged != new_value` check is False, so
        # no update happens - the correct behavior when nothing actually
        # needs merging.
        new_value = prompt.rsplit("NEW: ", 1)[-1].strip()
        return new_value
    if "entities" in prompt and "relationships" in prompt:
        return json.dumps({
            "facts": {"occupation": "AI Engineer"},
            "entities": [
                {"name": "Tuffy", "type": "thing"},
                {"name": "Jetson", "type": "thing"},
            ],
            "relationships": [
                {"source": "user", "relation": "builds", "target": "Tuffy"},
                {"source": "Tuffy", "relation": "runs_on", "target": "Jetson"},
            ],
        })
    return "NONE"


def _make_store(path):
    return Elastimem(
        str(path), complete_fn=_fake_llm, embed_fn=None,
        config=ElastimemConfig(disable_builtin_embedder=True),
        probe_fn=lambda: (32 * GIB, 20 * GIB),
    )


def test_full_pipeline_end_to_end(tmp_path):
    s = _make_store(tmp_path / "pipeline.db")
    assert s.profile.tier is Tier.FULL

    # Semantic memory: versioned facts, independent of the graph work.
    s.remember("occupation", "Student")
    s.remember("occupation", "Designer")

    # One exchange that yields both a fact and graph entities/relationships
    # via the SAME background extraction job.
    s.record_turn(
        "I am building a robot called Tuffy that runs on a Jetson",
        "That sounds like a fun project!",
    )
    assert s.drain(timeout=5)

    # Consolidation: decay, dedup, clustering + labeling, all FULL tier.
    s.end_session()

    # -- graph storage actually happened --
    conn = s._conn
    node_names = {
        r["canonical_name"] for r in conn.execute("SELECT canonical_name FROM graph_nodes")
    }
    assert {"tuffy", "jetson"} <= node_names
    edge_count = conn.execute("SELECT count(*) c FROM graph_edges").fetchone()["c"]
    assert edge_count >= 1

    # -- clustering ran and produced a labeled cluster --
    clusters = s.clusters()
    assert clusters, "expected at least one cluster after consolidation"
    assert clusters[0]["label"] == "Local AI"
    assert {"tuffy", "jetson"} <= set(clusters[0]["members"])

    # -- retrieval surfaces the graph-connected memory --
    hits = s.recall("tell me about my Jetson")
    assert hits
    assert any("Tuffy" in h.text for h in hits)

    # -- explain() shows the traversal that produced that result --
    explanation = s.explain("tell me about my Jetson")
    traversed_names = {step.canonical_name for step in explanation.graph_traversal}
    assert {"jetson", "tuffy"} <= traversed_names

    # -- timeline() resolves free text to the right fact key --
    # (the background extraction pass also stores "occupation": "AI
    # Engineer" from the fake LLM's facts payload, on top of the two
    # explicit remember() calls above, so the full chain is three deep.)
    timeline = s.timeline("what did I do before AI")
    assert timeline.key == "occupation"
    assert [v.value for v in timeline.versions] == ["Student", "Designer", "AI Engineer"]

    # -- build_context() surfaces the RELATED TOPICS section --
    plan = s.build_context("tell me about my Jetson")
    assert plan.sections["graph_context"]
    assert "Local AI" in plan.sections["graph_context"]
    rendered = plan.render()
    assert "RELATED TOPICS" in rendered

    s.close()


def test_full_pipeline_survives_reopen(tmp_path):
    """Everything the pipeline wrote (facts, graph, clusters) must persist
    across a close/reopen, same as the base store guarantees."""
    path = tmp_path / "reopen.db"
    s = _make_store(path)
    s.record_turn(
        "I am building a robot called Tuffy that runs on a Jetson",
        "Nice!",
    )
    assert s.drain(timeout=5)
    s.end_session()
    s.close()

    s2 = Elastimem(str(path), probe_fn=lambda: (32 * GIB, 20 * GIB))
    node_names = {
        r["canonical_name"] for r in s2._conn.execute("SELECT canonical_name FROM graph_nodes")
    }
    assert {"tuffy", "jetson"} <= node_names
    assert s2.clusters()
    s2.close()


def test_full_pipeline_lite_tier_touches_nothing_graph_related(tmp_path):
    """LITE tier must produce zero graph writes and empty results from
    every new API, without raising."""
    s = Elastimem(
        str(tmp_path / "lite.db"), complete_fn=_fake_llm,
        probe_fn=lambda: (4 * GIB, 2 * GIB),
    )
    assert s.profile.tier is Tier.LITE
    s.record_turn(
        "I am building a robot called Tuffy that runs on a Jetson", "Cool!",
    )
    assert s.drain(timeout=2)

    n_nodes = s._conn.execute("SELECT count(*) c FROM graph_nodes").fetchone()["c"]
    assert n_nodes == 0
    assert s.clusters() == []
    explanation = s.explain("tell me about my Jetson")
    assert explanation.graph_traversal == ()
    plan = s.build_context("tell me about my Jetson")
    assert plan.sections["graph_context"] == ""
    s.close()


def test_full_pipeline_via_memory_store(tmp_path):
    """The exact scenario that caught the :memory: background-worker bug —
    kept here as a standing regression guard at the integration level, on
    top of the dedicated unit-level regression test in test_degradation.py."""
    s = _make_store(":memory:")
    s.record_turn(
        "I am building a robot called Tuffy that runs on a Jetson", "Cool!",
    )
    assert s.drain(timeout=5)
    s.end_session()
    assert s.clusters()
    s.close()
