"""Context assembly: sections, budgets, ordering, rendering."""

from elastimem import Elastimem, ElastimemConfig, Tier
from elastimem.governor import GIB
from elastimem import assembly


def make_store(tmp_path, avail_gib=20.0, **cfg_kwargs):
    cfg = ElastimemConfig(**cfg_kwargs)
    return Elastimem(str(tmp_path / "a.db"), config=cfg,
                  probe_fn=lambda: (32 * GIB, int(avail_gib * GIB)))


def test_facts_and_lessons_sections(tmp_path):
    s = make_store(tmp_path)
    s.remember("name", "Alex")
    s.remember("favorite_food", "ramen")
    s.add_lesson("web_search wants plain keywords, not full sentences")

    plan = s.build_context("what should I cook tonight for dinner?")
    facts = plan.sections[assembly.SECTION_FACTS]
    assert "- name: Alex" in facts
    assert "favorite_food: ramen" in facts
    assert "web_search" in plan.sections[assembly.SECTION_LESSONS]
    rendered = plan.render()
    assert "WHAT YOU KNOW ABOUT THE USER" in rendered
    assert rendered.index("name: Alex") < rendered.index("favorite_food")  # profile first
    s.close()


def test_facts_budget_truncates(tmp_path):
    # A microscopic context squeezes the facts budget to a few lines.
    s = make_store(tmp_path, context_tokens=1400)
    for i in range(80):
        s.remember(f"note_{i:02d}", f"some remembered detail number {i} with padding text")
    plan = s.build_context()
    lines = plan.sections[assembly.SECTION_FACTS].splitlines()
    assert 0 < len(lines) < 80
    used = sum(assembly.estimate_tokens(l) for l in lines)
    assert used <= plan.profile.budgets.facts
    s.close()


def test_injected_facts_are_touched(tmp_path):
    s = make_store(tmp_path)
    s.remember("name", "Alex")
    s.build_context("hello there my good friend")
    row = s._conn.execute("SELECT access_count FROM facts WHERE key='name'").fetchone()
    assert row["access_count"] == 1
    s.close()


def test_lite_tier_plan(tmp_path):
    s = make_store(tmp_path, avail_gib=1.0)
    assert s.profile.tier is Tier.LITE
    s.remember("name", "Alex")
    plan = s.build_context("tell me about my past conversations please")
    assert plan.sections[assembly.SECTION_EPISODIC] == ""
    assert plan.keep_last_n_turns >= 2
    s.close()


def test_empty_input_skips_retrieval(tmp_path):
    s = make_store(tmp_path)
    plan = s.build_context("")
    assert plan.sections[assembly.SECTION_EPISODIC] == ""
    s.close()


def test_short_meaningful_query_still_triggers_retrieval(tmp_path):
    """Regression test: min_query_words (4) used to gate BOTH the expensive
    LLM-extraction path and the cheap local retrieval path with the same
    threshold, so a short-but-real question like "my birthday?" (2 words)
    got zero episodic/fact retrieval - silently weaker recall on exactly the
    kind of short follow-up that's common in real conversation. Retrieval
    now uses its own much lower min_retrieval_query_words gate (1) so only
    genuinely empty input is skipped; a short query that matches something
    in the store must actually retrieve it."""
    s = make_store(tmp_path)
    s.record_turn("my birthday is on October 3rd", "Got it, October 3rd — noted!")
    s.drain(timeout=5)
    # episodic_section deliberately excludes the CURRENT session (it's for
    # past sessions, not the live one) - end this session first so the next
    # build_context() call is a genuine "later session asks a short
    # follow-up" scenario, matching real usage.
    s.end_session()
    s.begin_session()
    plan = s.build_context("my birthday?")
    assert plan.sections[assembly.SECTION_EPISODIC] != "", (
        "a short (2-word) but meaningful query should still trigger retrieval"
    )
    s.close()


def test_graph_context_section_populated_on_full_tier(tmp_path):
    from elastimem import graph

    s = make_store(tmp_path, tier_override=Tier.FULL)
    conn = s._conn
    j = graph.upsert_node(conn, "thing", "Jetson", confidence=1.0)
    t = graph.upsert_node(conn, "thing", "Tuffy", confidence=1.0)
    graph.upsert_edge(conn, t, j, "runs_on", confidence=1.0)
    graph.store_clusters(conn, graph.compute_clusters(conn))
    conn.execute("UPDATE graph_nodes SET cluster_label='Local AI' WHERE cluster_id IS NOT NULL")

    plan = s.build_context("tell me about my Jetson")
    graph_text = plan.sections[assembly.SECTION_GRAPH_CONTEXT]
    assert "Local AI" in graph_text
    assert "jetson" in graph_text and "tuffy" in graph_text
    assert "RELATED TOPICS" in plan.render()
    s.close()


def test_graph_context_section_populated_at_lite_tier(tmp_path):
    """LITE tier now carries a small nonzero episodic budget (graph_hops=1,
    up from the old 0), so graph context — carved out of the episodic
    budget — is reachable here too, just smaller than STANDARD/FULL."""
    from elastimem import graph

    s = make_store(tmp_path, avail_gib=1.0)
    assert s.profile.tier is Tier.LITE
    conn = s._conn
    j = graph.upsert_node(conn, "thing", "Jetson")
    t = graph.upsert_node(conn, "thing", "Tuffy")
    graph.upsert_edge(conn, t, j, "runs_on")

    plan = s.build_context("tell me about my Jetson")
    graph_text = plan.sections[assembly.SECTION_GRAPH_CONTEXT]
    assert "jetson" in graph_text
    s.close()


def test_graph_context_section_empty_when_no_seed_matches(tmp_path):
    from elastimem import graph

    s = make_store(tmp_path, tier_override=Tier.FULL)
    graph.upsert_node(s._conn, "thing", "Jetson")

    plan = s.build_context("totally unrelated query about nothing stored")
    assert plan.sections[assembly.SECTION_GRAPH_CONTEXT] == ""
    s.close()


def test_graph_context_section_unclustered_entities_still_shown(tmp_path):
    """An entity with no cluster (below min_size, or clustering never ran)
    still surfaces in graph context, just without a topic label."""
    from elastimem import graph

    s = make_store(tmp_path, tier_override=Tier.FULL)
    graph.upsert_node(s._conn, "thing", "Jetson", confidence=1.0)

    plan = s.build_context("tell me about my Jetson")
    graph_text = plan.sections[assembly.SECTION_GRAPH_CONTEXT]
    assert "jetson" in graph_text
    s.close()


def test_build_context_never_raises_when_graph_leg_fails(tmp_path, monkeypatch):
    from elastimem import graph

    s = make_store(tmp_path, tier_override=Tier.FULL)
    graph.upsert_node(s._conn, "thing", "Jetson")

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(graph, "detect_seed_nodes", _boom)
    plan = s.build_context("tell me about my Jetson")
    assert plan.sections[assembly.SECTION_GRAPH_CONTEXT] == ""
    assert plan.profile is not None
    s.close()


def test_custom_tokenizer_is_used(tmp_path):
    calls = []

    def tok(text):
        calls.append(text)
        return len(text)  # brutal: 1 token per char → tiny effective budgets

    s = Elastimem(str(tmp_path / "t.db"), tokenizer_fn=tok,
               probe_fn=lambda: (32 * GIB, 20 * GIB))
    s.remember("name", "Alexandra von Longname")
    s.build_context()
    assert calls  # tokenizer consulted
    s.close()
