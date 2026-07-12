"""Context assembly: sections, budgets, ordering, rendering."""

from engram import Engram, EngramConfig, Tier
from engram.governor import GIB
from engram import assembly


def make_store(tmp_path, avail_gib=20.0, **cfg_kwargs):
    cfg = EngramConfig(**cfg_kwargs)
    return Engram(str(tmp_path / "a.db"), config=cfg,
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


def test_short_input_skips_retrieval(tmp_path):
    s = make_store(tmp_path)
    plan = s.build_context("hi")   # under min_query_words
    assert plan.sections[assembly.SECTION_EPISODIC] == ""
    s.close()


def test_custom_tokenizer_is_used(tmp_path):
    calls = []

    def tok(text):
        calls.append(text)
        return len(text)  # brutal: 1 token per char → tiny effective budgets

    s = Engram(str(tmp_path / "t.db"), tokenizer_fn=tok,
               probe_fn=lambda: (32 * GIB, 20 * GIB))
    s.remember("name", "Alexandra von Longname")
    s.build_context()
    assert calls  # tokenizer consulted
    s.close()
