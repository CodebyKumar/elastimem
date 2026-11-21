import pytest

import elastimem
from elastimem import Tier
from elastimem.config import normalize_key
from elastimem.storage import facts


def test_changing_a_fact_keeps_its_history(mem):
    assert mem.remember("occupation", "designer")
    assert mem.remember("occupation", "writer")
    assert not mem.remember("occupation", "writer")          # same value: no new version
    assert mem.facts() == {"occupation": "writer"}
    history = mem.history("occupation")
    assert [f.value for f in history] == ["designer", "writer"]
    assert history[0].until is not None and history[1].until is None


def test_forget_closes_the_current_version(mem):
    mem.remember("city", "Pune")
    assert mem.forget("city")
    assert mem.facts() == {}
    assert [f.value for f in mem.history("city")] == ["Pune"]
    assert not mem.forget("city")


def test_history_accepts_a_question(mem):
    mem.remember("occupation", "AI engineer")
    assert [f.value for f in mem.history("what engineer job do I have?")] == ["AI engineer"]
    assert mem.history("nothing like that") == []


@pytest.mark.parametrize("value, user, ok", [
    ("Bruno", "I adopted a puppy named Bruno", True),
    ("unknown", "I adopted a puppy named Bruno", False),           # placeholder
    ("an AI assistant", "who are you?", False),                    # about the agent
    ("Lisboa", "Eu moro em Lisboa agora", True),                   # any language
    ("I adopted a puppy named Bruno last week",
     "I adopted a puppy named Bruno last week", False),            # copy of the message
])
def test_grounding_check(mem, value, user, ok):
    changed, reason = facts.store(mem._conn, "k", value, "auto", user_text=user)
    assert changed is ok, reason
    assert bool(mem.rejected()) is not ok


def test_explicit_facts_skip_grounding(mem):
    assert mem.remember("note", "something the user never typed")


def test_blocked_keys(path):
    with elastimem.open(path, embedder=None, tier=Tier.FULL, blocked={"model"}) as mem:
        assert not mem.remember("model", "x")
        assert not mem.remember("preferred_model", "x")
        assert mem.remember("modeling_hobby", "clay")


def test_normalize_key():
    assert normalize_key("Favorite Color") == "favorite_color"
    assert normalize_key("  ciudad natal ") == "ciudad_natal"
    assert normalize_key("café-name") == "café_name"


def test_pinned_and_explicit_facts_lead_the_section(path):
    with elastimem.open(path, embedder=None, tier=Tier.FULL, pinned={"name"}) as mem:
        facts.store(mem._conn, "hobby", "chess", "auto", user_text="I play chess")
        mem.remember("color", "teal")
        mem.remember("name", "Asha")
        lines = mem.context("hello there").sections["facts"].splitlines()
        assert lines == ["- name: Asha", "- color: teal", "- hobby: chess"]


def test_unused_auto_facts_fade(mem):
    conn = mem._conn
    facts.store(conn, "hobby", "chess", "auto", user_text="I play chess")
    mem.remember("color", "teal")
    conn.execute("UPDATE facts SET since='2000-01-01T00:00:00+00:00'")
    conn.commit()
    assert facts.archive_stale(conn, 30.0) == 1          # explicit facts never fade
    assert mem.facts() == {"color": "teal"}
