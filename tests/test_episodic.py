"""Episodic memory: persistence, sessions, chunking, recall across sessions."""

from elastimem import Elastimem, ElastimemConfig
from elastimem.governor import GIB


def make_store(path, **cfg_kwargs):
    return Elastimem(str(path), config=ElastimemConfig(**cfg_kwargs),
                  probe_fn=lambda: (32 * GIB, 20 * GIB))


def test_record_turn_persists_messages_and_chunks(tmp_path):
    s = make_store(tmp_path / "e.db")
    s.record_turn("I'm planning a trip to Japan in October",
                  "Great! Autumn is a lovely season for Kyoto.")
    st = s.stats()
    assert st["messages"] == 2 and st["chunks"] == 1 and st["sessions"] == 1
    s.close()


def test_flagship_cross_session_recall(tmp_path):
    """Session 1 mentions the car; session 2 recalls it by paraphrase."""
    path = tmp_path / "car.db"
    a = make_store(path)
    a.record_turn("my car needs new brake pads soon, the fronts are squealing",
                  "Noted — worn brake pads squeal; get the fronts checked.")
    a.record_turn("also thinking about repainting the kitchen",
                  "Nice, what color?")
    a.end_session()
    a.close()

    b = make_store(path)
    hits = b.recall("what did we discuss about my car")
    assert hits, "expected episodic recall to find the brake-pad exchange"
    assert "brake pads" in hits[0].text
    # and it should surface in the prompt sections too
    plan = b.build_context("remind me what my car needed again?")
    assert "brake" in plan.sections["relevant_past_moments"]
    b.close()


def test_current_session_excluded_from_prompt_injection(tmp_path):
    s = make_store(tmp_path / "cur.db")
    s.record_turn("my dog is named Biscuit and he loves the beach",
                  "Biscuit sounds delightful!")
    plan = s.build_context("tell me something about my dog Biscuit")
    # It's in the live window already; injecting it back is duplication.
    assert "Biscuit" not in plan.sections["relevant_past_moments"]
    # but recall() (explicit search) does see it
    assert any("Biscuit" in h.text for h in s.recall("dog named Biscuit"))
    s.close()


def test_long_exchange_is_split_into_chunks(tmp_path):
    s = make_store(tmp_path / "long.db", chunk_target_tokens=50)
    long_reply = ". ".join(f"Sentence number {i} with several filler words in it"
                           for i in range(40))
    s.record_turn("please explain everything about sourdough baking", long_reply)
    assert s.stats()["chunks"] > 1
    s.close()


def test_rule_capture_stores_facts_inline(tmp_path):
    s = make_store(tmp_path / "r.db")
    s.record_turn("Hi! My name is Priya and I live in Mumbai. I'm allergic to peanuts.",
                  "Nice to meet you, Priya!")
    facts = s.facts()
    assert facts["name"] == "Priya"
    assert facts["location"] == "Mumbai"
    assert facts["allergies"] == "peanuts"
    s.record_turn("my favorite color is teal by the way", "Teal is lovely.")
    assert s.facts()["favorite_color"] == "teal"
    s.close()


def test_session_lifecycle_and_resume(tmp_path):
    path = tmp_path / "s.db"
    s = make_store(path)
    s.record_turn("let's plan my sister's birthday party", "Sure — when is it?")
    s.record_turn("it's on the 14th of March", "Got it, March 14th.")
    s.end_session()
    s.close()

    s2 = make_store(path)
    summary, tail = s2.resume_session()
    assert summary is None  # no LLM yet (phase 4 adds summaries)
    assert [m["role"] for m in tail] == ["user", "assistant", "user", "assistant"]
    assert "birthday" in tail[0]["content"]
    sessions = s2.sessions()
    assert sessions[0]["title"].startswith("let's plan my sister's")
    s2.close()


def test_orphan_sessions_closed_on_reopen(tmp_path):
    path = tmp_path / "o.db"
    s = make_store(path)
    s.record_turn("hello there, remember the blue bicycle", "Will do.")
    s.close()  # no end_session — simulates a crash

    s2 = make_store(path)
    assert all(sess["ended_at"] is not None for sess in s2.sessions())
    s2.close()


def test_recall_never_raises_on_weird_input(tmp_path):
    s = make_store(tmp_path / "w.db")
    assert s.recall('"; DROP TABLE facts; --') == []
    assert s.recall("") == []
    assert s.recall("!!! ??? ***") == []
    s.close()
