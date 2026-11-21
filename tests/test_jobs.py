import threading
import time

from conftest import StubLLM, toy_embed

import elastimem
from elastimem import Tier

PUPPY = ("I adopted a puppy named Bruno", "Lovely!")


def _open(path, tier=Tier.FULL, **kw):
    return elastimem.open(path, embedder=kw.pop("embedder", None), tier=tier, **kw)


def test_extraction_stores_grounded_facts_and_graph(path):
    llm = StubLLM(facts={"pet_name": "Bruno", "role": "AI assistant"},
                  entities=[{"name": "Bruno", "type": "thing"}])
    with _open(path, llm=llm) as mem:
        mem.record(*PUPPY)
        assert mem.flush()
        assert mem.facts() == {"pet_name": "Bruno"}
        assert [r["reason"] for r in mem.rejected()] == ["not grounded in the user's message"]
        assert mem._conn.execute("SELECT importance FROM chunks").fetchone()[0] == 0.8


def test_batched_cadence_waits_for_enough_turns(path):
    llm = StubLLM()
    with _open(path, Tier.STANDARD, llm=llm) as mem:
        mem.record("first turn here", "ok")
        time.sleep(0.3)
        assert llm.count("Extract") == 0
        mem.record("second turn here", "ok")
        deadline = time.time() + 5
        while llm.count("Extract") < 2 and time.time() < deadline:
            time.sleep(0.05)
        assert llm.count("Extract") == 2


def test_lite_extracts_only_at_session_end(path):
    llm = StubLLM()
    with _open(path, Tier.LITE, llm=llm) as mem:
        mem.record(*PUPPY)
        time.sleep(0.3)
        assert llm.count("Extract") == 0
        mem.end_session()
        assert llm.count("Extract") == 1


def test_no_llm_call_while_paused(path):
    llm = StubLLM()
    with _open(path, llm=llm) as mem:
        with mem.pause():
            mem.record(*PUPPY)
            time.sleep(0.3)
            assert llm.calls == []
        mem.flush()
        assert llm.count("Extract") == 1


def test_pause_can_end_on_another_thread(path):
    with _open(path, llm=StubLLM()) as mem:
        p = mem.pause()
        t = threading.Thread(target=p.end)
        t.start()
        t.join(2)
        assert mem.flush()


def test_failing_llm_is_not_retried_forever(path):
    llm = StubLLM(fail=True)
    with _open(path, llm=llm) as mem:
        mem.record(*PUPPY)
        assert mem.flush()
        assert llm.count("Extract") == 1
        assert mem.facts() == {}


def test_missed_extraction_resumes_after_restart(path):
    with _open(path) as mem:                 # no LLM: the turn is stored, not extracted
        mem.record(*PUPPY)
    llm = StubLLM(facts={"pet_name": "Bruno"})
    with _open(path, llm=llm) as mem:
        mem.flush()
        assert mem.facts() == {"pet_name": "Bruno"}


def test_vectors_backfill_when_an_embedder_appears(path):
    with _open(path, Tier.LITE) as mem:
        mem.record(*PUPPY)
    with _open(path, embedder=toy_embed) as mem:
        mem.flush()
        assert mem._conn.execute("SELECT count(*) FROM chunks WHERE embedding IS NULL").fetchone()[0] == 0


def test_facts_get_vectors_too(path):
    with _open(path, embedder=toy_embed) as mem:
        mem.remember("home_city", "Lisbon")
        mem.flush()
        assert mem._conn.execute("SELECT count(*) FROM facts WHERE embedding IS NULL").fetchone()[0] == 0


def test_llm_folds_the_window_when_the_tier_allows(path):
    llm = StubLLM()
    with _open(path, llm=llm) as mem:
        for i in range(mem.status.keep_turns + 1):
            mem.record(f"turn {i}", "ok")
        mem.flush()
        assert mem.context().summary == "A short summary."


def test_end_session_summarizes_and_consolidates(path):
    llm = StubLLM()
    with _open(path, llm=llm) as mem:
        mem.record("planning a trip", "ok")
        mem.record("to the mountains", "ok")
        mem.end_session()
        assert mem.sessions()[0]["summary"] == "A short summary."


def test_memory_store_works_across_threads():
    with elastimem.open(":memory:", embedder=toy_embed, tier=Tier.FULL,
                        llm=StubLLM(facts={"pet_name": "Bruno"})) as mem:
        mem.record(*PUPPY)
        assert mem.flush()
        assert mem.facts() == {"pet_name": "Bruno"}


def test_closing_inside_a_pause_does_not_hang(path):
    llm = StubLLM()
    mem = _open(path, llm=llm)
    mem.record(*PUPPY)
    done = threading.Event()
    with mem.pause():
        t = threading.Thread(target=lambda: (mem.close(), done.set()))
        t.start()
        t.join(3)
    assert done.is_set()
