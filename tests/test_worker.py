"""Background worker: extraction jobs, foreground gate, cadence, drain,
rolling/session summaries — all with a fake LLM."""

import json
import threading
import time

from elastimem import Elastimem, ElastimemConfig, Tier
from elastimem.governor import GIB


class FakeLLM:
    """Deterministic complete_fn that records calls and thread names."""

    def __init__(self, fact_json=None):
        self.calls = []
        self.lock = threading.Lock()
        self.fact_json = fact_json or {}
        self.active = 0
        self.max_concurrent = 0

    def __call__(self, prompt, *, max_tokens, temperature):
        with self.lock:
            self.active += 1
            self.max_concurrent = max(self.max_concurrent, self.active)
            self.calls.append({"prompt": prompt, "thread": threading.current_thread().name})
        time.sleep(0.02)  # simulate generation
        with self.lock:
            self.active -= 1
        if "extract durable facts" in prompt:
            return json.dumps(self.fact_json) if self.fact_json else "NONE"
        if "Condense" in prompt:
            return "The user discussed early topics that were evicted."
        if "Summarize what the user" in prompt:
            return "The user planned a trip and asked about food."
        return "NEW"


def full_store(path, llm, **cfg):
    return Elastimem(str(path), complete_fn=llm, config=ElastimemConfig(**cfg),
                  probe_fn=lambda: (32 * GIB, 20 * GIB))


def test_background_extraction_stores_facts(tmp_path):
    llm = FakeLLM(fact_json={"favorite_sport": "badminton"})
    s = full_store(tmp_path / "x.db", llm)
    s.record_turn("i really love playing badminton on weekends",
                  "That's a great hobby!")
    assert s.drain(timeout=5)
    assert s.facts().get("favorite_sport") == "badminton"
    # extraction happened on the worker thread, not the caller's
    assert any(c["thread"] == "elastimem-worker" for c in llm.calls)
    s.close()


def test_foreground_gate_blocks_llm_jobs(tmp_path):
    llm = FakeLLM(fact_json={"hobby": "chess"})
    s = full_store(tmp_path / "g.db", llm)
    with s.foreground():
        s.record_turn("i play chess every single evening", "Nice!")
        time.sleep(0.15)  # worker had time; the gate must have held it
        assert not any("extract" in c["prompt"][:80].lower() for c in llm.calls)
    assert s.drain(timeout=5)
    assert s.facts().get("hobby") == "chess"
    assert llm.max_concurrent == 1  # never overlapped with anything
    s.close()


def test_foreground_waits_for_inflight_background_job(tmp_path):
    """Closes the TOCTOU race: a background LLM job that's already dequeued
    and mid-execution (i.e. slipped past the advisory gate check) must not
    run concurrently with a foreground generation that starts right after.
    Regression test for a real bug: two threads calling into the same
    non-thread-safe llama.cpp instance concurrently produced garbled
    replies (a bare stray token) whenever a fact was saved right before the
    next turn started generating."""
    llm = FakeLLM(fact_json={"hobby": "chess"})
    s = full_store(tmp_path / "race.db", llm)

    # Foreground gate is OPEN (no one holds it yet) — enqueue an LLM job and
    # let the worker actually dequeue + start executing it, simulating the
    # real sequence: previous turn's record_turn() enqueues, gate is closed
    # by then (previous foreground() already exited), worker grabs it.
    s.record_turn("i play chess every single evening", "Nice!")
    # FakeLLM sleeps 20ms per call; give the worker time to dequeue and be
    # inside the call (but not yet finished) before we open the gate.
    deadline = time.monotonic() + 1.0
    while not llm.calls and time.monotonic() < deadline:
        time.sleep(0.001)
    assert llm.calls, "worker never started the background job in time"

    # Now the next turn begins: foreground_begin() must block until the
    # in-flight background call releases llm_lock, not race it.
    t0 = time.monotonic()
    with s.foreground():
        elapsed = time.monotonic() - t0
        # It should have waited out (part of) the background call's 20ms,
        # not returned instantly while the job was still running.
        assert llm.active == 0, "foreground entered while background call was still active"
        # Simulate the foreground's own generation call.
        llm(  # noqa: this direct call simulates the host's chat completion
            "simulated foreground generation", max_tokens=8, temperature=0.3
        )
    assert llm.max_concurrent == 1, "background and foreground calls overlapped"
    s.close()


def test_record_turn_returns_fast_despite_slow_llm(tmp_path):
    class SlowLLM(FakeLLM):
        def __call__(self, prompt, *, max_tokens, temperature):
            time.sleep(0.5)
            return "NONE"

    s = full_store(tmp_path / "f.db", SlowLLM())
    t0 = time.monotonic()
    for i in range(3):
        s.record_turn(f"message number {i} with enough words here", "ok!")
    elapsed = time.monotonic() - t0
    assert elapsed < 0.3, f"record_turn blocked on LLM work ({elapsed:.2f}s)"
    s.close()


def test_lite_tier_never_calls_llm(tmp_path):
    llm = FakeLLM(fact_json={"anything": "value"})
    s = Elastimem(str(tmp_path / "l.db"), complete_fn=llm,
               probe_fn=lambda: (4 * GIB, 2 * GIB))
    assert s.profile.tier is Tier.LITE
    s.record_turn("i adore mountain hiking every summer", "Wonderful!")
    s.drain(timeout=2)
    assert llm.calls == []  # extraction cadence OFF
    s.close()


def test_batched_cadence_coalesces(tmp_path):
    llm = FakeLLM(fact_json={"k": "v"})
    # STANDARD tier -> BATCHED every 3 turns
    s = Elastimem(str(tmp_path / "b.db"), complete_fn=llm,
               config=ElastimemConfig(batched_every_n_turns=3),
               probe_fn=lambda: (8 * GIB, 4 * GIB))
    s.record_turn("first message with plenty of words", "ok")
    s.record_turn("second message with plenty of words", "ok")
    time.sleep(0.1)
    n_before = len(llm.calls)
    s.record_turn("third message with plenty of words", "ok")  # triggers flush
    s.drain(timeout=5)
    assert n_before == 0 and len(llm.calls) >= 3
    s.close()


def test_rolling_summary_from_evictions(tmp_path):
    llm = FakeLLM()
    s = full_store(tmp_path / "r.db", llm)
    s.record_turn("let's talk about gardening tomatoes", "Sure!")
    s.report_evictions([("let's talk about gardening tomatoes", "Sure!")])
    s.drain(timeout=5)
    plan = s.build_context("and what about the watering schedule?")
    assert plan.rolling_summary == "The user discussed early topics that were evicted."
    s.close()


def test_rolling_summary_placeholder_without_llm(tmp_path):
    s = Elastimem(str(tmp_path / "p.db"), probe_fn=lambda: (32 * GIB, 20 * GIB))
    s.record_turn("hello there friend of mine", "hi!")
    s.report_evictions([("hello there friend of mine", "hi!")])
    plan = s.build_context()
    assert "omitted" in plan.rolling_summary
    s.close()


def test_end_session_writes_summary(tmp_path):
    llm = FakeLLM()
    s = full_store(tmp_path / "e.db", llm)
    s.record_turn("planning a trip to Japan in autumn", "Lovely!")
    s.record_turn("what food should I try in Osaka?", "Takoyaki!")
    s.end_session()
    sess = s.sessions()[0]
    assert sess["summary"] == "The user planned a trip and asked about food."
    assert sess["ended_at"] is not None
    s.close()


def test_close_drains_cleanly_with_pending_jobs(tmp_path):
    llm = FakeLLM(fact_json={"note_key": "note value"})
    s = full_store(tmp_path / "c.db", llm)
    for i in range(5):
        s.record_turn(f"turn {i} with sufficient words to extract", "ok")
    s.close()  # must not hang or raise
