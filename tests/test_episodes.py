import elastimem
from elastimem import Tier
from elastimem.storage import episodes


def test_record_stores_messages_and_chunks(mem):
    mem.record("hello", "hi there")
    stats = mem.stats()
    assert (stats["sessions"], stats["messages"], stats["chunks"]) == (1, 2, 1)


def test_long_exchange_is_split():
    text = "User: " + " ".join(f"Sentence number {i} is here." for i in range(400))
    chunks = episodes._split(text, None)
    assert len(chunks) > 1
    assert all(episodes.estimate_tokens(c) <= 400 + 20 for c in chunks)


def test_turns_leaving_the_window_are_folded_without_a_model(mem):
    keep = mem.status.keep_turns
    for i in range(keep + 3):
        mem.record(f"Question number {i}. Some detail.", "answer")
    summary = mem.context().summary
    assert summary.splitlines() == [f"- Question number {i}." for i in range(3)]


def test_summary_survives_a_restart(path):
    with elastimem.open(path, embedder=None, tier=Tier.LITE) as mem:
        for i in range(mem.status.keep_turns + 2):
            mem.record(f"Topic {i}.", "ok")
    with elastimem.open(path, embedder=None, tier=Tier.LITE) as mem:
        summary, messages = mem.resume()
        assert summary == "- Topic 0.\n- Topic 1."
        assert messages[-1] == {"role": "assistant", "content": "ok"}


def test_crashed_session_is_closed_on_next_open(path):
    mem = elastimem.open(path, embedder=None, tier=Tier.FULL)
    mem.record("My first message", "ok")
    mem._worker.stop()                     # simulate a crash: no end_session()
    with elastimem.open(path, embedder=None, tier=Tier.FULL) as again:
        assert again.sessions()[0]["ended_at"] is not None
        assert again.sessions()[0]["title"] == "My first message"


def test_new_session_ends_the_previous(mem):
    mem.record("one", "ok")
    second = mem.new_session("work")
    assert mem.sessions()[1]["ended_at"] is not None
    assert mem.sessions()[0]["id"] == second


def test_lessons(mem):
    assert mem.learn("Use metric units")
    assert mem.learn("Use metric units")       # duplicate: merged
    assert not mem.learn("   ")
    assert mem.lessons() == ["Use metric units"]
    assert "Use metric units" in str(mem.context("hi"))
