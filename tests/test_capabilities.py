"""What Elastimem can do, end to end and on every tier.

The unit tests check each module; these check the promises in the README:
facts survive change, past moments are found by wording and by meaning,
the graph reaches what words cannot, prompts stay inside their budgets,
and a smaller tier does less background work without losing the store.
"""

import threading
import time

import pytest
from conftest import StubLLM, toy_embed

import elastimem
from elastimem import Tier
from elastimem.runtime import governor
from elastimem.runtime.governor import GIB
from elastimem.storage import episodes, graph

TIERS = [Tier.LITE, Tier.STANDARD, Tier.FULL]

# (planted exchange, question that paraphrases it, word the answer must contain)
PLANTED = [
    ("My dentist appointment is on Friday morning", "when do I see the dentist", "Friday"),
    ("I adopted a puppy named Bruno", "tell me about the puppy I adopted", "Bruno"),
    ("The router admin password is pinecone42", "what is the router admin password", "pinecone42"),
    ("I am allergic to peanuts", "which peanuts am I allergic to", "peanuts"),
]
CHATTER = [f"Chat number {i} about the weather and traffic today" for i in range(40)]


def _history(mem):
    """Planted exchanges buried in chatter, over several sessions."""
    for i, text in enumerate(CHATTER):
        mem.record(text, "Sure.")
        if i % 10 == 9:
            mem.new_session()
    for text, _, _ in PLANTED:
        mem.record(text, "Noted.")
        mem.new_session()
    mem.flush()


@pytest.mark.parametrize("tier", TIERS)
def test_planted_memories_are_recalled_on_every_tier(path, tier):
    with elastimem.open(path, embedder=toy_embed, tier=tier) as mem:
        _history(mem)
        for _, question, answer in PLANTED:
            assert answer in " ".join(h.text for h in mem.search(question, k=3)), question
            assert answer in mem.context(question).sections["moments"], question


def test_ranking_is_the_same_on_every_tier(path):
    results = {}
    for tier in TIERS:
        with elastimem.open(f"{path}.{tier.name}", embedder=toy_embed, tier=tier) as mem:
            _history(mem)
            results[tier] = [[h.text for h in mem.search(q, k=3)] for _, q, _ in PLANTED]
    assert results[Tier.LITE] == results[Tier.STANDARD] == results[Tier.FULL]


@pytest.mark.parametrize("tier, cadence, hops, builtin", [
    (Tier.LITE, "session_end", 1, False),
    (Tier.STANDARD, "batched", 1, True),
    (Tier.FULL, "per_turn", 2, True),
])
def test_status_matches_the_tier(path, tier, cadence, hops, builtin):
    with elastimem.open(path, embedder=None, llm=StubLLM(), tier=tier) as mem:
        s = mem.status
        assert (s.tier, s.llm, s.graph_hops) == (tier, cadence, hops)
        assert s.spec.load_embedder is builtin


def test_budgets_do_not_depend_on_the_tier(path):
    seen = set()
    for tier in TIERS:
        with elastimem.open(f"{path}.{tier.name}", embedder=None, tier=tier) as mem:
            seen.add(mem.status.budgets)
    assert len(seen) == 1


@pytest.mark.parametrize("tier", TIERS)
def test_prompt_sections_fit_their_budgets(path, tier):
    with elastimem.open(path, embedder=None, llm=StubLLM(), tier=tier, context_tokens=2048,
                        reserved_tokens=512) as mem:
        for i in range(200):
            mem.remember(f"fact_{i}", f"value number {i} " * 5)
            mem.learn(f"Lesson {i}: do something different next time, please")
        for i in range(60):
            mem.record(f"Question {i} about gardens and tomatoes", "Tomatoes like sun. " * 10)
            if i % 10 == 9:
                mem.end_session()          # the LLM writes this session's summary
        b = mem.status.budgets
        sections = mem.context("gardens and tomatoes").sections
        for name in ("facts", "moments", "sessions", "lessons"):
            assert sections[name], name
            assert episodes.estimate_tokens(sections[name]) <= getattr(b, name) + 8, name


def test_a_changed_fact_replaces_the_old_one_in_the_prompt(mem):
    mem.remember("occupation", "designer")
    mem.new_session()
    mem.remember("occupation", "writer")
    text = str(mem.context("what do I do for work"))
    assert "occupation: writer" in text and "designer" not in text
    assert [f.value for f in mem.history("occupation")] == ["designer", "writer"]


@pytest.mark.parametrize("tier", TIERS)
def test_everything_survives_a_restart(path, tier):
    with elastimem.open(path, embedder=toy_embed, tier=tier) as mem:
        mem.remember("home_city", "Lisbon")
        mem.learn("Use metric units")
        _history(mem)
    with elastimem.open(path, embedder=toy_embed, tier=tier) as mem:
        assert mem.facts() == {"home_city": "Lisbon"}
        assert mem.lessons() == ["Use metric units"]
        assert "pinecone42" in mem.search("router admin password")[0].text
        assert len(mem.sessions()) >= 5


# --- the graph reaches what words cannot ------------------------------------------
def _tuffy_store(mem):
    """Jetson -> Tuffy -> CUDA, each told in a different conversation."""
    rows = [
        ("Tuffy runs on a Jetson", [("Tuffy", "thing"), ("Jetson", "thing")],
         [("Tuffy", "runs_on", "Jetson")]),
        ("Tuffy does its vision work with CUDA", [("Tuffy", "thing"), ("CUDA", "thing")],
         [("Tuffy", "uses", "CUDA")]),
        ("The latency target for the kernels is twelve milliseconds", [("CUDA", "thing")], []),
    ]
    for text, entities, relations in rows:
        mem.record(text, "Noted.")
        chunk = mem._conn.execute("SELECT max(id) FROM chunks").fetchone()[0]
        graph.store_extraction(
            mem._conn, [{"name": n, "type": t} for n, t in entities],
            [{"source": s, "relation": r, "target": t} for s, r, t in relations], [chunk])
        mem.new_session()


@pytest.mark.parametrize("tier, hops", [(Tier.LITE, 1), (Tier.STANDARD, 1), (Tier.FULL, 2)])
def test_graph_depth_follows_the_tier(path, tier, hops):
    with elastimem.open(path, embedder=None, tier=tier) as mem:
        _tuffy_store(mem)
        texts = " ".join(h.text for h in mem.search("what do I know about my Jetson", k=5))
        assert "Tuffy runs on a Jetson" in texts            # the keyword match
        assert "CUDA" in texts                              # one hop: Jetson - Tuffy - CUDA chunk
        assert ("twelve milliseconds" in texts) is (hops == 2)   # two hops: needs FULL


# --- tiers changing at runtime -------------------------------------------------------
@pytest.mark.parametrize("total_gib, pressure, tier", [
    (32, "normal", Tier.FULL), (8, "normal", Tier.STANDARD), (4, "normal", Tier.LITE),
    (32, "critical", Tier.LITE),
])
def test_open_picks_the_tier_from_the_machine(path, monkeypatch, total_gib, pressure, tier):
    monkeypatch.setattr(governor, "probe", lambda: (total_gib * GIB, pressure))
    with elastimem.open(path, embedder=None) as mem:
        assert mem.status.tier is tier
        assert "memory pressure" in mem.status.reasons["tier"]


def test_stepping_down_keeps_the_store_working(path, monkeypatch):
    monkeypatch.setattr(governor, "probe", lambda: (32 * GIB, "normal"))
    changes = []
    llm = StubLLM(facts={"pet_name": "Bruno"})
    with elastimem.open(path, embedder=toy_embed, llm=llm,
                        on_tier=lambda old, new: changes.append((old, new))) as mem:
        _history(mem)
        mem.report_pressure()
        assert mem.status.tier is Tier.STANDARD
        monkeypatch.setattr("elastimem.tiers.PRESSURE_COOLDOWN_SECONDS", 0)
        mem.report_pressure()
        assert mem.status.tier is Tier.LITE
        assert changes == [(Tier.FULL, Tier.STANDARD), (Tier.STANDARD, Tier.LITE)]
        # LITE defers the LLM to the session end, and search still works.
        before = llm.count("Extract")
        mem.record("I adopted a parrot named Kiwi", "Lovely.")
        time.sleep(0.3)
        assert llm.count("Extract") == before
        mem.new_session()                         # ends the session: extraction runs now
        assert llm.count("Extract") == before + 1
        assert mem.facts() == {"pet_name": "Bruno"}
        assert "Kiwi" in " ".join(h.text for h in mem.search("tell me about the parrot I adopted", k=3))


def test_a_lower_tier_resumes_work_a_higher_one_left_pending(path):
    with elastimem.open(path, embedder=None, tier=Tier.FULL) as mem:
        mem.record("I adopted a puppy named Bruno", "Lovely.")     # no LLM: stays pending
    llm = StubLLM(facts={"pet_name": "Bruno"})
    with elastimem.open(path, embedder=None, llm=llm, tier=Tier.LITE) as mem:
        time.sleep(0.3)
        assert mem.facts() == {}                 # LITE does not extract on its own
        mem.flush()                              # but an explicit flush clears the backlog
        assert mem.facts() == {"pet_name": "Bruno"}


# --- the host ------------------------------------------------------------------------
@pytest.mark.parametrize("tier", TIERS)
def test_record_and_search_from_many_threads(path, tier):
    errors = []

    def work(n):
        try:
            for i in range(15):
                mem.record(f"Thread {n} message {i} about zebras", "ok")
                assert mem.search("zebras")
                mem.context("zebras")
        except Exception as e:      # noqa: BLE001
            errors.append(e)

    with elastimem.open(path, embedder=toy_embed, tier=tier) as mem:
        threads = [threading.Thread(target=work, args=(n,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        assert errors == []
        assert mem.flush(30)
        assert mem.stats()["messages"] == 4 * 15 * 2


def test_switching_embedders_redoes_vectors_and_keeps_search(path):
    def other_embed(texts):           # a different model: 8 dimensions
        return [[float(len(t) % 7 + 1)] + [1.0] * 7 for t in texts]

    with elastimem.open(path, embedder=toy_embed, tier=Tier.FULL) as mem:
        _history(mem)
    with elastimem.open(path, embedder=other_embed, tier=Tier.FULL) as mem:
        mem.flush()
        assert mem._conn.execute("SELECT count(*) FROM chunks WHERE embedding IS NULL").fetchone()[0] == 0
        assert "pinecone42" in " ".join(h.text for h in mem.search("router admin password pinecone42", k=3))
