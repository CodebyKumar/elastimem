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


def test_build_context_never_raises(tmp_path):
    s = Elastimem(str(tmp_path / "n.db"), probe_fn=lambda: (32 * GIB, 20 * GIB))
    for weird in ["", "hi", '"; DROP TABLE chunks; --', "🦆" * 500, "a " * 3000]:
        plan = s.build_context(weird)
        assert plan.profile is not None
    s.close()
