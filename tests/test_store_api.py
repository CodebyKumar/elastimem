"""Store lifecycle: persistence across reopen, corruption recovery, lessons, stats."""

import os

from elastimem import Elastimem, ElastimemConfig


def test_facts_survive_reopen(tmp_path):
    path = str(tmp_path / "m.db")
    s = Elastimem(path)
    s.remember("city", "Bengaluru")
    s.close()
    s2 = Elastimem(path)
    assert s2.facts() == {"city": "Bengaluru"}
    s2.close()


def test_corrupt_db_is_quarantined_and_recreated(tmp_path):
    path = str(tmp_path / "m.db")
    with open(path, "wb") as f:
        f.write(b"this is not a sqlite database, definitely not")
    s = Elastimem(path)  # must not raise
    s.remember("name", "Alex")
    assert s.facts() == {"name": "Alex"}
    assert any(f.startswith("m.db.corrupt-") for f in os.listdir(tmp_path))
    s.close()


def test_lessons_dedupe_and_cap(tmp_path):
    cfg = ElastimemConfig(max_lessons=3, lessons_in_prompt=5)
    s = Elastimem(str(tmp_path / "l.db"), config=cfg)
    for i in range(5):
        s.add_lesson(f"lesson {i}")
    s.add_lesson("lesson 4")  # duplicate
    lessons = s.lessons()
    assert lessons == ["lesson 2", "lesson 3", "lesson 4"]  # capped at 3, oldest archived
    s.close()


def test_stats(store):
    store.remember("name", "Alex")
    st = store.stats()
    assert st["facts"] == 1
    assert st["fts_enabled"] is True
    assert st["db_bytes"] > 0


def test_zero_dependency_import():
    # elastimem must import with stdlib only; this suite runs without psutil/sqlite-vec
    import elastimem
    assert elastimem.__version__


def test_open_shorthand(tmp_path):
    import elastimem

    mem = elastimem.open(str(tmp_path / "o.db"))
    mem.remember("name", "Alex")
    assert mem.facts() == {"name": "Alex"}
    mem.close()


def test_inline_config_overrides(tmp_path):
    import elastimem

    mem = elastimem.open(str(tmp_path / "i.db"),
                         context_tokens=8192,
                         reserved_keys={"Model", "agent_name"})
    assert mem.config.context_tokens == 8192
    assert mem.config.reserved_keys == frozenset({"model", "agent_name"})
    changed, reason = mem.remember("model", "gpt-9")
    assert not changed and "reserved" in reason
    mem.close()


def test_external_config_plus_inline_override(tmp_path):
    import elastimem
    from elastimem import ElastimemConfig

    cfg = ElastimemConfig(context_tokens=8192, max_lessons=7)
    mem = elastimem.open(str(tmp_path / "c.db"), config=cfg, context_tokens=2048)
    assert mem.config.context_tokens == 2048   # inline wins
    assert mem.config.max_lessons == 7         # bundle preserved
    mem.close()


def test_unknown_config_option_raises_helpfully(tmp_path):
    import pytest
    import elastimem

    with pytest.raises(TypeError, match="unknown config option.*contxt_tokens"):
        elastimem.open(str(tmp_path / "u.db"), contxt_tokens=4096)


def test_llm_and_embedder_aliases(tmp_path):
    import elastimem

    def fake_llm(prompt, *, max_tokens, temperature):
        return "NONE"

    def fake_embed(texts):
        return [[0.0] * 4 for _ in texts]

    mem = elastimem.open(str(tmp_path / "a.db"), llm=fake_llm, embedder=fake_embed)
    assert mem.complete_fn is fake_llm and mem.embed_fn is fake_embed
    mem.close()
    # legacy names still accepted
    mem2 = elastimem.open(str(tmp_path / "a2.db"),
                          complete_fn=fake_llm, embed_fn=fake_embed)
    assert mem2.complete_fn is fake_llm and mem2.embed_fn is fake_embed
    mem2.close()
