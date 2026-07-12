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
