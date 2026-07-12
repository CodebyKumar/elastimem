"""Facts: storage, temporal versioning, guards, decay."""

import sqlite3

from engram import Engram, EngramConfig


def test_store_and_read(store):
    changed, reason = store.remember("favorite_color", "blue")
    assert changed and reason == "stored"
    assert store.facts() == {"favorite_color": "blue"}


def test_key_normalization(store):
    store.remember("Favorite Color!", "blue")
    assert "favorite_color" in store.facts()


def test_same_value_is_noop(store):
    store.remember("name", "Alex")
    changed, reason = store.remember("name", "Alex")
    assert not changed and reason == "already stored"
    assert len(store.fact_history("name")) == 1


def test_update_creates_version_chain(store):
    store.remember("occupation", "designer")
    store.remember("occupation", "writer")
    assert store.facts()["occupation"] == "writer"
    history = store.fact_history("occupation")
    assert [f.value for f in history] == ["designer", "writer"]
    assert history[0].invalidated_at is not None
    assert history[1].invalidated_at is None


def test_forget_tombstones_not_deletes(store):
    store.remember("nickname", "Ku")
    assert store.forget("nickname")
    assert "nickname" not in store.facts()
    # history survives
    assert [f.value for f in store.fact_history("nickname")] == ["Ku"]
    assert not store.forget("nickname")  # already gone


def test_placeholder_rejected(store):
    changed, reason = store.remember("goals", "unknown")
    assert not changed and "placeholder" in reason
    assert store.facts() == {}


def test_reserved_key_rejected(store_with_reserved):
    changed, reason = store_with_reserved.remember("model", "gpt-9")
    assert not changed and "reserved" in reason
    # suffix form too
    changed, _ = store_with_reserved.remember("my_agent_name", "Tuffy")
    assert not changed
    # but the user's own name is fine
    changed, _ = store_with_reserved.remember("name", "Alex")
    assert changed


def test_transcript_key_rejected_for_auto_only(store):
    changed, _ = store.remember("user_message", "hi there", source="auto")
    assert not changed
    changed, _ = store.remember("user_message", "hi there")  # explicit is allowed
    assert changed


def test_self_referential_value_rejected(store):
    changed, reason = store.remember("role", "I am a local AI agent", source="auto")
    assert not changed
    assert "agent" in reason


def test_auto_rejections_are_quarantined(store):
    store.remember("goals", "unknown", source="auto")
    entries = store.quarantine_entries()
    assert len(entries) == 1 and entries[0]["key"] == "goals"


def test_short_auto_value_cannot_clobber(store):
    store.remember("name", "Alexandra")
    changed, reason = store.remember("name", "A", source="auto")
    assert not changed and reason == "kept existing value"
    assert store.facts()["name"] == "Alexandra"


def test_profile_categorization(store):
    store.remember("name", "Alex")
    store.remember("favorite_food", "ramen")
    conn = sqlite3.connect(store.path)
    cats = dict(conn.execute(
        "SELECT key, category FROM facts WHERE invalidated_at IS NULL").fetchall())
    conn.close()
    assert cats == {"name": "profile", "favorite_food": "note"}


def test_decay_archives_only_unaccessed_auto_facts(tmp_path):
    from engram import semantic
    cfg = EngramConfig(fact_decay_half_life_days=60.0, fact_archive_threshold=0.15)
    s = Engram(str(tmp_path / "d.db"), config=cfg)
    s.remember("old_auto", "something", source="auto")
    s.remember("old_explicit", "keep me")
    # backdate both far beyond the decay horizon
    conn = s._conn
    with conn:
        conn.execute("UPDATE facts SET valid_from='2020-01-01T00:00:00+00:00'")
    archived = semantic.apply_decay(conn, cfg)
    assert archived == 1
    assert "old_auto" not in s.facts()
    assert s.facts()["old_explicit"] == "keep me"
    s.close()
