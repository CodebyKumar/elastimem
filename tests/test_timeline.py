"""timeline(): chronological version-history queries over facts — a query
layer on top of the already-existing fact_history() storage."""

from elastimem import Elastimem, ElastimemConfig
from elastimem.governor import GIB


def make_store(path, **cfg):
    cfg.setdefault("disable_builtin_embedder", True)
    return Elastimem(str(path), embed_fn=None, config=ElastimemConfig(**cfg),
                  probe_fn=lambda: (32 * GIB, 20 * GIB))


def test_timeline_exact_key_match(tmp_path):
    s = make_store(tmp_path / "e.db")
    s.remember("occupation", "Student")
    s.remember("occupation", "Designer")
    s.remember("occupation", "AI Engineer")

    result = s.timeline("occupation")
    assert result.key == "occupation"
    assert result.resolved_by == "exact"
    assert [v.value for v in result.versions] == ["Student", "Designer", "AI Engineer"]
    s.close()


def test_timeline_exact_key_normalizes(tmp_path):
    s = make_store(tmp_path / "n.db")
    s.remember("occupation", "Designer")
    result = s.timeline("Occupation!")
    assert result.key == "occupation"
    assert result.resolved_by == "exact"
    s.close()


def test_timeline_free_text_resolves_via_search(tmp_path):
    """The proposal's motivating example: 'what did I do before AI?' must
    resolve to the occupation key via fact search over stored values, no
    exact key match involved."""
    s = make_store(tmp_path / "s.db")
    s.remember("occupation", "Student")
    s.remember("occupation", "Designer")
    s.remember("occupation", "AI Engineer")

    result = s.timeline("what did I do before AI")
    assert result.key == "occupation"
    assert result.resolved_by == "search"
    assert [v.value for v in result.versions] == ["Student", "Designer", "AI Engineer"]
    s.close()


def test_timeline_single_version_fact(tmp_path):
    s = make_store(tmp_path / "sv.db")
    s.remember("favorite_color", "blue")
    result = s.timeline("favorite_color")
    assert result.resolved_by == "exact"
    assert len(result.versions) == 1
    assert result.versions[0].value == "blue"
    s.close()


def test_timeline_unresolvable_query_returns_none(tmp_path):
    s = make_store(tmp_path / "u.db")
    s.remember("favorite_color", "blue")
    result = s.timeline("something totally unrelated to anything stored")
    assert result.key is None
    assert result.resolved_by == "none"
    assert result.versions == ()
    s.close()


def test_timeline_empty_store_never_raises(tmp_path):
    s = make_store(tmp_path / "empty.db")
    result = s.timeline("anything")
    assert result.resolved_by == "none"
    assert result.versions == ()
    s.close()


def test_timeline_preserves_query(tmp_path):
    s = make_store(tmp_path / "q.db")
    result = s.timeline("what did I do before AI")
    assert result.query == "what did I do before AI"
    s.close()


def test_timeline_forgotten_key_still_shows_history(tmp_path):
    """forget() tombstones (invalidates) the current version but never
    deletes - the timeline must still show the full chain including the
    tombstoned tail, consistent with fact_history()'s existing contract."""
    s = make_store(tmp_path / "f.db")
    s.remember("nickname", "Ku")
    s.forget("nickname")
    result = s.timeline("nickname")
    assert result.resolved_by == "exact"
    assert [v.value for v in result.versions] == ["Ku"]
    assert result.versions[0].invalidated_at is not None
    s.close()


def test_timeline_error_degrades_silently(tmp_path, monkeypatch):
    from elastimem import semantic

    s = make_store(tmp_path / "err.db")
    s.remember("occupation", "Designer")

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(semantic, "fact_history", _boom)
    result = s.timeline("occupation")
    assert result.resolved_by == "none"
    assert result.versions == ()
    s.close()
