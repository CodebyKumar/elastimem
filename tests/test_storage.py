import shutil
import sqlite3
from pathlib import Path

import pytest

import elastimem
from elastimem import Tier
from elastimem.storage import db

FIXTURE = Path(__file__).parent / "fixtures" / "v3_store.db"


@pytest.fixture
def v3(tmp_path):
    target = tmp_path / "old.db"
    shutil.copy(FIXTURE, target)
    return str(target)


def test_fresh_store_has_v4_schema(path):
    conn, fts = db.open_store(path)
    assert conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0] == "4"
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"entities", "relations", "mentions", "rejected", "facts", "chunks"} <= tables


def test_migration_keeps_every_row(v3):
    with elastimem.open(v3, embedder=None, tier=Tier.FULL) as mem:
        assert mem.facts() == {"name": "Asha", "location": "Pune",
                               "occupation": "writer", "favorite_color": "teal"}
        assert [f.value for f in mem.history("occupation")] == ["designer", "writer"]
        assert mem.lessons() == ["The weather tool needs a city name"]
        assert len(mem.rejected()) == 1
        stats = mem.stats()
        assert (stats["messages"], stats["chunks"], stats["entities"]) == (4, 2, 3)
        assert any("Bruno" in h.text for h in mem.search("Bruno"))


def test_migration_maps_rule_facts_and_skips_old_turns(v3):
    with elastimem.open(v3, embedder=None, tier=Tier.FULL) as mem:
        conn = mem._conn
        assert conn.execute("SELECT source FROM facts WHERE key='name'").fetchone()[0] == "auto"
        # Old turns count as extracted: no LLM backlog of the whole history.
        assert conn.execute("SELECT count(*) FROM chunks WHERE extracted_at IS NULL").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM mentions").fetchone()[0] == 2


def test_vectors_from_another_embedder_are_redone(v3):
    from conftest import toy_embed

    conn = sqlite3.connect(v3)
    conn.execute("UPDATE chunks SET embedding_model='some-other-model' WHERE id=1")
    conn.commit()
    conn.close()
    with elastimem.open(v3, embedder=toy_embed, tier=Tier.FULL) as mem:
        rows = dict(mem._conn.execute("SELECT id, embedding IS NOT NULL FROM chunks").fetchall())
        assert rows == {1: 0, 2: 1}     # chunk 2 was written by this embedder ("host"): kept
        mem.flush()
        assert mem._conn.execute("SELECT count(*) FROM chunks WHERE embedding IS NULL").fetchone()[0] == 0


def test_newer_schema_is_refused(path):
    conn, _ = db.open_store(path)
    conn.execute("UPDATE meta SET value='99' WHERE key='schema_version'")
    conn.commit()
    with pytest.raises(RuntimeError, match="newer"):
        db.open_store(path)


def test_corrupt_file_is_moved_aside(path):
    Path(path).write_bytes(b"not a database" * 100)
    conn, _ = db.open_store(path)
    assert conn.execute("SELECT count(*) FROM facts").fetchone()[0] == 0
    assert list(Path(path).parent.glob("memory.db.corrupt-*"))
