import pytest

from engram import Engram, EngramConfig


@pytest.fixture
def store(tmp_path):
    s = Engram(str(tmp_path / "test.db"))
    yield s
    s.close()


@pytest.fixture
def store_with_reserved(tmp_path):
    cfg = EngramConfig(reserved_keys=frozenset({"model", "agent_name"}))
    s = Engram(str(tmp_path / "test.db"), config=cfg)
    yield s
    s.close()
