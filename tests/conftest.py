import pytest

from elastimem import Elastimem, ElastimemConfig


@pytest.fixture
def store(tmp_path):
    s = Elastimem(str(tmp_path / "test.db"))
    yield s
    s.close()


@pytest.fixture
def store_with_reserved(tmp_path):
    cfg = ElastimemConfig(reserved_keys=frozenset({"model", "agent_name"}))
    s = Elastimem(str(tmp_path / "test.db"), config=cfg)
    yield s
    s.close()
