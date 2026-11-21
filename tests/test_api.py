import pytest
from conftest import StubLLM

import elastimem
from elastimem import Context, Status, Tier


def test_three_call_flow(path):
    llm = StubLLM()
    with elastimem.open(path, llm=llm, embedder=None) as mem:
        context = mem.context("hi there")
        assert isinstance(context, Context) and isinstance(context.status, Status)
        reply = llm(f"{context}\n\nhi there", max_tokens=10, temperature=0)
        mem.record("hi there", reply)
    assert elastimem.open(path, embedder=None).stats()["messages"] == 2


def test_status_explains_itself(mem):
    s = mem.status
    assert s.tier is Tier.FULL and s.vectors is False and s.llm == "off"
    assert s.reasons["llm"] == "no llm given"
    assert "no embedder" in s.reasons["vectors"]
    assert s.reasons["tier"].startswith("FULL")




def test_hot_path_never_raises(path):
    def broken_embedder(texts):
        raise RuntimeError("embedder down")

    with elastimem.open(path, embedder=broken_embedder, tier=Tier.FULL,
                        llm=StubLLM(fail=True)) as mem:
        mem.record("hello", "hi")
        mem.flush()
        assert mem.search("hello")
        assert isinstance(mem.context("hello"), Context)
        assert "embedder failed" in mem.status.reasons["vectors"]
        mem._conns[0].close()                  # even a dead connection
        mem.record("x", "y")
        assert mem.search("x") == []
        assert mem.context("x").sections["facts"] == ""


def test_unknown_setting_is_rejected(path):
    with pytest.raises(TypeError, match="unknown setting"):
        elastimem.open(path, context_window=4096)


def test_config_is_a_copy_and_configure_applies(mem):
    mem.config.context_tokens = 99999
    assert mem.config.context_tokens == 4096
    status = mem.configure(context_tokens=8192)
    assert status.budgets.working > 0 and mem.config.context_tokens == 8192


def test_invalid_settings():
    with pytest.raises(ValueError):
        elastimem.ElastimemConfig(context_tokens=512)
    with pytest.raises(ValueError):
        elastimem.ElastimemConfig(reserved_tokens=4000)




def test_context_renders_sections(mem):
    mem.remember("name", "Asha")
    text = str(mem.context("hi"))
    assert text.startswith("## What you know about the user\n- name: Asha")


def test_public_names():
    assert set(elastimem.__all__) == {
        "open", "Elastimem", "ElastimemConfig", "Tier", "Status", "Budgets",
        "Context", "Fact", "Hit", "Explanation", "__version__",
    }
