"""The built-in model (needs the embed extra and a one-time download)."""

import pytest

import elastimem
from elastimem import Tier

pytest.importorskip("fastembed")
pytestmark = pytest.mark.slow


def test_builtin_model_finds_a_paraphrase(path):
    with elastimem.open(path, tier=Tier.FULL) as mem:
        mem.record("I adopted a puppy named Bruno", "Lovely.")
        mem.record("The weather is rainy today", "Bring an umbrella.")
        mem.new_session()
        mem.flush(120)
        assert mem.status.vectors
        assert "Bruno" in mem.search("how is my dog doing?")[0].text
