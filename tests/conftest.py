import hashlib
import json
import math

import pytest

import elastimem
from elastimem import Tier, governor


@pytest.fixture(autouse=True)
def plugged_in(monkeypatch):
    """Tests must not depend on whether the machine is on battery."""
    monkeypatch.setattr(governor, "on_battery", lambda: False)
    monkeypatch.delenv("ELASTIMEM_TIER", raising=False)


def toy_embed(texts):
    """Bag of word stems hashed into 64 dims: texts sharing words land close."""
    out = []
    for text in texts:
        vec = [0.0] * 64
        for word in text.lower().split():
            stem = word.strip(".,!?')(\"")[:5]
            if len(stem) >= 3:
                vec[int(hashlib.md5(stem.encode()).hexdigest(), 16) % 64] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        out.append([x / norm for x in vec])
    return out


class StubLLM:
    """Answers each kind of background prompt; records every call."""

    def __init__(self, facts=None, entities=None, relations=None, fail=False):
        self.facts, self.entities, self.relations = facts or {}, entities or [], relations or []
        self.fail = fail
        self.calls: list[str] = []

    def __call__(self, prompt, *, max_tokens, temperature):
        self.calls.append(prompt)
        if self.fail:
            raise RuntimeError("model crashed")
        if prompt.startswith("Extract information"):
            return json.dumps({"facts": self.facts, "entities": self.entities,
                               "relations": self.relations})
        if prompt.startswith("Are these two labels"):
            return "no"
        return "A short summary."

    def count(self, prefix):
        return sum(p.startswith(prefix) for p in self.calls)


@pytest.fixture
def path(tmp_path):
    return str(tmp_path / "memory.db")


@pytest.fixture
def mem(path):
    m = elastimem.open(path, embedder=None, tier=Tier.FULL)
    yield m
    m.close()
