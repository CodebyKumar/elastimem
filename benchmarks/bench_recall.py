"""Per-turn cost: how long do context(), search() and record() take?

Ingests N exchanges with 384-dim vectors (the built-in model's size) from a
fast hashing embedder, then times the calls a host makes every turn.
``ram`` mode instead reports the process memory Elastimem adds, with and
without the built-in embedder.

Run:  uv run python benchmarks/bench_recall.py [n_chunks]
      uv run python benchmarks/bench_recall.py ram [keyword|builtin]
"""

import hashlib
import math
import os
import random
import resource
import statistics
import sys
import tempfile
import time

import elastimem
from elastimem import Tier

WORDS = ("python rust berlin paris coffee hiking guitar startup mortgage dentist "
         "sister marathon kubernetes llama garden").split()


def toy_embed(texts, dim=384):
    """Deterministic bag-of-words vectors: shared words land close."""
    out = []
    for text in texts:
        vec = [0.0] * dim
        for word in text.lower().split():
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % dim] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        out.append([x / norm for x in vec])
    return out


def _ms(fn, n):
    times = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1000)
    return statistics.median(times)


def latency(n: int) -> None:
    mem = elastimem.open(os.path.join(tempfile.mkdtemp(), "bench.db"),
                         embedder=toy_embed, tier=Tier.FULL)
    rng = random.Random(1)
    for i in range(n):
        if i and i % 50 == 0:
            mem.new_session()       # history lives in past sessions, as in real use
        a, b = rng.sample(WORDS, 2)
        mem.record(f"I talked about {a} and {b} today, number {i}", f"Tell me more about {a}.")
    mem.flush(600)
    q = "what did I say about hiking in berlin"
    mem.context(q)   # first call loads the vector index
    print(f"{n:>6} chunks: context {_ms(lambda: mem.context(q), 10):6.1f} ms"
          f"  search {_ms(lambda: mem.search(q), 10):6.1f} ms"
          f"  record {_ms(lambda: mem.record('my favorite food is ramen', 'ok'), 50):5.2f} ms")
    mem.close()


def ram(mode: str) -> None:
    def peak_mb():   # ru_maxrss is bytes on macOS, KiB on Linux
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss / 1024 / (1024 if sys.platform == "darwin" else 1)

    base = peak_mb()
    mem = elastimem.open(os.path.join(tempfile.mkdtemp(), "ram.db"), tier=Tier.FULL,
                         embedder=None if mode == "keyword" else "auto")
    for i in range(500):
        mem.record(f"turn {i} about hiking and coffee", "nice")
    mem.flush(600)
    mem.search("hiking")
    print(f"{mode}: elastimem adds {peak_mb() - base:.0f} MB (process peak {peak_mb():.0f} MB)")
    mem.close()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ram":
        ram(sys.argv[2] if len(sys.argv) > 2 else "keyword")
    else:
        latency(int(sys.argv[1]) if len(sys.argv) > 1 else 10_000)
