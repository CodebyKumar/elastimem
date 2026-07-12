"""Recall latency benchmark: is brute-force cosine really fine at this scale?

Ingests N synthetic sessions (default 10k chunks), then measures FTS5-only
and hybrid (FTS5 + vector) recall latency. Targets: <100 ms FTS5, <150 ms
hybrid at 10k chunks on a laptop.

Run:  uv run python benchmarks/bench_recall.py [n_chunks]
"""

import random
import statistics
import sys
import time

sys.path.insert(0, "src")

from engram import Engram, EngramConfig  # noqa: E402
from engram.governor import GIB  # noqa: E402

sys.path.insert(0, "tests")
from test_retrieval import toy_embed  # noqa: E402

TOPICS = [
    "car brake pads squealing repair garage", "kitchen repainting color walls",
    "sourdough starter feeding flour bakery", "japan trip kyoto autumn temples",
    "marathon training schedule running shoes", "python project sqlite database",
    "birthday party planning cake candles", "guitar practice chords fingers",
    "vegetable garden tomatoes watering", "tax filing deadline documents",
]


def main(n_chunks: int = 10_000) -> None:
    store = Engram(":memory:", embed_fn=toy_embed,
                   probe_fn=lambda: (32 * GIB, 20 * GIB))
    rng = random.Random(42)

    t0 = time.monotonic()
    conn = store._conn
    from engram.db import utcnow
    from engram.embeddings import encode

    with conn:
        conn.execute("INSERT INTO sessions(started_at) VALUES (?)", (utcnow(),))
        rows = []
        for i in range(n_chunks):
            topic = rng.choice(TOPICS)
            words = topic.split()
            rng.shuffle(words)
            text = (f"User: tell me about {' '.join(words[:3])} number {i}\n"
                    f"Assistant: sure, {' '.join(words)} details here.")
            rows.append((1, 1, 2, text, utcnow(), encode(toy_embed([text])[0])))
        conn.executemany(
            "INSERT INTO chunks(session_id, first_msg_id, last_msg_id, text,"
            " created_at, embedding) VALUES (?,?,?,?,?,?)", rows)
    print(f"ingested {n_chunks} chunks in {time.monotonic()-t0:.1f}s")

    queries = ["what did we discuss about my car brakes",
               "remind me about the japan travel plans",
               "how was my sourdough starter doing",
               "anything about the vegetable garden tomatoes"]

    def bench(label, fn):
        times = []
        for _ in range(5):
            for q in queries:
                t = time.monotonic()
                hits = fn(q)
                times.append((time.monotonic() - t) * 1000)
                assert hits, f"no hits for {q!r}"
        print(f"{label}: median {statistics.median(times):.1f} ms, "
              f"p95 {sorted(times)[int(len(times)*0.95)]:.1f} ms")

    real_embed_fn = store.embed_fn
    store.embed_fn = None  # FTS-only
    bench("FTS5 only     ", lambda q: store.recall(q))
    store.embed_fn = real_embed_fn
    bench("hybrid (FTS+cos)", lambda q: store.recall(q))
    store.close()


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10_000)
