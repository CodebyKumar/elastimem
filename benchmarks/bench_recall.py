"""Recall latency benchmark: is brute-force cosine really fine at this scale?

Ingests N synthetic sessions (default 10k chunks), then measures FTS5-only
and hybrid (FTS5 + vector) recall latency. Targets: <100 ms FTS5, <150 ms
hybrid at 10k chunks on a laptop. The numbers this script produces are what
docs/schema.md's "Size expectations" section and embeddings.py's module
docstring cite — re-run it after any change to retrieval.py or
embeddings.py to keep those claims honest.

Run:  uv run python benchmarks/bench_recall.py [n_chunks]
"""

import hashlib
import math
import os
import random
import statistics
import sys
import time

from elastimem import Elastimem
from elastimem.governor import GIB


def toy_embed(texts):
    """Deterministic 'semantic' embedding: bag of word-stems hashed into 64
    dims. Paraphrases sharing words land close; good enough to exercise
    fusion without a real model. Kept in sync with tests/test_retrieval.py's
    copy — duplicated here rather than imported so this script has no
    dependency on the tests/ directory or an internal test module."""
    out = []
    for text in texts:
        vec = [0.0] * 64
        for word in text.lower().split():
            stem = word.strip(".,!?')(\"")[:5]
            if len(stem) < 3:
                continue
            idx = int(hashlib.md5(stem.encode()).hexdigest(), 16) % 64
            vec[idx] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        out.append([x / norm for x in vec])
    return out


TOPICS = [
    "car brake pads squealing repair garage", "kitchen repainting color walls",
    "sourdough starter feeding flour bakery", "japan trip kyoto autumn temples",
    "marathon training schedule running shoes", "python project sqlite database",
    "birthday party planning cake candles", "guitar practice chords fingers",
    "vegetable garden tomatoes watering", "tax filing deadline documents",
]


def main(n_chunks: int = 10_000) -> None:
    import tempfile

    db_path = tempfile.mktemp(suffix=".db")
    store = Elastimem(db_path, embed_fn=toy_embed,
                   probe_fn=lambda: (32 * GIB, 20 * GIB))
    rng = random.Random(42)

    t0 = time.monotonic()
    conn = store._conn
    from elastimem.db import utcnow
    from elastimem.embeddings import encode

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

    bench("hybrid (FTS+cos)", lambda q: store.recall(q))
    store.close()

    # embed_fn/tokenizer_fn/path are construction-time-only (see store.py) —
    # a fresh store over the same DB file is the correct way to get an
    # FTS5-only view of identical data, rather than mutating a live store.
    fts_only = Elastimem(db_path, disable_builtin_embedder=True,
                          probe_fn=lambda: (32 * GIB, 20 * GIB))
    bench("FTS5 only     ", lambda q: fts_only.recall(q))
    fts_only.close()

    os.remove(db_path)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10_000)
