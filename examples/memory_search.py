"""Basic memory & search, no chat loop: mem.remember(), mem.recall(),
mem.facts(), mem.fact_history() used directly. No LLM, no embedder required
— demonstrates the memory API on its own, outside of a conversation.

Run:  python examples/memory_search.py
"""

import sys

sys.path.insert(0, "src")

import elastimem  # noqa: E402


def main() -> None:
    mem = elastimem.open("./memory_search_demo.db")

    # Explicit facts (source="explicit" is the default).
    mem.remember("name", "Sam")
    mem.remember("favorite_color", "teal")
    changed, reason = mem.remember("favorite_color", "teal")  # no-op, same value
    print(f"re-storing the same value: changed={changed}, reason={reason!r}")

    mem.remember("favorite_color", "blue")  # value changed -> new version
    print("current facts:", mem.facts())
    print("favorite_color history:", mem.fact_history("favorite_color"))

    mem.forget("name")
    print("facts after forget('name'):", mem.facts())

    # A couple of turns, so recall() has something to search over.
    mem.record_turn("what's the capital of Japan?", "Tokyo is the capital of Japan.")
    mem.record_turn("what's 2+2?", "2+2 is 4.")
    mem.drain()

    hits = mem.recall("capital of Japan")
    print("recall('capital of Japan'):")
    for h in hits:
        print(f"  [{h.kind}] {h.text[:80]}")

    mem.close()


if __name__ == "__main__":
    main()
