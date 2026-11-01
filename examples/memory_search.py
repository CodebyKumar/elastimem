"""The memory API on its own, outside a conversation: facts, their
history, and search. No LLM needed.

Run:  python examples/memory_search.py
"""

import elastimem


def main() -> None:
    with elastimem.open("./memory_search_demo.db") as mem:
        mem.remember("name", "Sam")
        mem.remember("favorite_color", "teal")
        print("same value again changes nothing:", mem.remember("favorite_color", "teal"))
        mem.remember("favorite_color", "blue")
        print("facts:", mem.facts())
        print("history:", [(f.value, f.since, f.until) for f in mem.history("favorite_color")])

        mem.forget("name")
        print("after forget('name'):", mem.facts())

        mem.record("what's the capital of Japan?", "Tokyo is the capital of Japan.")
        mem.record("what's 2+2?", "2+2 is 4.")
        mem.new_session()            # search() looks at past sessions and facts
        mem.flush()
        for hit in mem.search("Japanese capital city"):
            print(f"  [{hit.kind} {hit.score:.2f}] {hit.text[:80]!r}")
        print("why:", mem.explain("Japanese capital city", k=1).items[0].parts)


if __name__ == "__main__":
    main()
