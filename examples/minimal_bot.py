"""Memory with no LLM and no setup: a bot that only stores and searches.

It keeps every exchange, finds past ones by meaning (with the built-in
embedder, ``pip install elastimem[embed]``) or by keyword (without it), and
stores facts you give it explicitly.

Run:  python examples/minimal_bot.py
Try:  "remember: city = Lisbon", restart, "facts", "search: <anything from last run>", "quit".
"""

import elastimem


def main() -> None:
    with elastimem.open("./minimal_bot_memory.db") as mem:
        print(f"[tier {mem.status.tier.name}, vectors: {mem.status.reasons['vectors']}]")
        while True:
            try:
                user = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user or user.lower() in ("quit", "exit"):
                break

            if user.lower().startswith("remember:") and "=" in user:
                key, value = user[9:].split("=", 1)
                reply = "stored" if mem.remember(key, value) else "already known"
            elif user.lower().startswith("search:"):
                hits = mem.search(user[7:])
                reply = "\n".join(f"[{h.date}] {h.text[:120]}" for h in hits) or "nothing found"
            elif user.lower() == "facts":
                reply = "\n".join(f"- {k}: {v}" for k, v in mem.facts().items()) or "no facts yet"
            else:
                # A real host sends str(mem.context(user)) and the message to its model.
                reply = f"(no model here; memory for this turn:)\n{mem.context(user)}"

            print(f"bot> {reply}")
            mem.record(user, reply)
    print("[saved]")


if __name__ == "__main__":
    main()
