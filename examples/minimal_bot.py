"""The zero-capability floor, demonstrated: a memory-enabled bot with NO LLM.

No model, no embeddings, no dependencies — yet it remembers your name across
restarts, recalls past conversations by keyword, and budgets its context.

Run:  python examples/minimal_bot.py
Try:  "my name is Sam", restart, "what do you know about me?",
      "recall: <anything you said last run>", "quit".
"""

import sys

sys.path.insert(0, "src")

from elastimem import Elastimem  # noqa: E402


def main() -> None:
    mem = Elastimem("./minimal_bot_memory.db")
    print(f"[elastimem tier: {mem.profile.tier.name}, "
          f"facts known: {len(mem.facts())}]")

    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user or user.lower() in ("quit", "exit"):
            break

        if user.lower().startswith("recall:"):
            hits = mem.recall(user[7:])
            reply = ("\n".join(f"[{h.date}] {h.text[:120]}" for h in hits)
                     or "nothing found")
        elif "know about me" in user.lower():
            facts = mem.facts()
            reply = ("\n".join(f"- {k}: {v}" for k, v in facts.items())
                     or "nothing yet — tell me about yourself!")
        else:
            # A real host would send build_context() + user to a model.
            plan = mem.build_context(user)
            known = plan.sections["user_facts"].replace("\n", ", ")
            reply = f"(no LLM here, but I'm listening{' — I know: ' + known if known else ''})"

        print(f"bot> {reply}")
        mem.record_turn(user, reply)   # rule capture happens in here

    mem.end_session()
    mem.close()
    print("[session saved]")


if __name__ == "__main__":
    main()
