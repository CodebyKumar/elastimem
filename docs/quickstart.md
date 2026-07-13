# Quickstart

Get from install to first recall in under a minute.

## 1. Install

```bash
pip install elastimem
```

No required dependencies — this works standalone.

## 2. Open a store

```python
import elastimem

mem = elastimem.open("~/.myagent/memory.db")
```

That's the whole setup. `elastimem.open(path, **kwargs)` creates the SQLite
file if it doesn't exist, and every capability beyond "remember facts,
recall past turns" is optional — see below.

## 3. Remember something

```python
mem.remember("name", "Priya")
mem.remember("favorite_color", "teal")

print(mem.facts())
# {'name': 'Priya', 'favorite_color': 'teal'}
```

## 4. Record a conversation turn

```python
mem.record_turn(
    "what's the capital of Japan?",
    "Tokyo is the capital of Japan.",
)
```

`record_turn` persists the exchange and runs free regex-based fact capture
(names, locations) inline — no LLM required.

## 5. Recall

```python
hits = mem.recall("capital of Japan")
for h in hits:
    print(h.kind, h.text[:80])
```

`recall()` never raises — a failed or degraded search just returns fewer
results, never an exception that could break your chat loop.

## 6. Wire it into a chat loop

The pattern every example in `examples/` follows:

```python
while True:
    user_input = get_user_input()

    plan = mem.build_context(user_input)   # budgeted prompt sections
    prompt = plan.render() + f"\nUser: {user_input}\nAssistant:"
    reply = my_llm(prompt)                  # your model call, any backend

    mem.record_turn(user_input, reply)
```

`build_context()` assembles facts, relevant past moments, session summaries,
and lessons into a token budget that's automatically sized to the machine
Elastimem is running on (see [governor.md](governor.md)) — you don't have to
think about how much context is "too much" for a small model on constrained
hardware.

## 7. End the session

```python
mem.end_session()   # summarizes, consolidates, closes the session
mem.close()          # stop background worker, close DB connections
```

## Adding an LLM and embedder

Everything above works with zero external dependencies. Pass an LLM and/or
embedder to unlock automatic fact extraction and semantic (vector) recall —
both are plain callables, so any backend works:

```python
mem = elastimem.open(
    "~/.myagent/memory.db",
    llm=my_complete_fn,       # (prompt, *, max_tokens, temperature) -> str
    embedder=my_embed_fn,     # (list[str]) -> list[list[float]]
    context_tokens=4096,      # match your model's context window
)
```

If you don't pass an embedder, Elastimem activates its own small built-in
one automatically (see [governor.md](governor.md)) — semantic recall works
out of the box, no setup required.

## Where to go next

- [installation.md](installation.md) — optional extras, supported Python versions
- [architecture.md](architecture.md) — how the pieces fit together
- [governor.md](governor.md) — how token budgets and degradation work
- [api.md](api.md) — every public method, in detail
- [api_stability.md](api_stability.md) — what's safe to depend on long-term
- `examples/` — runnable scripts: no-LLM, llama.cpp, OpenAI-compatible API, memory-only usage
