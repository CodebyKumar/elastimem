# Integration guide

Engram needs, at most, two callables from you:

```python
complete_fn(prompt: str, *, max_tokens: int, temperature: float) -> str
embed_fn(texts: list[str]) -> list[list[float]]
```

Both optional. Everything below is progressive enhancement.

## Level 0 — no LLM at all

```python
from engram import Engram

mem = Engram("~/.myapp/memory.db")
mem.record_turn(user_text, reply_text)   # transcripts + regex fact capture
ctx = mem.build_context(user_text)       # facts/lessons/episodic via FTS5
hits = mem.recall("that thing about the car")
```

You get: durable facts (rules + `remember`), full transcripts, keyword
episodic recall, budgeted context sections. See `examples/minimal_bot.py`.

## Level 1 — llama.cpp host (single model instance)

The one hard rule: **your model instance is not thread-safe, so bracket your
own generation with the foreground gate.** Engram's background jobs then
interleave between turns.

```python
from llama_cpp import Llama
from engram import Engram, EngramConfig

llm = Llama(model_path="model.gguf", n_ctx=4096, ...)

def complete_fn(prompt, *, max_tokens, temperature):
    out = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens, temperature=temperature)
    return out["choices"][0]["message"]["content"]

mem = Engram("~/.myagent/memory.db",
             complete_fn=complete_fn,
             config=EngramConfig(context_tokens=4096,
                                 static_prompt_tokens=measured_prompt_tokens,
                                 reserved_keys=frozenset({"model", "agent_name"})))

# per turn:
mem.tick()
plan = mem.build_context(user_input)
system_prompt = base_prompt + "\n\n" + plan.render()
with mem.foreground():                       # hold background LLM jobs
    reply = generate(system_prompt, history[-plan.keep_last_n_turns*2:], user_input)
mem.record_turn(user_input, reply)

# if you trim your history list, tell Engram:
mem.report_evictions(evicted_pairs)          # -> rolling summary
# feed plan.rolling_summary back as a system-side note next turn

# on exit / model switch:
mem.end_session()      # drains worker, writes session summary, consolidates
mem.close()
```

Embeddings: load a *separate tiny* embedding model (e.g. a MiniLM-class GGUF,
`Llama(model_path=..., embedding=True, n_ctx=512, n_gpu_layers=0)`) so
embedding never contends with chat generation, and pass its
`create_embedding` through `embed_fn`. If loading it fails or RAM is tight,
just don't pass `embed_fn` — retrieval stays FTS5.

On decode/OOM errors from your model, call `mem.report_pressure()` — the
governor sheds memory work first.

See `examples/llama_cpp_bot.py` for a complete program.

## Level 2 — OpenAI-compatible / API host

API models don't contend for local RAM, so the foreground gate matters less
(still harmless). Wire `complete_fn` to your chat endpoint and `embed_fn` to
an embeddings endpoint; set `context_tokens` to the model's window. You may
pin `ENGRAM_TIER=full` since the machine isn't running the model.

## Exposing memory to your agent as tools

Two natural tool bindings:

- `remember(key, value)` → `mem.remember(key, value)[1]` (returns the reason
  string — "stored", "reserved key…", etc.)
- `memory_search(query)` → format `mem.recall(query)` hits as
  `- [date] text` lines. Works in every tier; this is LITE's only episodic
  access, so always register it.

## Host checklist

- [ ] Pass `context_tokens` (your `n_ctx`) and measured `static_prompt_tokens`.
- [ ] Pass `reserved_keys` for identity fields your app owns.
- [ ] Call `tick()` once per turn; `report_pressure()` on model OOM.
- [ ] Bracket generation with `foreground()` when the model is local.
- [ ] Call `record_turn` after every exchange; `report_evictions` when you trim.
- [ ] Call `end_session()` on exit; `drain()` before unloading/switching models.
- [ ] Render `plan.sections` into your prompt (or use `plan.render()`).
