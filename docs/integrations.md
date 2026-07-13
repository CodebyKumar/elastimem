# Integration guide

Elastimem needs, at most, one callable from you — an LLM. Embeddings are
handled automatically:

```python
llm(prompt: str, *, max_tokens: int, temperature: float) -> str      # optional
embedder(texts: list[str]) -> list[list[float]]                      # optional — see below
```

**`llm` is fully optional** — without it you get rule-based fact capture,
full transcripts, and keyword (FTS5) episodic recall; no LLM extraction, no
LLM-condensed summaries, no consolidation merges.

**`embedder` is different: if you don't pass one, Elastimem activates its
own built-in embedder automatically** (real model — `BAAI/bge-small-en-v1.5`
via `fastembed`, not a hashing trick — see
[governor.md](governor.md#built-in-embedder-auto-activates-read-this-before-assuming-embeddings-are-off)
for the full story). This means semantic/paraphrase recall works out of the
box with zero setup, as long as:
1. the optional extra is installed (`pip install elastimem[embed]`), and
2. the store's tier is STANDARD or FULL (embeddings are never called at
   LITE, regardless of whether an embedder is configured — see
   [governor.md](governor.md#capability-tier)).

If you genuinely want FTS5-only behavior (no embeddings, ever, regardless
of what's installed), set `disable_builtin_embedder=True` — see below.

## Level 0 — no LLM at all

```python
import elastimem

mem = elastimem.open("~/.myapp/memory.db")
mem.record_turn(user_text, reply_text)   # transcripts + regex fact capture
ctx = mem.build_context(user_text)       # facts/lessons/episodic, FTS5 + (if
                                          # tier and the 'embed' extra allow)
                                          # the built-in embedder's vector leg
hits = mem.recall("that thing about the car")
```

You get: durable facts (rules + `remember`), full transcripts, hybrid
FTS5+vector episodic recall (vector leg via the built-in embedder if the
`embed` extra is installed and the tier allows it; FTS5-only otherwise —
either way, `build_context`/`recall` never raise and never require any
extra code from you). See `examples/minimal_bot.py`.

## Level 1 — llama.cpp host (single model instance)

The one hard rule: **your model instance is not thread-safe, so bracket your
own generation with the foreground gate.** Elastimem's background jobs then
interleave between turns.

```python
from llama_cpp import Llama
import elastimem

chat = Llama(model_path="model.gguf", n_ctx=4096, ...)

def llm(prompt, *, max_tokens, temperature):
    out = chat.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens, temperature=temperature)
    return out["choices"][0]["message"]["content"]

mem = elastimem.open(
    "~/.myagent/memory.db",
    llm=llm,
    context_tokens=4096,
    static_prompt_tokens=measured_prompt_tokens,
    reserved_keys={"model", "agent_name"},
)

# per turn:
mem.tick()
plan = mem.build_context(user_input)
system_prompt = base_prompt + "\n\n" + plan.render()
with mem.foreground():                       # hold background LLM jobs
    reply = generate(system_prompt, history[-plan.keep_last_n_turns*2:], user_input)
mem.record_turn(user_input, reply)

# if you trim your history list, tell Elastimem:
mem.report_evictions(evicted_pairs)          # -> rolling summary
# feed plan.rolling_summary back as a system-side note next turn

# on exit:
mem.end_session()      # drains worker, writes session summary, consolidates
mem.close()
```

**If your host can switch models mid-session** (a different `n_ctx`, e.g.
swapping a local model for a larger-context API model), budgets do **not**
update on their own — they're only recomputed on a tier change, which may
never happen. Call `reconfigure()` right after the switch:

```python
mem.drain()                      # finish queued work on the old model
mem.reconfigure(context_tokens=new_model_ctx,
               static_prompt_tokens=measured_prompt_tokens)
```

Forgetting this is not dangerous (nothing crashes), but a host that jumps
from a 4k-context local model to a 128k-context API model will silently keep
using the old, much smaller memory budgets until something else happens to
flip the tier.

Embeddings: **you usually don't need to do anything here.** If you don't
pass `embedder=`, Elastimem's built-in embedder (`fastembed` +
`BAAI/bge-small-en-v1.5`, ONNX Runtime — no torch, no extra GPU contention
with your chat model) activates automatically and runs entirely on the
background worker thread, so it never competes with your foreground
generation for the model instance itself. Install the extra
(`pip install elastimem[embed]`) and you're done.

Two reasons you might still want to supply your own `embedder=`:
- **Non-English or multilingual content** — the built-in model is
  English-tuned; a purpose-chosen multilingual embedding model will give
  materially better semantic recall for non-English-dominant use cases.
- **You already have an embedding model loaded for another reason** (e.g. a
  separate tiny embedding GGUF via
  `Llama(model_path=..., embedding=True, n_ctx=512, n_gpu_layers=0)`) and
  want to reuse it rather than let Elastimem load and cache its own.

If you supply your own `embedder=` and it fails (raises, or the model can't
load), the vector leg disables itself for the rest of the session (logged
once) and retrieval falls back to FTS5 — same floor as the built-in
embedder failing. To force FTS5-only behavior unconditionally (e.g. an
air-gapped environment that must never attempt the built-in embedder's
first-run download), set `disable_builtin_embedder=True` explicitly —
simply not passing `embedder=` is NOT enough to guarantee "no embeddings
ever," since that's exactly the case the built-in default fills.

On decode/OOM errors from your model, call `mem.report_pressure()` — the
governor sheds memory work first. Only call this for a **confirmed** memory
pressure signal from your inference backend, not for every exception your
generation call can raise — `report_pressure()` trusts every call at face
value (it rate-limits repeated calls within a 30s window, but does not and
cannot verify the call is actually about memory; see
[governor.md's Known limitations](governor.md#known-limitations-exhaustive-as-of-this-writing),
item 5). A host that catches a broad exception type and reports pressure
for every instance of it risks degrading memory quality (dropping to LITE,
disabling extraction/embeddings/consolidation) in response to failures that
have nothing to do with RAM.

See `examples/llama_cpp_bot.py` for a complete program.

## Level 2 — OpenAI-compatible / API host

API models don't contend for local RAM, so the foreground gate matters less
(still harmless). Wire `llm` to your chat endpoint; set `context_tokens` to
the model's window. You may pin `ELASTIMEM_TIER=full` since the machine
isn't running the model. For `embedder`, the same auto-activation applies as
everywhere else — leave it unset to get the built-in embedder (runs locally,
regardless of your chat model being remote), or wire your own
`embedder=` to an embeddings API endpoint if you'd rather keep embedding
calls on the same provider as chat.

## Exposing memory to your agent as tools

Two natural tool bindings:

- `remember(key, value)` → `mem.remember(key, value)[1]` (returns the reason
  string — "stored", "reserved key…", etc.)
- `memory_search(query)` → format `mem.recall(query)` hits as
  `- [date] text` lines. Works in every tier; this is LITE's only episodic
  access, so always register it.

## Host checklist

- [ ] Pass `context_tokens` (your `n_ctx`) and measured `static_prompt_tokens`.
      Check the actual resulting per-section budgets aren't hitting the
      256-token dynamic floor (see [governor.md](governor.md#budget-derivation))
      — a small model's context minus a large system prompt can leave almost
      nothing for memory even with everything else working correctly.
- [ ] Pass `reserved_keys` for identity fields your app owns.
- [ ] Call `tick()` once per turn; `report_pressure()` ONLY on a confirmed
      memory-pressure signal, never as a catch-all for any exception your
      inference call can raise (see the Level 1 section above).
- [ ] Bracket generation with `foreground()` when the model is local.
- [ ] Call `record_turn` after every exchange; `report_evictions` when you trim.
- [ ] Call `end_session()` on exit; `drain()` before unloading/switching models.
- [ ] Render `plan.sections` into your prompt (or use `plan.render()`).
- [ ] Decide your embedder story deliberately: leave `embedder=` unset (get
      the built-in default — install `elastimem[embed]` for it to actually
      activate), pass your own, or set `disable_builtin_embedder=True` if you
      want a guarantee of FTS5-only behavior. Don't assume "I didn't pass
      `embedder=`" means "no embeddings" — as of the built-in default, it
      doesn't.
- [ ] After any model switch, call `reconfigure_for_model`-equivalent logic
      on your side (rebuild `context_tokens`/`static_prompt_tokens` and call
      `mem.reconfigure(reprobe=True, ...)`) — including after any operation
      that reopens/reconstructs your `Elastimem` object (e.g. a "clear all
      memory" feature), not just an explicit model switch. A freshly
      reopened store resets to whatever `context_tokens` you pass it at
      that moment, not whatever was correct before.

## Example programs

- `examples/minimal_bot.py` — Level 0, no LLM at all.
- `examples/llama_cpp_bot.py` — full-capability, local `llama-cpp-python`.
- `examples/openai_agent.py` — any OpenAI-compatible HTTP endpoint (OpenAI,
  Groq, Together, a local vLLM/llama.cpp server) via a plain `requests` call
  — no vendor SDK required.
- `examples/memory_search.py` — `remember()`/`recall()`/`facts()` used
  directly, no chat loop, no LLM.

All four use the same `llm=` shape — `(prompt, *, max_tokens, temperature) ->
str` — so switching backends is a matter of swapping which example you start
from, not restructuring how you call Elastimem.
