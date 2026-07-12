# Elastimem Architecture

Elastimem is a memory *library* for AI agents, not an agent runtime. It owns one
SQLite file and answers two questions for its host, every turn:

1. **What should the model see right now?** → `build_context(user_input)`
2. **What just happened that's worth keeping?** → `record_turn(user, reply)`

Everything else — extraction, summarization, embedding, consolidation,
forgetting — happens behind those two calls, paced by the machine's actual
capacity.

## Design principles

- **Host-agnostic.** Elastimem never loads a model, imports an inference
  library, or calls an API. The host injects `llm` (text completion)
  and `embedder` (embeddings) as plain callables. Both are optional.
- **Elastic.** The [Memory Governor](governor.md) probes RAM and the model's
  context size, then budgets every capability. Same code on a 4 GB Jetson
  and a 128 GB workstation.
- **Defined floors.** Every capability degrades to a documented fallback
  rather than failing (see the [degradation matrix](governor.md#degradation-matrix)).
  `record_turn`, `recall`, and `build_context` never raise into the host.
- **Never destroy.** Fact updates version; forgetting archives; corrupt
  databases are quarantined and rebuilt. The raw transcript is always kept,
  so future (better) models can re-index the past.

## The four memory layers

```
                       ┌───────────────────────────────┐
   host message list ◄─┤ WORKING   window plan +       │
                       │           rolling summary     │
                       ├───────────────────────────────┤
 build_context() ◄─────┤ SEMANTIC  versioned facts     │  facts, facts_fts
                       ├───────────────────────────────┤
                       │ EPISODIC  transcripts+chunks  │  messages, chunks, chunks_fts
                       ├───────────────────────────────┤
                       │ PROCEDURAL lessons            │  lessons
                       └───────────────────────────────┘
```

### Working memory (`working` plan inside `ContextPlan`)
Elastimem does not own the host's message list. `build_context()` returns
`keep_last_n_turns` (how many verbatim turns the working budget affords) and
`rolling_summary` (an LLM-condensed digest of turns the host evicted and
reported via `report_evictions()`). Without an LLM, the rolling summary
degrades to a marker line telling the model that older turns exist and are
searchable.

### Episodic memory (`episodic.py`)
`record_turn()` writes each exchange verbatim to `messages` and condensed to
`chunks` (~1 chunk per exchange, sentence-split beyond
`chunk_target_tokens`). Chunks are indexed by FTS5 and, when an embedder
exists, by float32 vector BLOBs. Retrieval at turn start injects the top
chunks from *previous* sessions as `RELEVANT PAST MOMENTS` (the current
session is already in the window). `recall(query)` searches everything,
including the current session — it backs an agent-visible `memory_search`
tool and user-facing `/memory search` commands.

### Semantic memory (`semantic.py`)
Facts are `key → value` rows with **temporal versioning**: an update
invalidates the old row (`invalidated_at`, `invalidated_by`) and inserts a
new one. `fact_history(key)` is the audit chain; `forget(key)` tombstones.
Profile-category facts (name, location, …) are always injected; note facts
compete by `relevance × importance × recency`. Never-accessed auto-extracted
facts decay and archive; explicit facts never auto-archive.

Three write paths, in increasing cost:
1. **Rules** (`rules.py`) — ~10 compiled regexes on the user text, inline,
   microseconds, every tier ("my name is…", "I'm allergic to…").
2. **LLM extraction** (`extraction.py`) — a background reflection pass per
   exchange, validated by `guards.py`, cadence set by the governor.
3. **Explicit** — `remember(key, value)`, synchronous and durable.

All three funnel through the same guards: placeholder junk, transcript
echoes, self-referential values, and host-reserved keys are rejected —
automatic rejects land in `quarantine` for inspection.

### Procedural memory (`procedural.py`)
Short operational lessons (`add_lesson`), deduplicated, oldest archived
beyond the cap, top-N injected.

## The background worker (`worker.py`)

One daemon thread, one queue. Job kinds: `extract`, `rolling_summary`,
`session_summary`, `embed`, `consolidate`.

The critical invariant is **foreground-wins**: most local hosts have exactly
one model instance and it is not thread-safe. The host brackets its own
generation with `with mem.foreground():`; while held, the worker will not
start LLM jobs (it holds them and polls). Background LLM calls are capped at
`worker_max_tokens` (default 96) so even a mistimed overlap costs
sub-seconds on a 2B model. `drain()` finishes the queue before the host
unloads its model; `close()` drains and stops.

## Consolidation

Runs at session end (all tiers except OFF-level work) and after 90 s of
idle in FULL tier:
1. decay/archival math over auto facts (no LLM),
2. FULL only: LLM contradiction-merge for keys whose value changed within a
   week ("lives in Austin" vs "moving to Austin in May" → merged value,
   still versioned),
3. cap maintenance (quarantine ≤ 200).

## Data flow per turn

```
user input
  │
  ├─ mem.tick()                    governor re-checks RAM (cheap)
  ├─ plan = mem.build_context(q)   FTS+vector retrieval, budgeted sections
  ├─ host renders plan into its system prompt, trims window to plan
  ├─ with mem.foreground():        model generates the reply
  ├─ mem.record_turn(q, reply)     persist + rule capture (inline, fast)
  │     └─ worker: extract facts, embed chunks     (background)
  └─ mem.report_evictions(...)     if the host trimmed its window
        └─ worker: fold into rolling summary       (background)
```

## Reference

- [governor.md](governor.md) — tiers, budget math, degradation matrix
- [schema.md](schema.md) — every table and index
- [integration.md](integration.md) — wiring guides (llama.cpp, API, no-LLM)
- [api.md](api.md) — public API reference
