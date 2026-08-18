# Elastimem — agent integration prompt

**Audience: a coding agent (Claude Code, Cursor, Copilot, an SDK agent) that
has been asked to add Elastimem to a project, or to upgrade an existing
integration.** Paste this whole file into the agent's context, or point the
agent at its URL. It is written to be self-sufficient: every capability,
every config knob, every gotcha, and the exact upgrade steps.

Current version: **0.2.0**. If the target project pins an older version,
read [§9 Upgrading](#9-upgrading-an-existing-integration) before anything
else.

---

## 1. What Elastimem is, in one paragraph

Elastimem is a local-first, resource-adaptive memory framework for AI
agents. It gives a chat application persistent memory — facts about the
user, past conversations, learned lessons, and a knowledge graph — stored
in **one SQLite file**, with **zero required dependencies**. Its defining
component is the **Memory Governor**, which measures available RAM and
continuously decides how much memory work the machine can afford, emitting
per-turn token budgets and capability flags. Every capability degrades to a
defined floor rather than failing, and after construction **nothing raises
into the host's chat loop**.

What it is *not*: a vector database, a RAG framework, a hosted service, or
something that manages your prompt for you. It hands you budgeted text
sections; you decide how to assemble your prompt.

---

## 2. Decision checklist — read before writing code

Answer these from the target project. They determine the whole integration.

| Question | Where it lands |
|---|---|
| What is the chat model's context window? | `context_tokens=` (**required for sane budgets**) |
| How many tokens is the project's fixed system prompt? | `static_prompt_tokens=` |
| Is there an LLM callable available for background work? | `llm=` (optional) |
| Is there already an embedding model in the process? | `embedder=` (optional; see §6) |
| Is there one shared, non-thread-safe model instance? | You **must** use `foreground()` — see §5 |
| Air-gapped, or must never download anything? | `disable_builtin_embedder=True` |
| Where should the DB file live? | first positional arg |

**The single most common integration bug is leaving `context_tokens` at its
4096 default when the host model has a 32K or 128K window.** Every budget
derives from it, so memory silently gets a fraction of the space available.

---

## 3. Install

Not on PyPI. Install from GitHub, pinned to a tag:

```bash
pip install "elastimem @ git+https://github.com/CodebyKumar/elastimem.git@v0.2.0"
```

Extras (all optional, all additive):

```bash
# psutil — accurate RAM probing. Effectively REQUIRED on Windows, which has
# no stdlib probe and otherwise assumes a fixed 8/4 GiB.
pip install "elastimem[system] @ git+https://github.com/CodebyKumar/elastimem.git@v0.2.0"

# fastembed — activates the built-in semantic embedder (BAAI/bge-small-en-v1.5,
# ONNX, no torch). ~130MB downloaded on first use, then cached.
pip install "elastimem[embed] @ git+https://github.com/CodebyKumar/elastimem.git@v0.2.0"

# combine
pip install "elastimem[system,embed] @ git+https://github.com/CodebyKumar/elastimem.git@v0.2.0"
```

`elastimem[vec]` is **declared but not wired in** — installing it currently
has no effect. Do not add it expecting a speedup.

Requires Python 3.10–3.13.

---

## 4. Minimum viable integration

```python
import elastimem

mem = elastimem.open(
    "~/.myapp/memory.db",
    llm=my_complete_fn,          # optional: (prompt, *, max_tokens, temperature) -> str
    embedder=my_embed_fn,        # optional: (list[str]) -> list[list[float]]
    context_tokens=8192,         # your model's real n_ctx
    static_prompt_tokens=900,    # measured size of your own system prompt
)

def chat(user_input: str) -> str:
    mem.tick()                                   # cheap RAM re-check, once per turn
    plan = mem.build_context(user_input)         # budgeted sections, never raises

    messages = [{"role": "system", "content": SYSTEM_PROMPT + "\n\n" + plan.render()}]
    if plan.rolling_summary:
        messages.append({"role": "system", "content": plan.rolling_summary})
    messages += window[-plan.keep_last_n_turns * 2:]
    messages.append({"role": "user", "content": user_input})

    with mem.foreground():                       # background LLM work pauses
        reply = my_model.generate(messages)

    mem.record_turn(user_input, reply)           # persist + extract, never raises
    return reply

# on shutdown
mem.end_session()
mem.close()
```

That is the whole required surface: `tick` → `build_context` → generate
inside `foreground()` → `record_turn`. Everything in §7 is optional depth.

---

## 5. The five host obligations

An agent integrating Elastimem must wire all five. Skipping any of them
causes a specific, known failure.

1. **`tick()` once per turn.** Without it the tier never re-evaluates, so
   the governor cannot react to memory pressure.
2. **Bracket your own generation in `foreground()`.** Elastimem runs
   background LLM jobs on a worker thread. Most local hosts have exactly one
   model instance and it is **not thread-safe** — concurrent calls corrupt
   llama.cpp state and produce garbled or truncated replies. `foreground()`
   holds a real lock, not an advisory flag. Use
   `foreground_begin()`/`foreground_end()` only when generation spans
   separate callbacks (token-by-token streaming) and cannot be bracketed in
   one `with` block; every begin must be matched by exactly one end.
3. **`report_evictions(turns)` when you drop turns from your window.** This
   is how evicted content reaches the rolling summary. Without it, dropped
   turns are still persisted and searchable, but the model loses the thread.
4. **`end_session()` before shutdown**, and `close()` after. `end_session()`
   drains the worker, writes a session summary, and runs exit consolidation.
   Killing the process without it loses queued background work.
5. **`report_pressure()` on OOM or decode failure.** Your explicit signal
   that the machine is in trouble; downgrades a tier immediately. Repeat
   calls within 30s coalesce, so a burst of retries won't cascade.

---

## 6. The Memory Governor and tiers — what an agent must understand

Elastimem classifies the machine as **FULL** (≥16 GiB total / ≥6 available),
**STANDARD** (≥8 / ≥2.5), or **LITE** (below that), re-checked every
`tick()`. Downgrades are immediate; upgrades need 10 consecutive healthy
ticks and never exceed the startup tier.

| Capability | FULL | STANDARD | LITE |
|---|---|---|---|
| Indexing new chunks (`embeddings_enabled`) | yes | yes | never |
| Vector leg when searching (`vector_recall_enabled`) | yes | yes | yes, over already-indexed chunks, if an embedder is already resident |
| Loading the built-in embedder (`embedder_load_allowed`) | yes | yes | never |
| Episodic injection (`episodic_top_k`) | top 5 | top 4 | top 3 |
| LLM fact extraction | per turn | batched every 2 turns | off by default; `lite_llm_extraction=True` opts in, deferred to session end |
| Rolling summary (`rolling_summary_mode`) | LLM | LLM | extractive, no model call |
| Consolidation | full, incl. LLM merge | dedupe + decay | dedupe + decay |
| Knowledge graph (`graph_hops`) | 2-hop | 1-hop | 1-hop |
| Rule capture, transcripts, `remember`/`recall`/`forget` | always | always | always |

**LITE's floor is "spend no new resources," not "least capability
possible."** It refuses exactly three things by default: an LLM call, a
model load, and any sustained per-item cost. Everything that is just local
SQLite runs there too.

**Three separate embedding flags, not one** — this trips up integrations:

- `embeddings_enabled` — may we embed **newly recorded chunks**? (`False` at LITE)
- `vector_recall_enabled` — may we run the **vector leg during retrieval**? (`True` everywhere)
- `embedder_load_allowed` — may we trigger a **first load** of the built-in model? (`False` at LITE)

A host-supplied `embedder=` is always considered resident — it lives in your
process, so declining to call it would free nothing. None of these flags
means an embedder is actually configured; check `mem.embed_fn is not None`
separately.

**The built-in embedder auto-activates.** If you pass no `embedder=`,
Elastimem wires in its own (`BAAI/bge-small-en-v1.5`) once the `embed`
extra is installed. Passing `embedder=None` explicitly does *not* opt out —
`None` is exactly what triggers activation. Use
`disable_builtin_embedder=True` to force FTS5-only.

Env override for testing: `ELASTIMEM_TIER=lite|standard|full`. Config
override: `tier_override=`. Both pin the tier and disable re-classification.

---

## 7. Complete capability list

Everything Elastimem can do. An integration does not need all of it, but an
agent should know it all exists before deciding what to skip.

### 7.1 The five memory layers

| Layer | Holds | Storage |
|---|---|---|
| **Working** | current window + rolling summary of evicted turns | host's message list, planned by Elastimem |
| **Episodic** | full past transcripts, chunked and indexed | `messages`, `chunks` (+FTS5, +vectors) |
| **Semantic** | facts about the user, temporally versioned, importance-decayed | `facts` |
| **Procedural** | lessons the agent learned about its own behavior | `lessons` |
| **Graph** | entities, relationships, topic clusters | `graph_nodes`, `graph_edges` |

### 7.2 Per-turn API

| Method | Purpose |
|---|---|
| `tick() -> MemoryProfile` | cheap hardware re-check; once per turn |
| `build_context(user_input="") -> ContextPlan` | budgeted sections + window plan; never raises |
| `record_turn(user_text, assistant_text)` | persist, rule-capture, enqueue extraction/embedding; never raises |
| `report_evictions(turns)` | fold evicted `(user, assistant)` pairs into the rolling summary |
| `foreground()` / `foreground_begin()` / `foreground_end()` | hold background LLM jobs during host generation |
| `report_pressure() -> MemoryProfile` | OOM/decode-failure signal; immediate downgrade |
| `reconfigure(*, reprobe=False, **overrides) -> MemoryProfile` | change config after construction and rebuild budgets |

### 7.3 Memory operations

| Method | Purpose |
|---|---|
| `remember(key, value, source="explicit") -> (changed, reason)` | validated, synchronous, durable fact write |
| `facts() -> dict[str, str]` | current facts |
| `fact_history(key) -> list[Fact]` | full version chain, oldest first |
| `forget(key) -> bool` | tombstone the current version |
| `recall(query, k=5) -> list[Hit]` | search chunks + facts; never raises; works at every tier |
| `timeline(query) -> TimelineResult` *(Experimental)* | resolve free text to a fact key, return its version history |
| `explain(query, k=5) -> ExplainResult` *(Experimental)* | `recall`'s ranking plus every per-leg score and the graph path |
| `clusters() -> list[dict]` *(Experimental)* | knowledge-graph topic clusters, largest first |
| `add_lesson(text, tag=None) -> bool` / `lessons(n=None) -> list[str]` | procedural memory |

### 7.4 Sessions

| Method | Purpose |
|---|---|
| `begin_session(host_tag=None) -> int` | explicit start (`record_turn` starts one lazily) |
| `end_session()` | drain worker, session summary, exit consolidation |
| `sessions(n=20) -> list[dict]` | recent sessions, newest first |
| `resume_session(session_id=None) -> (rolling_summary, tail_messages)` | reload a past session into the host's window |

### 7.5 Lifecycle and inspection

| Method | Purpose |
|---|---|
| `drain(timeout=5.0) -> bool` | finish queued background work (also flushes session-end-held jobs) |
| `close()` | drain, stop worker, close connections |
| `stats() -> dict` | keys: `path`, `db_bytes`, `fts_enabled`, `messages`, `chunks`, `facts`, `sessions`, `lessons`, `quarantine` |
| `quarantine_entries(n=20) -> list[dict]` | auto-extractions that failed validation |
| `profile -> MemoryProfile` | current governor snapshot |
| `config -> ElastimemConfig` | read-only snapshot; mutating it does nothing, use `reconfigure()` |

### 7.6 Value types

- **`ContextPlan`** — `sections: dict[str, str]` with keys `user_facts`,
  `graph_context`, `relevant_past_moments`, `previous_sessions`, `lessons`;
  plus `rolling_summary: str | None`, `keep_last_n_turns: int`,
  `profile: MemoryProfile`, `render() -> str`.
- **`MemoryProfile`** — `tier`, `budgets`, `embeddings_enabled`,
  `vector_recall_enabled`, `embedder_load_allowed`, `llm_extraction_enabled`,
  `extraction_cadence`, `rolling_summary_mode`, `rolling_summary_enabled`
  (derived), `consolidation_level`, `episodic_top_k`, `graph_hops`,
  `window_min_turns`.
- **`Budgets`** — `working`, `facts`, `episodic`, `sessions`, `lessons`
  (tokens), `memory_total` (derived).
- **`Hit`** — `kind` (`chunk`|`fact`), `text`, `date`, `score`, `session_id`.
- **`Fact`** — `key`, `value`, `category`, `source`, `importance`,
  `valid_from`, `invalidated_at`, `archived`.
- **`ExplainResult`** — `query`, `graph_hops`, `chunk_breakdowns`,
  `fact_breakdowns`, `graph_traversal`, `hits`.
- **`TimelineResult`** — `query`, `key`, `resolved_by`, `versions`.
- **Enums** — `Tier` (`LITE<STANDARD<FULL`), `Cadence`
  (`PER_TURN`/`BATCHED`/`SESSION_END`/`OFF`), `ConsolidationLevel`
  (`FULL`/`DEDUPE_ONLY`/`OFF`), `RollingSummaryMode`
  (`LLM`/`EXTRACTIVE`/`MARKER`).

### 7.7 Automatic behaviors (no host code required)

- **Regex rule capture** — ~10 high-precision patterns run on every turn at
  every tier, free: "my name is X", "call me X", "I live/stay in X", "I'm
  from X", "I work as/at X", "I'm an X by profession", "I'm allergic to X",
  "I'm vegetarian/vegan/pescatarian", "my favorite X is Y".
- **LLM fact extraction** — background pass over each exchange, producing
  facts *and* graph entities/relationships in **one** completion.
- **Fact validation and quarantine** — rejected extractions are stored for
  inspection, not silently dropped.
- **Temporal versioning** — facts are never overwritten; old versions are
  invalidated, so `fact_history()`/`timeline()` always work.
- **Importance decay and archival** — facts and graph nodes/edges decay from
  last reinforcement and archive below a threshold.
- **Automatic schema migration** — v1/v2/v3 stores migrate losslessly on open.
- **Corrupt-DB recovery** — a corrupt file is renamed
  `<path>.corrupt-<ts>` and a fresh store is created, with a warning.

---

## 8. Full configuration reference

Every field, with its default. Pass any of them inline to
`elastimem.open(...)` or via `config=ElastimemConfig(...)`.

**Budget inputs**
| Field | Default | Notes |
|---|---|---|
| `context_tokens` | `4096` | your model's real context window. **Set this.** Must be ≥1024 |
| `output_reserve` | `512` | generation headroom |
| `tool_reserve` | `600` | mid-turn tool observations (ReAct etc.) |
| `static_prompt_tokens` | `0` | your own fixed prompt, measured by you |
| `working_share` | `0.55` | fraction of the dynamic pool for the working window |
| `memory_split` | `facts .40, episodic .30, sessions .15, lessons .15` | must sum to 1.0, exactly these keys |

**Fact hygiene**
| Field | Default | Notes |
|---|---|---|
| `profile_keys` | `{name, email, location, age, occupation, pronouns}` | always injected |
| `reserved_keys` | `frozenset()` | host-owned keys Elastimem must reject |
| `quarantine_cap` | `200` | |
| `fact_merge_review_window_days` | `7.0` | lookback for LLM contradiction-merge |

**Episodic / retrieval**
| Field | Default | Notes |
|---|---|---|
| `chunk_target_tokens` | `400` | split exchanges bigger than this |
| `min_query_words` | `4` | gates **LLM extraction** (a real cost) |
| `min_retrieval_query_words` | `1` | gates **local retrieval** (free) — deliberately much lower |

**Embeddings**
| Field | Default | Notes |
|---|---|---|
| `disable_builtin_embedder` | `False` | `True` forces FTS5-only, never downloads |

**Forgetting**
| Field | Default |
|---|---|
| `fact_decay_half_life_days` | `60.0` |
| `fact_archive_threshold` | `0.15` |
| `episodic_recency_half_life_days` | `30.0` |
| `fact_recency_half_life_days` | `90.0` |

**Governor**
| Field | Default | Notes |
|---|---|---|
| `tier_override` | from `ELASTIMEM_TIER` | pins the tier |
| `upgrade_healthy_ticks` | `10` | |
| `idle_consolidate_seconds` | `90.0` | |
| `tier_thresholds_gib` | `full (16, 6)`, `standard (8, 2.5)` | `(total, available)` |

**Background worker**
| Field | Default | Notes |
|---|---|---|
| `worker_max_tokens` | `96` | cap on **every** background LLM call |
| `batched_every_n_turns` | `2` | STANDARD cadence |
| `lite_llm_extraction` | `False` | **new in 0.2.0** — opt LITE into session-end extraction |

**Knowledge graph**
| Field | Default |
|---|---|
| `graph_node_cap` | `2000` |
| `graph_edge_cap` | `5000` |
| `graph_decay_half_life_days` | `30.0` |
| `graph_archive_threshold` | `0.15` |
| `graph_merge_review_window_days` | `7.0` |

**Procedural**
| Field | Default |
|---|---|
| `max_lessons` | `15` |
| `lessons_in_prompt` | `5` |

---

## 9. Upgrading an existing integration

### Step 1 — find the installed version

```python
import elastimem; print(elastimem.__version__)
```

Also check the project's `requirements.txt` / `pyproject.toml` / lockfile
for a pinned `@v...` ref. `0.0.0+dev` means an editable checkout with no
distribution metadata — check `git describe --tags` in that checkout.

### Step 2 — upgrade the pin

```bash
pip install --upgrade "elastimem @ git+https://github.com/CodebyKumar/elastimem.git@v0.2.0"
```

Update the pin in the project's dependency file too, not just the
environment.

### Step 3 — apply the version-specific migration below

### `0.1.0a1` → `0.2.0`

**No code changes are required.** The stores migrate automatically, and
every 0.1.0a1 call site keeps working. Everything below is optional
adoption. Two things to be aware of:

- `MemoryProfile.rolling_summary_enabled` became a **derived property**.
  Reading it is unchanged and still means "does this cost a model call?"
  Only code that *constructed* `MemoryProfile(rolling_summary_enabled=...)`
  breaks — that is not a supported thing to do, but grep for it.
- LITE tier now does noticeably more work (all of it local SQLite). If the
  project deliberately pinned `ELASTIMEM_TIER=lite` as a "do almost
  nothing" mode, re-check that assumption — it now means "spend no new
  resources," not "least capability possible."

**Optional new capabilities to adopt:**

1. **Check the new profile flags** if the project branches on
   `embeddings_enabled` to decide whether to show a "semantic search on"
   indicator. Indexing and querying are now separate:

   ```python
   # before
   if mem.profile.embeddings_enabled: ...
   # after — for "can this search semantically right now?"
   if mem.profile.vector_recall_enabled and mem.embed_fn is not None: ...
   ```

2. **Consider `lite_llm_extraction=True`** if the project classifies as
   LITE because of *other* processes rather than a genuinely small machine,
   or if its model is unloaded between sessions. Extraction is deferred to
   `end_session()` and never competes with foreground generation.

3. **`rolling_summary_mode`** if the project displayed the old
   `[N turns omitted]` marker in a UI — LITE now produces real extractive
   text instead.

### From `0.1.0a1` if the graph features were never adopted

0.1.0a1 also shipped the knowledge graph. If the integration predates it,
consider adopting:

- `plan.sections["graph_context"]` — a "RELATED TOPICS" block. Already
  included in `plan.render()`; only relevant if the project assembles
  sections manually by key.
- `mem.explain(query)` for a retrieval-debugging view.
- `mem.timeline(query)` for "what did I do before X" questions.
- `mem.clusters()` for topic grouping.

### Step 4 — verify

```python
import elastimem
mem = elastimem.open("path/to/existing.db", context_tokens=8192)
print(elastimem.__version__)          # 0.2.0
print(mem.stats())                    # existing rows intact, schema migrated
print(mem.profile)                    # new flags present
print(mem.facts())                    # pre-upgrade facts still readable
mem.close()
```

---

## 10. Anti-patterns

Things agents get wrong. Check each before declaring an integration done.

- **Leaving `context_tokens` at 4096** when the model's window is larger.
  Everything downstream is starved.
- **Generating without `foreground()`.** Works in testing, corrupts output
  under load. Non-negotiable with a single model instance.
- **Reassigning `mem.embed_fn` / `embed_query_fn` / `tokenizer_fn` /
  `mem.path` after construction.** Raises `AttributeError` by design — the
  worker reads them unsynchronized. Open a new store instead.
- **Mutating `mem.config` directly.** It is a snapshot; the write is
  silently ignored. Use `reconfigure(**overrides)`.
- **Assuming construction never raises.** `Elastimem(path, ...)` *can*
  raise (unwritable path, unknown config key → `TypeError`). The "never
  raises" guarantee covers `build_context`/`record_turn`/`recall` *after*
  construction succeeds.
- **Passing `embedder=None` to disable embeddings.** That is exactly what
  activates the built-in. Use `disable_builtin_embedder=True`.
- **Forgetting `end_session()`.** Queued background work is lost.
- **Calling `reconfigure()` without `reprobe=True` right after loading a
  big model.** The startup probe measured RAM the model had not yet
  claimed, so the tier is stale.
- **Registering no `memory_search` tool.** `recall()` reaches the *full*
  history; injection only reaches what fits the budget. Always expose it.
- **Treating a `TypeError` from an unknown config key as a bug.** It lists
  the valid options — read the message.

---

## 11. Verification script

Run this after integrating. It exercises every layer and asserts the
governor is behaving.

```python
import elastimem
from elastimem import Tier

mem = elastimem.open(":memory:", context_tokens=8192)

# semantic
assert mem.remember("favorite_color", "blue")[0]
assert mem.facts()["favorite_color"] == "blue"
mem.remember("favorite_color", "green")
assert len(mem.fact_history("favorite_color")) == 2      # versioned, not overwritten

# episodic + rule capture
mem.record_turn("my name is Priya and I live in Mysore", "Nice to meet you!")
assert mem.facts()["name"] == "Priya"
assert mem.facts()["location"] == "Mysore"
assert mem.recall("where does the user live")

# procedural
mem.add_lesson("Do not suggest restarting before checking the logs.")
assert mem.lessons()

# context assembly
plan = mem.build_context("tell me about myself")
assert plan.sections["user_facts"]
assert plan.keep_last_n_turns >= 2
assert plan.profile.tier in (Tier.LITE, Tier.STANDARD, Tier.FULL)
print(plan.render())

# governor
print("tier:", mem.profile.tier.name, "budgets:", mem.profile.budgets)
print("stats:", mem.stats())

mem.end_session()
mem.close()
print("Elastimem integration OK")
```

---

## 12. Where to read more

| Doc | Covers |
|---|---|
| [quickstart.md](quickstart.md) | shortest path to a working store |
| [installation.md](installation.md) | extras, editable installs, platform notes |
| [integrations.md](integrations.md) | Level 0 (no LLM), llama.cpp, OpenAI-compatible; host checklist |
| [governor.md](governor.md) | normative tier spec, budget math, degradation matrix, known limitations |
| [tiers.md](tiers.md) | task-first companion: what each capability does per tier |
| [api.md](api.md) | full public API reference |
| [api_stability.md](api_stability.md) | Stable / Experimental / Internal boundaries |
| [architecture.md](architecture.md) | pipeline design and rationale |
| [schema.md](schema.md) | SQLite tables, versioning, graph storage |
| [CHANGELOG.md](../CHANGELOG.md) | release history |

---

## 13. Maintaining this document

**This file must be updated in the same commit as any user-visible change.**
It is the single artifact downstream agents rely on to discover what
Elastimem can do; if it drifts, every integration built from it silently
misses features.

Update when you change any of:

- a public method, its signature, or its return shape → §7
- an `ElastimemConfig` field, or its default → §8
- a `MemoryProfile` field, or per-tier behavior → §6 and §7.6
- the version number → the header, §3, and §9
- anything requiring migration or adoption steps → a new subsection in §9
- a new failure mode worth warning about → §10

Keep §9 append-only: add a new `X → Y` subsection per release rather than
rewriting the old ones, so an agent upgrading across several versions can
follow the chain.
