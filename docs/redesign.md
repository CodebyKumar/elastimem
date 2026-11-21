# Elastimem 0.3 redesign

Status: draft. Records the decisions made so far, the reasoning behind them,
and the change plan. Items marked **open** still need a decision.

## 1. Summary

Elastimem gives a local AI agent long-term memory in a single SQLite file.
Version 0.2 works, but its behavior is hard to predict: a RAM-based governor
switches off features that cost no RAM, ranking is spread across three
differently written formulas, some derived data is never produced, and
English word lists are wired into validation and search.

The redesign rests on four ideas:

1. **Messages are the source of truth.** Everything else (chunks, vectors,
   facts, graph, summaries) is derived from them and can be rebuilt.
   Pending work is whatever derived data is missing, read straight from
   SQL, so a crash loses nothing.
2. **One scoring equation, with fixed weights per tier.** Every memory type
   is ranked by the same formula, so any result can be explained term by
   term.
3. **Each capability is gated by what it actually costs.** RAM gates loading
   a model, measured time gates heavy per-turn work, and the context window
   gates token budgets.
4. **No word lists.** Checks are structural (is this fact grounded in what
   the user said?) or data-driven (is this term common in this store?), not
   lookups in hand-written English vocabularies.

## 2. Findings in 0.2 (measured on an 8 GB Mac, Nov 2026)

| # | Finding | Effect |
|---|---|---|
| F1 | The tier is chosen from "available RAM". On macOS that counts only free + inactive pages and ignores compressed memory | An 8 GB Mac reads 1.5 GB available and is LITE permanently |
| F2 | LITE disables LLM extraction and LLM summaries | These use the host's already-loaded model and add no RAM |
| F3 | LITE never embeds chunks and never loads the built-in embedder | Vector search never runs on LITE machines |
| F4 | `embed_pending()` exists but is never called | Chunks recorded while on LITE never get vectors, even after a tier upgrade |
| F5 | STANDARD keeps batched extraction jobs in an in-memory list | A crash loses them, and missed turns are never re-extracted |
| F6 | The extractive (LITE) rolling summary is never written to the database | `resume_session()` returns no summary on LITE |
| F7 | Vector scoring is a pure-Python cosine over every chunk, every turn | 93 ms at 5k chunks, 185 ms at 10k (384-dim vectors), about 85% of a turn's cost |
| F8 | Graph seed detection scans all nodes in Python and runs twice per turn. Graph scoring does a `LIKE '%name%'` scan of chunks per entity | Cost grows with graph size and chunk count |
| F9 | Ranking exists in three versions: chunks (max + nudges), facts (product), `explain()` (copied fact code) | Hard to reason about and easy to drift; a comment in `assembly.py` already describes LITE behavior that no longer holds |
| F10 | Validation and search depend on English word lists (Section 9) | Brittle, English-only, and each list needs maintenance |
| F11 | `store.py` (666 lines) and `retrieval.py` (701 lines) each mix several jobs | Hard to see where a behavior lives |

## 3. Terminology

0.2 used "semantic" for two unrelated things, which caused confusion.

| Old term | New term | Meaning |
|---|---|---|
| Semantic memory | **Facts** | Key-value statements about the user (`location: Berlin`) |
| Semantic search | **Vector search** | Finding text by meaning, using embeddings |
| Procedural memory | **Lessons** | Notes the agent keeps about its own behavior |
| Episodic memory | **Episodes** | What was said: messages and their chunks |

## 4. Memory layers

Each layer answers one question.

| Layer | Question it answers | Stored in | Found by | Vectors |
|---|---|---|---|---|
| Working | What is happening right now? | Not stored by Elastimem (the host's message list) | n/a; Elastimem returns `keep_last_n_turns` and the rolling summary | No |
| Episodes | What happened before? | `messages` (verbatim), `chunks` (search units) | Keyword + vector + graph, combined by the equation | **Yes** |
| Facts | What is true about the user now? | `facts` (versioned, never deleted) | Keyword + vector, combined by the same equation | **Yes (new in 0.3)** |
| Graph | What is connected to what? | `graph_nodes`, `graph_edges`, `graph_mentions` (new) | Entity names in the query, then SQL hops along edges | No |
| Lessons | What should the agent do differently? | `lessons` | Most recent N (no search) | No |
| Sessions | What were past conversations about? | `sessions.summary` | Most recent N (no search) | No |

Vectors are a search method, not a layer. They earn their cost only when the
wording of a question differs from the stored text ("how is my dog?" vs. "I
adopted a puppy"). That happens with episodes and with facts ("where do I
live?" vs. key `location`), so those are the only two layers that get
vectors.

## 5. Data flow

### 5.1 Write path

`record_turn(user, assistant)` synchronously writes:

- two `messages` rows (the source of truth)
- one or more `chunks` rows; FTS5 indexing happens through triggers

Nothing else is written inline. All derived work is found by query:

| Derived data | Pending when | Done by |
|---|---|---|
| Chunk vector | `chunks.embedding IS NULL` | Worker, when an embedder is resident or may be loaded |
| Fact vector | `facts.embedding IS NULL` | Worker, same rule |
| LLM extraction (facts + graph) | `chunks.extracted_at IS NULL` | Worker, when an LLM is present, at the tier's cadence |
| Rolling summary | Evicted turns reported | Inline (extractive) or worker (LLM); **always persisted** |
| Session summary | Session ended without one | `end_session()`, or on the next startup |

The worker's queue becomes a wake-up signal only. If the process dies, the
next start finds the same missing rows and continues. Moving up a tier
backfills automatically (fixes F4, F5, F6).

### 5.2 Read path

```
query
  -> analyse once: query terms, query vector (if allowed), graph seeds
  -> candidates: keyword hits, vector hits, graph-linked chunks
  -> signals: each scaled to 0-1
  -> score = tier equation
  -> fill each prompt section to its token budget
```

The query is analysed once per turn and shared by every section and by
`recall()`. Today each section repeats that work (fixes part of F8).

## 6. Scoring

### 6.1 Signals

Every signal is in 0-1.

| Signal | Source | Scaling |
|---|---|---|
| `keyword` | FTS5 BM25 score | Divided by the best score for this query (BM25 is not comparable across queries) |
| `vector` | Dot product of normalized vectors | Clamped at 0, otherwise unscaled: already 0-1 and comparable between moments and facts |
| `graph` | Chunk or fact mentions an entity reachable from a query seed | Entity confidence x 0.5 per hop, divided by the best |
| `importance` | Provenance (explicit 1.0, import 0.8, auto 0.5) and whether the chunk produced a fact | Already 0-1 |
| `recency` | `exp(-age_days / half_life)` | Already 0-1 |

### 6.2 Equation

```
relevance = w_vector * vector + w_keyword * keyword + w_graph * graph
score     = relevance + w_importance * importance + w_recency * recency
```

Only items with `relevance > 0` are ranked. Importance and recency break ties
between relevant items; they can never lift an irrelevant item into the
results. 0.2 had to fix this bug once already.

### 6.3 The equation per tier

The weights are constants in `tiers.py`. They were tuned on the evaluation
set (Section 15): vector 0.8 / keyword 0.1 gave the best MRR, and nothing
in the data justified different weights per tier, so every tier uses the
same equation. A tier changes which terms are *available* (whether the
vector term can run, how far the graph term reaches), not what a term is
worth.

| Tier | w_vector | w_keyword | w_graph | w_importance | w_recency | Graph hops |
|---|---|---|---|---|---|---|
| FULL | 0.80 | 0.10 | 0.10 | 0.05 | 0.05 | 2 |
| STANDARD | 0.80 | 0.10 | 0.10 | 0.05 | 0.05 | 1 |
| LITE | 0.80 | 0.10 | 0.10 | 0.05 | 0.05 | 1 |

The graph weight is untested: the evaluation runs without an LLM, so it
has no graph. When a term found nothing for a query (no embedder, no
entities named), its weight is shared among the remaining relevance terms
in proportion to their weights. That is the only runtime adjustment.

Facts and chunks use the same equation. `explain()` shows the tier, each
term's weighted contribution, and which terms were missing and why.

## 7. Governor

### 7.1 Signals

| Signal | Read from | Used for |
|---|---|---|
| Memory pressure | OS pressure level (macOS `kern.memorystatus_vm_pressure_level`, Linux PSI / `MemAvailable`, or psutil) | Whether a model may be loaded |
| Measured cost | Elastimem times its own per-turn work and background LLM calls | Lowering the tier when memory work exceeds a per-turn budget (default 50 ms) for several turns |
| Battery | `psutil.sensors_battery()` when psutil is installed | Moving background LLM work to session end while on battery |
| Host capabilities | Whether `llm=` / `embedder=` were given | Which terms and jobs exist at all |
| Context window | `context_tokens`, `static_prompt_tokens` | Token budgets (the same on every tier) |

Measured cost is the main signal. It is the same on every kind of machine
and costs nothing to collect, because the work is being done anyway.

### 7.2 What each tier allows

| Capability | FULL | STANDARD | LITE |
|---|---|---|---|
| Load the built-in embedder | Yes | Yes | No |
| Use an embedder that is already resident (host-supplied, or already loaded) | Yes | Yes | Yes |
| Embed new chunks and facts | Yes | Yes | If resident |
| LLM extraction (when `llm=` is given) **open, see D1** | Every turn | Every N turns | Session end |
| Rolling summary | LLM | LLM | Extractive |
| Consolidation with LLM merges | Yes | No | No |
| Graph hops | 2 | 1 | 1 |
| Token budgets | From the context window | Same | Same |

## 8. Memory without an LLM

Elastimem works without any LLM call: on LITE, or with no `llm=`, 0.2 makes
none. What that mode can and cannot do:

| Works without an LLM | Needs an LLM |
|---|---|
| Verbatim storage of every exchange | Pulling structured facts out of free conversation |
| Keyword + vector search over episodes | Graph entities and relationships |
| Facts the host stores explicitly with `remember()` | Abstractive session and rolling summaries |
| Extractive rolling summaries | Merging contradictory fact versions |
| Lessons, sessions, version history | Cluster labels |

Recall of what happened depends on episodes plus vector search, not on an
LLM. With a good embedder, LLM-free memory answers most "what did I say
about X" questions. LLM extraction adds structure (an up-to-date profile, an
entity graph) on top of that.

How 0.3 strengthens the LLM-free mode:

- **Host tool calls.** The host's own model already runs every turn. Exposing
  `remember()` as a tool lets it store facts during its normal reply, with no
  extra LLM call by Elastimem. The integration prompt documents this.
- **Fact vectors** make facts findable by meaning without extraction.
- **Recency in the equation** makes the newest of two conflicting episodes
  rank first ("lived in Berlin" vs. "moved to Paris").

Whether LLM-free memory is "good" should be measured, not argued: the
evaluation set (Section 11) reports results with and without an LLM.

## 9. Removing hard-coded words

| 0.2 location | What it does | 0.3 replacement |
|---|---|---|
| `retrieval._STOPWORDS` (~80 English words, plus "user" and "assistant") | Drops common words from queries | **Data-driven:** drop query terms that appear in more than half the chunks (read from an `fts5vocab` table). If every term is dropped, keep the rarest one |
| `"User: ... Assistant: ..."` labels inside chunk text | Pollute keyword search | Handled by the term-frequency filter above: the labels appear in every chunk |
| `guards._PLACEHOLDER_VALUES` ("unknown", "n/a", ...) | Rejects empty-meaning facts | **Grounding check:** an auto-extracted value must share a token with the user's message |
| `guards._TRANSCRIPT_KEY_MARKERS` ("message", "reply", ...) | Rejects facts that copy the exchange | Grounding check, plus: reject a value that is most of the user's message |
| `guards._SELF_REFERENTIAL_MARKERS` ("ai assistant", ...) | Rejects facts about the agent | Grounding check: the value must come from the user's text, not the assistant's |
| `guards._VOWELLESS_ALLOWLIST` and the vowel check | Rejects garbled keys | Removed. Grounding catches garbled output, and quarantine keeps it inspectable |
| `rules.py` (11 English regexes plus 2 word lists) | Captures name, location, ... without an LLM | **Removed (open, see D2).** Replaced by host tool calls (Section 8) |
| `graph._ARTICLES` ("the", "a", "an") | Normalizes entity names | Casefold + collapse whitespace only |
| `config.DEFAULT_PROFILE_KEYS` (name, email, ...) | Chooses always-shown facts | Explicit facts (`remember()`) lead the section. Hosts may pin keys; the default is empty |
| FTS tokenizer `porter` | English stemming | Kept for now as config `fts_tokenizer`, default unchanged. Multi-language is out of scope |

What is not a word list and stays: the source importance table (provenance,
not vocabulary), the entity type enum (a schema value), sentence splitting
on punctuation, and the LLM prompts.

Magic numbers (0.15 graph cap, 0.1 recency nudge, 0.3 fact floor, LITE
budget shares 0.45/0.75, RRF k=60) disappear into the tier table in
Section 6.3 or are removed.

## 10. Efficiency

| Hot spot (0.2) | Change | Expected (estimate, to verify) |
|---|---|---|
| Pure-Python cosine over all chunks (F7) | Store normalized vectors; keep an in-RAM matrix (numpy, which fastembed already requires); pure-Python dot product fallback | 185 ms -> a few ms at 10k chunks; matrix ~15 MB per 10k chunks, counted as RAM |
| Graph seeds: Python scan of all nodes, twice per turn | Same scan, once per turn (14) | Half |
| Graph scoring: `LIKE '%name%'` over chunks | `graph_mentions(node_id, chunk_id)` written at extraction time | Index lookup |
| `tick()` on macOS without psutil spawns 2 subprocesses | One pressure read, rate-limited (at most every few seconds) | Near zero per turn |
| Decay sweeps loop over rows in Python | One SQL `UPDATE` using `julianday()` | Faster sweeps |
| `build_context()` and `recall()` repeat query analysis | Shared analysis object (Section 5.2) | Half the work |

Memory footprint targets: without the built-in embedder ~10-30 MB; with it
~200-350 MB, almost all of it the ONNX model. Section 11 replaces these
estimates with measurements.

## 11. Testing and evaluation

**Baseline first.** Run the evaluation set, benchmark and RAM measurement on
0.2 before changing anything, so every claim in this document can be
checked.

**Evaluation set.** Synthetic multi-session conversations with planted
information and paraphrased questions (puppy/dog, live/location, a
corrected fact). Metrics: recall@5 and MRR. Reported for each tier
equation, with and without an embedder, with and without an LLM. That table
is the headline result of any write-up.

**Test matrix** (on top of the existing 204 tests):

| Axis | Values |
|---|---|
| Tier | FULL, STANDARD, LITE, a tier change mid-session |
| LLM | none, well-behaved stub, junk-output stub, raising stub |
| Embedder | none, host stub, built-in (slow marker), raising stub |
| FTS5 | available, unavailable |
| Storage | file, `:memory:` |
| Lifecycle | normal, crash mid-batch then restart (backlog resumes), corrupt file |

**Benchmarks.** Per-turn latency at 1k, 10k and 50k chunks; process RAM
before and after the embedder loads.

## 12. Code structure

The package started flat. Two large files are split, and modules are named
after the layer they own. After the rewrite the modules were grouped into
`storage/`, `retrieval/` and `runtime/` (see `architecture.md`); the table
below lists them by name only.

| 0.3 module | Contains | From 0.2 |
|---|---|---|
| `__init__.py` | Public API (unchanged) | same |
| `store.py` | The `Elastimem` facade only: public methods that delegate | `store.py` (split) |
| `config.py` | `ElastimemConfig` and value types | same |
| `tiers.py` | **New.** The tier table: capabilities and equation weights | parts of `governor.py` |
| `governor.py` | Signals -> tier | `governor.py` |
| `db.py` | Schema and migrations (v4) | same |
| `episodes.py` | Messages, chunks, sessions | `episodic.py` |
| `facts.py` | Versioned facts and the grounding check | `semantic.py` + `guards.py` |
| `graph.py` | Entities, edges, mentions, clusters | same, trimmed |
| `lessons.py` | Lessons | `procedural.py` |
| `vectors.py` | Encode, store, score; built-in embedder loading | `embeddings.py` + `default_embedder.py` |
| `search.py` | Query analysis and candidate generation | `retrieval.py` (split) |
| `scoring.py` | **New.** The equation | `retrieval.py` + `assembly.py` (merged) |
| `context.py` | Prompt sections and `ContextPlan` | `assembly.py` + section builders |
| `inspection.py` | `explain()`, `timeline()` | `retrieval.py` (split) |
| `jobs.py` | Worker loop and job bodies | `worker.py` + `store._execute_job` |
| `llm.py` | Prompts and output parsing | `extraction.py` |

Deleted: `rules.py`, the word lists in `guards.py`, and the duplicates
(`_days_since` x2, graph traversal SQL x2, fact scoring x2).

All of these submodules are Internal per `api_stability.md`, so moving them
breaks no supported code. `Elastimem` methods and `elastimem.*` names keep
working.

## 13. Public API and naming

### 13.1 Naming rules

1. **One or two words.** `history(key)`, not `fact_history(key)`. No chains
   like `min_retrieval_query_words` or `embedder_load_allowed`.
2. **Verbs for actions, nouns for data.** `mem.search()` does something;
   `mem.facts()` returns something.
3. **No internal jargon in the API.** Users see "facts", "lessons",
   "search", not "episodic", "semantic", "procedural", "quarantine".
4. **One word per concept, everywhere.** The same term in the API, the
   docs, the code and the database.
5. **Booleans read as questions,** without `_enabled` / `_allowed`
   suffixes: `status.vectors`, not `profile.vector_recall_enabled`.
6. **Defaults do the right thing.** A setting exists only if a typical host
   would actually change it.

### 13.2 Usage flow

0.2 (from `examples/llama_cpp_bot.py`), 9 distinct calls:

```python
mem = elastimem.open(path, llm=llm, embedder=embed)
mem.tick()
plan = mem.build_context(user)
prompt = plan.render()
mem.report_evictions(evicted)
with mem.foreground():
    reply = llm(prompt)
mem.report_pressure()           # on failure
mem.record_turn(user, reply)
mem.end_session()
mem.close()
```

0.3, 3 required calls:

```python
mem = elastimem.open(path, llm=llm, embedder=embed)
context = mem.context(user)     # also re-checks the governor
reply = llm(f"{context}\n\n{user}")
mem.record(user, reply)
mem.close()                     # ends the session, finishes background work
```

What made the other calls unnecessary:

| 0.2 call | Why 0.3 doesn't need it |
|---|---|
| `tick()` | `context()` runs the governor check itself |
| `plan.render()` | `str(context)` renders |
| `report_evictions(turns)` | Elastimem stores every turn and already tells the host how many to keep, so it knows which turns fell out of the window |
| `end_session()` | `close()` ends the session. Kept for hosts that run several sessions in one process |
| `foreground()` | Still needed by hosts that share one model with the worker, renamed `pause()` (13.3). Not needed when `llm=` is a separate model or an API |
| `report_pressure()` | Optional. The measured-time signal (7.1) covers most cases |

### 13.3 Method names

| 0.2 | 0.3 | Notes |
|---|---|---|
| `build_context(user_input)` | `context(query)` | Returns `Context` |
| `record_turn(user_text, assistant_text)` | `record(user, reply)` | |
| `recall(query, k)` | `search(query, k)` | |
| `explain(query, k)` | `explain(query, k)` | Unchanged |
| `remember(key, value, source)` | `remember(key, value)` | `source` becomes internal |
| `facts()` | `facts()` | Unchanged |
| `fact_history(key)` + `timeline(query)` | `history(key_or_question)` | One method: exact key first, then search |
| `forget(key)` | `forget(key)` | Unchanged |
| `add_lesson(text, tag)` | `learn(text)` | |
| `lessons(n)` | `lessons(n)` | Unchanged |
| `clusters()` | `topics()` | Matches the "topics" prompt section |
| `sessions(n)` | `sessions(n)` | Unchanged |
| `begin_session(host_tag)` | `new_session(tag)` | Optional; sessions start on the first `record()` |
| `resume_session(id)` | `resume(id)` | |
| `foreground()` | `pause()` | `with mem.pause():` holds background LLM work |
| `foreground_begin()` / `foreground_end()` | **open, see D9** | For streaming hosts, whose pause starts and ends in different callbacks |
| `reconfigure(**opts)` | `configure(**opts)` | |
| `drain(timeout)` | `flush(timeout)` | |
| `quarantine_entries(n)` | `rejected(n)` | |
| `profile` (property) | `status` (property) | Avoids clashing with "user profile" |
| `tick()`, `report_evictions()` | removed | See 13.2 |

### 13.4 Returned objects

| 0.2 | 0.3 | Fields (0.3) |
|---|---|---|
| `ContextPlan` | `Context` | `sections`, `summary`, `keep_turns`, `status`; `str()` renders |
| Section keys `user_facts`, `graph_context`, `relevant_past_moments`, `previous_sessions`, `lessons` | `facts`, `topics`, `moments`, `sessions`, `lessons` | |
| `MemoryProfile` (13 fields) | `Status` | `tier`, `budgets`, `vectors`, `llm`, `graph_hops`, `reasons` (why each one is on or off) |
| `Hit` | `Hit` | `kind`, `text`, `date`, `score`, `session` |
| `Fact` (9 fields) | `Fact` | `key`, `value`, `source`, `since`, `until` |
| `Cadence`, `ConsolidationLevel`, `RollingSummaryMode` | internal | Tier details, not something hosts set |

### 13.5 Constructor and settings

The constructor keeps `path`, `llm`, `embedder`, `tokenizer` and `on_tier`.
`embedder` accepts a function, `"auto"` (the default: built-in model when
installed) or `None` (keyword search only, which replaces
`disable_builtin_embedder`). The `complete_fn`/`embed_fn` aliases and
`probe_fn` are removed (tests inject the probe privately).

`ElastimemConfig` drops from 32 fields to 8. The rest become constants in
`tiers.py`.

| 0.3 setting | Default | Replaces |
|---|---|---|
| `context_tokens` | 4096 | same |
| `reserved_tokens` | 1112 | `output_reserve` + `tool_reserve` + `static_prompt_tokens` |
| `tier` | auto | `tier_override` |
| `pinned` | empty | `profile_keys` |
| `blocked` | empty | `reserved_keys` |
| `turn_budget_ms` | 50 | new (7.1) |
| `half_life_days` | 30 | `fact_decay_half_life_days`, `episodic_recency_half_life_days`, `fact_recency_half_life_days`, `graph_decay_half_life_days` |
| `max_lessons` | 15 | `max_lessons` + `lessons_in_prompt` |

Moved to `tiers.py` constants: `chunk_target_tokens`, `min_query_words`,
`min_retrieval_query_words`, `quarantine_cap`, `fact_archive_threshold`,
`graph_archive_threshold`, `fact_merge_review_window_days`,
`graph_merge_review_window_days`, `graph_node_cap`, `graph_edge_cap`,
`upgrade_healthy_ticks`, `idle_consolidate_seconds`, `tier_thresholds_gib`,
`working_share`, `memory_split`, `worker_max_tokens`,
`batched_every_n_turns`. Removed by D1: `lite_llm_extraction`.

### 13.6 Database names

The schema follows the same words: `chunks` stays, `quarantine` becomes
`rejected`, `graph_nodes`/`graph_edges` become `entities`/`relations`, the new
`graph_mentions` table (5.1) is created as `mentions`, and
`facts.valid_from`/`invalidated_at` become `since`/`until`. The migration
renames in place; no data is lost.

## 14. Decisions

Settled on 2026-11-01. The proposals were taken as defaults so
implementation could start; any of them can still be revisited.

| # | Question | Decision |
|---|---|---|
| D1 | Does LLM extraction depend on the tier? | No. It runs whenever `llm=` is given; the tier sets only the cadence (7.2) |
| D2 | Keep `rules.py`? | Removed. Facts without an LLM come from `remember()`, typically exposed to the host's model as a tool |
| D3 | Embed facts? | Yes |
| D4 | Flat package or subpackages? | Flat (Section 12) |
| D5 | Per-turn time budget for memory work | 50 ms (`turn_budget_ms`) |
| D6 | Version | 0.3.0, every behavior change listed in the CHANGELOG |
| D7 | Stemmer | `porter` stays the default |
| D8 | Deprecated aliases for the 0.2 names? | No. 0.2 was never published to PyPI, so nothing outside this repo depends on it. Clean break; `api_stability.md` is rewritten for 0.3 |
| D9 | Ending a pause from a different callback | `pause()` returns a handle: `with mem.pause():` or `p = mem.pause()` ... `p.end()`. One method instead of two |

Two simplifications made while planning the code:

- No separate role column for chunks. The data-driven term filter (Section
  9) already ignores "User"/"Assistant", because they appear in every chunk.
- Graph seed detection stays a Python scan of entity names (0.6 ms at 420
  entities, measured), but runs once per turn instead of twice. The cost in
  F8 came from the duplicate run and the `LIKE` scans over chunks, which
  the `mentions` table removes.

## 15. Task list

The 0.2 names, schema and scoring all change, so the code is rewritten
module by module into the Section 12 layout instead of being moved first
and renamed later.

**Phase 0: baseline on 0.2**
- [x] Evaluation set and script (`benchmarks/eval_recall.py`): planted memories, paraphrased questions, recall@5 and MRR
- [x] Baseline scores: keyword only, and with the built-in embedder
- [x] Baseline per-turn latency at 1k / 10k chunks with 384-dim vectors
- [x] Baseline process RAM with and without the built-in embedder
- [x] v3 store fixture for the migration test (`tests/fixtures/v3_store.db`)

Baseline results (0.2.0, 8 GB M-series Mac):

| Measurement | Result |
|---|---|
| Recall@5 / MRR, keyword only | 0.50 / 0.51 |
| Recall@5 / MRR, built-in embedder, FULL | 0.94 / 0.84 |
| Recall@5 / MRR, built-in embedder, LITE | 0.50 / 0.51 (vectors never run, F3) |
| `build_context()` per turn, 1k chunks | 20.7 ms |
| `build_context()` per turn, 10k chunks | 206 ms |
| `record_turn()` | 0.2 ms |
| RAM added, keyword only | 3 MB |
| RAM added, built-in embedder | 360 MB |

Also found: `close()` closes only the calling thread's connection, so the
worker's SQLite connection leaks (F12).

**Resume point (updated 2026-11-21).** All docs, CHANGELOG and version 0.3.0 are written and uncommitted in the working tree. Remaining: test, tag, then integrate into Tuffy. Earlier note, from 2026-11-01:

**Resume point (2026-11-01, end of day).** The 0.3 code, tests,
benchmarks and examples are done and committed on the `redesign-0.3`
branch (85 tests pass, plus 1 slow test for the built-in model). Rewritten
docs so far: `architecture.md`, `api.md`, `schema.md`, `integrations.md`,
`api_stability.md`. Next session: the remaining docs (below), then the
release steps. Until then, README, `agent_integration_prompt.md`,
`governor.md`, `tiers.md`, `quickstart.md` and `installation.md` still
describe 0.2.

**Phase 1: storage**
- [x] `db.py`: schema v4 (`entities`, `relations`, `mentions`, `rejected`, `facts.since/until/embedding`, `chunks.extracted_at`, `sessions.summary_turn`), v3 -> v4 migration in one transaction. Verified by hand and by `tests/test_storage.py`
- Also in `db.py`: `utcnow()`, `days_since()`, `decay()` (registered as a SQL function) so no module keeps its own copy

**Phase 2: layers**
- [x] `config.py` (8 settings, `Tier`, `normalize_key`) and `tiers.py` (capabilities, weights, constants)
- [x] `facts.py` (versioned facts, grounding check, `rejected` table helpers)
- [x] `episodes.py` (messages, chunks, sessions, backlog queries, extractive folding, `fit_lines`/`estimate_tokens`)
- [x] `lessons.py`: `learn(conn, text, max_lessons) -> bool`, `recent(conn, n) -> list[str]`
- [x] `graph.py` (entities, relations, mentions, topics, decay, merges)
- [x] `vectors.py` (normalized vectors, in-RAM index with numpy and pure-Python paths, built-in embedder)

Design details settled while coding, for the remaining modules:

- `graph.py`: the `user` node (the token our extraction prompt uses for the
  user) is never expanded through, otherwise it connects every entity to
  every other. Seed matching is word-boundary aware (`" name "` in the
  space-joined query words), not raw substring. One `expand()` returning
  `{entity_id: depth}` replaces `expand`/`expand_with_depth`. Extraction
  writes `mentions` for the turn's chunk ids. Decay is one SQL `DELETE`
  using `decay()`.
- `vectors.py`: one `Vectors` object owns the embedder (`"auto"`, `None` or
  a function) and an in-RAM index per table (`chunks`, `facts`), keyed by
  row id so a re-embedded fact replaces its old vector. Loaded lazily on
  the first search, then appended to directly by the embed job. Vectors
  are normalized on write and on load (v3 vectors were not). Rows whose
  dimension differs from the current embedder are skipped. numpy when
  importable, pure-Python dot product otherwise. A failing embedder sets a
  sticky `failed` reason.
- `search.py`: keyword score is the BM25 score divided by the best one
  (reciprocal rank with k=60 barely separates rank 1 from rank 20). The
  common-term filter counts each query term with an FTS `MATCH` count;
  the `LIKE` fallback skips the filter.
- `scoring.py`: a term is "missing" when it cannot contribute for this
  query (no query vector or empty index, no keyword terms, no graph seeds)
  and its weight moves to the other relevance terms.
- `jobs.py`: the worker reads pending work from SQL in this order:
  embeddings (chunks, then facts), LLM extraction (cadence from the tier;
  `session_end` only during `flush()`), LLM folding, then idle
  consolidation. A failed LLM call still marks the turn extracted, so no
  turn retries forever. `pause()` holds `llm_lock` and the worker skips LLM
  steps while paused. `flush()` waits for a full pass that finds nothing.
- `store.py`: keeps every thread's connection in a list so `close()`
  closes all of them (F12). `record()` folds the window inline when the
  tier uses extractive summaries; `keep_turns` comes from the last
  `Status`. `remember()` returns `bool`.
- `governor.py`: capacity tier from total RAM (16 / 8 GiB). Pressure from
  `sysctl kern.memorystatus_vm_pressure_level` on macOS (1 normal, 2 warn,
  4 critical; this Mac reads 2), `MemAvailable/MemTotal` on Linux, psutil
  elsewhere. Warn caps at STANDARD, critical forces LITE. Probes are
  rate-limited to `PROBE_SECONDS`.
- `context.py`: `str(Context)` renders sections under sentence-case
  headings, plus the rolling summary.

**Phase 3: retrieval**
- [x] `search.py` (query analysis once per turn, term-frequency filter, candidates)
- [x] `scoring.py` (the equation, missing-term reweighting)
- [x] `context.py` (`Context`, sections, budgets)
- [x] `inspection.py` (`explain()`, `history()`); renamed from `inspect.py`, which shadows the standard library module

**Phase 4: runtime**
- [x] `governor.py` (pressure, measured cost, battery, `Status` with reasons)
- [x] `llm.py` (prompts, parsing)
- [x] `jobs.py` (backlog-driven worker, `pause()` handle, `flush()`)
- [x] `store.py` (the facade) and `__init__.py`

**Phase 5: verification**
- [x] New test suite covering the matrix in Section 11
- [x] Re-run the evaluation, latency and RAM measurements; tune weights

Results (same 8 GB Mac; eval grown to 19 questions, 3 of them exact
identifiers, and both versions re-run on it):

| Measurement | 0.2 | 0.3 |
|---|---|---|
| Recall@5 / MRR, keyword only | 0.58 / 0.58 | 0.68 / 0.62 |
| Recall@5 / MRR, built-in embedder, FULL | 0.95 / 0.86 | 1.00 / 0.94 |
| `context()` per turn, 1k chunks | 20.7 ms | 1.7 ms |
| `context()` per turn, 10k chunks | 206 ms | 11.9 ms |
| `record()` | 0.2 ms | 0.3 ms |
| RAM added, keyword only / built-in embedder | 3 / 360 MB | 3 / 368 MB |
| Tier on this Mac | LITE | STANDARD |
| Test suite | 204 tests, 4.6 s | 83 tests, 1.75 s |

What moved the numbers, found by measuring rather than planned:
- Weight sweep: vector 0.8 / keyword 0.1 gave the best MRR, including on
  the identifier questions, and nothing in the data justified different
  weights per tier. Every tier now uses one equation (`tiers._EQUATION`);
  tiers differ in which terms are available. This replaces the per-tier
  weights in 6.3.
- Exact-duplicate exchanges are shown once (they filled the top 5 before).
- Vector similarity is used unscaled. Scaling each table by its best match
  let the only stored fact score 1.0 for any query (found by running
  `examples/memory_search.py`). This cost 0.02 MRR on the moments-only
  eval (0.96 -> 0.94) and fixes mixed fact/moment rankings, which the eval
  does not measure.
- Keyword scores use BM25 values, not reciprocal rank (a sign bug in the
  first draft emptied the keyword signal; caught by the smoke test).
- Excluding the current session from vector search moved after ranking: a
  long session made the exclude list as big as the store.
- Known limit: a question word that is rare among stored statements
  ("what") can still match as a keyword. With keyword at 0.1 weight it no
  longer decides rankings in the eval.
- [x] Delete the 0.2 modules and tests
- [x] Delete or rewrite stale docs (with phase 6): delete `governor.md`, `tiers.md`, `quickstart.md`, `installation.md` (absorbed into `architecture.md` and `integrations.md`)

**Phase 6: docs and release**
- [x] Update examples (all four rewritten; the two offline ones run)
- [x] Rewrite `architecture.md`, `api.md`, `schema.md`, `integrations.md`, `api_stability.md`
- [x] Rewrite README
- [x] Rewrite `agent_integration_prompt.md` for 0.3, adding a `0.2.0 -> 0.3.0` subsection to its section 9 (append-only; the old subsections stay)
- [x] Update `CONTRIBUTING.md` links to the deleted docs
- [x] CHANGELOG and version 0.3.0; drop the unused `vec` extra from `pyproject.toml`
