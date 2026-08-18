# Elastimem Architecture

Elastimem is a memory *library* for AI agents, not an agent runtime. It owns one
SQLite file and answers two questions for its host, every turn:

1. **What should the model see right now?** → `build_context(user_input)`
2. **What just happened that's worth keeping?** → `record_turn(user, reply)`

Everything else — extraction, summarization, embedding, consolidation,
forgetting — happens behind those two calls, paced by the machine's actual
capacity.

## Design principles

- **Host-agnostic, with one deliberate exception.** Elastimem never loads a
  chat/completion model or calls an inference API itself — the host injects
  `llm` (text completion) as a plain callable, and it's fully optional
  (see the [degradation matrix](governor.md#degradation-matrix): no `llm`
  means rule-based fact capture only, no LLM extraction/summarization/
  consolidation). Embeddings are different: if the host doesn't supply
  `embedder`, Elastimem activates its **own built-in embedder**
  (`default_embedder.py`, via the optional `elastimem[embed]` extra) rather
  than leaving semantic search off by default — see
  [governor.md's "Built-in embedder" section](governor.md#built-in-embedder-auto-activates-read-this-before-assuming-embeddings-are-off)
  for the full story (lazy loading, first-run download, silent fallback to
  FTS5 if the extra isn't installed, the `disable_builtin_embedder` opt-out).
  This is intentional: a host that does nothing beyond
  `elastimem.open(path)` still gets real semantic recall, not just
  keyword matching, once resources allow it (STANDARD tier or above).
- **Elastic.** The [Memory Governor](governor.md) probes RAM and the model's
  context size, then budgets every capability. Same code on a 4 GB Jetson
  and a 128 GB workstation.
- **Defined floors.** Every capability degrades to a documented fallback
  rather than failing (see the [degradation matrix](governor.md#degradation-matrix)).
  `record_turn`, `recall`, and `build_context` never raise into the host —
  this guarantee starts *after* construction succeeds; `Elastimem(path, ...)`
  itself can raise (bad path, unknown config key), see [api.md](api.md).
- **Never destroy.** Fact updates version; forgetting archives; corrupt
  databases are quarantined and rebuilt. The raw transcript is always kept,
  so future (better) models can re-index the past.

## The four memory layers (plus the graph that connects them)

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
                                    ▲
                                    │ one more retrieval signal, not
                                    │ a separate store
                       ┌───────────────────────────────┐
                       │ GRAPH  entities + relationships│ graph_nodes, graph_edges
                       │        + topic clusters        │
                       └───────────────────────────────┘
```

The graph is deliberately drawn separately above: it isn't a fifth
peer memory layer that competes with the other four for the host's
attention. It's an index *over* them — entities/relationships extracted
from the same exchanges that feed episodic and semantic memory, used to
widen retrieval (`recall`) and to populate one more `build_context()`
section (`graph_context`, "RELATED TOPICS"). See "The knowledge graph"
below.

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
exists (host-supplied or the built-in default — see the design principles
above), by float32 vector BLOBs. Retrieval at turn start injects the top
chunks from *previous* sessions as `RELEVANT PAST MOMENTS` — deliberately
excluding the current, still-live session (it's already in the working
window, so re-injecting it would be redundant and would waste budget).
`recall(query)` searches everything, including the current session — it
backs an agent-visible `memory_search` tool and user-facing `/memory
search` commands.

**Ranking (`retrieval.search_chunks`)**: the FTS5 leg and vector leg are
fused differently on purpose. FTS5's own BM25 score isn't cheaply
comparable across different queries, so that leg stays rank-based
(Reciprocal Rank Fusion, `1/(60+rank)`). The vector leg's cosine similarity
IS directly meaningful — a 0.60 match really is stronger than a 0.15 one,
for the same query — so `similar_chunks()` returns real `(chunk_id, score)`
pairs, and `search_chunks()` min-max normalizes those against the best
score in the current result set, giving each leg its own comparable 0-1
relevance signal before fusing them (`max()` of the two, not a sum — the
two legs' raw scales aren't the same unit, so summing would let FTS's much
smaller RRF numbers get silently dominated by cosine, or vice versa
depending on normalization order). Only after that fused relevance score
exists do `importance` and `recency` apply, as small **additive** nudges
(not multipliers) — enough to break a near-tie between two similarly
relevant chunks, not enough to let a topically-irrelevant-but-fact-bearing
chunk (importance gets bumped when a chunk yields a stored fact, see
`episodic.bump_importance`) outrank a chunk that's actually relevant to the
query. This replaced an earlier design where relevance × importance ×
recency were multiplied together directly — since RRF's raw scores are
deliberately flat across ranks (by design, so no single ranker dominates),
that multiplication let importance's ~1.6x swing (0.5 → 0.8) trivially flip
rankings on real queries. If you're touching this code, `search_chunks()`'s
own docstring in `retrieval.py` has the fuller before/after story. A third
signal, the knowledge graph, was added later using the exact same
additive-nudge principle — see "The knowledge graph" below.

### Semantic memory (`semantic.py`)
Facts are `key → value` rows with **temporal versioning**: an update
invalidates the old row (`invalidated_at`, `invalidated_by`) and inserts a
new one. `fact_history(key)` is the audit chain; `forget(key)` tombstones.
Profile-category facts (name, location, …) are always injected; note facts
compete on **relevance first**, with importance and recency as small
additive tie-breaking nudges (not multipliers — see the ranking note in
episodic memory below, since chunks and facts share the same underlying
principle even though the exact formulas differ per query surface). Never-
accessed auto-extracted facts decay and archive; explicit facts never
auto-archive.

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

## The knowledge graph (`graph.py`)

Entities and relationships ride the **same** background LLM completion
`extraction.py` already makes for facts — one call per turn, not two. The
extraction prompt asks for `facts`/`entities`/`relationships` in a single
JSON object; a model that ignores the new shape and returns the old flat
`{key: value}` JSON is still handled (backward-compatible fallback), so
this never risked regressing plain fact extraction.

Storage is two plain tables in the same file, no separate database:
- **`graph_nodes`** — one row per distinct entity, write-time deduped by
  normalized identity (`ON CONFLICT DO UPDATE` — repeated mentions bump a
  running-average `confidence` and `mention_count` rather than inserting a
  new row). Carries `cluster_id`/`cluster_label` once clustering has run.
- **`graph_edges`** — directed relationships between two nodes, same
  write-time dedup pattern on `(source, target, relationship)`.

**Retrieval** (`retrieval.graph_relevance`): query-time entity detection is
a plain substring scan over `canonical_name`/aliases (no NER call), then a
`WITH RECURSIVE` SQL traversal expands N hops from those seeds — no graph
library. The result folds into `search_chunks`/`search_all` as a small
**additive** score nudge, same "nudge not multiplier" philosophy as
importance/recency: the nudge's ceiling is a fraction of *that query's own*
top relevance score (not a fixed constant), and each match is weighted by
the entity's own confidence and match specificity. This means a graph
match can break a near-tie or surface an associatively-connected memory
that shares zero vocabulary with the query — but it can never manufacture
a hit out of a query with zero real FTS/vector relevance to begin with.

**Maintenance** piggybacks on the existing `consolidate` job (no new job
kind, no new scheduler) — see "Consolidation" below.

**Governor gating** (`MemoryProfile.graph_hops`): LITE=1, STANDARD=1,
FULL=2 hops. The graph leg is reachable at every tier — traversal is a
bounded `WITH RECURSIVE` query over tables that row-count caps already
bound, so it costs a starved machine nothing beyond a local SQLite read.
Only the second hop, whose fan-out is not bounded per query, is reserved
for FULL.

**`explain(query)`** (Experimental) and **`clusters()`** (Experimental)
expose this machinery directly for debugging/UI: `explain` returns the
per-leg score breakdown and the traversal path behind a `recall()`-
equivalent search; `clusters` returns the topic groups (connected
components over `graph_edges`, union-find — see "Consolidation" below —
optionally LLM-labeled).

## The background worker (`worker.py`)

One daemon thread, one queue. Job kinds: `extract`, `rolling_summary`,
`session_summary`, `embed`, `consolidate`.

The critical invariant is **foreground-wins**: most local hosts have exactly
one model instance and it is not thread-safe. The host brackets its own
generation with `with mem.foreground():`. This is enforced by a real
mutual-exclusion lock (`Worker.llm_lock`), not just an advisory flag —
`foreground_begin()` blocks until it acquires the lock, and the worker holds
the same lock for the full duration of every `complete_fn` call. A job that
was already dequeued and mid-call when the foreground gate opened is waited
out rather than raced with; background LLM calls are capped at
`worker_max_tokens` (default 96) so that wait is sub-second on a 2B model.
`drain()` finishes the queue before the host unloads its model; `close()`
drains and stops.

## Consolidation

Runs at session end (all tiers except OFF-level work) and after 90 s of
idle in FULL tier:
1. decay/archival math over auto facts (no LLM),
2. FULL only: LLM contradiction-merge for keys whose value changed within a
   week ("lives in Austin" vs "moving to Austin in May" → merged value,
   still versioned),
3. **graph decay/archival** (any tier above OFF, no LLM): a node/edge's
   confidence decays exponentially from its last reinforcement, same shape
   as fact decay; below threshold, the row is hard-deleted (the graph has
   no audit-trail requirement, unlike facts — decay removes rather than
   soft-archives),
4. FULL only: **graph duplicate-entity merging** — same-type node pairs
   sharing a token get one LLM yes/no question each (capped at 5/sweep);
   on "yes" the newer node's edges are repointed and it's deleted,
5. **cluster recomputation** (any tier above OFF, no LLM) — connected
   components over the now-settled graph, run *after* steps 3-4 so stale/
   duplicate nodes don't fragment or pollute a topic group; FULL only asks
   the LLM for a short label per *new* cluster,
6. cap maintenance (quarantine ≤ 200, plus the graph's own
   `graph_node_cap`/`graph_edge_cap` enforced at write time, not here).

## Data flow per turn

```
user input
  │
  ├─ mem.tick()                    governor re-checks RAM (cheap)
  ├─ plan = mem.build_context(q)   FTS+vector+graph retrieval, budgeted sections
  ├─ host renders plan into its system prompt, trims window to plan
  ├─ with mem.foreground():        model generates the reply
  ├─ mem.record_turn(q, reply)     persist + rule capture (inline, fast)
  │     └─ worker: extract facts + graph entities/relationships,
  │                embed chunks                    (background, one LLM call)
  └─ mem.report_evictions(...)     if the host trimmed its window
        └─ worker: fold into rolling summary       (background)
```

## Reference

- [quickstart.md](quickstart.md) — install to first recall
- [installation.md](installation.md) — extras, supported Python versions
- [governor.md](governor.md) — tiers, budget math, degradation matrix
- [schema.md](schema.md) — every table and index
- [integrations.md](integrations.md) — wiring guides (llama.cpp, API, no-LLM)
- [api.md](api.md) — public API reference
- [api_stability.md](api_stability.md) — what's safe to depend on long-term
