# The Memory Governor — normative spec

The governor makes Elastimem *elastic*: it measures what the machine can afford
and sizes every memory capability accordingly, continuously. This document is
the specification; `governor.py` implements it and `tests/test_governor.py` +
`tests/test_degradation.py` enforce it.

## Inputs

| Input | Source | When |
|---|---|---|
| RAM total / available | `psutil` if installed, else `/proc/meminfo` (Linux) or `sysctl` + `vm_stat` (macOS); **Windows and any other platform have no stdlib probe and assume 8/4 GiB** (a guess, not a measurement) — install the `system` extra (`psutil`) for accurate probing on Windows | startup + every `tick()` |
| `context_tokens` | `ElastimemConfig` (host passes its model's `n_ctx`) | construction |
| `static_prompt_tokens` | `ElastimemConfig` (host measures its fixed prompt) | construction |
| Tier override | `ELASTIMEM_TIER=lite\|standard\|full` or `ElastimemConfig.tier_override` | construction (pins the tier) |
| Pressure signal | host calls `report_pressure()` on OOM / decode failure | runtime, rate-limited (see below) |

## Tier classification

```
FULL      total ≥ 16 GiB  and  available ≥ 6 GiB
STANDARD  total ≥ 8 GiB   and  available ≥ 2.5 GiB
LITE      otherwise
```

Movement rules:
- **Downgrade immediately** when available RAM < 1.2 GiB, or when measured
  classification drops (checked on every `tick()`, so at most one turn of
  lag behind reality).
- **`report_pressure()` downgrades one tier immediately on the FIRST call**,
  then enters a 30-second cooldown (`_PRESSURE_REPORT_COOLDOWN_SECONDS`
  in `governor.py`): further calls within that window are coalesced into
  the same downgrade instead of dropping the tier again. This exists
  because a single bad turn can plausibly produce several pressure reports
  in quick succession (a decode-retry loop, or the same underlying failure
  surfacing through more than one call site in the host) — without the
  cooldown, one real event could cascade the tier down multiple levels and
  need far more healthy ticks than the event actually warranted to recover
  from. The cooldown does **not** weaken "downgrades are immediate": the
  first report in any burst still drops the tier on the spot, exactly as
  before. A genuinely new, separate pressure event after the cooldown
  elapses downgrades again, normally.
- **Upgrade cautiously**: only after `upgrade_healthy_ticks` (default 10)
  consecutive healthy ticks, one tier at a time, never above the startup
  tier (a host that started at STANDARD can recover to STANDARD, never to
  FULL, even if RAM later becomes abundant — the assumption is that
  whatever else the process is doing at that RAM footprint hasn't gone
  away just because a temporary dip cleared).
- `on_tier_change(old, new)` fires so the host can react (e.g. unload its
  embedding model when dropping to LITE). **Elastimem's own built-in
  embedder (see below) does NOT need this** — it already checks
  `profile.embeddings_enabled` on every call and simply isn't invoked when
  the tier says no; there is nothing for the host to unload since the
  built-in model, once loaded, just sits idle rather than holding a lock
  or consuming CPU.

## Budget derivation

```
dynamic = context_tokens − output_reserve(512) − tool_reserve(600) − static_prompt_tokens
          (floor: 256)

working  = 55% of dynamic
memory   = 45% of dynamic, split:
             facts 40% · episodic 30% · sessions 15% · lessons 15%
```

LITE additionally keeps only `LITE_EPISODIC_SHARE` (45%) of the episodic
share and `LITE_SESSIONS_SHARE` (75%) of the sessions share, returning the
rest to the working window — on a starved machine, immediate coherence
still beats recall in general.

These fractions were 10% and 50% before 0.2.0, which was too aggressive:
on a typical 4K-context host with a real system prompt, 10% of the episodic
share rounds down to a handful of tokens, so `fit_lines()` kept zero or one
line and episodic injection was effectively absent rather than merely
reduced. Note this is a **token-split** decision, not a resource one —
reading three rows out of SQLite instead of one costs a RAM-starved machine
nothing, so there is no reason to starve the section beyond what the
context window actually forces.

Worked example (defaults, `n_ctx=4096`, static prompt 1200 tokens):
dynamic = 1784 → working ≈ 981, facts ≈ 321, episodic ≈ 240, sessions ≈ 120,
lessons ≈ 120. At LITE: working ≈ 1143, facts ≈ 321, episodic ≈ 108,
sessions ≈ 90, lessons ≈ 120.

**The 256-token floor is real and gets hit in practice, not just a
theoretical edge case.** A small local model (`n_ctx=4096`) whose host has a
sizeable fixed system prompt (persona + a large tool catalog can easily run
1500-2600+ tokens) leaves very little `dynamic` budget once
`output_reserve` (512) and `tool_reserve` (600) are subtracted — e.g.
`4096 - 512 - 600 - 2584 = 400`, barely above the floor. At that point
`working≈220`, and the ENTIRE memory pool (facts + episodic + sessions +
lessons combined) is only ~180 tokens, meaning each individual section gets
single-digit-to-low-double-digit token budgets — `fit_lines()` then keeps
0-1 lines per section, which reads to a user as "memory barely knows
anything" even though every other part of the pipeline (extraction,
storage, retrieval) is working correctly. **This is not a bug to patch
inside the governor** — the floor exists precisely to prevent a negative or
zero budget from crashing budget math, and it's doing its job. If you hit
this, the fix is on the host side: either increase the model's `n_ctx`
(the actual fix Tuffy applied — see its own model config), or reduce
`static_prompt_tokens` (a shorter persona / fewer always-on tool
descriptions), or both. There is no governor-side knob that manufactures
context tokens that don't exist.

Token estimation uses chars/4 unless the host passes `tokenizer_fn`. This
proxy is a real approximation, not exact — for non-English text or
code-heavy content, actual tokenization can diverge meaningfully from
chars/4, which combined with a tight budget (see above) can make
`fit_lines()` either truncate more aggressively than necessary or, in the
opposite direction, let slightly more text through than the model's real
tokenizer would allow. Pass `tokenizer_fn` if this matters for your use
case — it costs nothing extra elsewhere in the pipeline.

### Two separate query-length gates, not one

Retrieval and LLM-based extraction are gated by **different** thresholds,
because they have very different costs:

| Gate | Config field | Default | Applies to | Why |
|---|---|---|---|---|
| Extraction gate | `min_query_words` | 4 | LLM fact-extraction job (`store.py`'s `_after_record_turn`) | A real model call has real latency/cost — not worth it for "ok" or "thanks". |
| Retrieval gate | `min_retrieval_query_words` | 1 | FTS5/vector search inside `build_context()` | Local, free, no LLM involved — excluding anything beyond genuinely empty input (a single word still carries real query intent, e.g. "birthday?" or a one-word follow-up) would silently weaken recall on exactly the kind of short question that's common in real conversation. |

These used to share one threshold (`min_query_words=4` gating both), which
meant a short-but-real question like "my birthday?" (2 words) got **zero**
episodic/fact retrieval just from being under 4 words — indistinguishable
from "memory isn't working" from the outside, with nothing logged (retrieval
never raises, by design). If you're tuning either value, remember they now
move independently — raising `min_query_words` to save LLM calls does not
also raise the bar for local retrieval, and vice versa.

## Capability × tier

| Capability | FULL | STANDARD | LITE |
|---|---|---|---|
| Indexing new chunks (`embeddings_enabled`) | yes | yes | **never** |
| Vector leg at query time (`vector_recall_enabled`) | yes | yes | yes, **but only over already-embedded chunks and only with an embedder that is already resident** — see below |
| Loading the built-in embedder (`embedder_load_allowed`) | yes | yes | **never** |
| Episodic injection (`build_context`) | top 5 | top 4 | top 3 |
| LLM fact extraction | background, per turn | batched every 2 turns | off (opt in with `lite_llm_extraction`, then deferred to session end) |
| Rolling summary (`rolling_summary_mode`) | LLM | LLM | extractive (no model call); marker line only if nothing usable |
| Consolidation | full (incl. LLM merge), idle + exit | dedupe + decay, exit | dedupe + decay, exit |
| Knowledge graph (`graph_hops`) | 2-hop expansion; decay + clustering + LLM dedupe/labeling | 1-hop expansion; decay + clustering | 1-hop expansion; decay + clustering (no LLM-gated steps — those need FULL) |
| Rule capture | always | always | always |
| Transcript persistence | always | always | always |
| `remember` / `recall` / `forget` | always | always | always |

**The single most important thing to internalize about the tier system:
LITE's floor is "spend no new resources," not "least capability possible."**
Concretely, LITE refuses exactly three things by default:

1. **No LLM call.** Not extraction, not rolling summaries, not
   consolidation merges. (`lite_llm_extraction` is the one opt-in — see
   below.)
2. **No model load.** Elastimem will not materialize the built-in
   embedder's ~130MB of weights at LITE.
3. **No sustained per-item cost.** New chunks are not embedded, because
   that is one encode per chunk forever, not a one-off.

Everything else is fair game, and this is where LITE changed in 0.2.0. The
tier used to be defined by what it turned *off*, which swept in several
capabilities that are pure local SQLite and therefore cost a starved
machine nothing. Each of these is independent of the others — that is the
whole reason they can be decided separately:

- **Consolidation runs** (`DEDUPE_ONLY`). Fact decay/archival, quarantine
  trimming, graph decay/archival, cluster recompute: all SQL, no model.
  Leaving this `OFF` meant a LITE store was the one store that never
  pruned anything and grew without bound — the opposite of what a
  RAM-constrained host wants. It also silently disabled graph maintenance,
  which lives inside the same sweep.
- **The vector leg still runs at query time**, provided the vectors already
  exist and the embedder is already resident. This is the subtle one, so
  it gets its own section below.
- **Rolling summaries are extractive.** Condensing evicted turns by
  selecting from text already in the database is string manipulation, not
  inference. LITE gets real content instead of `[3 earlier turn(s)
  omitted]`.
- **Episodic injection is top 3, not top 1**, with a real token share.
  Reading three rows instead of one costs a starved machine nothing; this
  is a token-budget tradeoff, not a resource one, and the old share left
  episodic with single-digit token budgets on a typical 4K-context host.

What a LITE host still genuinely gives up:
- Fact capture is rule-based only (`rules.py`'s ~10 regexes) — genuinely
  hardcoded, will never learn a new phrasing on its own. See "Known
  limitations" below.
- No *new* content becomes semantically searchable, since new chunks go
  unembedded. Chunks embedded during an earlier STANDARD/FULL stretch stay
  searchable.
- No LLM contradiction-merge, duplicate-entity merge, or cluster labeling.

### Vector recall vs. embedding: two decisions, not one

Before 0.2.0 a single `embeddings_enabled` flag covered both "index new
chunks" and "score existing ones." Those have very different costs, so
they are now separate flags:

| Flag | Governs | LITE |
|---|---|---|
| `embeddings_enabled` | embedding newly recorded chunks (one encode per chunk, forever) | `False` |
| `vector_recall_enabled` | running the vector leg during retrieval at all | `True` |
| `embedder_load_allowed` | triggering a *first* load of the built-in model | `False` |

The reasoning: a **host-supplied `embed_fn` is always already resident** —
it lives in the host's own process and was loaded before Elastimem ever saw
it. Refusing to call it at LITE frees exactly zero bytes; it only throws
away recall quality. Likewise, cosine-scoring vectors that are already
sitting in the database costs one query encode and a loop. What actually
costs RAM is materializing a model that isn't loaded yet, and that is what
`embedder_load_allowed` blocks.

So a machine that ran at STANDARD, embedded its history, and then dipped to
LITE keeps semantic recall over everything it already indexed, instead of
falling off a cliff to keyword-only. A machine that has been at LITE since
startup with no host embedder gets FTS5 keyword search, exactly as before —
`embeddings.embedder_resident()` reports the built-in's load state *without
triggering a load*, since probing must not be the thing that allocates.

### Opting LITE into LLM extraction

`ElastimemConfig.lite_llm_extraction` (default `False`) lets a host allow
fact extraction at LITE. When enabled, LITE's cadence becomes
`Cadence.SESSION_END`: extraction jobs are held by the worker and released
on `end_session()`/`drain()`, so they never compete with a live foreground
generation, and each is still capped at `worker_max_tokens`.

Leave it off unless you know the machine can afford it. It is useful when a
host classifies as LITE because of *other* processes rather than a
genuinely small machine, or when the model is unloaded between sessions
anyway. The default keeps rule 1 above intact: no LLM call is ever
attempted at LITE unless you ask for one.
- No background summarization — sessions fall back to their title-only
  floor (first 80 chars of the first user message, the same default every
  tier uses absent an LLM-generated summary) with no LLM-condensed 1-2
  sentence summary layered on top.
- No consolidation — contradictory fact updates (e.g. "I live in Austin"
  then weeks later "I live in Denver") are versioned correctly (the old
  value is never lost, see `schema.md`) but never LLM-merged into a single
  coherent value; the host sees only the latest version.

None of this is a bug. It is the intended floor: a host on a genuinely
constrained machine gets a working memory system with hard guarantees
(nothing crashes, nothing is silently lost — raw transcripts persist in
every tier) rather than a broken one. But it means **"memory doesn't seem
to be learning things"** on a real deployment is very often actually
**"this process has been running at LITE tier the whole time"** — check
`mem.profile.tier` (or the host's own `/memory` status command, if it
surfaces one) before assuming a capture/retrieval bug. The tier can only be
confirmed above LITE if the RAM thresholds in "Tier classification" above
are actually met at the moment of measurement.

## Built-in embedder (auto-activates — read this before assuming embeddings are off)

As of the `embed` optional extra, **Elastimem embeds by default.** If the
host constructs `Elastimem(...)`/`elastimem.open(...)` without passing
`embedder=`/`embed_fn=` (i.e. that argument is `None` after resolving both
spellings), the store activates its own built-in embedder
(`default_embedder.py`) automatically — no host code required.

- **Model**: `BAAI/bge-small-en-v1.5` via `fastembed` (ONNX Runtime, no
  torch dependency) — a real MTEB-benchmarked retrieval model, not a
  hashing trick or bag-of-words approximation.
- **Lazy, always**: nothing is imported, downloaded, or loaded at
  `Elastimem()` construction time — only on the first actual embed call
  (which only happens via the background worker's `"embed"` job, itself
  gated on `profile.embeddings_enabled`, i.e. never at LITE tier). This
  means construction never blocks on network access, and a host on a
  machine that starts at LITE tier never triggers the download at all —
  `embedder_load_allowed` is `False` there, so even the retrieval path,
  which may otherwise use an already-resident embedder at LITE, will not
  cause a first load.
- **First use downloads ~130MB** from Hugging Face Hub, cached under
  `fastembed`'s own platform cache directory afterward (subsequent runs on
  the same machine reuse the cache — no repeat download). This download
  happens on a background worker thread, never the host's main/foreground
  thread, so it cannot block an interactive turn — but it does mean the
  FIRST embed job after a fresh install can take significantly longer
  (tens of seconds) than subsequent ones.
- **Fails silently, always degrades to FTS5**: if the `embed` extra isn't
  installed (`pip install elastimem[embed]`), or the download fails (no
  network, disk full, HF Hub unreachable), `default_embedder` catches the
  failure, logs a warning once, and the store behaves exactly as if no
  embedder were configured at all — FTS5/keyword retrieval only. This is
  the same "vector leg disabled for the session" floor documented below for
  a host-supplied embedder that raises.
- **Asymmetric query/passage encoding**: bge-small (like most modern
  retrieval-tuned embedding models) is trained with different encoding for
  queries vs. the passages being searched — using the wrong one measurably
  hurts retrieval accuracy. The built-in embedder handles this correctly
  via `Store.embed_query_fn` (set automatically alongside `embed_fn` when
  the built-in activates); a **host-supplied** `embedder=` is assumed
  symmetric (true of most embedding APIs, e.g. OpenAI's) and
  `embed_query_fn` stays `None` in that case — retrieval then reuses the
  host's `embed_fn` for both queries and passages, exactly as it always
  has. If your own embedder is also asymmetrically trained, you currently
  have no way to pass a separate query-side encoder through the public
  API — file an issue if you need this; the plumbing exists internally
  (`Elastimem.embed_query_fn`) but isn't yet part of the constructor's
  public parameter list.
- **Opting out entirely**: set `disable_builtin_embedder=True` in
  `ElastimemConfig` (or pass it as a config override). Use this for an
  air-gapped environment that must never attempt the optional extra's
  first-run download, or any host that wants to guarantee FTS5-only
  behavior regardless of what's installed. Passing `embedder=None`
  explicitly is **not** enough on its own to mean "no embedder" — `None`
  is genuinely ambiguous with "the host simply didn't pass this optional
  parameter," which is exactly the case the auto-activation is designed
  for. `disable_builtin_embedder=True` is the unambiguous opt-out.

## Knowledge graph

Entities and relationships are extracted alongside facts by the same
per-turn LLM completion (`extraction.py`) — never a second model call —
and stored as plain tables in the same SQLite file (`graph_nodes`,
`graph_edges`; see [schema.md](schema.md)). The graph is one more
retrieval signal, not a separate store or a separate retrieval path:
`recall()` still returns FTS5/vector hits exactly as before, with a small
score nudge when a hit's chunk or fact also mentions an entity reachable
from the query via the graph.

`graph_hops` (LITE=1, STANDARD=1, FULL=2, from the table above) bounds a
`WITH RECURSIVE` SQL traversal — no graph library. Read/traversal at LITE
is real: seed-entity detection and the 1-hop expansion both run, the same
as STANDARD.

**Writes are a separate story.** New nodes/edges are only ever created
inside `extraction.extract_facts`'s per-turn LLM completion (see its
`graph_hops` gate) — the same completion that produces facts. That
completion does not run at LITE by default (see `lite_llm_extraction` for
the opt-in), so **LITE traverses whatever graph already exists but does not
normally grow it.** In practice this means
a store that has only ever run at LITE has an empty graph and the traversal
finds nothing; a store that spent time at STANDARD/FULL before dropping to
LITE (RAM pressure, `report_pressure()`, etc.) keeps querying the graph it
already built, just without adding to it until the tier recovers.

The nudge is deliberately small and self-relative rather than a fixed
constant: it can never exceed 15% of a query's own top FTS/vector
relevance score, and each matched entity's contribution is weighted by
that entity's own extraction-time confidence (a running average across
repeated extractions) and match specificity. In practice this means the
graph can break a near-tie or surface an associatively-connected memory
that shares no vocabulary with a chunk that *does* share vocabulary with
the query — but it can never manufacture a hit out of a query with zero
FTS/vector relevance to begin with. A query with no keyword or semantic
overlap with anything in the store returns nothing, same as before the
graph existed.

### Graph maintenance (decay, dedup, LLM-assisted merging)

Write-time dedup (`graph.py`'s `ON CONFLICT DO UPDATE` upserts) and the
row caps (`graph_node_cap`/`graph_edge_cap`) are the *floor* — they stop
unbounded growth but don't improve graph quality over time. Real upkeep
piggybacks on the same `consolidate` job that already handles fact decay
and quarantine trimming (`extraction.consolidate`), triggered the same way
(idle sweep or session end on FULL tier, exit-only dedupe on STANDARD and
LITE) — no new job kind, no new scheduling. Graph upkeep therefore runs at
LITE too, as of 0.2.0: it is all local SQLite, and the LLM-gated steps
(duplicate-entity merge, cluster labeling) are gated separately on FULL.

- **Decay/archival** (`graph.apply_decay`, runs whenever consolidation
  runs, any tier above OFF): a node/edge's confidence decays
  exponentially from its last reinforcement (`updated_at`/`last_seen`),
  same shape as `semantic.effective_importance`. Below
  `graph_archive_threshold` (default 0.15, half-life
  `graph_decay_half_life_days` = 30 days), the row is hard-deleted —
  unlike facts, the graph has no audit-trail requirement, so decay removes
  rather than soft-archives. An entity that keeps coming up in
  conversation resets its own clock on every re-extraction and never
  decays away.
- **Duplicate-entity merging** (`graph.merge_duplicates`, FULL tier only,
  same `llm_merge` gate as the fact contradiction-merge): reviews recently
  created node pairs of the same type that share a token (a cheap
  pre-filter — no fuzzy-matching library, see `graph._canonicalize`'s
  docstring for why) and asks the LLM a single yes/no question per
  candidate pair, capped at 5 pairs per sweep. On "yes", the newer node's
  edges are repointed to the older node and the newer node is deleted.
  This is the one piece of graph maintenance that costs an LLM call, so it
  never runs at STANDARD/LITE.
- **Semantic clustering** (`graph.compute_clusters`/`store_clusters`, runs
  whenever consolidation runs, any tier above OFF — always runs, unlike
  duplicate merging, since it costs no LLM call): groups entities into
  topics via **connected components** over `graph_edges` — no clustering
  library, same union-find approach any graph toolkit uses internally
  under the hood. A cluster is simply "all entities reachable from each
  other," with no distance/similarity threshold, since a graph edge is the
  only relationship signal available. Runs after decay/dedup, not before,
  so stale or duplicate nodes don't fragment or pollute a topic group.
  Singleton components (an entity with no surviving edges) aren't
  considered a topic and get no `cluster_id`. **Labeling**
  (`graph.label_clusters`, FULL tier only, same `llm_merge` gate) asks the
  LLM for a short topic name (e.g. "Local AI") per unlabeled cluster, one
  completion per *new* cluster — already-labeled clusters are skipped on
  later sweeps. An unlabeled cluster is still a fully usable retrieval
  grouping; the label is presentation sugar, not a requirement.

```python
for cluster in mem.clusters():   # largest first
    print(cluster["label"] or "(unlabeled)", cluster["members"])
```

### Graph context in `build_context()`

`build_context()`'s existing sections (facts, episodic, sessions, lessons)
gained one more: `RELATED TOPICS`, populated from the same query-time
graph traversal `recall()`'s graph leg already computes — entities
reachable from the query, grouped by cluster label where one exists. This
is **additive only**: the change adds a new `ContextPlan.sections` key
without touching the existing four, their order, or their behavior, and
without adding a new top-level `Budgets` field (which would touch the
Stable `memory_split` contract). Its budget is instead a fixed share
(`assembly.GRAPH_CONTEXT_SHARE`, 25%) carved out of the existing episodic
budget — which means it inherits episodic's tier gating for free, scaling
down automatically at LITE tier's much smaller episodic allocation without
a separate gate. It renders empty whenever the graph has nothing to offer
for the query (including a LITE store whose graph was never populated in
the first place, see "Knowledge graph" above) — never raises; degrades to
an empty section like every other piece of retrieval.

### `explain(query, k=5) -> ExplainResult`

Retrieval transparency: runs the same ranking `recall()` does, but returns
every per-leg score instead of collapsing them into one number —
`ChunkScoreBreakdown`/`FactScoreBreakdown` (FTS, vector, fused relevance,
importance/recency/graph nudges, matched entity names) plus
`graph_traversal` (the hop path — which entities were detected in the
query, which were reached and at what distance). Intended for debugging
retrieval quality and building "why was this retrieved" views, not the
per-turn hot path: it recomputes rather than reusing `recall()`'s work.
Never raises; a failed leg (e.g. a corrupted graph table) degrades that
one leg to empty, same as every other retrieval failure mode in this
module.

```python
result = mem.explain("what do I know about my Jetson")
for step in result.graph_traversal:
    print(step.canonical_name, step.hop_distance, step.is_seed)
for b in result.chunk_breakdowns:
    print(b.total, "fts=", b.fts, "vector=", b.vector, "graph=", b.graph_nudge)
```

## Timeline query

Facts have always been versioned in full (`facts.valid_from`/
`invalidated_at`/`invalidated_by` — see [schema.md](schema.md); a fact
update never overwrites in place, it invalidates the old row and inserts a
new one, chaining the two). `fact_history(key)` has exposed that chain
since Phase 1. `timeline(query) -> TimelineResult` is the query layer on
top: no new storage, just resolving a natural-language question to the
right key.

Resolution is two-step, cheapest first:
1. **Exact**: `query` normalized as a key (same normalization
   `remember()`/`forget()` use) matches a key that has ever been stored —
   `mem.timeline("occupation")`.
2. **Search**: falls back to the existing fact search
   (`retrieval.fact_relevance`, the same FTS5/LIKE search that backs
   `recall()`) and takes the top-scoring fact's key — handles a free-text
   question like *"what did I do before AI?"* matching the **value** "AI
   Engineer" stored under the `occupation` key. No new NER or fuzzy-
   matching logic; this reuses the fact search that already exists.

```python
mem.remember("occupation", "Student")
mem.remember("occupation", "Designer")
mem.remember("occupation", "AI Engineer")

result = mem.timeline("what did I do before AI")
# result.key == "occupation", result.resolved_by == "search"
for fact in result.versions:   # oldest first
    print(fact.value, fact.valid_from)
```

An unresolvable query (no exact key, no fact search match) returns
`TimelineResult(key=None, resolved_by="none", versions=())` rather than
raising — same never-raises contract as every other retrieval-adjacent
function in this codebase. `forget(key)` tombstones rather than deletes,
so a forgotten key's timeline still shows its full history, ending in an
invalidated (but not erased) final version.

## Degradation matrix

Every capability has a defined floor. **Nothing in this table raises to the
host.**

| Capability | Fallback 1 | Floor |
|---|---|---|
| Vector search | FTS5 BM25 only | `LIKE` term matching (no FTS5 in sqlite build) |
| `embedder` raises (host-supplied OR built-in) | vector leg disabled for the session (logged once), FTS5-only | — |
| Built-in embedder extra not installed / first download fails | same as "embedder raises" above — FTS5-only, logged once | — |
| `llm` absent or raises | regex rule capture | explicit `remember` only |
| Rolling summary | extractive selection from evicted turns (no model call) | `[k earlier turn(s) omitted — memory search can recall them]` |
| Session summary | — | title = first user message (80 chars) |
| Corrupt DB file | renamed `<path>.corrupt-<ts>`, fresh store created, warning logged | — |
| RAM probe fails | assume 8 GiB total / 4 GiB available (STANDARD-ish) | — |
| `recall()` / `build_context()` / `record_turn()` internal error | logged, empty result / empty section / skipped write | never raises |

## Known limitations (exhaustive, as of this writing)

This section exists so a future reader — maintainer or downstream host —
doesn't have to rediscover these by hitting them in production. If you fix
one, update this list.

1. **Rule-based fact capture (`rules.py`) is a fixed, hardcoded regex list.**
   It runs at every tier including LITE, is essentially free, and is
   deliberately conservative (high precision over recall) — but it will
   never learn a phrasing it wasn't written for. Known-covered patterns as
   of this writing: `"my name is X"` / `"my <adjective> name is X"` (e.g.
   "my full name is X"), `"call me X"`, `"I live/stay in X"`,
   `"I'm/I am from X"`, `"I work as/at X"`, `"I'm an X by profession/trade"`,
   `"I'm allergic to X"`, `"I'm vegetarian/vegan/pescatarian"`, and
   `"my favorite/preferred <subject> is X"`. Deliberately NOT attempted:
   typo-tolerant matching (e.g. "you can **all** me kumar" for "call me
   kumar") — the false-positive risk of loosening these patterns outweighs
   the recall benefit for a zero-cost regex layer. Real coverage of
   "everything else" is the LLM extraction pass, which at LITE is **off by
   default** (see the tier table above) — a host running at LITE gets
   rules-only capture until the tier recovers, or until it opts in via
   `lite_llm_extraction`, which defers extraction to session end.
2. **The chars/4 token-estimation proxy is not exact.** Every budget
   computation in the governor, and every `fit_lines()` truncation decision,
   uses `len(text)//4` unless the host supplies `tokenizer_fn`. This is a
   reasonable average for English prose but can diverge meaningfully for
   non-English text, code, or unusual formatting — worth passing a real
   tokenizer if budget precision matters for your use case.
3. **The 256-token dynamic-budget floor is a real, observed failure mode**
   on small-context local models with a large fixed system prompt — see the
   worked example above. It is not a bug (the floor correctly prevents
   negative/zero budgets from crashing), but it can silently reduce every
   memory section to near-nothing. There is no governor-side fix; the host
   must either grow `context_tokens` (increase the model's real `n_ctx`) or
   shrink `static_prompt_tokens`.
4. **Retrieval ranking is relevance-first with importance/recency as small
   tie-breaking nudges, not equal-weight multipliers** (this was a real bug,
   fixed — see `retrieval.py`'s `search_chunks()` docstring for the full
   history). If you're extending the ranking formula, do not reintroduce a
   multiplicative importance/recency term against a bounded 0-1 relevance
   score — it will let a durable-but-topically-irrelevant chunk outrank a
   genuinely relevant one on real queries, exactly as it used to.
5. **`report_pressure()`'s cooldown coalesces reports, it does not validate
   them.** There is no way for the governor to distinguish a genuine
   hardware OOM from a host mis-classifying some other failure as pressure
   (a real example: a naive host that catches every bare `RuntimeError`
   from its inference library and assumes it means OOM, when some other,
   unrelated exception type could in principle share that class — the fix
   belongs in the host's own error classification, not here). Any call to
   `report_pressure()` is trusted at face value, once per cooldown window.
   If your host's OOM detection is noisy or heuristic, tighten it upstream
   rather than relying on the governor to filter false positives; the
   cooldown only limits blast radius, it doesn't distinguish real from
   spurious.
6. **The built-in embedder is English-tuned** (`bge-small-en-v1.5`).
   Multilingual or non-English-dominant hosts will get materially worse
   semantic retrieval quality from the built-in default than from a
   purpose-chosen multilingual embedding model passed via `embedder=`.
   There is currently no multilingual built-in option — an explicit
   `embedder=` override is the only path to better non-English coverage
   today.
7. **No mid-flight cancellation of a downloading built-in embedder.** If the
   first-use download is interrupted (process killed, network drops
   mid-download), `fastembed`/`huggingface-hub`'s own retry/resume behavior
   governs what happens on the next attempt — Elastimem does not add its
   own retry or partial-download cleanup logic on top.
8. **Two-tier upgrade cap.** Upgrades never exceed the tier measured at
   process startup (`_startup_tier`), even after `upgrade_healthy_ticks`
   consecutive healthy ticks. A process that starts at STANDARD (e.g.
   because RAM was briefly under pressure from something else at launch)
   can recover to STANDARD but will never reach FULL for the rest of that
   process's lifetime, even if RAM later becomes genuinely abundant. Only a
   fresh process restart re-measures the ceiling.
9. **No stdlib RAM probe on Windows.** `probe_ram()`'s fallback path (used
   when `psutil` isn't installed) only implements Linux (`/proc/meminfo`)
   and macOS (`sysctl`/`vm_stat`). On Windows — or any other platform — it
   silently assumes 8 GiB total / 4 GiB available, a guess rather than a
   measurement, which can misclassify the tier in either direction. Install
   the `system` extra (`psutil`) for accurate probing on Windows; this is
   the only platform where that extra is effectively required rather than
   a precision upgrade.

## Why this design

Cloud memory systems can assume the model is far away and infinitely
patient. A local agent shares one machine between the model, the memory
system, and the user's actual work. The governor exists so that memory is
always the *first* thing to shrink under pressure and the *last* thing to
crash: dropping episodic injection frees context and cycles; dropping
extraction frees the model; nothing the user said is ever lost, because raw
transcripts persist in every tier and can be re-indexed when capacity
returns.
