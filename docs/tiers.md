# Tiers by task

Elastimem runs one of three tiers — `FULL`, `STANDARD`, `LITE` — chosen from
available RAM and re-checked on every `tick()` (see
[governor.md](governor.md) for classification thresholds, upgrade/downgrade
rules, and budget math). This page is the task-first companion to that spec:
for each thing Elastimem *does*, what actually happens at each tier.

Check `mem.profile.tier` before assuming a task is misbehaving — most
"memory isn't working" reports turn out to be "this process has been running
at LITE the whole time."

**LITE's floor is "spend no new resources" — not "least capability
possible."** By default LITE refuses exactly three things: an LLM call, a
model load, and any sustained per-item cost (embedding every new chunk).
Those are the guarantees a RAM-starved host actually needs. Everything
else — anything that is just a local SQLite read or some string
manipulation — runs at LITE too: consolidation, 1-hop graph traversal,
graph decay/clustering, extractive summaries, and scoring vectors that
already exist. Those are independent decisions, which is exactly why they
can be made separately.

## Fact capture

| Tier | Behavior |
|---|---|
| FULL | Regex rules (always) **+** LLM extraction in the background, per turn |
| STANDARD | Regex rules (always) **+** LLM extraction, batched every 2 turns |
| LITE | Regex rules only. Set `lite_llm_extraction=True` to opt in to LLM extraction here, deferred to session end |

Regex rules (`rules.py`) are ~10 hardcoded, high-precision patterns — "my
name is X", "call me X", "I live in X", "I work as X", allergies, diet,
"my favorite X is Y". They run at every tier for free but never learn a new
phrasing. LLM extraction is what catches everything else, and it's the
first thing that disappears going into LITE.

## Retrieval (`recall()`, `memory_search`)

| Tier | Behavior |
|---|---|
| FULL | Vector (embedder) + FTS5 + 2-hop knowledge graph nudge |
| STANDARD | Vector (embedder) + FTS5 + 1-hop knowledge graph nudge |
| LITE | FTS5 + 1-hop graph nudge, **plus** the vector leg over chunks that are already embedded — if an embedder is already resident |

The vector leg at LITE is narrower than it looks, and the distinction
matters: LITE never *indexes* new chunks and never *loads* the built-in
model. What it will do is score chunks embedded during an earlier
STANDARD/FULL stretch, using an embedder that is already in memory — a
host-supplied `embedder=` always is, since it lives in your process. So a
store that dips to LITE mid-session keeps semantic recall over its existing
history instead of falling off a cliff to keyword-only; a store that has
only ever run at LITE with no host embedder gets FTS5 alone.

The knowledge graph nudge can only ever refine an existing FTS/vector hit —
it never manufactures a result from a query with zero keyword or semantic
overlap.

## Episodic injection (`build_context()`)

| Tier | Behavior |
|---|---|
| FULL | Top 5 recent turns injected |
| STANDARD | Top 4 recent turns injected |
| LITE | Top 3 recent turns injected, on a reduced (but real) token share — full history always reachable via explicit `recall()` regardless of tier |

How many rows get read is a token-budget question, not a resource one:
pulling three rows out of SQLite instead of one costs a starved machine
nothing. LITE keeps 45% of the normal episodic token share and 75% of the
sessions share, returning the rest to the working window.

## Rolling / session summaries

| Tier | Rolling summary (window eviction) |
|---|---|
| FULL | LLM-generated |
| STANDARD | LLM-generated |
| LITE | Extractive: the leading clause of each evicted user turn, fitted to the sessions budget. No model call. Falls back to the marker line `"[k earlier turn(s) omitted — memory search can recall them]"` only if nothing usable can be extracted |

Extractive summarization is selection, not inference — it reuses text
already persisted in the database, so it costs no model call and no
embedder. It is bounded by the sessions token budget, with the oldest
content ageing out first, so it cannot grow without limit across a long
session.

Session **titles** are a separate mechanism and not tier-gated: every
session title is the first 80 chars of the first user message, at every
tier — `end_session()`/`close_orphan_sessions()` set it via
`COALESCE(title, ?)`, so it's the baseline default, not a LITE fallback.
An LLM-generated session **summary** (a fuller 1-2 sentence recap, distinct
from the title) is layered on top when the host supplies one; nothing in
the tier system currently gates that call — it happens if and only if the
host explicitly generates and passes a summary at `end_session()` time.

## Consolidation (contradiction merging, decay, quarantine trimming)

| Tier | Behavior |
|---|---|
| FULL | Full LLM-assisted merge, runs on idle sweep + session exit |
| STANDARD | Dedupe + decay only (no LLM merge), exit only |
| LITE | Dedupe + decay only (no LLM merge), exit only — same as STANDARD |

Consolidation at the dedupe+decay level is entirely local SQLite: fact
decay/archival, quarantine trimming, graph decay/archival, cluster
recompute. It used to be `Off` at LITE, which made a starved machine the
one machine that never pruned anything and grew monotonically — and, since
graph upkeep lives in the same sweep, silently disabled that too. The
LLM-assisted merge steps remain FULL-only.

Contradictory fact updates (e.g. "I live in Austin" → weeks later "I live
in Denver") are always versioned correctly — the old value is never lost —
but only FULL tier LLM-merges them into one coherent value. Below FULL, the
host just sees the latest version.

## Knowledge graph

| Tier | Behavior |
|---|---|
| FULL | 2-hop traversal; decay + clustering + LLM duplicate-merge + LLM cluster labeling |
| STANDARD | 1-hop traversal; decay + clustering (no LLM merge/labeling) |
| LITE | 1-hop traversal; decay + clustering — same as STANDARD, minus the LLM-gated merge/labeling |

Decay and connected-components clustering are free (no LLM call) and run
at every tier including LITE; duplicate-entity merging and topic labeling
cost an LLM call each, so they're FULL-only.

**LITE reads the graph but does not normally grow it.** Traversal is a
local SQLite query with no LLM involved, so it runs at LITE same as
STANDARD. But new nodes/edges are only ever written by the same per-turn
LLM completion that produces facts (`extraction.py`), and that completion
does not run at LITE unless you set `lite_llm_extraction=True`. So a store
that has only ever run at LITE with the default config has an empty graph —
traversal finds nothing to nudge with. A store that spent time at
STANDARD/FULL before dropping to LITE keeps querying whatever graph it
already built.

## Always on, every tier

- Rule-based fact capture
- Raw transcript persistence
- `remember()` / `recall()` / `forget()` (explicit API calls)
- Fact versioning and `timeline()` / `fact_history()`
- `explain()` retrieval-transparency query
- 1-hop knowledge graph traversal, decay, and clustering
- Consolidation at the dedupe + decay level (LLM merge steps need FULL)

Nothing the user says is ever lost regardless of tier — raw transcripts
persist everywhere and can be re-indexed once capacity recovers.

## Degradation, not failure

Every task above has a defined floor and none of them raise to the host.
Losing a tier means losing *quality* (semantic recall on new content →
keyword recall, LLM merge → versioned-but-unmerged facts, LLM summary →
extractive summary) — never a crash, and never silent data loss. See governor.md's "Degradation matrix"
for the full fallback chain per capability, and "Known limitations" for the
sharp edges (Windows RAM probe, chars/4 token estimate, the 256-token
budget floor, English-only built-in embedder).

## See also

- [governor.md](governor.md) — full normative spec: tier classification
  thresholds, upgrade/downgrade timing, budget derivation formulas, and the
  complete degradation matrix.
- [architecture.md](architecture.md) — how these tasks fit into the overall
  pipeline.
- [schema.md](schema.md) — how fact versioning and graph tables are stored.
