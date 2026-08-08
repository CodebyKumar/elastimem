# Tiers by task

Elastimem runs one of three tiers — `FULL`, `STANDARD`, `LITE` — chosen from
available RAM and re-checked on every `tick()` (see
[governor.md](governor.md) for classification thresholds, upgrade/downgrade
rules, and budget math). This page is the task-first companion to that spec:
for each thing Elastimem *does*, what actually happens at each tier.

Check `mem.profile.tier` before assuming a task is misbehaving — most
"memory isn't working" reports turn out to be "this process has been running
at LITE the whole time."

**LITE's floor is "no LLM call, ever" — not "least capability possible."**
Anything that costs an LLM completion (extraction, rolling summaries,
consolidation merges) or an embedder call (vector search) is fully off at
LITE, unconditionally, because that's the actual resource guarantee a
RAM-starved host needs. But local, no-LLM operations — a sliver of episodic
injection, a 1-hop graph traversal, graph decay/clustering — don't threaten
that guarantee, so LITE now includes a small amount of each rather than
zeroing them outright.

## Fact capture

| Tier | Behavior |
|---|---|
| FULL | Regex rules (always) **+** LLM extraction in the background, per turn |
| STANDARD | Regex rules (always) **+** LLM extraction, batched every 2 turns |
| LITE | Regex rules only — no LLM call is ever attempted |

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
| LITE | FTS5 keyword search only — **no embedder call, ever** |

Semantic/paraphrase matching requires the embedder, which is fully off at
LITE regardless of whether one is configured and working. The knowledge
graph nudge can only ever refine an existing FTS/vector hit — it never
manufactures a result from a query with zero keyword or semantic overlap.

## Episodic injection (`build_context()`)

| Tier | Behavior |
|---|---|
| FULL | Top 5 recent turns injected |
| STANDARD | Top 4 recent turns injected |
| LITE | Top 1 recent turn injected, on a small token sliver reclaimed from the working-window carve-out — always reachable in full via explicit `recall()` regardless of tier |

## Rolling / session summaries

| Tier | Rolling summary (window eviction) |
|---|---|
| FULL | LLM-generated |
| STANDARD | LLM-generated |
| LITE | Marker line only: `"[k earlier turn(s) omitted — memory search can recall them]"` |

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
| LITE | Off |

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

**LITE reads the graph but never grows it.** Traversal is a local SQLite
query with no LLM involved, so it runs at LITE same as STANDARD. But new
nodes/edges are only ever written by the same per-turn LLM completion that
produces facts (`extraction.py`), and that completion never runs at LITE.
So a store that has only ever run at LITE has an empty graph — traversal
finds nothing to nudge with. A store that spent time at STANDARD/FULL
before dropping to LITE keeps querying whatever graph it already built; it
just stops growing until the tier recovers.

## Always on, every tier

- Rule-based fact capture
- Raw transcript persistence
- `remember()` / `recall()` / `forget()` (explicit API calls)
- Fact versioning and `timeline()` / `fact_history()`
- `explain()` retrieval-transparency query
- 1-hop knowledge graph traversal, decay, and clustering (as of this update)

Nothing the user says is ever lost regardless of tier — raw transcripts
persist everywhere and can be re-indexed once capacity recovers.

## Degradation, not failure

Every task above has a defined floor and none of them raise to the host.
Losing a tier means losing *quality* (semantic recall → keyword recall, LLM
merge → versioned-but-unmerged facts, LLM summary → a marker line) — never
a crash, and never silent data loss. See governor.md's "Degradation matrix"
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
