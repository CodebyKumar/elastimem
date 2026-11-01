# Architecture

How Elastimem stores memory, finds it again, and decides how much work the
machine can afford. The reasoning behind each decision, and the
measurements that settled it, are in [redesign.md](redesign.md).

## Principles

1. **Messages are the source of truth.** Everything else (chunks, vectors,
   facts, the graph, summaries) is derived from them. Pending work is
   whatever derived data is missing, read from SQL, so a crash loses
   nothing and a tier upgrade backfills on its own.
2. **One ranking equation** for every kind of memory, so any result can be
   explained term by term.
3. **Each capability is gated by what it costs.** RAM gates loading a
   model, measured time gates per-turn work, the context window gates token
   budgets.
4. **No word lists.** Checks are structural (is this fact grounded in what
   the user said?) or data-driven (is this word common in this store?).
5. **Never break the host.** After construction, `context()`, `record()`
   and `search()` never raise; a failing part degrades to empty.

## Memory layers

| Layer | Answers | Stored in | Found by | Vectors |
|---|---|---|---|---|
| Working | What is happening now? | The host's message list | `keep_turns` + the rolling summary | No |
| Moments | What happened before? | `messages`, `chunks` | Keyword + vector + graph | Yes |
| Facts | What is true about the user? | `facts` (versioned) | Keyword + vector + graph | Yes |
| Graph | What is connected to what? | `entities`, `relations`, `mentions` | Entity names in the query, then hops | No |
| Lessons | What should the agent do differently? | `lessons` | Most recent | No |
| Sessions | What were past conversations about? | `sessions.summary` | Most recent | No |

Vectors are a search method, not a layer. They help when the wording of a
question differs from the stored text ("how is my dog?" vs "I adopted a
puppy"), which happens with moments and facts.

## Writing

`record(user, reply)` writes, synchronously and in well under a
millisecond:

- two `messages` rows (verbatim);
- one or more `chunks` (the search unit; long exchanges split at sentence
  ends), indexed for keyword search by SQLite triggers.

Then it wakes the background worker, which looks for missing derived data
in this order:

| Derived data | Pending when | Needs |
|---|---|---|
| Chunk and fact vectors | `embedding IS NULL` | An embedder the tier allows |
| Facts, entities, relations | `chunks.extracted_at IS NULL` | `llm=`, at the tier's cadence |
| Rolling summary (LLM) | Turns left the window, not folded | `llm=` and a tier with LLM summaries |
| Consolidation | Idle, FULL only | Nothing (LLM merges need `llm=`) |

Without an LLM, or on LITE, `record()` folds turns that left the window
into the rolling summary itself, by keeping the first sentence of each
user turn. No model call.

Automatic facts must be **grounded**: the value shares a word with the
user's message and does not just copy it. That one rule rejects
placeholders ("unknown"), echoed transcripts and statements about the
assistant ("an AI assistant"), in any language. Rejected facts are kept in
`rejected` for inspection.

## Searching

A query is analysed once per turn: its words, its vector (if allowed), and
the entities it names plus everything within `graph_hops` of them. Three
relevance signals follow, each in 0-1:

| Signal | Computed from |
|---|---|
| keyword | FTS5 BM25 over the query's distinctive words, divided by the best score. A word in more than half the rows of a table is ignored (this is what drops "the", "User:", ...) |
| vector | Similarity of unit vectors, unscaled (comparable between moments and facts) |
| graph | The item mentions an entity the query reaches: confidence x 0.5 per hop, divided by the best |

They combine in one equation:

```
relevance = 0.80*vector + 0.10*keyword + 0.10*graph
score     = relevance + 0.05*importance + 0.05*recency
```

- A term that found nothing for this query hands its weight to the others
  in proportion, so `relevance` keeps the same scale.
- Only items with `relevance > 0` are ranked. Importance (explicit facts
  1.0, auto 0.5, a turn that produced a fact 0.8) and recency
  (`exp(-age / half_life_days)`) break ties; they never surface an
  unrelated memory.
- Exchanges repeated word for word are shown once.

The weights were tuned on `benchmarks/eval_recall.py`. The graph weight is
untested there, because the evaluation runs without an LLM.

## The context

`context(query)` returns a `Context` whose sections are filled to fixed
token budgets. The budgets come from `context_tokens - reserved_tokens`:
55% for the host's recent turns (`keep_turns`), the rest split 40% facts,
30% moments (a quarter of it for topics), 15% sessions, 15% lessons. They
are the same on every tier.

The facts section always lists pinned keys first, then facts the user
stated explicitly, then facts matching the query, then the rest by
importance and age.

## The governor and tiers

The tier is chosen from three signals:

| Signal | Source | Effect |
|---|---|---|
| Capacity | Total RAM | 16 GiB+ FULL, 8 GiB+ STANDARD, else LITE |
| Pressure | macOS `kern.memorystatus_vm_pressure_level`; Linux `MemAvailable/MemTotal`; psutil elsewhere | warn caps at STANDARD, critical forces LITE |
| Measured cost | How long `context()` took | Over `turn_budget_ms` (50) for 3 turns: one tier down |

Free-page counts are not used: macOS keeps most memory compressed or
cached, so they read low on healthy machines. Going down is immediate;
going up waits for 10 healthy turns and never passes what capacity and
pressure allow. Hardware is read at most every 5 seconds. On battery
(needs psutil), background LLM work waits for the session end.

What each tier allows (all values live in `tiers.py`):

| | FULL | STANDARD | LITE |
|---|---|---|---|
| Load the built-in embedder (~360 MB) | yes | yes | no |
| Use an embedder already in memory | yes | yes | yes |
| LLM extraction | every turn | every 2 turns | session end |
| Rolling summary | LLM | LLM | extractive |
| LLM merges, topic names | yes | no | no |
| Graph hops | 2 | 1 | 1 |
| Ranking equation, token budgets | same | same | same |

A host-supplied `embedder` is always "in memory": it lives in the host's
process, so LITE uses it. LITE never makes the built-in model appear.

## Memory without an LLM

With no `llm=`, Elastimem makes no model calls and still keeps every
exchange, finds past ones by keyword and meaning, folds the window
extractively, and stores facts given through `remember()`. The usual way
to get facts without a background LLM is to expose `remember()` to the
host's own model as a tool, so it stores facts during its normal reply. An
LLM adds structure on top: facts extracted from free conversation,
entities and relations, written summaries, merged contradictions.

On the evaluation set (19 questions, no LLM), recall@5 is 0.68 with
keyword search only and 1.00 with the built-in embedder.

## Threads and locks

- The host calls public methods on its own thread(s). One daemon worker
  thread does derived work.
- All writes share one `RLock`. Each thread has its own SQLite connection
  (WAL); `:memory:` stores share one connection, because each new
  `:memory:` connection is a separate empty database. `close()` closes
  every thread's connection.
- Every background LLM call holds `llm_lock`. `pause()` takes it too, so
  the host's own generation never overlaps a background call; the worker
  skips LLM work while paused. A pause may end on a different thread
  (streaming callbacks).

## Modules

| Module | Owns |
|---|---|
| `store.py` | `Elastimem`, the public class |
| `config.py` | `ElastimemConfig`, `Tier` |
| `tiers.py` | Every tunable constant: tier capabilities, the equation, budgets, caps |
| `governor.py` | Tier choice, `Status`, `Budgets` |
| `db.py` | Schema, migrations, connections, time helpers |
| `episodes.py` | Messages, chunks, sessions, rolling summary |
| `facts.py` | Versioned facts, grounding, rejected facts |
| `graph.py` | Entities, relations, mentions, topics |
| `lessons.py` | Lessons |
| `vectors.py` | Embedders and the in-RAM vector index |
| `search.py` | Query analysis and signals |
| `scoring.py` | The equation |
| `context.py` | `Context` and its sections |
| `inspection.py` | `explain()`, `history()` |
| `llm.py` | Prompts and parsing for background calls |
| `jobs.py` | The worker, `pause()`, consolidation |
