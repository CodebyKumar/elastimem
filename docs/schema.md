# Storage schema

One store is one SQLite file in WAL mode (schema version 4). The schema is
defined in `src/elastimem/db.py`; this page explains it.

## Tables

| Table | Holds | Notes |
|---|---|---|
| `meta` | `schema_version` | |
| `sessions` | One row per conversation | `rolling_summary` and `summary_turn` (how far folding has got) survive restarts |
| `messages` | Every message, verbatim | The source of truth; never rewritten |
| `chunks` | Search units, one or more per exchange | `embedding` (float32 BLOB) and `extracted_at` are filled in by the worker; `NULL` means pending |
| `facts` | Versioned key-value facts | One open version per key (`until IS NULL`); `replaced_by` links the chain; `embedding` as for chunks |
| `lessons` | Lessons for the agent | Unique text; beyond `max_lessons` the oldest are archived |
| `rejected` | Automatic facts that failed validation | Capped at 200 rows |
| `entities` | Things mentioned in conversation | Unique on `(type, name)`; `seen` and `confidence` grow with repeats; `topic_id`/`topic` set by consolidation |
| `relations` | Directed links between entities | Unique on `(source, target, relation)` |
| `mentions` | Which chunks mention which entity | Feeds the graph signal |
| `chunks_fts`, `facts_fts` | FTS5 indexes (`porter unicode61`) | Kept in sync by triggers; absent when SQLite lacks FTS5 |

## Pending work is a query

| Work | Query |
|---|---|
| Vectors to compute | `chunks.embedding IS NULL`, `facts.embedding IS NULL` (current facts) |
| Turns to extract | `chunks.extracted_at IS NULL`, grouped by `first_msg_id` |
| Turns to fold | `max(messages.turn) - keep_turns > sessions.summary_turn` |

Vectors carry `embedding_model`. On open, vectors written by a different
embedder are cleared so they are redone.

## Size

At 384 dimensions a vector is 1.5 KB, so 10,000 chunks add about 15 MB to
the file and the same to the in-RAM index while vector search is on.
Entities and relations are capped at 2,000 and 5,000 rows.

## Migrations

Opening an older store migrates it in one transaction; a failed migration
leaves it untouched.

- **v3 (Elastimem 0.2) -> v4**: renames (`quarantine` -> `rejected`,
  `graph_nodes`/`graph_edges` -> `entities`/`relations`,
  `valid_from`/`invalidated_at` -> `since`/`until`), adds
  `chunks.extracted_at`, `facts.embedding`, `sessions.summary_turn` and
  `mentions` (backfilled from chunk text). `rule` facts become `auto`.
  Turns recorded before v4 count as extracted, so the history is not
  queued for the LLM.
- **v1/v2**: open once with Elastimem 0.2 to reach v3 first.
- A store newer than the installed package is refused.

A corrupt file is moved to `<path>.corrupt-<timestamp>` and a fresh store
is created.
