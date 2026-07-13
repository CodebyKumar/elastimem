# Storage schema

One store = one SQLite file. WAL journal mode, `synchronous=NORMAL`,
foreign keys ON. All timestamps are ISO-8601 UTC strings. Schema is
versioned via `meta.schema_version`; migrations run in `db.open_store()`.

## Tables

### `meta`
`key TEXT PK, value TEXT` — schema version, install markers, checkpoints.

### `sessions`
| column | notes |
|---|---|
| `id` | PK |
| `started_at`, `ended_at` | `ended_at` NULL while live; orphans (crash) are closed on next open |
| `title` | first user message, ≤ 80 chars — the no-LLM summary floor |
| `summary` | LLM 1–2 sentence episodic summary (nullable) |
| `rolling_summary` | last working-memory digest, used by `resume_session` |
| `message_count` | maintained by `record_turn` |
| `host_tag` | free-form host label (model id, app version…) |

### `messages` — the permanent verbatim record
| column | notes |
|---|---|
| `session_id → sessions` | `ON DELETE CASCADE` |
| `turn` | user-turn index within the session |
| `role` | `user \| assistant \| system` |
| `content` | text; media as `[image: <ref>]` placeholders |
| `token_estimate` | chars/4 at write time; used by resume budgeting |

Index: `(session_id, turn)`.

### `chunks` — episodic retrieval units
One per exchange (`User: …\nAssistant: …`), sentence-split beyond
`chunk_target_tokens` (default 400).

| column | notes |
|---|---|
| `first_msg_id`, `last_msg_id` | span of source messages |
| `text` | the condensed exchange |
| `embedding` | little-endian float32 BLOB (`array('f')`), NULL until embedded |
| `embedding_model` | label; lets a re-index detect stale vectors. `"elastimem:bge-small-en-v1.5"` for vectors from the built-in default embedder, `"host"` for any host-supplied `embedder=` (the label doesn't distinguish between different host embedders — a host that swaps its own embedding model should re-embed or track that distinction itself) |
| `importance` | default 0.5; consolidation may raise/lower later |

Indexes: `(session_id)`; partial `WHERE embedding IS NULL` (the embed queue).
FTS5: `chunks_fts` (content-linked, `porter unicode61`, trigger-maintained).

### `facts` — temporally versioned semantic memory
| column | notes |
|---|---|
| `key` | normalized snake_case |
| `value` | short string |
| `category` | `profile` (always injected) \| `note` (competes by score) |
| `source` | `explicit(1.0) \| rule(0.7) \| auto(0.5) \| import(0.8)` → `importance` |
| `valid_from`, `invalidated_at` | NULL `invalidated_at` = current version |
| `invalidated_by → facts` | the superseding row (audit chain) |
| `archived` | decay-forgotten; excluded from prompts, never deleted |
| `last_accessed_at`, `access_count` | bumped when injected; feeds decay |

Indexes: unique partial `(key) WHERE invalidated_at IS NULL` (one live
version per key — updates must invalidate before insert); `(key, valid_from)`
for history. FTS5: `facts_fts` over `(key, value)`.

### `lessons`
`text UNIQUE, tag, use_count, archived` — procedural memory; capped by
archiving the oldest beyond `max_lessons`.

### `quarantine`
Rejected automatic extractions: `ts, key, value, reason, source`. Capped at
`quarantine_cap` (200). Never injected into prompts; exists so extractor
misbehavior is inspectable.

## Size expectations

384-dim float32 vector = 1.5 KB/chunk → 10k chunks ≈ 15 MB of vectors.
Benchmarked recall at 10k chunks (pure Python): FTS5 ~2 ms, hybrid ~43 ms.
`sqlite-vec`, when installed, can index the same BLOBs for much larger
stores; it is optional and unused by default.
