# Public API reference

The surface documented here is stable — see [api_stability.md](api_stability.md)
for the precise Stable/Experimental/Internal boundaries. Anything not listed
there (submodules like `elastimem.governor`, underscore-prefixed members) is
internal even though Python does not prevent importing it.

`elastimem.open(path, **kwargs) -> Elastimem`

The friendly front door — identical to constructing `Elastimem(path, **kwargs)`.

`Elastimem(path, *, llm=None, embedder=None, config=None, tokenizer_fn=None, probe_fn=None, on_tier_change=None, **config_overrides)`

One persistent store backed by one SQLite file (`":memory:"` supported).

- `llm(prompt, *, max_tokens, temperature) -> str` — optional LLM.
  (`complete_fn=` still accepted as a legacy alias.)
- `embedder(texts: list[str]) -> list[list[float]]` — optional embedder.
  (`embed_fn=` still accepted as a legacy alias.) **Leaving this `None`
  (i.e. not passing it) auto-activates Elastimem's own built-in embedder**
  (`BAAI/bge-small-en-v1.5` via the optional `fastembed` extra) rather than
  leaving semantic search disabled — see
  [governor.md](governor.md#built-in-embedder-auto-activates-read-this-before-assuming-embeddings-are-off).
  Set `disable_builtin_embedder=True` in config to guarantee FTS5-only
  behavior instead; passing `embedder=None` explicitly does NOT opt out on
  its own, since `None` is exactly what triggers the auto-activation.
- `**config_overrides` — any `ElastimemConfig` field passed inline, e.g.
  `context_tokens=4096, reserved_keys={"model"}`. May be combined with
  `config=`; inline values override the config object. Unknown names raise
  `TypeError` listing the valid options.
- `tokenizer_fn(text) -> int` — optional exact token counter (default chars/4).
- `probe_fn() -> (total_bytes, available_bytes)` — override hardware probe.
- `on_tier_change(old: Tier, new: Tier)` — governor callback.

### Attributes (post-construction, read-only in normal use)

- `embed_fn` — the resolved passage-side embedder (host-supplied, the
  built-in default, or `None` if neither is active).
- `embed_query_fn` — the resolved query-side embedder for an asymmetrically-
  trained embedder (set automatically alongside `embed_fn` when the
  built-in default activates; `None` for a host-supplied `embedder=`, which
  is assumed symmetric — retrieval then reuses `embed_fn` for queries too).
- `config` — read-only snapshot (`ElastimemConfig`) of the active
  configuration. Mutating the returned object has no effect; use
  `reconfigure()` to change settings, which keeps derived budgets in sync.
  There is no public `governor` attribute — the equivalent operations are
  `profile`, `tick()`, `report_pressure()`, and `reconfigure()` on `Elastimem`
  itself.

`embed_fn`, `embed_query_fn`, `tokenizer_fn`, `path`, and `session_id` are
construction-time only: set them via `open()`/`Elastimem(...)` and don't
reassign after construction (the background worker may hold a reference to
the original callable).

### Per-turn

| method                                                                             | purpose                                                                                                          |
| ---------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `tick() -> MemoryProfile`                                                        | cheap hardware re-check; call once per turn                                                                      |
| `build_context(user_input="") -> ContextPlan`                                    | budgeted prompt sections + window plan; never raises                                                             |
| `foreground()` (context manager) / `foreground_begin()` / `foreground_end()` | hold background LLM jobs while the host generates                                                                |
| `record_turn(user_text, assistant_text)`                                         | persist exchange, rule capture, enqueue extraction/embedding; never raises                                       |
| `report_evictions(turns: list[tuple[str, str]])`                                 | fold host-evicted (user, assistant) pairs into the rolling summary                                               |
| `report_pressure() -> MemoryProfile`                                             | OOM/decode-failure signal; downgrades one tier on first call, coalesces repeat calls within a 30s cooldown (see [governor.md](governor.md)) |
| `reconfigure(**config_overrides) -> MemoryProfile`                               | update config (e.g.`context_tokens` after a model switch) and rebuild budgets immediately — the only supported way to change config after construction |

### Memory operations

| method                                                           | purpose                                    |
| ---------------------------------------------------------------- | ------------------------------------------ |
| `remember(key, value, source="explicit") -> (changed, reason)` | validated, synchronous, durable fact write |
| `facts() -> dict[str, str]`                                    | current facts                              |
| `fact_history(key) -> list[Fact]`                              | full version chain, oldest first           |
| `forget(key) -> bool`                                          | tombstone the current version              |
| `recall(query, k=5) -> list[Hit]`                              | search chunks + facts; never raises        |
| `add_lesson(text, tag=None)` / `lessons(n=None)`             | procedural memory                          |

### Sessions

| method                                                                  | purpose                                               |
| ----------------------------------------------------------------------- | ----------------------------------------------------- |
| `begin_session(host_tag=None) -> int`                                 | explicit start (`record_turn` starts one lazily)    |
| `end_session()`                                                       | drain worker, LLM session summary, exit consolidation |
| `sessions(n=20) -> list[dict]`                                        | recent sessions, newest first                         |
| `resume_session(session_id=None) -> (rolling_summary, tail_messages)` | reload a past session into the host's window          |

### Lifecycle / inspection

| method                                     | purpose                                 |
| ------------------------------------------ | --------------------------------------- |
| `drain(timeout=5.0) -> bool`             | finish queued background work           |
| `close()`                                | drain, stop worker, close connections   |
| `stats() -> dict`                        | row counts, file size, FTS availability |
| `quarantine_entries(n=20) -> list[dict]` | rejected auto-extractions               |
| `profile -> MemoryProfile`               | current governor snapshot               |

## Value types

- **`ContextPlan`** — `sections: dict[str, str]` (keys: `user_facts`,
  `relevant_past_moments`, `previous_sessions`, `lessons`),
  `rolling_summary: str | None`, `keep_last_n_turns: int`,
  `profile: MemoryProfile`, `render() -> str`.
- **`MemoryProfile`** — `tier`, `budgets` (`working/facts/episodic/sessions/ lessons`, tokens), `embeddings_enabled`, `llm_extraction_enabled`,
  `extraction_cadence`, `rolling_summary_enabled`, `consolidation_level`,
  `episodic_top_k`. Note: `embeddings_enabled` reflects the GOVERNOR's
  tier-based permission only (`False` at LITE, `True` otherwise) — it does
  NOT mean an embedder is actually configured and working. Check
  `store.embed_fn is not None` separately if you need to know whether
  vector search is actually possible right now (both conditions must hold
  for the vector leg to run).
- **`Hit`** (from `elastimem.retrieval`) — `kind` (`chunk|fact`), `text`,
  `date`, `score`, `session_id`.
- **`Fact`** — `key`, `value`, `category`, `source`, `importance`,
  `valid_from`, `invalidated_at`, `archived`.
- **`ElastimemConfig`** — see docstrings in `elastimem/config.py`; notable fields:
  `context_tokens`, `static_prompt_tokens`, `reserved_keys`, `profile_keys`,
  `chunk_target_tokens`, `worker_max_tokens`, `tier_override`,
  `min_query_words` (gates LLM extraction, default 4),
  `min_retrieval_query_words` (gates local FTS5/vector retrieval, default 1
  — deliberately much lower than `min_query_words` since retrieval is free
  and a short query still carries real intent), `disable_builtin_embedder`
  (default `False` — set `True` to force FTS5-only retrieval and prevent
  the built-in embedder from ever activating, e.g. for an air-gapped host),
  `tier_thresholds_gib` (RAM thresholds the governor uses to classify
  FULL/STANDARD/LITE), `memory_split` (how the non-working token pool
  divides across facts/episodic/sessions/lessons), `working_share`
  (fraction of the dynamic token pool reserved for the working window).
- **`Tier`** — `LITE < STANDARD < FULL`; env override `ELASTIMEM_TIER`.
