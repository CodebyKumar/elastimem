# API Stability Policy

Elastimem's implementation (retrieval strategy, storage schema, extraction
prompts, governor internals) is expected to change across releases. The
surface listed below as **Stable** will not — code written against it keeps
working without modification across minor and patch releases.

## Stable

Safe to depend on long-term. Changes to this surface follow semantic
versioning: additive changes bump the minor version, breaking changes bump
the major version and are called out in `CHANGELOG.md`.

**Top-level (`import elastimem`)**

```python
elastimem.open(path, **kwargs)
elastimem.Elastimem                # the facade class
elastimem.ElastimemConfig
elastimem.MemoryProfile
elastimem.Budgets
elastimem.Tier
elastimem.Cadence
elastimem.ConsolidationLevel
elastimem.Fact
elastimem.__version__
```

**`Elastimem` instance methods and properties**

| Member | Purpose |
|---|---|
| `profile` (property) | current `MemoryProfile` snapshot |
| `config` (property) | read-only snapshot of active `ElastimemConfig` |
| `tick()` | per-turn cheap hardware re-check |
| `report_pressure()` | signal OOM/decode failure |
| `reconfigure(*, reprobe=False, **overrides)` | update config, rebuild budgets |
| `build_context(user_input="")` | assemble budgeted prompt sections |
| `begin_session(host_tag=None)` | start a session explicitly |
| `record_turn(user_text, assistant_text)` | persist an exchange |
| `foreground()` / `foreground_begin()` / `foreground_end()` | bracket host LLM generation |
| `report_evictions(turns)` | fold evicted turns into rolling summary |
| `drain(timeout=5.0)` | finish queued background work |
| `recall(query, k=5)` | search past conversations/facts |
| `sessions(n=20)` | recent sessions |
| `resume_session(session_id=None)` | reload a past session |
| `end_session()` | close session, summarize, consolidate |
| `remember(key, value, source="explicit")` | store one fact |
| `facts()` | current facts dict |
| `fact_history(key)` | version chain of a fact |
| `forget(key)` | tombstone a fact |
| `add_lesson(text, tag=None)` | store a procedural lesson |
| `lessons(n=None)` | load lessons |
| `quarantine_entries(n=20)` | rejected auto-extractions |
| `stats()` | row counts, file size, FTS flag |
| `close()` | stop worker, close connections |

`config` is read-only: assigning into the object it returns (e.g.
`mem.config.context_tokens = 8192`) has no effect. Use `reconfigure()` for
any change that should take effect.

`complete_fn`, `embed_fn`, `embed_query_fn`, `tokenizer_fn`, `path`, and
`session_id` are readable attributes on `Elastimem` but are **construction-time
only** — pass them via `open()`/`Elastimem(...)` and do not reassign them
after construction. Reassignment is not guarded against today, but is
unsupported: the background worker may hold a reference to the original
callable.

## Experimental

None yet. Future additions the team wants public feedback on before
committing to stability will be exposed under `elastimem.experimental` and
may change or disappear at any time, including in patch releases.

## Internal

Everything not listed above, specifically including:

- Any name beginning with `_` on any object (e.g. `Elastimem._governor`,
  `Elastimem._conn`, `Elastimem._execute_job`).
- Submodules other than the top-level `elastimem` package — `elastimem.governor`,
  `elastimem.retrieval`, `elastimem.extraction`, `elastimem.episodic`,
  `elastimem.semantic`, `elastimem.procedural`, `elastimem.db`,
  `elastimem.worker`, `elastimem.embeddings`, `elastimem.assembly`,
  `elastimem.guards`, `elastimem.rules`, `elastimem.default_embedder` — are
  implementation, not API, even though Python does not prevent importing
  from them. This includes module-level constants such as
  `elastimem.governor.GIB`.
- The `Governor` class and its methods (`tick`, `report_pressure`,
  `reconfigure`, `profile`) — these are wrapped by identically-named methods
  on `Elastimem` itself, which is the supported entry point.

Internal surface may change or be removed without notice, in any release,
including patch releases. If you find yourself reaching into it, that's a
signal a corresponding stable method is missing — please open an issue
rather than depending on the internal path.

## Why this split exists

The framework's implementation (how retrieval works, how facts are scored,
how the governor classifies hardware) is expected to improve over time.
Locking those details into the public API would force a breaking release
every time the implementation gets better. The stable surface above is
intentionally described in terms of user intent (`recall`, `remember`,
`forget`) rather than mechanism (no `retrieve_fts`, no `run_governor_cycle`),
so the framework is free to change how it fulfills that intent without
changing how you call it.
