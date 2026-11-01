# API reference

Everything below is importable from `elastimem`. Submodules are internal
(see [api_stability.md](api_stability.md)).

## Opening a store

```python
mem = elastimem.open(path, *, llm=None, embedder="auto", tokenizer=None,
                     on_tier=None, config=None, **settings)
```

Same as `elastimem.Elastimem(path, ...)`. Usable as a context manager:
`with elastimem.open(...) as mem:` closes it on exit.

| Argument | Meaning |
|---|---|
| `path` | SQLite file, created if missing. `":memory:"` for a throwaway store |
| `llm` | `(prompt, *, max_tokens, temperature) -> str` for background work. Optional |
| `embedder` | `(list[str]) -> list[list[float]]`; `"auto"` = the built-in model when `elastimem[embed]` is installed; `None` = keyword search only |
| `tokenizer` | `(str) -> int` token counter; default is about four characters per token |
| `on_tier` | `(old: Tier, new: Tier) -> None`, called when the tier changes |
| `config` | an `ElastimemConfig`; inline `**settings` override it |

Raises `TypeError` for an unknown setting and `ValueError` for an invalid
value.

## Each turn

| Method | Returns | Notes |
|---|---|---|
| `context(query="")` | `Context` | Memory for this turn's prompt. Re-checks the governor. Never raises |
| `record(user, reply)` | `None` | Stores one exchange; derived work runs in the background. Never raises |
| `pause()` | `Pause` | `with mem.pause():` around the host's own generation, when the host and Elastimem share one model. Or `p = mem.pause()` ... `p.end()`, from any thread |

## Searching

| Method | Returns | Notes |
|---|---|---|
| `search(query, k=5)` | `list[Hit]` | Past moments and facts, best first. Never raises |
| `explain(query, k=5)` | `Explanation` | The same results with each score split into its terms |

## Facts and lessons

| Method | Returns | Notes |
|---|---|---|
| `remember(key, value)` | `bool` | True if new or changed. Keys are normalized to snake_case |
| `forget(key)` | `bool` | Closes the current version; history is kept |
| `facts()` | `dict[str, str]` | Current facts |
| `history(key_or_question)` | `list[Fact]` | Every version, oldest first. A question is matched against current facts |
| `learn(lesson)` | `bool` | Stores a lesson for the agent itself |
| `lessons(n=5)` | `list[str]` | Newest lessons, oldest first |
| `topics()` | `list[dict]` | `{"id", "name", "members"}` groups of related entities |
| `rejected(n=20)` | `list[dict]` | Automatic facts that failed validation, for debugging |

## Sessions

| Method | Returns | Notes |
|---|---|---|
| `sessions(n=20)` | `list[dict]` | Newest first: `id, started_at, ended_at, title, summary, message_count` |
| `new_session(tag=None)` | `int` | Ends the current session and starts another. Optional: `record()` starts one |
| `end_session()` | `None` | Finishes background work, summarizes (with `llm`), tidies memory |
| `resume(session_id=None)` | `(str \| None, list[dict])` | Rolling summary and trailing messages of a past session (default: the latest) |

## Status and lifecycle

| Member | Returns | Notes |
|---|---|---|
| `status` | `Status` | What memory will do this turn, and why |
| `config` | `ElastimemConfig` | A copy; changing it has no effect |
| `configure(**settings)` | `Status` | Change settings, e.g. `context_tokens` after a model switch |
| `report_pressure()` | `Status` | The host hit an OOM or failed decode: one tier down |
| `flush(timeout=5.0)` | `bool` | Finish all pending background work; False on timeout |
| `stats()` | `dict` | Row counts, file size, FTS5 availability |
| `close()` | `None` | Ends the session, stops the worker, closes connections. Safe to call twice |

## Settings (`ElastimemConfig`)

| Field | Default | Meaning |
|---|---|---|
| `context_tokens` | 4096 | The chat model's context window. Set it |
| `reserved_tokens` | 1112 | Tokens the host needs for its own prompt, tool output and the reply |
| `tier` | `None` | Fix the tier; `None` lets the governor decide. Env: `ELASTIMEM_TIER=lite\|standard\|full` |
| `pinned` | empty | Fact keys always listed first |
| `blocked` | empty | Fact keys the host owns; never stored (also matches `*_<key>`) |
| `turn_budget_ms` | 50 | Memory work per turn before the tier steps down |
| `half_life_days` | 30 | How fast unused memories fade |
| `max_lessons` | 15 | Older lessons are archived |

Everything else that could be tuned is a constant in `elastimem/tiers.py`.

## Types

**`Context`**: `sections: dict[str, str]` with keys `facts`, `topics`,
`moments`, `sessions`, `lessons` (`""` when empty); `summary: str | None`
(this session's turns that left the window); `keep_turns: int` (turns the
host should keep verbatim); `status: Status`. `str(context)` renders the
summary and non-empty sections under Markdown headings.

**`Status`**: `tier: Tier`; `budgets: Budgets`; `keep_turns: int`;
`vectors: bool`; `llm: str` (`per_turn`, `batched`, `session_end` or
`off`); `graph_hops: int`; `reasons: dict[str, str]` with keys `tier`,
`vectors`, `llm`.

**`Budgets`**: token budgets `working, facts, moments, topics, sessions,
lessons`.

**`Hit`**: `kind` (`"moment"` or `"fact"`), `text`, `date`
(`YYYY-MM-DD`), `score`, `session` (moments only).

**`Fact`**: `key`, `value`, `source` (`explicit`, `auto` or `import`),
`since`, `until` (`None` for the current version).

**`Explanation`**: `query`, `tier`, `weights` (the tier's equation),
`missing` (term -> why it did not contribute), `items` (each with `hit`
and `parts`: the weighted contribution of each term, summing to the score).

**`Tier`**: `LITE < STANDARD < FULL`.
