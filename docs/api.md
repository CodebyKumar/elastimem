# Public API reference

Everything importable from `elastimem` is public; everything else is internal.

## `Elastimem(path, *, complete_fn=None, embed_fn=None, config=None, tokenizer_fn=None, probe_fn=None, on_tier_change=None)`

One persistent store backed by one SQLite file (`":memory:"` supported).

- `complete_fn(prompt, *, max_tokens, temperature) -> str` — optional LLM.
- `embed_fn(texts: list[str]) -> list[list[float]]` — optional embedder.
- `tokenizer_fn(text) -> int` — optional exact token counter (default chars/4).
- `probe_fn() -> (total_bytes, available_bytes)` — override hardware probe.
- `on_tier_change(old: Tier, new: Tier)` — governor callback.

### Per-turn
| method | purpose |
|---|---|
| `tick() -> MemoryProfile` | cheap hardware re-check; call once per turn |
| `build_context(user_input="") -> ContextPlan` | budgeted prompt sections + window plan; never raises |
| `foreground()` (context manager) / `foreground_begin()` / `foreground_end()` | hold background LLM jobs while the host generates |
| `record_turn(user_text, assistant_text)` | persist exchange, rule capture, enqueue extraction/embedding; never raises |
| `report_evictions(turns: list[tuple[str, str]])` | fold host-evicted (user, assistant) pairs into the rolling summary |
| `report_pressure() -> MemoryProfile` | OOM/decode-failure signal; downgrades one tier |

### Memory operations
| method | purpose |
|---|---|
| `remember(key, value, source="explicit") -> (changed, reason)` | validated, synchronous, durable fact write |
| `facts() -> dict[str, str]` | current facts |
| `fact_history(key) -> list[Fact]` | full version chain, oldest first |
| `forget(key) -> bool` | tombstone the current version |
| `recall(query, k=5) -> list[Hit]` | search chunks + facts; never raises |
| `add_lesson(text, tag=None)` / `lessons(n=None)` | procedural memory |

### Sessions
| method | purpose |
|---|---|
| `begin_session(host_tag=None) -> int` | explicit start (`record_turn` starts one lazily) |
| `end_session()` | drain worker, LLM session summary, exit consolidation |
| `sessions(n=20) -> list[dict]` | recent sessions, newest first |
| `resume_session(session_id=None) -> (rolling_summary, tail_messages)` | reload a past session into the host's window |

### Lifecycle / inspection
| method | purpose |
|---|---|
| `drain(timeout=5.0) -> bool` | finish queued background work |
| `close()` | drain, stop worker, close connections |
| `stats() -> dict` | row counts, file size, FTS availability |
| `quarantine_entries(n=20) -> list[dict]` | rejected auto-extractions |
| `profile -> MemoryProfile` | current governor snapshot |

## Value types

- **`ContextPlan`** — `sections: dict[str, str]` (keys: `user_facts`,
  `relevant_past_moments`, `previous_sessions`, `lessons`),
  `rolling_summary: str | None`, `keep_last_n_turns: int`,
  `profile: MemoryProfile`, `render() -> str`.
- **`MemoryProfile`** — `tier`, `budgets` (`working/facts/episodic/sessions/
  lessons`, tokens), `embeddings_enabled`, `llm_extraction_enabled`,
  `extraction_cadence`, `rolling_summary_enabled`, `consolidation_level`,
  `episodic_top_k`.
- **`Hit`** (from `elastimem.retrieval`) — `kind` (`chunk|fact`), `text`,
  `date`, `score`, `session_id`.
- **`Fact`** — `key`, `value`, `category`, `source`, `importance`,
  `valid_from`, `invalidated_at`, `archived`.
- **`ElastimemConfig`** — see docstrings in `elastimem/config.py`; notable fields:
  `context_tokens`, `static_prompt_tokens`, `reserved_keys`, `profile_keys`,
  `chunk_target_tokens`, `worker_max_tokens`, `tier_override`.
- **`Tier`** — `LITE < STANDARD < FULL`; env override `ELASTIMEM_TIER`.
