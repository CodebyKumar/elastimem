# Elastimem: agent integration prompt

**Audience: a coding agent (Claude Code, Cursor, Copilot, an SDK agent) asked
to add Elastimem to a project or upgrade an existing integration.** Paste
this whole file into the agent's context. It is self-sufficient: every
method, every setting, the upgrade steps and the usual mistakes.

Current version: **0.3.0**. If the project pins an older version, read
[section 8](#8-upgrading) first.

---

## 1. What Elastimem is

A local-first memory library for chat agents. It stores messages, facts
about the user (versioned), lessons, and an entity graph in **one SQLite
file** with **no required dependencies**. It never loads a chat model or
calls an API itself: the host passes an `llm` and an `embedder` function if
it has them. A governor measures the machine and decides how much
background work is affordable (tier LITE / STANDARD / FULL). After
construction, `context()`, `record()` and `search()` never raise.

It is not a vector database, a RAG framework or a hosted service. It hands
the host text sections; the host builds its own prompt.

## 2. Questions to answer from the target project

| Question | Where it lands |
|---|---|
| The chat model's context window? | `context_tokens=` (default 4096 starves memory on bigger models) |
| Tokens needed for the project's own system prompt, tool output and reply? | `reserved_tokens=` |
| Is there a function that completes a prompt cheaply (the same model is fine)? | `llm=`. Without it, facts only come from `remember()` |
| One local model instance shared with Elastimem (llama.cpp)? | Wrap each generation in `mem.pause()` |
| Must it never download anything? | `embedder=None` or your own function |
| Does the app own identity keys (`agent_name`, `model`)? | `blocked=` |
| Does the host's own model support tools? | Expose `remember` and `memory_search` (section 5) |

## 3. Install

Python 3.10-3.13. Not on PyPI; pin a tag in the project's dependency file:

```bash
pip install "elastimem @ git+https://github.com/CodebyKumar/elastimem.git@v0.3.0"
pip install "elastimem[embed] @ git+...@v0.3.0"    # fastembed: built-in embedder, ~130 MB download, ~360 MB RAM when loaded
pip install "elastimem[system] @ git+...@v0.3.0"   # psutil: battery detection; memory readings on Windows
```

Without `embed`, search is keyword-only unless an `embedder` is passed.
numpy is used for vector search when importable and is optional.

## 4. Minimum integration

```python
import elastimem

mem = elastimem.open("~/.myapp/memory.db", llm=my_complete_fn,
                     context_tokens=8192, reserved_tokens=1500)

def chat(user: str) -> str:
    context = mem.context(user)                        # never raises
    history[:] = history[-2 * context.keep_turns:]     # older turns live in context.summary
    with mem.pause():                                  # only if the model instance is shared
        reply = model(system=f"{SYSTEM_PROMPT}\n\n{context}", messages=history + [user])
    mem.record(user, reply)                            # never raises
    return reply

mem.close()                                            # on exit
```

`str(context)` renders the rolling summary and every non-empty section
under headings. Use `context.sections["facts"]` (and `topics`, `moments`,
`sessions`, `lessons`) to place them yourself.

## 5. Memory as tools

Expose two tools to the host's model. They are the recommended way to get
facts when there is no background `llm`, and `memory_search` reaches the
full history instead of what fits the prompt budget.

```python
def call_tool(name, args):
    if name == "remember":                      # args: key (snake_case), value
        return "stored" if mem.remember(args["key"], args["value"]) else "already known"
    hits = mem.search(args["query"])            # name == "memory_search"
    return "\n".join(f"- [{h.date}] {h.text}" for h in hits) or "nothing found"
```

## 6. API

All importable from `elastimem`. Everything else is internal.

`elastimem.open(path, *, llm=None, embedder="auto", tokenizer=None,
on_tier=None, config=None, **settings)` is the same as
`Elastimem(path, ...)`; it works as a context manager. Unknown setting:
`TypeError`; invalid value: `ValueError`. `":memory:"` gives a throwaway
store.

| Method | Returns | Notes |
|---|---|---|
| `context(query="")` | `Context` | Per turn. Re-checks the governor |
| `record(user, reply)` | `None` | Per turn, after the reply |
| `pause()` | `Pause` | `with mem.pause():` or `p = mem.pause()` ... `p.end()` from any thread |
| `search(query, k=5)` | `list[Hit]` | Moments and facts, best first |
| `explain(query, k=5)` | `Explanation` | Per-term score breakdown |
| `remember(key, value)` | `bool` | True if new or changed. Key normalized to snake_case |
| `forget(key)` | `bool` | Closes the version; history kept |
| `facts()` | `dict[str, str]` | Current facts |
| `history(key_or_question)` | `list[Fact]` | All versions, oldest first |
| `learn(lesson)` / `lessons(n=5)` | `bool` / `list[str]` | Notes about the agent's own behavior |
| `topics()` | `list[dict]` | Groups of related entities |
| `rejected(n=20)` | `list[dict]` | Automatic facts that failed validation |
| `sessions(n=20)` | `list[dict]` | Newest first |
| `new_session(tag=None)` | `int` | `record()` starts one on its own |
| `end_session()` | `None` | Finishes background work, summarizes, tidies |
| `resume(session_id=None)` | `(summary, messages)` | Default: the latest session |
| `status` | `Status` | Tier, budgets, `keep_turns`, `vectors`, `llm`, `graph_hops`, `reasons` |
| `config` | `ElastimemConfig` | A copy; mutating it does nothing |
| `configure(**settings)` | `Status` | Change settings; re-reads the hardware |
| `report_pressure()` | `Status` | OOM or failed decode: one tier down |
| `flush(timeout=5.0)` | `bool` | Finish pending background work |
| `stats()` | `dict` | Row counts, file size, FTS5 availability |
| `close()` | `None` | Safe to call twice |

Types: `Context` (`sections`, `summary`, `keep_turns`, `status`), `Hit`
(`kind`, `text`, `date`, `score`, `session`), `Fact` (`key`, `value`,
`source`, `since`, `until`), `Status`, `Budgets`, `Explanation`, `Tier`
(`LITE < STANDARD < FULL`).

Behavior that needs no host code: keyword and vector search over every
exchange; exact repeats shown once; versioned facts (a new value closes
the old one); grounding check on automatic facts; rolling summary of turns
that left the window (LLM when allowed, first sentences otherwise); entity
graph and topics when an `llm` is given; decay of unused memories; vectors
cleared and redone when the embedder changes.

## 7. Settings

Passed to `open(...)` inline or as `config=ElastimemConfig(...)`.

| Field | Default | Meaning |
|---|---|---|
| `context_tokens` | 4096 | The chat model's context window |
| `reserved_tokens` | 1112 | Host's own prompt, tool output and reply |
| `tier` | `None` | Fixed tier; `None` = governor. Env `ELASTIMEM_TIER=lite\|standard\|full` |
| `pinned` | empty | Fact keys always listed first |
| `blocked` | empty | Fact keys the host owns; never stored (also matches `*_<key>`) |
| `turn_budget_ms` | 50 | Memory work per turn before the tier steps down |
| `half_life_days` | 30 | How fast unused memories fade |
| `max_lessons` | 15 | Older lessons are archived |

Tier behavior: FULL (16 GiB+) and STANDARD (8 GiB+) may load the built-in
embedder; LITE never does but uses a host-supplied embedder. LLM extraction
runs every turn (FULL), every 2 turns (STANDARD) or at session end (LITE).
Ranking and token budgets are identical on every tier. Pressure caps or
lowers the tier; slow turns lower it; it recovers after 10 healthy turns.
Other constants live in `elastimem/tiers.py` (internal).

Call `configure(context_tokens=..., reserved_tokens=...)` after switching
models. Call `report_pressure()` on a confirmed OOM, not on every
exception. `on_tier=callback` fires on each tier change (e.g. to unload an
embedding model).

## 8. Upgrading

1. Find the installed version: `python -c "import elastimem; print(elastimem.__version__)"`
   (`0.0.0+dev` is an editable checkout: use `git describe --tags`). Also
   check the project's dependency file for a `@v...` pin.
2. Change the pin to `@v0.3.0` in the dependency file, not only the environment.
3. Apply the matching subsection below. Subsections are append-only, so
   follow every one between the installed version and the current one.
4. Verify with section 10.

### `0.2.0` -> `0.3.0` (breaking: no aliases were kept)

Stores migrate automatically on first open (schema v3 -> v4, one
transaction, untouched if it fails). Code must change. Grep for each old
name:

| 0.2 | 0.3 |
|---|---|
| `build_context(q)` -> `ContextPlan` | `context(q)` -> `Context` (`.sections`, `.summary`, `.keep_turns`, `.status`; `str(ctx)` renders it) |
| `plan.keep_last_n_turns`, `plan.rolling_summary` | `context.keep_turns`, `context.summary` |
| `record_turn(u, a)` | `record(u, a)` |
| `recall(q)` | `search(q)` -> `list[Hit]` |
| `with mem.foreground():`, `foreground_begin/end` | `with mem.pause():` or `p = mem.pause()` ... `p.end()` |
| `tick()` | Gone: `context()` re-checks the governor |
| `reconfigure(reprobe=True, ...)` | `configure(...)` (always re-reads hardware) |
| `profile` (`MemoryProfile`) | `status` (`Status`) |
| `drain(timeout)` | `flush(timeout)` |
| `report_evictions(...)` | Gone: use `context.keep_turns` and `context.summary` |
| `begin_session(tag)`, `resume_session(id)` | `new_session(tag)`, `resume(id)` |
| `remember(k, v, source=)` -> `(bool, str)` | `remember(k, v)` -> `bool` |
| `fact_history(key)`, `timeline(q)` | `history(key_or_question)` |
| `clusters()` | `topics()` |
| `add_lesson(text, tag)` | `learn(text)` |
| `quarantine_entries()` | `rejected()` |
| `static_prompt_tokens`, `output_reserve`, `tool_reserve` | One `reserved_tokens` (sum of the three) |
| `reserved_keys` | `blocked` |
| `tier_override` | `tier` |
| `embed_fn=`, `disable_builtin_embedder=True` | `embedder=` (`None` = keyword only; `"auto"` = built-in) |
| Any other `ElastimemConfig` field | Gone: the field is now a constant in `tiers.py`, or removed |
| `Tier.LITE/STANDARD/FULL`, `ElastimemConfig` | Unchanged |

Behavior changes to know about:

- No regex rule capture: facts come from `remember()` (expose it as a tool)
  or from an `llm`. Without both, no facts are created.
- LLM extraction depends on `llm=` being given, not on the tier; the tier
  only sets the cadence (every turn / every 2 turns / session end).
- Facts are now searchable by meaning (they have vectors).
- Per-tier ranking weights are gone: one equation for all tiers.
- `ELASTIMEM_TIER` still works.

0.1.0a1 and 0.2.0 were never on PyPI; a project on either follows the
table above.

## 9. Mistakes to avoid

- Leaving `context_tokens` at 4096 on a larger model.
- Generating without `pause()` when one model instance is shared with
  Elastimem. It works in testing and collides under load.
- Calling `flush()`, `end_session()` or `close()` inside `with mem.pause():`.
  They return, but LLM work stays pending until a flush outside the pause.
- Passing `embedder=None` to mean "the default". It means keyword search
  only; the default is `"auto"`.
- Mutating `mem.config`. It is a copy; use `configure()`.
- Assuming construction never raises: an unwritable path or an unknown
  setting does. Calls after construction do not.
- Forgetting `close()` on exit: pending background work is lost.
- Not exposing `memory_search` as a tool: `context()` only reaches what
  fits the budget.
- Keeping all history in the host after `context.keep_turns`: the rolling
  summary already covers older turns.

## 10. Verification

```python
import elastimem
from elastimem import Tier

with elastimem.open(":memory:", embedder=None) as mem:
    assert mem.remember("home_city", "Lisbon")
    assert mem.facts() == {"home_city": "Lisbon"}
    mem.record("I adopted a puppy called Rex", "Congratulations!")
    mem.flush()

    assert any("Rex" in h.text for h in mem.search("what is my dog called"))
    context = mem.context("where do I live")
    assert "Lisbon" in str(context)
    assert context.keep_turns >= 2
    assert context.status.tier in (Tier.LITE, Tier.STANDARD, Tier.FULL)

    mem.learn("Check the logs before suggesting a restart.")
    assert mem.lessons()
    print("tier:", mem.status.tier.name, "stats:", mem.stats())
print("Elastimem integration OK", elastimem.__version__)
```

For an existing store, also open it and confirm `mem.stats()` shows the
old rows and `mem.facts()` the old facts.

## 11. Where to read more

| Doc | Covers |
|---|---|
| [integrations.md](integrations.md) | Install, the loop, tools, host checklist |
| [api.md](api.md) | Full reference |
| [api_stability.md](api_stability.md) | Stable vs internal names |
| [architecture.md](architecture.md) | Storage, search, governor, threads |
| [schema.md](schema.md) | Tables and migrations |
| [redesign.md](redesign.md) | Why 0.3 is shaped this way, with measurements |
| [CHANGELOG.md](../CHANGELOG.md) | Release history |

## 12. Maintaining this document

**Update it in the same commit as any user-visible change.** It is what
downstream agents read to learn what Elastimem can do.

- Public method, signature or return shape -> section 6
- `ElastimemConfig` field or default -> section 7
- Tier behavior -> section 7
- Version number -> header, section 3, section 8 pin
- Anything needing migration -> a new subsection in section 8 (append-only)
- A new failure mode -> section 9
