# Elastimem

**Long-term memory for local AI agents, in one SQLite file.**

Elastimem remembers what was said, what is true about the user, what the
agent learned, and how those things connect. It has no required
dependencies, never loads a model or calls an API on its own, and sizes its
background work to the machine it runs on, so the same code serves a 4 GB
Jetson and a 128 GB workstation.

Not on PyPI; install from this repository:

```
pip install "elastimem @ git+https://github.com/CodebyKumar/elastimem.git"           # no dependencies
pip install "elastimem[embed] @ git+https://github.com/CodebyKumar/elastimem.git"    # + built-in embedder (fastembed)
pip install "elastimem[system] @ git+https://github.com/CodebyKumar/elastimem.git"   # + psutil
```

For development: `git clone`, then `uv sync` (or `pip install -e .`).

## Quickstart

```python
import elastimem

with elastimem.open("~/.myagent/memory.db", llm=my_llm, context_tokens=8192) as mem:
    context = mem.context(user)                  # memory for this turn's prompt
    reply = my_llm(f"{context}\n\n{user}")
    mem.record(user, reply)                      # store it; derived work runs in the background

    mem.remember("diet", "vegetarian")           # explicit, durable fact
    for hit in mem.search("what did we say about my car"):
        print(hit.date, hit.text)
```

Every argument beyond the path is optional. `llm` is
`(prompt, *, max_tokens, temperature) -> str` and enables extraction of
facts, entities and relations from conversation plus written summaries.
`embedder` is `(list[str]) -> list[list[float]]`; the default `"auto"` uses
the built-in model when `elastimem[embed]` is installed, and `None` means
keyword search only. Without an `llm`, exchanges are still kept and
searchable, and facts come from `remember()`.

## What it keeps

| Layer | Answers | Found by |
|---|---|---|
| Working | What is happening now? | The host's message list, trimmed to `context.keep_turns`, plus a rolling summary |
| Moments | What happened before? | Keyword + vector + graph |
| Facts | What is true about the user? (versioned, never overwritten) | Keyword + vector + graph |
| Graph | What is connected to what? | Entity names in the query, then hops |
| Lessons | What should the agent do differently? | Most recent |
| Sessions | What were past conversations about? | Most recent |

One equation ranks moments and facts, and `explain()` splits any score into
its terms. Automatic facts must be grounded in what the user said, so
placeholders and echoed transcripts are rejected without any word lists.

## Elastic by design

A governor picks a tier (LITE, STANDARD, FULL) from total RAM, current
memory pressure and how long `context()` actually took. A tier decides what
background work is allowed: whether the built-in embedder may load, how often
the LLM extracts, how deep the graph is followed. Ranking and token budgets
are the same on every tier. `mem.status` says what is on and why.

Measured on an 8 GB Mac with a 19-question evaluation set:

| | |
|---|---|
| Recall@5, keyword only / built-in embedder | 0.68 / 1.00 |
| `context()` per turn at 10k chunks | 12 ms |
| RAM added, keyword only / built-in embedder | 3 / 368 MB |

## Docs

| | |
|---|---|
| [integrations.md](docs/integrations.md) | Install, the turn loop, memory as tools, host checklist |
| [api.md](docs/api.md) | Every method, setting and type |
| [architecture.md](docs/architecture.md) | How storage, search, the governor and the worker fit together |
| [schema.md](docs/schema.md) | SQLite tables and migrations |
| [api_stability.md](docs/api_stability.md) | What is stable and what is not |
| [redesign.md](docs/redesign.md) | Why 0.3 looks the way it does, with measurements |
| [agent_integration_prompt.md](docs/agent_integration_prompt.md) | One document to hand a coding agent (Claude Code, Cursor) to integrate or upgrade |

Runnable samples are in [examples/](examples/).

## Status

Pre-1.0 (0.3.0). The names listed in
[api_stability.md](docs/api_stability.md) are stable; everything else,
including the database schema (which migrates automatically), may change.
0.3.0 broke the 0.2 API; see [CHANGELOG.md](CHANGELOG.md).

Built as the memory engine for [Tuffy](https://github.com/CodebyKumar/tuffy).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT
