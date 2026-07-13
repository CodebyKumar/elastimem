# Elastimem

**Elastic, resource-adaptive memory for local-first AI agents.**

Elastimem gives your agent long-term memory — facts, past conversations, learned
lessons — in a single SQLite file, with **zero required dependencies**. Its
defining feature is the **Memory Governor**: Elastimem probes the machine it's
running on and elastically sizes every memory capability to fit, so the same
code serves a 4 GB Jetson and a 128 GB workstation. Every capability has a
documented degradation floor; nothing ever hard-fails because the host is
small.

Not on PyPI — install directly from this repository:

```
pip install git+https://github.com/CodebyKumar/elastimem.git                          # zero dependencies
pip install "elastimem[system] @ git+https://github.com/CodebyKumar/elastimem.git"    # + psutil for richer hardware probing
pip install "elastimem[embed] @ git+https://github.com/CodebyKumar/elastimem.git"     # + fastembed, activates the built-in embedder
```

Or clone and install locally (editable, for hacking on the source):

```
git clone https://github.com/CodebyKumar/elastimem.git
cd elastimem
pip install -e .
```

## Why another memory framework?

Mem0, Letta/MemGPT, and Zep all assume a big cloud model and unbounded
resources. Local agents live in a different world: a 2B model with a 4096-token
context on a machine that's also running the model itself. Elastimem is designed
for that world:

- **Host-agnostic.** Elastimem is a library, not a runtime. It never loads a
  model or calls an API. You inject `llm` / `embedder` callables;
  Elastimem degrades gracefully around whatever you don't provide.
- **Elastic.** Token budgets, retrieval depth, window size, extraction
  cadence — all derived from your model's context size and the machine's RAM,
  re-evaluated at runtime under memory pressure.
- **Honest storage.** One SQLite file. Facts are temporally versioned, never
  destructively overwritten ("I'm a writer now" invalidates "designer" and
  keeps the audit trail). Forgetting archives; it doesn't delete.
- **Small-model-proof.** Every write from an automatic extractor passes
  validation guards (placeholder junk, transcript echoes, the model storing
  facts about *itself*); rejects are quarantined for inspection, not silently
  dropped.

## Quickstart

```python
import elastimem

mem = elastimem.open("~/.myagent/memory.db")        # that's it — memory works

# each turn:
ctx = mem.build_context(user_input)        # budgeted prompt sections + window plan
reply = run_my_agent(ctx, user_input)
mem.record_turn(user_input, reply)         # persist + rule capture

# anytime:
mem.remember("dietary_restriction", "vegetarian")   # explicit, durable
hits = mem.recall("what did we discuss about my car")
mem.end_session()                          # summarize, consolidate, close
```

Give it an LLM and an embedder when you have them — every capability is
optional and unlocks more:

```python
mem = elastimem.open(
    "~/.myagent/memory.db",
    llm=my_llm,               # (prompt, *, max_tokens, temperature) -> str
    embedder=my_embedder,     # (list[str]) -> list[list[float]]
    context_tokens=4096,      # any config field can be passed inline...
    reserved_keys={"model"},
)

# ...or bundle settings in a reusable config object:
from elastimem import ElastimemConfig
cfg = ElastimemConfig(context_tokens=4096, reserved_keys=frozenset({"model"}))
mem = elastimem.open("~/.myagent/memory.db", llm=my_llm, config=cfg)
```

No `llm` → no LLM extraction/summaries (regex rules + explicit memory still
work). No `embedder` → keyword (FTS5 BM25) retrieval. No psutil → stdlib
hardware probe. No FTS5 in your sqlite build → `LIKE` retrieval. It always
works; it's just progressively less clever.

## The four memory layers

| Layer                | What it holds                                                  | Where                                       |
| -------------------- | -------------------------------------------------------------- | ------------------------------------------- |
| **Working**    | current conversation window + rolling summary of evicted turns | host's message list, planned by Elastimem   |
| **Episodic**   | full past transcripts, chunked and indexed for recall          | `messages` / `chunks` (+FTS5, +vectors) |
| **Semantic**   | facts about the user, temporally versioned, importance-decayed | `facts`                                   |
| **Procedural** | lessons the agent learned about its own behavior               | `lessons`                                 |

## The Memory Governor

At startup (and on every `tick()`), Elastimem classifies the machine into a tier
and derives token budgets from your model's context size:

| Capability                | FULL (≥16 GB)       | STANDARD (≥8 GB)     | LITE               |
| ------------------------- | -------------------- | --------------------- | ------------------ |
| Embeddings used           | yes                  | yes, if provided      | never              |
| Episodic injection        | hybrid, top 4        | FTS5, top 3           | `recall()` only  |
| Working window            | ~8–10 turns         | ~5–6 turns           | 2 turns + newest   |
| Rolling summary           | LLM, every eviction  | LLM, every 2nd batch  | placeholder marker |
| LLM fact extraction       | background, per turn | batched every 3 turns | off (rules only)   |
| Rule capture, transcripts | always               | always                | always             |

Full spec: [docs/governor.md](docs/governor.md). Architecture and rationale:
[docs/architecture.md](docs/architecture.md). Integration guides (llama.cpp,
OpenAI-compatible, no-LLM): [docs/integrations.md](docs/integrations.md).

## Status

Pre-release alpha (0.1.0a1). The core API (`open`, `remember`, `recall`,
`record_turn`, `build_context`, see [docs/api_stability.md](docs/api_stability.md)
for the exact Stable surface) is expected to stay stable; internal
implementation details and advanced features may still evolve based on
feedback from early adopters. See [CHANGELOG.md](CHANGELOG.md) for release
history.

Built as the memory engine for [Tuffy](https://github.com/CodebyKumar/tuffy)

## Contributing

Bug reports, feature requests, and PRs are welcome — see
[CONTRIBUTING.md](CONTRIBUTING.md) for dev setup and what to read before
touching the governor, retrieval ranking, or the worker's locking (several
current behaviors replaced an earlier, buggier design — the docs explain
why). See [CHANGELOG.md](CHANGELOG.md) for release history.

## License

MIT
