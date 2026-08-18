# Elastimem

**Elastic, resource-adaptive memory for local-first AI agents.**

Elastimem gives your agent long-term memory — facts, past conversations, learned
lessons, and an **embedded knowledge graph** connecting them — in a single
SQLite file, with **zero required dependencies**. Its defining feature is
the **Memory Governor**: Elastimem probes the machine it's running on and
elastically sizes every memory capability to fit, so the same code serves a
4 GB Jetson and a 128 GB workstation. Every capability has a documented
degradation floor; nothing ever hard-fails because the host is small.

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
- **Associative, not just keyword/semantic.** Entities and relationships are
  extracted alongside facts (same background LLM pass, no extra model call)
  and stored as an embedded knowledge graph in the same SQLite file. Asking
  about "my Jetson" can surface a memory that only mentions "Tuffy" — because
  the graph knows Tuffy runs on the Jetson — even though the two share no
  vocabulary. This is one more retrieval signal, not a separate database;
  the Memory Governor gates it exactly like everything else (1-hop at LITE
  and STANDARD, deeper at FULL).

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

## The knowledge graph

With an `llm` configured, Elastimem extracts entities and relationships
alongside facts from the same background reflection pass — no second model
call, no NER library, no graph database. They're stored as plain tables
(`graph_nodes`/`graph_edges`) in the same SQLite file and folded into
retrieval as one more signal:

```python
mem.record_turn("I'm building a robot called Tuffy that runs on a Jetson",
                "Sounds like a fun project!")
mem.record_turn("Tuffy uses CUDA for the vision pipeline",
                "Nice, that should help with real-time inference.")
mem.end_session()   # graph maintenance runs here: decay, dedup, clustering

# a query sharing no vocabulary with the CUDA memory still finds it,
# because Jetson -> Tuffy -> CUDA are graph-connected:
hits = mem.recall("what do I know about my Jetson")

# see exactly why, with per-signal scores and the traversal path:
result = mem.explain("what do I know about my Jetson")
for step in result.graph_traversal:
    print(step.canonical_name, step.hop_distance)

# related entities auto-group into topics (connected components + an
# optional LLM-generated label, e.g. "Local AI"):
for cluster in mem.clusters():
    print(cluster["label"], cluster["members"])

# facts are versioned; resolve a natural-language question to the right
# key and see the full history:
mem.remember("occupation", "Student")
mem.remember("occupation", "AI Engineer")
mem.timeline("what did I do before AI")   # -> Student -> AI Engineer
```

Depth adapts with the Memory Governor exactly like everything else — 2-hop
graph expansion at FULL tier, 1-hop at STANDARD and LITE (traversal is a
bounded SQL query over capped tables, so the first hop costs a starved
machine nothing; only the unbounded second hop needs FULL). Full design and
the reasoning behind it:
[governor.md's Knowledge graph section](docs/governor.md#knowledge-graph).

## The five memory layers

| Layer                | What it holds                                                  | Where                                       |
| -------------------- | -------------------------------------------------------------- | ------------------------------------------- |
| **Working**    | current conversation window + rolling summary of evicted turns | host's message list, planned by Elastimem   |
| **Episodic**   | full past transcripts, chunked and indexed for recall          | `messages` / `chunks` (+FTS5, +vectors) |
| **Semantic**   | facts about the user, temporally versioned, importance-decayed | `facts`                                   |
| **Procedural** | lessons the agent learned about its own behavior               | `lessons`                                 |
| **Graph**      | entities and relationships connecting the above, plus topic clusters | `graph_nodes` / `graph_edges`         |

## The Memory Governor

At startup (and on every `tick()`), Elastimem classifies the machine into a tier
and derives token budgets from your model's context size:

| Capability                | FULL (≥16 GB)       | STANDARD (≥8 GB)     | LITE               |
| ------------------------- | -------------------- | --------------------- | ------------------ |
| Indexing new chunks       | yes                  | yes                   | never              |
| Vector leg when searching | yes                  | yes                   | yes, over already-indexed chunks, if the embedder is already resident |
| Episodic injection        | hybrid, top 5        | hybrid, top 4         | top 3              |
| Working window            | ~8–10 turns         | ~5–6 turns           | 2 turns + newest   |
| Rolling summary           | LLM, every eviction  | LLM, every eviction   | extractive, no model call |
| LLM fact extraction       | background, per turn | batched every 2 turns | off by default (`lite_llm_extraction` opts in, deferred to session end) |
| Consolidation             | full, incl. LLM merge | dedupe + decay        | dedupe + decay     |
| Knowledge graph           | 2-hop expansion       | 1-hop expansion        | 1-hop expansion    |
| Rule capture, transcripts | always               | always                | always             |

LITE's floor is **"spend no new resources," not "least capability
possible."** It makes no LLM call, loads no model, and pays no sustained
per-chunk cost — but anything that is just local SQLite (consolidation,
graph traversal, extractive summarization, scoring vectors that already
exist) runs there too. Those decisions are independent of each other, which
is exactly why they can be made separately. See
[docs/governor.md](docs/governor.md#capability-tier).

Full spec: [docs/governor.md](docs/governor.md). Architecture and rationale:
[docs/architecture.md](docs/architecture.md). Integration guides (llama.cpp,
OpenAI-compatible, no-LLM): [docs/integrations.md](docs/integrations.md).

## Status

Beta (0.2.0), out of pre-release. The core API (`open`, `remember`,
`recall`, `record_turn`, `build_context`, see
[docs/api_stability.md](docs/api_stability.md) for the exact Stable surface)
has settled and now moves under the documented versioning policy: additive
changes bump the minor version, breaking changes bump the major version.
Internal implementation details and advanced features may still evolve
based on feedback. The knowledge-graph query surface
(`explain()`, `timeline()`, `clusters()`) is marked **Experimental** —
the underlying storage is stable, but their exact return shapes may still
change based on real usage. See [CHANGELOG.md](CHANGELOG.md) for release
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
