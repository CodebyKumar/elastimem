# Changelog

All notable changes to Elastimem are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versioning follows
[PEP 440](https://peps.python.org/pep-0440/) pre-release identifiers while
in alpha (`0.1.0a1`, `0.1.0a2`, ... → `0.1.0b1` → `0.1.0`), then the policy
in [docs/api_stability.md](docs/api_stability.md): additive changes to the
Stable surface bump the minor version, breaking changes bump the major
version.

## [Unreleased]

### Added
- **Embedded Semantic Knowledge Graph (ESKG)**: entities and relationships
  are extracted alongside facts by the same background LLM completion
  (`extraction.py`) — no second model call, no NER library — and stored as
  plain SQLite tables (`graph_nodes`/`graph_edges`) in the same store file.
  Multi-hop expansion is a `WITH RECURSIVE` SQL query, gated by the Memory
  Governor (`MemoryProfile.graph_hops`: LITE=0, STANDARD=1, FULL=2 — off
  entirely at LITE, zero extra reads or writes). Folded into `recall()` as
  a confidence-weighted, query-relative additive score nudge (never a
  fixed constant, never able to override real FTS/vector relevance) — see
  [docs/governor.md](docs/governor.md#knowledge-graph).
- **Graph maintenance**: write-time dedup plus row-count caps
  (`graph_node_cap`/`graph_edge_cap`) bound growth; a background
  consolidation sweep additionally decays and hard-deletes low-confidence
  nodes/edges (`graph_decay_half_life_days`/`graph_archive_threshold`) and
  offers the LLM a yes/no duplicate-entity merge review (FULL tier only,
  same gate as the existing fact contradiction-merge).
- **Semantic clustering**: `Elastimem.clusters()` — entities group into
  topics via connected components over `graph_edges` (union-find, no
  clustering library), optionally given a short LLM-generated label (e.g.
  "Local AI"). Computed during the same consolidation sweep as graph
  maintenance.
- **`explain(query, k=5) -> ExplainResult`** *(Experimental)*: retrieval
  transparency — the same ranking `recall()` produces, plus every per-leg
  score (FTS, vector, graph, importance, recency) and the graph traversal
  path that produced it.
- **`timeline(query) -> TimelineResult`** *(Experimental)*: resolves a
  natural-language question ("what did I do before AI?") to a fact key —
  by exact key match, or by falling back to the existing fact search over
  stored values — and returns its full version history. No new storage;
  built entirely on the fact-versioning that has existed since day one.
- **`build_context()` gains a `graph_context` section** ("RELATED TOPICS"):
  entities connected to the query, grouped by cluster label. Purely
  additive — a new `ContextPlan.sections` key, budgeted from a fixed share
  of the existing episodic budget rather than a new `Budgets` field, so
  the Stable `memory_split` contract and every existing section's
  behavior are unchanged.
- `MemoryProfile.graph_hops`, `ElastimemConfig.graph_node_cap`/
  `graph_edge_cap`/`graph_decay_half_life_days`/`graph_archive_threshold`/
  `graph_merge_review_window_days`.
- Storage schema bumped to v3 (`graph_nodes`/`graph_edges` tables in v2;
  `cluster_id`/`cluster_label` columns added in v3). Existing v1/v2 stores
  migrate automatically and losslessly on open.

### Fixed
- `":memory:"` stores silently dropped every write made by the background
  worker thread (LLM extraction, embedding, consolidation) — `sqlite3`
  gives each new connection to `":memory:"` its own separate, empty
  database, and `Elastimem._conn` opened a fresh connection per thread
  unconditionally, so the worker thread never saw the main thread's
  schema or data. `":memory:"` stores now share a single connection
  across every thread that touches them, matching what the code's own
  (previously unimplemented) docstring already claimed.

### Documentation
- `docs/schema.md`, `docs/governor.md`, `docs/api.md`,
  `docs/api_stability.md`, `docs/architecture.md`, and `README.md` updated
  for the knowledge graph, maintenance, clustering, `explain()`,
  `timeline()`, and the new `build_context()` section.

## [0.1.0a1]

Elastimem is in **alpha**. The core API (`open`/`remember`/`recall`/
`record_turn`/`build_context` — see
[docs/api_stability.md](docs/api_stability.md) for the exact Stable
surface) is expected to remain stable across alpha iterations, but internal
implementation details and advanced features may still evolve based on
feedback from early adopters. Not published to PyPI — install directly
from this repository, see [docs/installation.md](docs/installation.md).

### Added
- Core memory store (`Elastimem` / `elastimem.open()`) over a single SQLite
  file: working, episodic, semantic, and procedural memory layers.
- Memory Governor: RAM-tier classification (LITE/STANDARD/FULL), elastic
  token budgeting, pressure reporting, tier-change callbacks.
- Zero required dependencies; optional extras for richer RAM probing
  (`system`, via `psutil`) and a built-in zero-setup semantic embedder
  (`embed`, via `fastembed`, `BAAI/bge-small-en-v1.5`).
- Background worker with a foreground-wins LLM lock (`foreground()` /
  `foreground_begin()`/`foreground_end()`) so host and background LLM calls
  never race on a single local model instance.
- Temporally-versioned facts (`remember`/`recall`/`forget`/`fact_history`),
  regex rule capture, LLM-based fact extraction with validation guards,
  quarantine for rejected extractions.
- Hybrid episodic retrieval (FTS5 BM25 + cosine similarity, reciprocal-rank
  fused) with a `LIKE`-based fallback when FTS5 isn't available.
- `reconfigure()` for updating budgets immediately after a config change
  (e.g. switching the underlying model's context size).
- Full documentation set (`docs/`): architecture, governor spec, schema,
  integration guides, public API reference, API stability boundaries.
- Runnable examples: a zero-dependency minimal bot, a llama.cpp integration,
  an OpenAI-compatible HTTP integration, and a memory-only (no chat loop)
  demo.
- `Elastimem.embed_fn`, `embed_query_fn`, `tokenizer_fn`, and `path` are
  enforced as construction-time-only: reassigning them after construction
  raises `AttributeError` instead of silently racing the background worker,
  which reads them without synchronization.
- CI (`.github/workflows/ci.yml`): pytest across Python 3.10–3.13 on Linux
  and macOS, plus a separate job exercising all optional extras.
- `CONTRIBUTING.md`.

### Documentation
- Clarified that `Elastimem(path, ...)` construction itself can raise (bad
  path, unknown config key) — the "never raises into the host" guarantee
  applies to `build_context`/`record_turn`/`recall` after construction
  succeeds, not to construction itself.
- Documented that `governor.py`'s stdlib RAM-probing fallback has no
  Windows implementation and silently assumes 8 GiB total / 4 GiB
  available on Windows (or any platform besides Linux/macOS) unless the
  `system` extra (`psutil`) is installed. Added matching `Operating System`
  classifiers to `pyproject.toml` (Linux/macOS only — Windows omitted
  deliberately).
- The `vec` optional extra (`elastimem[vec]`) is documented as
  reserved/not-yet-implemented everywhere it's mentioned (README,
  installation.md, schema.md, embeddings.py) — `sqlite-vec` is declared but
  never imported anywhere in the codebase; brute-force cosine runs
  unconditionally today regardless of whether the extra is installed.
- `examples/*.py` and `benchmarks/bench_recall.py` no longer use
  `sys.path.insert(0, "src")` — that only worked from an editable checkout
  and silently did nothing useful for a `pip install`ed user copying an
  example into their own project.

[Unreleased]: https://github.com/CodebyKumar/elastimem/compare/v0.1.0a1...HEAD
[0.1.0a1]: https://github.com/CodebyKumar/elastimem/tree/v0.1.0a1
