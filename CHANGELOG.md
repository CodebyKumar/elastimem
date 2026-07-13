# Changelog

All notable changes to Elastimem are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versioning follows
[PEP 440](https://peps.python.org/pep-0440/) pre-release identifiers while
in alpha (`0.1.0a1`, `0.1.0a2`, ... → `0.1.0b1` → `0.1.0`), then the policy
in [docs/api_stability.md](docs/api_stability.md): additive changes to the
Stable surface bump the minor version, breaking changes bump the major
version.

## [Unreleased]

Nothing yet.

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
