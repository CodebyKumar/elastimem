# Contributing to Elastimem

Thanks for considering a contribution. Elastimem is a small, deliberately
minimal library — the bar for adding anything is "does this earn its
complexity," not "does this exist in other memory frameworks." Read
[docs/architecture.md](docs/architecture.md) and
[docs/redesign.md](docs/redesign.md) first; they document not just what the
code does but *why*, with the measurements behind each decision. If you're
about to touch ranking, the governor or the worker's locking, they will save
you from re-introducing a problem that was already found and fixed once.

## Development setup

```bash
git clone https://github.com/CodebyKumar/elastimem
cd elastimem
uv sync --all-extras   # or: pip install -e ".[system,embed]" pytest
```

The base library has zero required dependencies; `--all-extras` pulls in
`psutil` and `fastembed` so you can exercise every code path locally.

## Running tests

```bash
uv run pytest              # fast suite, ~3s, excludes the `slow` marker
uv run pytest -m slow      # includes the built-in embedder's real model download
uv run pytest -m ""        # everything
```

Tests must pass clean before a PR is opened. For ranking or retrieval
changes, also run `python benchmarks/eval_recall.py` and compare with the
results in `docs/redesign.md`.

## Before opening a PR

- **New capability?** Ask first (open an issue) if it's not an obvious bug
  fix. The design principle is "does this need to exist", not feature
  parity with other memory frameworks.
- **Touching ranking, the governor or the worker's locking?** Read
  `docs/architecture.md` and the matching section of `docs/redesign.md`
  first. Tune values in `tiers.py` against `benchmarks/eval_recall.py`, not
  by feel.
- **Changing the public API surface?** Check
  [docs/api_stability.md](docs/api_stability.md) for what's Stable
  vs. internal. A breaking change to the Stable surface needs a version bump
  as described there and a `CHANGELOG.md` entry.
- **Changing the schema?** `storage/db.py` has a `SCHEMA_VERSION` + `_migrate()`
  mechanism (see [docs/schema.md](docs/schema.md); fixtures in `tests/fixtures/`) — add a migration step
  there, don't just alter the `CREATE TABLE` statements in place.
- **Update `CHANGELOG.md`** under `[Unreleased]` for anything
  user-observable: new features, behavior changes, fixes, deprecations.
- **Update relevant docs in the same PR.** Elastimem's docs describe actual
  behavior, not aspiration — a PR that changes behavior without touching
  the doc that describes it will be asked to include the doc update.
- **Update
  [docs/agent_integration_prompt.md](docs/agent_integration_prompt.md) for
  ANY user-visible change.** This one is easy to forget because it is not
  the doc you were editing. It is the single artifact downstream coding
  agents are handed to discover what Elastimem can do, so when it drifts,
  every integration built from it silently misses the feature you just
  added. Its final section lists exactly which part to update for which
  kind of change. New public method or signature, new setting, changed
  per-tier behavior, or a version bump: all of them require an edit there.

## Code style

- No enforced formatter/linter is currently wired into CI — match the
  surrounding code's style (stdlib-only, minimal abstraction, comments that
  explain *why* a non-obvious choice was made rather than *what* the code
  does).
- Prefer extending an existing module over adding a new one. Each module
  owns one job (see the module table in `docs/architecture.md`); new code
  almost always belongs inside one of them.
- No word lists. Checks must be structural or data-driven (`docs/redesign.md`
  section 9).

## Reporting bugs / requesting features

Open a GitHub issue. For a bug report, the most useful things to include
are: `mem.status` at the time (its `reasons` say why vectors or LLM work are
off; a LITE tier is often working as designed), a
minimal reproduction, and whether you're on the base install or with
specific extras.

## License

By contributing, you agree your contributions are licensed under the
project's [MIT License](LICENSE).
