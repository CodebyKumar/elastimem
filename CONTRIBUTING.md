# Contributing to Elastimem

Thanks for considering a contribution. Elastimem is a small, deliberately
minimal library — the bar for adding anything is "does this earn its
complexity," not "does this exist in other memory frameworks." Read
[docs/architecture.md](docs/architecture.md) and
[docs/governor.md](docs/governor.md) first; both document not just what the
code does but *why*, including several design decisions that replaced an
earlier, buggier approach. If you're about to touch retrieval ranking, the
governor's budget math, or the worker's locking, those docs will save you
from re-introducing a bug that was already found and fixed once.

## Development setup

```bash
git clone https://github.com/CodebyKumar/elastimem
cd elastimem
uv sync --all-extras   # or: pip install -e ".[system,vec,embed]" pytest
```

The base library has zero required dependencies; `--all-extras` pulls in
`psutil`, `sqlite-vec` (declared but not yet wired in, see below), and
`fastembed` so you can exercise every code path locally.

## Running tests

```bash
uv run pytest              # fast suite, ~3s, excludes the `slow` marker
uv run pytest -m slow      # includes the built-in embedder's real model download
uv run pytest -m ""        # everything
```

Tests must pass clean before a PR is opened. If you're changing behavior
covered by `tests/test_degradation.py` or `tests/test_governor.py`, run
those explicitly and read the assertions — they encode specific, previously
observed failure modes (see governor.md's "Known limitations"), not
incidental coverage.

## Before opening a PR

- **New capability?** Ask first (open an issue) if it's not an obvious bug
  fix. Elastimem's design principle is "does this need to exist," not
  feature parity with other memory frameworks — see the project's own
  publication-readiness notes in `docs/` for the reasoning.
- **Touching the governor, retrieval ranking, or the worker's locking?**
  Read the relevant "why" section in `docs/governor.md` /
  `docs/architecture.md` first. Several current behaviors (additive not
  multiplicative importance/recency nudges, the foreground-wins real lock
  instead of an advisory flag, the two separate query-length gates) replaced
  an earlier design that had a real bug — the docs explain what broke and
  why the current shape is correct. Don't revert to the old shape without
  reading why it changed.
- **Changing the public API surface?** Check
  [docs/api_stability.md](docs/api_stability.md) for what's Stable
  (semver-protected) vs. Experimental/Internal. A breaking change to the
  Stable surface needs a major-version bump and a `CHANGELOG.md` entry.
- **Changing the schema?** `db.py` has a `SCHEMA_VERSION` + `_migrate()`
  mechanism (see [docs/schema.md](docs/schema.md)) — add a migration step
  there, don't just alter the `CREATE TABLE` statements in place.
- **Update `CHANGELOG.md`** under `[Unreleased]` for anything
  user-observable: new features, behavior changes, fixes, deprecations.
- **Update relevant docs in the same PR.** Elastimem's docs describe actual
  behavior, not aspiration — a PR that changes behavior without touching
  the doc that describes it will be asked to include the doc update.

## Code style

- No enforced formatter/linter is currently wired into CI — match the
  surrounding code's style (stdlib-only, minimal abstraction, comments that
  explain *why* a non-obvious choice was made rather than *what* the code
  does).
- Prefer extending an existing module over adding a new one. The module
  boundaries (`governor` / `guards` / `episodic` / `semantic` / `procedural`
  / `retrieval` / `extraction` / `assembly` / `worker` / `store`) are
  intentional and each owns one responsibility — new code almost always
  belongs inside one of them rather than beside them.

## Reporting bugs / requesting features

Open a GitHub issue. For a bug report, the most useful things to include
are: the `mem.profile.tier` at the time (LITE-tier "memory doesn't seem to
work" is very often working as designed, not a bug — see governor.md), a
minimal reproduction, and whether you're on the base install or with
specific extras.

## License

By contributing, you agree your contributions are licensed under the
project's [MIT License](LICENSE).
