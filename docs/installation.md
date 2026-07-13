# Installation

## Requirements

- Python 3.10, 3.11, 3.12, or 3.13.
- No required third-party dependencies — the base install is stdlib-only.
- Linux and macOS are actively supported, including the stdlib (no-`psutil`)
  hardware-probing fallback. **Windows has no stdlib RAM probe** — without
  the `system` extra (`psutil`), tier classification on Windows falls back
  to a fixed 8/4 GiB assumption rather than a real measurement; see
  [governor.md](governor.md#inputs) for details. Installing `elastimem[system]`
  is effectively required for correct tier behavior on Windows.

## Base install

Elastimem is not published on PyPI — install directly from GitHub:

```bash
pip install git+https://github.com/CodebyKumar/elastimem.git
```

This gives you the full public API (`open`, `remember`, `recall`,
`record_turn`, `build_context`, sessions, lessons, the governor) with:

- SQLite storage (stdlib `sqlite3`).
- FTS5 keyword search when the local SQLite build supports it (most do);
  falls back to a `LIKE`-based scan otherwise — check `mem.stats()["fts_enabled"]`
  to see which mode is active, and see the
  [degradation matrix](governor.md#degradation-matrix) for the full fallback
  chain.
- Hardware probing via `/proc/meminfo` (Linux) or `vm_stat`/`sysctl` (macOS)
  when `psutil` isn't installed.
- A built-in semantic embedder that activates automatically once the
  `embed` extra (below) is installed — no code changes required.

## Optional extras

Extras work the same way through a git install — reference them with the
`@ git+...` form:

```bash
pip install "elastimem[system] @ git+https://github.com/CodebyKumar/elastimem.git"   # + psutil, richer hardware probing
pip install "elastimem[vec] @ git+https://github.com/CodebyKumar/elastimem.git"      # reserved for a future accelerated vector index — no effect yet
pip install "elastimem[embed] @ git+https://github.com/CodebyKumar/elastimem.git"    # + fastembed, activates the built-in embedder
```

You can combine them:

```bash
pip install "elastimem[system,vec,embed] @ git+https://github.com/CodebyKumar/elastimem.git"
```

If you've already cloned the repo, the same extras work with a local
editable install too — see [Editable / development install](#editable--development-install)
below:

```bash
pip install -e ".[system,vec,embed]"
```

None of these change how you call the API — they only change what runs
underneath `recall()`/`build_context()`. Elastimem degrades gracefully
without any of them; see [governor.md](governor.md) for exactly what each
extra unlocks and what happens when it's absent.

| Extra | Adds | Without it |
|---|---|---|
| `system` | `psutil`-based RAM probing | stdlib-only probing (still works, slightly less precise) |
| `vec` | nothing yet — reserved for a future `sqlite-vec`-backed index, not wired in | brute-force cosine similarity in Python (fine at typical scales) — this is what runs today regardless of whether `vec` is installed |
| `embed` | `fastembed`, activates the built-in embedder | FTS5/keyword-only retrieval, unless you supply your own `embedder=` |

## Bringing your own LLM / embedder

Elastimem never loads a model or calls an API itself — you inject plain
callables. Nothing else to install for this; whatever library you use to
talk to your model (`llama-cpp-python`, `requests` for an HTTP API, an
official SDK, etc.) is your project's dependency, not Elastimem's. See
`examples/` for working integrations with llama.cpp and an OpenAI-compatible
HTTP endpoint.

## Verifying the install

```bash
python -c "import elastimem; print(elastimem.__version__)"
python -c "import elastimem; help(elastimem.open)"
```

## Editable / development install

```bash
git clone https://github.com/CodebyKumar/elastimem
cd elastimem
pip install -e .
# or, with uv:
uv sync
uv run pytest
```
