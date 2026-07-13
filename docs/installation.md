# Installation

## Requirements

- Python 3.10, 3.11, 3.12, or 3.13.
- No required third-party dependencies — the base install is stdlib-only.

## Base install

```bash
pip install elastimem
```

This gives you the full public API (`open`, `remember`, `recall`,
`record_turn`, `build_context`, sessions, lessons, the governor) with:

- SQLite storage (stdlib `sqlite3`).
- FTS5 keyword search when the local SQLite build supports it (most do);
  falls back to a `LIKE`-based scan otherwise — see
  [troubleshooting](governor.md) for how to check `mem.stats()["fts_enabled"]`.
- Hardware probing via `/proc/meminfo` (Linux) or `vm_stat`/`sysctl` (macOS)
  when `psutil` isn't installed.
- A built-in semantic embedder that activates automatically once the
  `embed` extra (below) is installed — no code changes required.

## Optional extras

```bash
pip install elastimem[system]   # + psutil, richer hardware probing
pip install elastimem[vec]      # + sqlite-vec, accelerated vector search
pip install elastimem[embed]    # + fastembed, activates the built-in embedder
```

You can combine them:

```bash
pip install "elastimem[system,vec,embed]"
```

None of these change how you call the API — they only change what runs
underneath `recall()`/`build_context()`. Elastimem degrades gracefully
without any of them; see [governor.md](governor.md) for exactly what each
extra unlocks and what happens when it's absent.

| Extra | Adds | Without it |
|---|---|---|
| `system` | `psutil`-based RAM probing | stdlib-only probing (still works, slightly less precise) |
| `vec` | `sqlite-vec` accelerated vector search | brute-force cosine similarity in Python (fine at typical scales) |
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
