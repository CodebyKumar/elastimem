# Integration guide

## Install

Python 3.10-3.13, no required dependencies. Not on PyPI; install from
GitHub:

```bash
pip install "elastimem @ git+https://github.com/CodebyKumar/elastimem.git@v0.3.0"
pip install "elastimem[embed] @ git+https://github.com/CodebyKumar/elastimem.git@v0.3.0"   # + built-in embedder
pip install "elastimem[system] @ git+https://github.com/CodebyKumar/elastimem.git@v0.3.0"  # + psutil
```

- `embed` adds fastembed and the BAAI/bge-small-en-v1.5 model (about 130 MB
  on disk, downloaded on first use; about 360 MB of RAM once loaded).
  Without it, search uses keywords only unless you pass an `embedder`.
- `system` adds psutil: battery detection everywhere, and memory readings
  on Windows, which otherwise assumes a mid-size machine (STANDARD).

For development: `git clone ... && cd elastimem && uv sync` (or `pip
install -e .`), then `uv run pytest`.

## The loop

```python
import elastimem

mem = elastimem.open("~/.myapp/memory.db", llm=llm, context_tokens=8192,
                     reserved_tokens=1500)

def turn(user: str) -> str:
    context = mem.context(user)
    history[:] = history[-2 * context.keep_turns:]    # older turns are in context.summary
    with mem.pause():                                  # only if you share one model with Elastimem
        reply = model(system=f"{PERSONA}\n\n{context}", messages=history + [user])
    mem.record(user, reply)
    return reply

mem.close()                                            # on exit
```

That is the whole required surface. What to pass depends on the host:

| Host | `llm` | `embedder` | `pause()` |
|---|---|---|---|
| No model at all | omit | default (`"auto"`) | no |
| Local model, one instance (llama.cpp) | your completion function | your embedding model, or `"auto"` | **yes**, around every generation |
| API model (OpenAI-compatible) | your completion function | `"auto"` or the API's embeddings | no |
| Must never download anything | as above | `None`, or your own function | as above |

`llm` has the shape `(prompt, *, max_tokens, temperature) -> str`;
`embedder` is `(list[str]) -> list[list[float]]`.

## Settings that matter

- **`context_tokens`**: your model's real context window. The default
  (4096) starves memory on larger models.
- **`reserved_tokens`**: what you need for yourself: your system prompt,
  tool output, and the reply. Memory budgets the rest.
- **`blocked`**: fact keys your app owns (`{"agent_name", "model"}`), so
  extraction never overwrites them.

After switching models, call `mem.configure(context_tokens=...,
reserved_tokens=...)`. It also re-reads the hardware, so call it after a
large model finishes loading.

## Memory as tools

Two tools give the host's own model direct access to memory. They are the
recommended way to store facts when there is no background `llm`, and they
reach the full history rather than only what fits in the prompt.

```python
TOOLS = [
    {"name": "remember", "description": "Store a fact about the user.",
     "parameters": {"type": "object", "required": ["key", "value"], "properties": {
         "key": {"type": "string", "description": "snake_case name, e.g. home_city"},
         "value": {"type": "string"}}}},
    {"name": "memory_search", "description": "Search past conversations and known facts.",
     "parameters": {"type": "object", "required": ["query"], "properties": {
         "query": {"type": "string"}}}},
]

def call_tool(name, args):
    if name == "remember":
        return "stored" if mem.remember(args["key"], args["value"]) else "already known"
    hits = mem.search(args["query"])
    return "\n".join(f"- [{h.date}] {h.text}" for h in hits) or "nothing found"
```

## Signals from the host

- `report_pressure()`: call on a confirmed out-of-memory or failed decode,
  not on every exception. It steps down one tier; repeats within 30 seconds
  count once.
- `on_tier=callback`: notified on every tier change, e.g. to unload an
  embedding model.
- `mem.status.reasons`: why vectors or LLM work are on or off right now.
  Show it in a debug panel.

## Checklist

- [ ] `context_tokens` and `reserved_tokens` set from the real model and prompt.
- [ ] `blocked` lists the app's own identity keys.
- [ ] `pause()` around generation when the model instance is shared.
- [ ] `record()` after every exchange.
- [ ] The host keeps only `context.keep_turns` turns and shows `context.summary`
      (`str(context)` includes it).
- [ ] `remember` and `memory_search` exposed as tools.
- [ ] `close()` on exit; `flush()` before unloading the model.
- [ ] `configure(...)` after a model switch.

## Mistakes to avoid

- Calling `flush()`, `end_session()` or `close()` inside a `pause()`: they
  return, but LLM work (extraction, summaries) is skipped and stays pending
  until a later flush outside the pause.
- Assuming `embedder=None` means "the default": it means keyword search
  only. The default is `"auto"`.
- Changing `mem.config` fields: it is a copy. Use `configure()`.
- Expecting facts without an `llm` and without the `remember` tool: only
  explicit `remember()` calls create facts.

## Examples

| File | Shows |
|---|---|
| `examples/minimal_bot.py` | No model: storing, searching, explicit facts |
| `examples/memory_search.py` | The memory API without a chat loop, including `explain()` |
| `examples/llama_cpp_bot.py` | A local model shared with Elastimem: `pause()`, `keep_turns`, `report_pressure()` |
| `examples/openai_agent.py` | An API model through a plain HTTP call |
