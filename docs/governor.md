# The Memory Governor — normative spec

The governor makes Engram *elastic*: it measures what the machine can afford
and sizes every memory capability accordingly, continuously. This document is
the specification; `governor.py` implements it and `tests/test_governor.py` +
`tests/test_degradation.py` enforce it.

## Inputs

| Input | Source | When |
|---|---|---|
| RAM total / available | `psutil` if installed, else `/proc/meminfo` (Linux) or `sysctl` + `vm_stat` (macOS); unknown platforms assume 8/4 GiB | startup + every `tick()` |
| `context_tokens` | `EngramConfig` (host passes its model's `n_ctx`) | construction |
| `static_prompt_tokens` | `EngramConfig` (host measures its fixed prompt) | construction |
| Tier override | `ENGRAM_TIER=lite\|standard\|full` or `EngramConfig.tier_override` | construction (pins the tier) |
| Pressure signal | host calls `report_pressure()` on OOM / decode failure | runtime |

## Tier classification

```
FULL      total ≥ 16 GiB  and  available ≥ 6 GiB
STANDARD  total ≥ 8 GiB   and  available ≥ 2.5 GiB
LITE      otherwise
```

Movement rules:
- **Downgrade immediately** when available RAM < 1.2 GiB, when measured
  classification drops, or on `report_pressure()` (one tier per report).
- **Upgrade cautiously**: only after `upgrade_healthy_ticks` (default 10)
  consecutive healthy ticks, one tier at a time, never above the startup
  tier.
- `on_tier_change(old, new)` fires so the host can react (e.g. unload its
  embedding model when dropping to LITE).

## Budget derivation

```
dynamic = context_tokens − output_reserve(512) − tool_reserve(600) − static_prompt_tokens
          (floor: 256)

working  = 55% of dynamic
memory   = 45% of dynamic, split:
             facts 40% · episodic 30% · sessions 15% · lessons 15%
```

LITE additionally zeroes the episodic share and halves sessions, returning
the freed tokens to the working window — on a starved machine, immediate
coherence beats recall (which remains available through `recall()`).

Worked example (defaults, `n_ctx=4096`, static prompt 1200 tokens):
dynamic = 1784 → working ≈ 981, facts ≈ 321, episodic ≈ 240, sessions ≈ 120,
lessons ≈ 120.

Token estimation uses chars/4 unless the host passes `tokenizer_fn`.

## Capability × tier

| Capability | FULL | STANDARD | LITE |
|---|---|---|---|
| `embed_fn` called | yes | yes | **never** |
| Episodic injection (`build_context`) | top 4 | top 3 | none (`recall()` only) |
| LLM fact extraction | background, per turn | batched every 3 turns | off |
| Rolling summary | LLM | LLM | marker line |
| Consolidation | full (incl. LLM merge), idle + exit | dedupe + decay, exit | off |
| Rule capture | always | always | always |
| Transcript persistence | always | always | always |
| `remember` / `recall` / `forget` | always | always | always |

## Degradation matrix

Every capability has a defined floor. **Nothing in this table raises to the
host.**

| Capability | Fallback 1 | Floor |
|---|---|---|
| Vector search | FTS5 BM25 only | `LIKE` term matching (no FTS5 in sqlite build) |
| `embed_fn` raises | vector leg disabled for the session (logged once), FTS5-only | — |
| `complete_fn` absent or raises | regex rule capture | explicit `remember` only |
| Rolling summary | `[k earlier turn(s) omitted — memory search can recall them]` | plain eviction |
| Session summary | — | title = first user message (80 chars) |
| Corrupt DB file | renamed `<path>.corrupt-<ts>`, fresh store created, warning logged | — |
| RAM probe fails | assume 8 GiB total / 4 GiB available (STANDARD-ish) | — |
| `recall()` / `build_context()` / `record_turn()` internal error | logged, empty result / empty section / skipped write | never raises |

## Why this design

Cloud memory systems can assume the model is far away and infinitely
patient. A local agent shares one machine between the model, the memory
system, and the user's actual work. The governor exists so that memory is
always the *first* thing to shrink under pressure and the *last* thing to
crash: dropping episodic injection frees context and cycles; dropping
extraction frees the model; nothing the user said is ever lost, because raw
transcripts persist in every tier and can be re-indexed when capacity
returns.
