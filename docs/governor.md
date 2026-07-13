# The Memory Governor — normative spec

The governor makes Elastimem *elastic*: it measures what the machine can afford
and sizes every memory capability accordingly, continuously. This document is
the specification; `governor.py` implements it and `tests/test_governor.py` +
`tests/test_degradation.py` enforce it.

## Inputs

| Input | Source | When |
|---|---|---|
| RAM total / available | `psutil` if installed, else `/proc/meminfo` (Linux) or `sysctl` + `vm_stat` (macOS); **Windows and any other platform have no stdlib probe and assume 8/4 GiB** (a guess, not a measurement) — install the `system` extra (`psutil`) for accurate probing on Windows | startup + every `tick()` |
| `context_tokens` | `ElastimemConfig` (host passes its model's `n_ctx`) | construction |
| `static_prompt_tokens` | `ElastimemConfig` (host measures its fixed prompt) | construction |
| Tier override | `ELASTIMEM_TIER=lite\|standard\|full` or `ElastimemConfig.tier_override` | construction (pins the tier) |
| Pressure signal | host calls `report_pressure()` on OOM / decode failure | runtime, rate-limited (see below) |

## Tier classification

```
FULL      total ≥ 16 GiB  and  available ≥ 6 GiB
STANDARD  total ≥ 8 GiB   and  available ≥ 2.5 GiB
LITE      otherwise
```

Movement rules:
- **Downgrade immediately** when available RAM < 1.2 GiB, or when measured
  classification drops (checked on every `tick()`, so at most one turn of
  lag behind reality).
- **`report_pressure()` downgrades one tier immediately on the FIRST call**,
  then enters a 30-second cooldown (`_PRESSURE_REPORT_COOLDOWN_SECONDS`
  in `governor.py`): further calls within that window are coalesced into
  the same downgrade instead of dropping the tier again. This exists
  because a single bad turn can plausibly produce several pressure reports
  in quick succession (a decode-retry loop, or the same underlying failure
  surfacing through more than one call site in the host) — without the
  cooldown, one real event could cascade the tier down multiple levels and
  need far more healthy ticks than the event actually warranted to recover
  from. The cooldown does **not** weaken "downgrades are immediate": the
  first report in any burst still drops the tier on the spot, exactly as
  before. A genuinely new, separate pressure event after the cooldown
  elapses downgrades again, normally.
- **Upgrade cautiously**: only after `upgrade_healthy_ticks` (default 10)
  consecutive healthy ticks, one tier at a time, never above the startup
  tier (a host that started at STANDARD can recover to STANDARD, never to
  FULL, even if RAM later becomes abundant — the assumption is that
  whatever else the process is doing at that RAM footprint hasn't gone
  away just because a temporary dip cleared).
- `on_tier_change(old, new)` fires so the host can react (e.g. unload its
  embedding model when dropping to LITE). **Elastimem's own built-in
  embedder (see below) does NOT need this** — it already checks
  `profile.embeddings_enabled` on every call and simply isn't invoked when
  the tier says no; there is nothing for the host to unload since the
  built-in model, once loaded, just sits idle rather than holding a lock
  or consuming CPU.

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

**The 256-token floor is real and gets hit in practice, not just a
theoretical edge case.** A small local model (`n_ctx=4096`) whose host has a
sizeable fixed system prompt (persona + a large tool catalog can easily run
1500-2600+ tokens) leaves very little `dynamic` budget once
`output_reserve` (512) and `tool_reserve` (600) are subtracted — e.g.
`4096 - 512 - 600 - 2584 = 400`, barely above the floor. At that point
`working≈220`, and the ENTIRE memory pool (facts + episodic + sessions +
lessons combined) is only ~180 tokens, meaning each individual section gets
single-digit-to-low-double-digit token budgets — `fit_lines()` then keeps
0-1 lines per section, which reads to a user as "memory barely knows
anything" even though every other part of the pipeline (extraction,
storage, retrieval) is working correctly. **This is not a bug to patch
inside the governor** — the floor exists precisely to prevent a negative or
zero budget from crashing budget math, and it's doing its job. If you hit
this, the fix is on the host side: either increase the model's `n_ctx`
(the actual fix Tuffy applied — see its own model config), or reduce
`static_prompt_tokens` (a shorter persona / fewer always-on tool
descriptions), or both. There is no governor-side knob that manufactures
context tokens that don't exist.

Token estimation uses chars/4 unless the host passes `tokenizer_fn`. This
proxy is a real approximation, not exact — for non-English text or
code-heavy content, actual tokenization can diverge meaningfully from
chars/4, which combined with a tight budget (see above) can make
`fit_lines()` either truncate more aggressively than necessary or, in the
opposite direction, let slightly more text through than the model's real
tokenizer would allow. Pass `tokenizer_fn` if this matters for your use
case — it costs nothing extra elsewhere in the pipeline.

### Two separate query-length gates, not one

Retrieval and LLM-based extraction are gated by **different** thresholds,
because they have very different costs:

| Gate | Config field | Default | Applies to | Why |
|---|---|---|---|---|
| Extraction gate | `min_query_words` | 4 | LLM fact-extraction job (`store.py`'s `_after_record_turn`) | A real model call has real latency/cost — not worth it for "ok" or "thanks". |
| Retrieval gate | `min_retrieval_query_words` | 1 | FTS5/vector search inside `build_context()` | Local, free, no LLM involved — excluding anything beyond genuinely empty input (a single word still carries real query intent, e.g. "birthday?" or a one-word follow-up) would silently weaken recall on exactly the kind of short question that's common in real conversation. |

These used to share one threshold (`min_query_words=4` gating both), which
meant a short-but-real question like "my birthday?" (2 words) got **zero**
episodic/fact retrieval just from being under 4 words — indistinguishable
from "memory isn't working" from the outside, with nothing logged (retrieval
never raises, by design). If you're tuning either value, remember they now
move independently — raising `min_query_words` to save LLM calls does not
also raise the bar for local retrieval, and vice versa.

## Capability × tier

| Capability | FULL | STANDARD | LITE |
|---|---|---|---|
| `embedder` called (host-supplied OR built-in — see below) | yes | yes | **never** |
| Episodic injection (`build_context`) | top 4 | top 3 | none (`recall()` only) |
| LLM fact extraction | background, per turn | batched every 3 turns | off |
| Rolling summary | LLM | LLM | marker line |
| Consolidation | full (incl. LLM merge), idle + exit | dedupe + decay, exit | off |
| Rule capture | always | always | always |
| Transcript persistence | always | always | always |
| `remember` / `recall` / `forget` | always | always | always |

**This table is the single most important thing to internalize about
Elastimem's tier system: LITE is not a slightly-degraded mode, it is
"local search + explicit facts only."** No LLM call is ever attempted at
LITE — not extraction, not rolling summaries, not consolidation merges —
and no embedder is ever called either, even if one is configured and
working. A host running on a RAM-constrained machine (common for local-model
deployments: the model itself, plus an embedding model, plus the app,
competing for RAM on an 8GB machine can easily land at LITE even though
each piece individually would run fine alone) will, correctly and silently,
get:
- Rule-based fact capture only (`rules.py`'s ~10 regexes) — genuinely
  hardcoded, will never learn a new phrasing on its own. See "Known
  limitations" below.
- FTS5 keyword search only for `recall()`/`memory_search` — no semantic/
  paraphrase matching, regardless of whether the built-in or a host
  embedder is configured.
- No background summarization — sessions get a title-only floor (first 80
  chars of the first user message), no LLM-condensed 1-2 sentence summary.
- No consolidation — contradictory fact updates (e.g. "I live in Austin"
  then weeks later "I live in Denver") are versioned correctly (the old
  value is never lost, see `schema.md`) but never LLM-merged into a single
  coherent value; the host sees only the latest version.

None of this is a bug. It is the intended floor: a host on a genuinely
constrained machine gets a working memory system with hard guarantees
(nothing crashes, nothing is silently lost — raw transcripts persist in
every tier) rather than a broken one. But it means **"memory doesn't seem
to be learning things"** on a real deployment is very often actually
**"this process has been running at LITE tier the whole time"** — check
`mem.profile.tier` (or the host's own `/memory` status command, if it
surfaces one) before assuming a capture/retrieval bug. The tier can only be
confirmed above LITE if the RAM thresholds in "Tier classification" above
are actually met at the moment of measurement.

## Built-in embedder (auto-activates — read this before assuming embeddings are off)

As of the `embed` optional extra, **Elastimem embeds by default.** If the
host constructs `Elastimem(...)`/`elastimem.open(...)` without passing
`embedder=`/`embed_fn=` (i.e. that argument is `None` after resolving both
spellings), the store activates its own built-in embedder
(`default_embedder.py`) automatically — no host code required.

- **Model**: `BAAI/bge-small-en-v1.5` via `fastembed` (ONNX Runtime, no
  torch dependency) — a real MTEB-benchmarked retrieval model, not a
  hashing trick or bag-of-words approximation.
- **Lazy, always**: nothing is imported, downloaded, or loaded at
  `Elastimem()` construction time — only on the first actual embed call
  (which only happens via the background worker's `"embed"` job, itself
  gated on `profile.embeddings_enabled`, i.e. never at LITE tier). This
  means construction never blocks on network access, and a host on a
  machine that starts at LITE tier may never trigger the download at all.
- **First use downloads ~130MB** from Hugging Face Hub, cached under
  `fastembed`'s own platform cache directory afterward (subsequent runs on
  the same machine reuse the cache — no repeat download). This download
  happens on a background worker thread, never the host's main/foreground
  thread, so it cannot block an interactive turn — but it does mean the
  FIRST embed job after a fresh install can take significantly longer
  (tens of seconds) than subsequent ones.
- **Fails silently, always degrades to FTS5**: if the `embed` extra isn't
  installed (`pip install elastimem[embed]`), or the download fails (no
  network, disk full, HF Hub unreachable), `default_embedder` catches the
  failure, logs a warning once, and the store behaves exactly as if no
  embedder were configured at all — FTS5/keyword retrieval only. This is
  the same "vector leg disabled for the session" floor documented below for
  a host-supplied embedder that raises.
- **Asymmetric query/passage encoding**: bge-small (like most modern
  retrieval-tuned embedding models) is trained with different encoding for
  queries vs. the passages being searched — using the wrong one measurably
  hurts retrieval accuracy. The built-in embedder handles this correctly
  via `Store.embed_query_fn` (set automatically alongside `embed_fn` when
  the built-in activates); a **host-supplied** `embedder=` is assumed
  symmetric (true of most embedding APIs, e.g. OpenAI's) and
  `embed_query_fn` stays `None` in that case — retrieval then reuses the
  host's `embed_fn` for both queries and passages, exactly as it always
  has. If your own embedder is also asymmetrically trained, you currently
  have no way to pass a separate query-side encoder through the public
  API — file an issue if you need this; the plumbing exists internally
  (`Elastimem.embed_query_fn`) but isn't yet part of the constructor's
  public parameter list.
- **Opting out entirely**: set `disable_builtin_embedder=True` in
  `ElastimemConfig` (or pass it as a config override). Use this for an
  air-gapped environment that must never attempt the optional extra's
  first-run download, or any host that wants to guarantee FTS5-only
  behavior regardless of what's installed. Passing `embedder=None`
  explicitly is **not** enough on its own to mean "no embedder" — `None`
  is genuinely ambiguous with "the host simply didn't pass this optional
  parameter," which is exactly the case the auto-activation is designed
  for. `disable_builtin_embedder=True` is the unambiguous opt-out.

## Degradation matrix

Every capability has a defined floor. **Nothing in this table raises to the
host.**

| Capability | Fallback 1 | Floor |
|---|---|---|
| Vector search | FTS5 BM25 only | `LIKE` term matching (no FTS5 in sqlite build) |
| `embedder` raises (host-supplied OR built-in) | vector leg disabled for the session (logged once), FTS5-only | — |
| Built-in embedder extra not installed / first download fails | same as "embedder raises" above — FTS5-only, logged once | — |
| `llm` absent or raises | regex rule capture | explicit `remember` only |
| Rolling summary | `[k earlier turn(s) omitted — memory search can recall them]` | plain eviction |
| Session summary | — | title = first user message (80 chars) |
| Corrupt DB file | renamed `<path>.corrupt-<ts>`, fresh store created, warning logged | — |
| RAM probe fails | assume 8 GiB total / 4 GiB available (STANDARD-ish) | — |
| `recall()` / `build_context()` / `record_turn()` internal error | logged, empty result / empty section / skipped write | never raises |

## Known limitations (exhaustive, as of this writing)

This section exists so a future reader — maintainer or downstream host —
doesn't have to rediscover these by hitting them in production. If you fix
one, update this list.

1. **Rule-based fact capture (`rules.py`) is a fixed, hardcoded regex list.**
   It runs at every tier including LITE, is essentially free, and is
   deliberately conservative (high precision over recall) — but it will
   never learn a phrasing it wasn't written for. Known-covered patterns as
   of this writing: `"my name is X"` / `"my <adjective> name is X"` (e.g.
   "my full name is X"), `"call me X"`, `"I live/stay in X"`,
   `"I'm/I am from X"`, `"I work as/at X"`, `"I'm an X by profession/trade"`,
   `"I'm allergic to X"`, `"I'm vegetarian/vegan/pescatarian"`, and
   `"my favorite/preferred <subject> is X"`. Deliberately NOT attempted:
   typo-tolerant matching (e.g. "you can **all** me kumar" for "call me
   kumar") — the false-positive risk of loosening these patterns outweighs
   the recall benefit for a zero-cost regex layer. Real coverage of
   "everything else" is the LLM extraction pass, which is **only available
   at STANDARD/FULL tier** (see the tier table above) — a host running at
   LITE gets rules-only capture, permanently, until the tier recovers.
2. **The chars/4 token-estimation proxy is not exact.** Every budget
   computation in the governor, and every `fit_lines()` truncation decision,
   uses `len(text)//4` unless the host supplies `tokenizer_fn`. This is a
   reasonable average for English prose but can diverge meaningfully for
   non-English text, code, or unusual formatting — worth passing a real
   tokenizer if budget precision matters for your use case.
3. **The 256-token dynamic-budget floor is a real, observed failure mode**
   on small-context local models with a large fixed system prompt — see the
   worked example above. It is not a bug (the floor correctly prevents
   negative/zero budgets from crashing), but it can silently reduce every
   memory section to near-nothing. There is no governor-side fix; the host
   must either grow `context_tokens` (increase the model's real `n_ctx`) or
   shrink `static_prompt_tokens`.
4. **Retrieval ranking is relevance-first with importance/recency as small
   tie-breaking nudges, not equal-weight multipliers** (this was a real bug,
   fixed — see `retrieval.py`'s `search_chunks()` docstring for the full
   history). If you're extending the ranking formula, do not reintroduce a
   multiplicative importance/recency term against a bounded 0-1 relevance
   score — it will let a durable-but-topically-irrelevant chunk outrank a
   genuinely relevant one on real queries, exactly as it used to.
5. **`report_pressure()`'s cooldown coalesces reports, it does not validate
   them.** There is no way for the governor to distinguish a genuine
   hardware OOM from a host mis-classifying some other failure as pressure
   (a real example: a naive host that catches every bare `RuntimeError`
   from its inference library and assumes it means OOM, when some other,
   unrelated exception type could in principle share that class — the fix
   belongs in the host's own error classification, not here). Any call to
   `report_pressure()` is trusted at face value, once per cooldown window.
   If your host's OOM detection is noisy or heuristic, tighten it upstream
   rather than relying on the governor to filter false positives; the
   cooldown only limits blast radius, it doesn't distinguish real from
   spurious.
6. **The built-in embedder is English-tuned** (`bge-small-en-v1.5`).
   Multilingual or non-English-dominant hosts will get materially worse
   semantic retrieval quality from the built-in default than from a
   purpose-chosen multilingual embedding model passed via `embedder=`.
   There is currently no multilingual built-in option — an explicit
   `embedder=` override is the only path to better non-English coverage
   today.
7. **No mid-flight cancellation of a downloading built-in embedder.** If the
   first-use download is interrupted (process killed, network drops
   mid-download), `fastembed`/`huggingface-hub`'s own retry/resume behavior
   governs what happens on the next attempt — Elastimem does not add its
   own retry or partial-download cleanup logic on top.
8. **Two-tier upgrade cap.** Upgrades never exceed the tier measured at
   process startup (`_startup_tier`), even after `upgrade_healthy_ticks`
   consecutive healthy ticks. A process that starts at STANDARD (e.g.
   because RAM was briefly under pressure from something else at launch)
   can recover to STANDARD but will never reach FULL for the rest of that
   process's lifetime, even if RAM later becomes genuinely abundant. Only a
   fresh process restart re-measures the ceiling.
9. **No stdlib RAM probe on Windows.** `probe_ram()`'s fallback path (used
   when `psutil` isn't installed) only implements Linux (`/proc/meminfo`)
   and macOS (`sysctl`/`vm_stat`). On Windows — or any other platform — it
   silently assumes 8 GiB total / 4 GiB available, a guess rather than a
   measurement, which can misclassify the tier in either direction. Install
   the `system` extra (`psutil`) for accurate probing on Windows; this is
   the only platform where that extra is effectively required rather than
   a precision upgrade.

## Why this design

Cloud memory systems can assume the model is far away and infinitely
patient. A local agent shares one machine between the model, the memory
system, and the user's actual work. The governor exists so that memory is
always the *first* thing to shrink under pressure and the *last* thing to
crash: dropping episodic injection frees context and cycles; dropping
extraction frees the model; nothing the user said is ever lost, because raw
transcripts persist in every tier and can be re-indexed when capacity
returns.
