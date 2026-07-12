"""The Elastimem store: the one class hosts interact with.

Thread model: public methods may be called from the host's main thread; the
background worker (phase 4) owns deferred writes. All writes acquire the
store-level ``_write_lock``; reads use per-thread connections (WAL-safe).
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from typing import Callable

import contextlib

from . import assembly, episodic, extraction, procedural, rules, semantic
from .assembly import ContextPlan
from .config import ConsolidationLevel, ElastimemConfig, MemoryProfile
from .db import open_store, connect
from .governor import Governor
from .worker import Job, Worker

log = logging.getLogger("elastimem")

CompleteFn = Callable[..., str]
EmbedFn = Callable[[list[str]], list[list[float]]]


def _resolve_config(
    config: ElastimemConfig | None, overrides: dict
) -> ElastimemConfig:
    """Merge an optional external config with inline keyword overrides."""
    import dataclasses

    valid = {f.name for f in dataclasses.fields(ElastimemConfig)}
    unknown = set(overrides) - valid
    if unknown:
        raise TypeError(
            f"unknown config option(s): {', '.join(sorted(unknown))} "
            f"(valid: {', '.join(sorted(valid))})"
        )
    if config is None:
        return ElastimemConfig(**overrides)
    if overrides:
        return dataclasses.replace(config, **overrides)
    return config


class Elastimem:
    """One persistent memory store backed by a single SQLite file.

    Simplest form — everything is optional except the path::

        mem = Elastimem("~/.myagent/memory.db")

    Full form — inject an LLM and an embedder, tune config inline::

        mem = Elastimem(
            "~/.myagent/memory.db",
            llm=my_complete_fn,        # (prompt, *, max_tokens, temperature) -> str
            embedder=my_embed_fn,      # (list[str]) -> list[list[float]]
            context_tokens=4096,       # any ElastimemConfig field works here
            reserved_keys={"model"},
        )

    Config fields may be passed directly as keyword arguments (as above), or
    bundled in an external :class:`ElastimemConfig` via ``config=``; inline
    keywords override the bundle. Elastimem degrades gracefully around any
    capability you don't provide (see docs/governor.md).
    """

    def __init__(
        self,
        path: str,
        *,
        llm: CompleteFn | None = None,
        embedder: EmbedFn | None = None,
        config: ElastimemConfig | None = None,
        tokenizer_fn: Callable[[str], int] | None = None,
        probe_fn: Callable[[], tuple[int, int]] | None = None,
        on_tier_change: Callable | None = None,
        complete_fn: CompleteFn | None = None,   # legacy alias of llm
        embed_fn: EmbedFn | None = None,         # legacy alias of embedder
        **config_overrides,
    ) -> None:
        self.path = os.path.expanduser(path) if path != ":memory:" else path
        self.config = _resolve_config(config, config_overrides)
        self.complete_fn = complete_fn if complete_fn is not None else llm
        self.embed_fn = embed_fn if embed_fn is not None else embedder
        self.tokenizer_fn = tokenizer_fn
        self._write_lock = threading.RLock()
        self._local = threading.local()

        conn, self.fts_enabled = open_store(self.path)
        self._local.conn = conn

        governor_kwargs = {"on_tier_change": on_tier_change}
        if probe_fn is not None:
            governor_kwargs["probe_fn"] = probe_fn
        self.governor = Governor(self.config, **governor_kwargs)

        # Session state (lazy: first record_turn opens one).
        self.session_id: int | None = None
        self._turn = 0
        self._rolling: str | None = None
        with self._write_lock:
            episodic.close_orphan_sessions(conn)

        self._worker = Worker(self)

    # ------------------------------------------------------------------ #
    # connections
    # ------------------------------------------------------------------ #
    @property
    def _conn(self) -> sqlite3.Connection:
        """Per-thread connection (:memory: stores share the creating thread's)."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = connect(self.path)
            self._local.conn = conn
        return conn

    # ------------------------------------------------------------------ #
    # governor / context
    # ------------------------------------------------------------------ #
    @property
    def profile(self) -> MemoryProfile:
        """The governor's current capability/budget snapshot."""
        return self.governor.profile

    def tick(self) -> MemoryProfile:
        """Call once per turn: cheap hardware re-check, may change the tier."""
        return self.governor.tick()

    def report_pressure(self) -> MemoryProfile:
        """Host signal that an OOM/decode failure happened; downgrades a tier."""
        return self.governor.report_pressure()

    def build_context(self, user_input: str = "") -> ContextPlan:
        """Assemble this turn's budgeted memory sections.

        ``user_input`` (when given) makes retrieval query-aware; retrieval
        never raises — on any failure a section is simply empty.
        """
        profile = self.governor.profile
        conn = self._conn

        relevance: dict[int, float] = {}
        episodic_text = ""
        if user_input and len(user_input.split()) >= self.config.min_query_words:
            try:
                from . import retrieval

                relevance = retrieval.fact_relevance(
                    conn, user_input, fts=self.fts_enabled
                )
                if profile.budgets.episodic > 0:
                    episodic_text = retrieval.episodic_section(
                        self, user_input, profile, tokenizer_fn=self.tokenizer_fn
                    )
            except Exception:
                log.exception("elastimem: retrieval failed; sections degraded to empty")

        facts_text, fact_ids = assembly.build_facts_section(
            conn, self.config, profile, relevance, self.tokenizer_fn
        )
        if fact_ids:
            with self._write_lock:
                semantic.touch(conn, fact_ids)

        sections = {
            assembly.SECTION_FACTS: facts_text,
            assembly.SECTION_EPISODIC: episodic_text,
            assembly.SECTION_SESSIONS: self._sessions_section(profile),
            assembly.SECTION_LESSONS: assembly.build_lessons_section(
                conn, self.config, profile, self.tokenizer_fn
            ),
        }
        return ContextPlan(
            sections=sections,
            rolling_summary=self._rolling_summary(),
            keep_last_n_turns=assembly.estimate_window_turns(
                profile.budgets.working, min_turns=profile.window_min_turns
            ),
            profile=profile,
            fact_ids=tuple(fact_ids),
        )

    def _sessions_section(self, profile: MemoryProfile) -> str:
        """Recent session summaries (populated from phase 3 onward)."""
        rows = self._conn.execute(
            "SELECT summary FROM sessions WHERE summary IS NOT NULL AND summary != ''"
            " ORDER BY id DESC LIMIT 5"
        ).fetchall()
        lines = [f"- {r['summary']}" for r in reversed(rows)]
        return "\n".join(
            assembly.fit_lines(lines, profile.budgets.sessions, self.tokenizer_fn)
        )

    def _rolling_summary(self) -> str | None:
        """Current session's rolling summary (maintained by the worker)."""
        return self._rolling

    # ------------------------------------------------------------------ #
    # episodic memory (turns, sessions, recall)
    # ------------------------------------------------------------------ #
    def begin_session(self, host_tag: str | None = None) -> int:
        """Start a new session explicitly (record_turn does this lazily)."""
        with self._write_lock:
            if self.session_id is not None:
                episodic.end_session(self._conn, self.session_id)
            self.session_id = episodic.begin_session(self._conn, host_tag)
            self._turn = 0
            self._rolling = None
        return self.session_id

    def record_turn(self, user_text: str, assistant_text: str) -> None:
        """Persist one exchange and run inline rule capture.

        Never raises: memory writes must not break the host's chat loop.
        """
        try:
            with self._write_lock:
                if self.session_id is None:
                    self.session_id = episodic.begin_session(self._conn)
                self._turn += 1
                chunk_ids = episodic.record_turn(
                    self._conn, self.config, self.session_id, self._turn,
                    user_text, assistant_text,
                )
            for key, value in rules.capture(user_text):
                self.remember(key, value, source="rule")
            self._after_record_turn(user_text, assistant_text, chunk_ids)
        except Exception:
            log.exception("elastimem: record_turn failed (chat unaffected)")

    def _after_record_turn(
        self, user_text: str, assistant_text: str, chunk_ids: list[int]
    ) -> None:
        """Enqueue deferred work: LLM extraction, chunk embedding."""
        profile = self.governor.profile
        if (self.complete_fn is not None
                and profile.llm_extraction_enabled
                and len(user_text.split()) >= self.config.min_query_words):
            self._worker.submit(Job(
                "extract", needs_llm=True,
                payload={"user": user_text, "assistant": assistant_text},
            ))
        if (self.embed_fn is not None and profile.embeddings_enabled and chunk_ids):
            self._worker.submit(Job("embed", needs_llm=False,
                                    payload={"chunk_ids": chunk_ids}))

    @contextlib.contextmanager
    def foreground(self):
        """Bracket the host's own LLM generation: while held, the worker
        will not start background LLM jobs (single-model-instance safety)."""
        self._worker.foreground_begin()
        try:
            yield
        finally:
            self._worker.foreground_end()

    def foreground_begin(self) -> None:
        self._worker.foreground_begin()

    def foreground_end(self) -> None:
        self._worker.foreground_end()

    def report_evictions(self, turns: list[tuple[str, str]]) -> None:
        """Host reports (user, assistant) pairs it evicted from its window;
        they fold into the rolling summary (already persisted verbatim)."""
        if not turns:
            return
        profile = self.governor.profile
        if profile.rolling_summary_enabled and self.complete_fn is not None:
            self._worker.submit(Job(
                "rolling_summary", needs_llm=True, payload={"turns": turns},
            ))
        else:
            # Degradation floor: a marker line, so the model knows history
            # exists and recall() can retrieve it.
            self._rolling = (
                f"[{len(turns)} earlier turn(s) omitted — memory search can"
                " recall them]"
                if self._rolling is None
                else self._rolling + f" [+{len(turns)} more turn(s) omitted]"
            )

    def drain(self, timeout: float = 5.0) -> bool:
        """Finish queued background work (call before unloading the model)."""
        return self._worker.drain(timeout)

    # ------------------------------------------------------------------ #
    # background job execution (called from the worker thread)
    # ------------------------------------------------------------------ #
    def _execute_job(self, job: Job) -> None:
        conn = self._conn  # worker thread gets its own connection
        if job.kind == "extract" and self.complete_fn is not None:
            extraction.extract_facts(
                conn, self.config, self.complete_fn,
                job.payload["user"], job.payload["assistant"],
                store_fn=lambda k, v, s: self.remember(k, v, source=s),
            )
        elif job.kind == "rolling_summary" and self.complete_fn is not None:
            summary = extraction.rolling_summary(
                self.complete_fn, self.config, self._rolling, job.payload["turns"]
            )
            if summary:
                self._rolling = summary
                if self.session_id is not None:
                    with self._write_lock:
                        conn.execute(
                            "UPDATE sessions SET rolling_summary=? WHERE id=?",
                            (summary, self.session_id),
                        )
                        conn.commit()
        elif job.kind == "embed":
            from . import embeddings

            embeddings.embed_chunks(self, job.payload["chunk_ids"])
        elif job.kind == "consolidate":
            level = self.governor.profile.consolidation_level
            if level is not ConsolidationLevel.OFF:
                with self._write_lock:
                    extraction.consolidate(
                        conn, self.config, self.complete_fn,
                        llm_merge=(level is ConsolidationLevel.FULL
                                   and self.complete_fn is not None),
                    )

    def recall(self, query: str, k: int = 5) -> list:
        """Search past conversations and facts. Works in every tier; returns
        a list of :class:`elastimem.retrieval.Hit`, best first. Never raises."""
        try:
            from . import retrieval

            return retrieval.search_all(self, query, k=k)
        except Exception:
            log.exception("elastimem: recall failed")
            return []

    def sessions(self, n: int = 20) -> list[dict]:
        """Recent sessions, newest first."""
        return episodic.list_sessions(self._conn, n)

    def resume_session(self, session_id: int | None = None) -> tuple[str | None, list[dict]]:
        """Return ``(rolling_summary, tail_messages)`` of a past session so the
        host can reload them into its message list. Defaults to the most
        recently ended session. Starts a fresh session for new turns."""
        conn = self._conn
        if session_id is None:
            row = conn.execute(
                "SELECT id FROM sessions WHERE ended_at IS NOT NULL"
                " AND message_count > 0 ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row is None:
                return None, []
            session_id = row["id"]
        return episodic.session_tail(
            conn, session_id, self.governor.profile.budgets.working
        )

    def end_session(self) -> None:
        """Close the current session: flush deferred work, summarize (when a
        completion callable exists), run exit consolidation."""
        if self.session_id is None:
            return
        self._worker.drain()

        summary = None
        if self.complete_fn is not None:
            user_turns = [
                r["content"] for r in self._conn.execute(
                    "SELECT content FROM messages WHERE session_id=? AND"
                    " role='user' ORDER BY id", (self.session_id,)
                )
            ]
            summary = extraction.session_summary(
                self.complete_fn, self.config, user_turns
            ) or None

        with self._write_lock:
            episodic.end_session(self._conn, self.session_id, summary)
            level = self.governor.profile.consolidation_level
            if level is not ConsolidationLevel.OFF:
                extraction.consolidate(
                    self._conn, self.config, self.complete_fn,
                    llm_merge=(level is ConsolidationLevel.FULL
                               and self.complete_fn is not None),
                )
            self.session_id = None
            self._turn = 0
            self._rolling = None

    # ------------------------------------------------------------------ #
    # semantic memory (facts)
    # ------------------------------------------------------------------ #
    def remember(self, key: str, value: str, source: str = "explicit") -> tuple[bool, str]:
        """Durably store one fact, synchronously. Returns ``(changed, reason)``."""
        with self._write_lock:
            return semantic.store_fact(self._conn, self.config, key, value, source)

    def facts(self) -> dict[str, str]:
        """Current facts as a merged ``{key: value}`` dict."""
        return semantic.facts_as_dict(self._conn)

    def fact_history(self, key: str) -> list[semantic.Fact]:
        """Every version ever stored for ``key`` — the audit chain."""
        return semantic.fact_history(self._conn, key)

    def forget(self, key: str) -> bool:
        """Invalidate the current version of ``key`` (non-destructive tombstone)."""
        with self._write_lock:
            return semantic.forget(self._conn, key)

    # ------------------------------------------------------------------ #
    # procedural memory (lessons)
    # ------------------------------------------------------------------ #
    def add_lesson(self, text: str, tag: str | None = None) -> bool:
        with self._write_lock:
            return procedural.add_lesson(self._conn, self.config, text, tag)

    def lessons(self, n: int | None = None) -> list[str]:
        return procedural.load_lessons(self._conn, n or self.config.lessons_in_prompt)

    # ------------------------------------------------------------------ #
    # inspection / maintenance
    # ------------------------------------------------------------------ #
    def quarantine_entries(self, n: int = 20) -> list[dict]:
        """Recently rejected automatic extractions (debugging; never prompted)."""
        rows = self._conn.execute(
            "SELECT ts, key, value, reason, source FROM quarantine"
            " ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def stats(self) -> dict:
        counts = {
            table: self._conn.execute(f"SELECT count(*) c FROM {table}").fetchone()["c"]
            for table in ("facts", "sessions", "messages", "chunks", "lessons", "quarantine")
        }
        size = os.path.getsize(self.path) if self.path != ":memory:" else 0
        return {"path": self.path, "db_bytes": size, "fts_enabled": self.fts_enabled,
                **counts}

    def close(self) -> None:
        """Stop the worker (draining pending jobs) and close connections."""
        self._worker.stop()
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None
