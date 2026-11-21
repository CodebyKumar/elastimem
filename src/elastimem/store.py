"""The ``Elastimem`` class: the one object hosts use.

Typical turn::

    context = mem.context(user)          # memory for the prompt
    reply = llm(f"{context}\\n\\n{user}")
    mem.record(user, reply)              # store it; derived work runs in the background

Threads: public methods run on the host's thread; one background worker
fills in vectors, extraction and summaries. Writes share one lock; each
thread has its own SQLite connection. Methods that write memory on the hot
path (``context``, ``record``, ``search``) never raise.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import sqlite3
import threading
import time
from typing import Callable

from . import tiers
from .config import ElastimemConfig, Tier
from .retrieval import context as context_mod
from .retrieval import inspection, search
from .retrieval.context import Context
from .retrieval.vectors import Vectors
from .runtime import llm
from .runtime.governor import Governor, Status, budgets, keep_turns
from .runtime.jobs import Pause, Worker, consolidate
from .storage import db, episodes, facts, graph, lessons

log = logging.getLogger("elastimem")


def _settings(config: ElastimemConfig | None, overrides: dict) -> ElastimemConfig:
    valid = {f.name for f in dataclasses.fields(ElastimemConfig)}
    unknown = set(overrides) - valid
    if unknown:
        raise TypeError(f"unknown setting(s): {', '.join(sorted(unknown))}"
                        f" (valid: {', '.join(sorted(valid))})")
    return dataclasses.replace(config or ElastimemConfig(), **overrides)


class Elastimem:
    """A persistent memory store in one SQLite file.

    ``llm`` is ``(prompt, *, max_tokens, temperature) -> str``. ``embedder``
    is ``(list[str]) -> list[list[float]]``, ``"auto"`` (the built-in model
    when ``elastimem[embed]`` is installed) or ``None`` (keyword search
    only). Settings are :class:`ElastimemConfig` fields, passed inline or as
    ``config=``.
    """

    def __init__(
        self,
        path: str,
        *,
        llm: Callable[..., str] | None = None,
        embedder: Callable | str | None = "auto",
        tokenizer: Callable[[str], int] | None = None,
        on_tier: Callable[[Tier, Tier], None] | None = None,
        config: ElastimemConfig | None = None,
        **settings,
    ) -> None:
        self.path = path if path == ":memory:" else os.path.expanduser(path)
        self._config = _settings(config, settings)
        self._llm, self._tokenizer = llm, tokenizer
        self._vectors = Vectors(embedder)
        self._write_lock = threading.RLock()
        self._local = threading.local()
        self._conns: list[sqlite3.Connection] = []
        self._conns_lock = threading.Lock()
        self._closed = False

        conn, self._fts = db.open_store(self.path)
        self._register(conn)
        self._governor = Governor(self._config, on_change=on_tier)
        self._session: int | None = None
        self._turn = 0
        with self._write_lock:
            episodes.close_orphans(conn)
        self._vectors.forget_other_models(conn, self._write_lock)
        self._worker = Worker(self)

    # -- connections ------------------------------------------------------------
    def _register(self, conn: sqlite3.Connection) -> None:
        self._local.conn = conn
        with self._conns_lock:
            self._conns.append(conn)

    @property
    def _conn(self) -> sqlite3.Connection:
        if self.path == ":memory:":
            return self._conns[0]       # each new :memory: connection is a separate empty db
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = db.connect(self.path)
            self._register(conn)
        return conn

    # -- status and settings ----------------------------------------------------
    @property
    def status(self) -> Status:
        """What memory will do this turn, and why."""
        tier = self._governor.tier
        spec = tiers.TIERS[tier]
        b = budgets(self._config)
        reasons = {"tier": f"{tier.name}: {self._governor.reason}"}
        vector_reason = self._vectors.unavailable(spec.load_embedder)
        reasons["vectors"] = vector_reason or "on"
        if self._llm is None:
            cadence, reasons["llm"] = "off", "no llm given"
        elif self._governor.on_battery and spec.extraction != "session_end":
            cadence, reasons["llm"] = "session_end", "on battery: background LLM work waits for session end"
        else:
            cadence, reasons["llm"] = spec.extraction, f"{spec.extraction} ({tier.name})"
        return Status(tier, b, keep_turns(b), vector_reason is None, cadence,
                      spec.graph_hops, reasons)

    @property
    def config(self) -> ElastimemConfig:
        """A copy of the active settings; change them with :meth:`configure`."""
        return dataclasses.replace(self._config)

    def configure(self, **settings) -> Status:
        """Change settings (for example ``context_tokens`` after a model switch)."""
        self._config = _settings(self._config, settings)
        self._governor.reconfigure(self._config)
        return self.status

    def report_pressure(self) -> Status:
        """Tell memory the host hit an out-of-memory or decode failure."""
        self._governor.report_pressure()
        return self.status

    # -- the turn ---------------------------------------------------------------
    def context(self, query: str = "") -> Context:
        """Memory for this turn's prompt: ``str(context)`` renders it."""
        started = time.perf_counter()
        self._governor.check()
        try:
            ctx, shown = context_mod.build(self, query)
            with self._write_lock:
                facts.touch(self._conn, shown)
        except Exception:
            log.exception("elastimem: building context failed; returning an empty one")
            status = self.status
            ctx = Context(dict.fromkeys(context_mod.TITLES, ""), None, status.keep_turns, status)
        self._governor.observe((time.perf_counter() - started) * 1000)
        return ctx

    def record(self, user: str, reply: str) -> None:
        """Store one exchange. Never raises."""
        try:
            status = self.status
            with self._write_lock:
                if self._session is None:
                    self._session = episodes.begin(self._conn)
                self._turn += 1
                episodes.record(self._conn, self._session, self._turn, user, reply, self._tokenizer)
                if not (status.spec.llm_summary and self._llm):
                    self._fold(status)
            self._worker.wake()
        except Exception:
            log.exception("elastimem: record failed (chat unaffected)")

    def _fold(self, status: Status) -> None:
        """Fold turns that left the window into the summary, without a model."""
        up_to, previous, turns = episodes.evicted(self._conn, self._session, status.keep_turns)
        if turns:
            summary = episodes.extractive_summary(
                previous, turns, status.budgets.sessions, self._tokenizer)
            episodes.save_summary(self._conn, self._session, summary, up_to)

    def pause(self) -> Pause:
        """Hold background LLM work while the host's own model generates::

            with mem.pause():
                reply = llm(prompt)

        Only needed when the host and Elastimem share one model instance.
        """
        return self._worker.pause()

    def flush(self, timeout: float = 5.0) -> bool:
        """Finish all pending background work. Returns False on timeout."""
        return self._worker.flush(timeout)

    # -- searching ----------------------------------------------------------------
    def search(self, query: str, k: int = 5) -> list[search.Hit]:
        """Past moments and facts matching ``query``, best first. Never raises."""
        try:
            return search.search(self, query, k)
        except Exception:
            log.exception("elastimem: search failed")
            return []

    def explain(self, query: str, k: int = 5) -> inspection.Explanation:
        """Like :meth:`search`, with each result's score broken into its terms."""
        return inspection.explain(self, query, k)

    # -- facts and lessons ------------------------------------------------------
    def remember(self, key: str, value: str) -> bool:
        """Store a fact. True if it was new or changed."""
        with self._write_lock:
            changed, _ = facts.store(self._conn, key, value, blocked=self._config.blocked)
        self._worker.wake()
        return changed

    def forget(self, key: str) -> bool:
        with self._write_lock:
            return facts.forget(self._conn, key)

    def facts(self) -> dict[str, str]:
        return facts.as_dict(self._conn)

    def history(self, key_or_question: str) -> list[facts.Fact]:
        """Every version of a fact, oldest first."""
        return inspection.history(self, key_or_question)

    def learn(self, lesson: str) -> bool:
        with self._write_lock:
            return lessons.learn(self._conn, lesson, self._config.max_lessons)

    def lessons(self, n: int = tiers.LESSONS_SHOWN) -> list[str]:
        return lessons.recent(self._conn, n)

    def topics(self) -> list[dict]:
        """Groups of related entities: ``[{"id", "name", "members"}]``."""
        return graph.list_topics(self._conn)

    def rejected(self, n: int = 20) -> list[dict]:
        """Recently rejected automatic facts, for debugging."""
        return facts.list_rejected(self._conn, n)

    # -- sessions -------------------------------------------------------------------
    def sessions(self, n: int = 20) -> list[dict]:
        return episodes.list_sessions(self._conn, n)

    def new_session(self, tag: str | None = None) -> int:
        """Close the current session (if any) and start another."""
        self.end_session()
        with self._write_lock:
            self._session = episodes.begin(self._conn, tag)
        return self._session

    def resume(self, session_id: int | None = None) -> tuple[str | None, list[dict]]:
        """``(summary, messages)`` of a past session (default: the latest
        ended one) for the host to reload. New turns go to a new session."""
        if session_id is None:
            row = self._conn.execute(
                "SELECT id FROM sessions WHERE ended_at IS NOT NULL AND message_count > 0"
                " ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row is None:
                return None, []
            session_id = row["id"]
        return episodes.tail(self._conn, session_id, self.status.budgets.working)

    def end_session(self) -> None:
        """Finish background work, summarize the session and tidy memory."""
        if self._session is None:
            return
        # Inside a pause() the host holds llm_lock; waiting for it would hang,
        # so LLM steps are skipped and the turns stay pending for next time.
        use_llm = self._llm is not None and not self._worker._paused
        if use_llm:
            self._worker.flush()
        summary = None
        if use_llm:
            turns = episodes.user_turns(self._conn, self._session)
            with self._worker.llm_lock:
                summary = llm.session_summary(self._llm, turns) or None
        try:
            consolidate(self, use_llm)
        except Exception:
            log.exception("elastimem: consolidation failed")
        with self._write_lock:
            episodes.end(self._conn, self._session, summary)
            self._session, self._turn = None, 0

    # -- lifecycle -----------------------------------------------------------------
    def stats(self) -> dict:
        tables = ("facts", "sessions", "messages", "chunks", "lessons", "entities", "rejected")
        counts = {t: self._conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in tables}
        size = os.path.getsize(self.path) if self.path != ":memory:" else 0
        return {"path": self.path, "bytes": size, "fts": self._fts, **counts}

    def close(self) -> None:
        """End the session, stop the worker and close every connection."""
        if self._closed:
            return
        self._closed = True
        try:
            self.end_session()
        except Exception:      # every turn is already stored; finish shutting down
            log.exception("elastimem: ending the session on close failed")
        finally:
            self._worker.stop()
            with self._conns_lock:
                for conn in self._conns:
                    conn.close()
                self._conns.clear()

    def __enter__(self) -> "Elastimem":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
