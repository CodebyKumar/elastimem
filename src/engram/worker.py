"""The background worker: one daemon thread that owns all deferred memory work.

Design constraints, in order:

1. **The foreground always wins.** Most local hosts have exactly one model
   instance, and it is not thread-safe. The host brackets its own generation
   with ``foreground()`` (or ``foreground_begin()/end()``); while the gate is
   held, the worker will not *start* an LLM job. Jobs it does run are capped
   tiny (``worker_max_tokens``) so a mistimed overlap costs well under a
   second on a 2B model.
2. **Nothing is lost silently.** ``drain()`` finishes the queue (with a
   timeout) before the host unloads its model or exits.
3. **Nothing ever raises.** A failed job is logged and dropped; the chat
   loop must never feel memory maintenance.

Cadence (from the governor) decides when extraction jobs run: PER_TURN as
they arrive, BATCHED coalesced every N turns, SESSION_END held until the
session closes, OFF dropped at enqueue time.
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from .config import Cadence

if TYPE_CHECKING:
    from .store import Engram

log = logging.getLogger("engram")

_SENTINEL = object()


@dataclass
class Job:
    kind: str                       # 'extract' | 'rolling_summary' | 'session_summary'
                                    # | 'embed' | 'consolidate'
    needs_llm: bool
    payload: dict[str, Any] = field(default_factory=dict)


class Worker:
    def __init__(self, store: "Engram") -> None:
        self._store = store
        self._queue: "queue.Queue[Job | object]" = queue.Queue()
        self._foreground_busy = threading.Event()
        self._idle = threading.Event()
        self._idle.set()
        self._held_llm_jobs: list[Job] = []      # waiting for the foreground gate
        self._batch: list[Job] = []              # BATCHED-cadence extractions
        self._turns_since_batch = 0
        self._thread = threading.Thread(
            target=self._run, name="engram-worker", daemon=True
        )
        self._thread.start()

    # ------------------------------------------------------------------ #
    # host-facing controls
    # ------------------------------------------------------------------ #
    def foreground_begin(self) -> None:
        """The host is about to use the LLM; hold new LLM jobs."""
        self._foreground_busy.set()

    def foreground_end(self) -> None:
        self._foreground_busy.clear()

    def submit(self, job: Job) -> None:
        cadence = self._store.profile.extraction_cadence
        if job.kind == "extract":
            if cadence is Cadence.OFF:
                return
            if cadence in (Cadence.BATCHED, Cadence.SESSION_END):
                self._batch.append(job)
                self._turns_since_batch += 1
                if (cadence is Cadence.BATCHED
                        and self._turns_since_batch
                        >= self._store.config.batched_every_n_turns):
                    self.flush_batch()
                return
        self._idle.clear()
        self._queue.put(job)

    def flush_batch(self) -> None:
        """Release held extraction jobs into the queue (batch/session-end)."""
        for job in self._batch:
            self._idle.clear()
            self._queue.put(job)
        self._batch.clear()
        self._turns_since_batch = 0

    def drain(self, timeout: float = 5.0) -> bool:
        """Block until queued work (incl. gate-held jobs) finishes, or timeout.
        The foreground gate is ignored during a drain — the host asked."""
        self.flush_batch()
        self._foreground_busy.clear()
        return self._idle.wait(timeout)

    def stop(self, timeout: float = 5.0) -> None:
        self.drain(timeout)
        self._queue.put(_SENTINEL)
        self._thread.join(timeout)

    def pending(self) -> int:
        return self._queue.qsize() + len(self._held_llm_jobs) + len(self._batch)

    # ------------------------------------------------------------------ #
    # worker loop
    # ------------------------------------------------------------------ #
    def _run(self) -> None:
        while True:
            job = self._next_job()
            if job is _SENTINEL:
                return
            if job is None:
                continue
            try:
                self._execute(job)
            except Exception:
                log.exception("engram worker: %s job failed", job.kind)
            finally:
                if (self._queue.empty() and not self._held_llm_jobs):
                    self._idle.set()

    def _next_job(self) -> Job | object | None:
        # Gate-held LLM jobs run as soon as the foreground frees up.
        if self._held_llm_jobs and not self._foreground_busy.is_set():
            return self._held_llm_jobs.pop(0)
        # While jobs are gate-held, poll briefly so we notice the gate
        # clearing; otherwise sleep the full idle interval.
        holding = bool(self._held_llm_jobs)
        timeout = 0.05 if holding else self._store.config.idle_consolidate_seconds
        try:
            job = self._queue.get(timeout=timeout)
        except queue.Empty:
            if not holding:
                self._maybe_idle_consolidate()
            return None
        if job is _SENTINEL:
            return job
        if job.needs_llm and self._foreground_busy.is_set():
            self._held_llm_jobs.append(job)
            return None
        return job

    def _maybe_idle_consolidate(self) -> None:
        """FULL tier runs a consolidation sweep after a quiet stretch."""
        from .config import ConsolidationLevel

        profile = self._store.profile
        if (profile.consolidation_level is ConsolidationLevel.FULL
                and not self._foreground_busy.is_set()
                and getattr(self._store, "session_id", None) is not None):
            self._idle.clear()
            self._queue.put(Job("consolidate", needs_llm=True, payload={}))

    def _execute(self, job: Job) -> None:
        self._store._execute_job(job)
