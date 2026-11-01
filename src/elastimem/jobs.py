"""The background worker: one thread that fills in derived memory.

There is no in-memory job queue. Pending work is read from the database
(chunks with no vector, turns with no ``extracted_at``, turns that left the
window but are not folded), so a crash loses nothing and a tier upgrade
backfills automatically. ``wake()`` only tells the thread to look.

Order of work: vectors, then LLM extraction (at the tier's cadence), then
LLM folding of the rolling summary, then idle consolidation (FULL only).

The host's model is usually a single instance that is not thread-safe.
``pause()`` takes ``llm_lock``, which every background LLM call holds, so
the host's own generation never overlaps one; while paused, the worker
skips LLM work and only does the rest.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from . import episodes, facts, graph, llm, tiers

if TYPE_CHECKING:
    from .store import Elastimem

log = logging.getLogger("elastimem")


class Pause:
    """Held while the host uses its model. ``with mem.pause():`` or
    ``p = mem.pause()`` ... ``p.end()``; ``end()`` may run on another thread."""

    def __init__(self, worker: "Worker") -> None:
        self._worker = worker
        self._held = True
        worker._pause_begin()

    def end(self) -> None:
        if self._held:
            self._held = False
            self._worker._pause_end()

    def __enter__(self) -> "Pause":
        return self

    def __exit__(self, *exc) -> None:
        self.end()


class Worker:
    def __init__(self, store: "Elastimem") -> None:
        self._store = store
        # A plain Lock, not an RLock: a streaming host may end a pause on a
        # different thread than the one that began it.
        self.llm_lock = threading.Lock()
        self._paused = 0
        self._pause_guard = threading.Lock()
        self._wake = threading.Event()
        self._done = threading.Condition()
        self._requested = self._finished = 0
        self._stopping = False
        self._thread = threading.Thread(target=self._run, name="elastimem-worker", daemon=True)
        self._thread.start()

    def wake(self) -> None:
        self._wake.set()

    def pause(self) -> Pause:
        return Pause(self)

    def _pause_begin(self) -> None:
        with self._pause_guard:
            if self._paused == 0:
                self.llm_lock.acquire()   # waits for a background call to finish
            self._paused += 1

    def _pause_end(self) -> None:
        with self._pause_guard:
            self._paused -= 1
            if self._paused == 0:
                self.llm_lock.release()
        self.wake()

    def flush(self, timeout: float = 5.0) -> bool:
        """Do all pending work, ignoring cadence. While paused, LLM work stays
        pending. True if it finished."""
        with self._done:
            self._requested += 1
            target = self._requested
            self._wake.set()
            return self._done.wait_for(lambda: self._finished >= target, timeout)

    def stop(self, timeout: float = 5.0) -> None:
        self.flush(timeout)
        self._stopping = True
        self._wake.set()
        self._thread.join(timeout)

    # -- loop -----------------------------------------------------------------
    def _run(self) -> None:
        while not self._stopping:
            woke = self._wake.wait(tiers.IDLE_SECONDS)
            self._wake.clear()
            with self._done:
                target = self._requested
            flushing = target > self._finished
            try:
                while not self._stopping and self._step(flushing):
                    pass
                if (not woke and not self._paused and self._store._session is not None
                        and self._store.status.spec.idle_consolidation):
                    consolidate(self._store)
            except Exception:
                log.exception("elastimem: background work failed")
            if flushing:
                with self._done:
                    self._finished = target
                    self._done.notify_all()

    def _step(self, flushing: bool) -> bool:
        """Do one piece of pending work. False when there is nothing to do."""
        store = self._store
        status, conn = store.status, store._conn
        if status.vectors:
            for table in ("chunks", "facts"):
                if store._vectors.embed_pending(conn, store._write_lock, table, tiers.EMBED_BATCH):
                    return True
        # A flush overrides cadence, never a pause: the host holds llm_lock.
        if store._llm is None or self._paused:
            return False

        cadence = status.llm
        waiting = episodes.count_unextracted(conn) if cadence == "batched" else 1
        due = (cadence == "per_turn"
               or (cadence == "batched" and waiting >= tiers.BATCH_TURNS)
               or (cadence != "off" and flushing))
        if due:
            # A batch is extracted whole; per-turn work goes one turn at a
            # time so a pause() is honoured between calls.
            pending = episodes.unextracted(conn, max(1, waiting))
            for turn in pending:
                self._extract(turn)
            if pending:
                return True

        if status.spec.llm_summary and store._session is not None:
            up_to, previous, turns = episodes.evicted(conn, store._session, status.keep_turns)
            if turns:
                with self.llm_lock:
                    summary = llm.fold(store._llm, previous, turns)
                summary = summary or episodes.extractive_summary(
                    previous, turns, status.budgets.sessions, store._tokenizer)
                with store._write_lock:
                    episodes.save_summary(conn, store._session, summary, up_to)
                return True
        return False

    def _extract(self, turn: dict) -> None:
        store, conn = self._store, self._store._conn
        with self.llm_lock:
            out = llm.extract(store._llm, turn["user"], turn["reply"])
        with store._write_lock:
            stored = sum(
                facts.store(conn, key, value, "auto", blocked=store._config.blocked,
                            user_text=turn["user"])[0]
                for key, value in out.get("facts", {}).items()
            )
            graph.store_extraction(conn, out.get("entities"), out.get("relations"), turn["chunk_ids"])
            if stored:
                episodes.mark_important(conn, turn["chunk_ids"])
            # Marked even when the call failed, so a broken turn is not retried forever.
            episodes.mark_extracted(conn, turn["chunk_ids"])


def consolidate(store: "Elastimem", allow_llm: bool = True) -> None:
    """Maintenance: retire faded memories, then (FULL with an LLM) merge
    changed facts and duplicate entities and name new topics."""
    conn, lock, worker = store._conn, store._write_lock, store._worker
    use_llm = allow_llm and store._llm is not None and store.status.spec.llm_maintenance
    half_life = store._config.half_life_days
    with lock:
        facts.archive_stale(conn, half_life)
        graph.retire_stale(conn, half_life)
        with conn:
            facts.trim_rejected(conn)
    if use_llm:
        for row in facts.recent_changes(conn):
            with worker.llm_lock:
                merged = llm.merge_value(store._llm, row["key"], row["old"], row["new"])
            if merged and merged != row["new"]:
                with lock:
                    facts.set_value(conn, row["new_id"], merged)
        for a, b in graph.merge_candidates(conn):
            with worker.llm_lock:
                same = llm.same_thing(store._llm, a["name"], b["name"])
            if same:
                with lock:
                    graph.merge(conn, a["id"], b["id"])
    with lock:
        graph.recompute_topics(conn)
    if use_llm:
        for topic_id, names in graph.unnamed_topics(conn).items():
            with worker.llm_lock:
                name = llm.topic_name(store._llm, names)
            if name:
                with lock:
                    graph.name_topic(conn, topic_id, name)
