"""The Memory Governor: Elastimem's defining component.

The governor answers one question, continuously: *what can this machine
afford right now?* It probes hardware at startup, re-checks cheaply on every
``tick()`` (call it once per turn), and emits a frozen
:class:`~elastimem.config.MemoryProfile` that every other component consumes —
token budgets per prompt section, whether embeddings run, how often
background LLM extraction fires, how aggressive consolidation is.

Rules of movement:

* **Downgrades are immediate.** Low available RAM or a host-reported pressure
  event (OOM, decode failure) drops the tier on the next profile read.
* **Upgrades are cautious.** Only after ``upgrade_healthy_ticks`` consecutive
  healthy ticks, one tier at a time, and never above the startup tier.

Hardware probing prefers ``psutil`` when installed (the ``[system]`` extra)
and falls back to stdlib-only reads (``/proc/meminfo`` on Linux, ``sysctl`` /
``vm_stat`` on macOS). When nothing works, the governor assumes STANDARD —
being wrong by one tier degrades quality, never correctness.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
import time
from typing import Callable

from .config import (
    Budgets,
    Cadence,
    ConsolidationLevel,
    ElastimemConfig,
    MemoryProfile,
    Tier,
)

log = logging.getLogger("elastimem")

GIB = 1024**3

_PRESSURE_AVAILABLE = int(1.2 * GIB)   # below this, downgrade immediately
# A burst of host-reported pressure events (several decode failures in the
# same bad turn, or one failure surfacing through more than one call site)
# must not cascade the tier down multiple levels for what is really one
# underlying event. This does NOT weaken "downgrades are immediate" - the
# FIRST report in a burst still downgrades right away, same as before; only
# additional reports within the window are coalesced into that same
# downgrade rather than each dropping the tier again.
_PRESSURE_REPORT_COOLDOWN_SECONDS = 30.0


def _tier_thresholds_bytes(cfg: ElastimemConfig) -> dict[Tier, tuple[int, int]]:
    """User-tunable GiB thresholds (config.py) converted to bytes."""
    full_total, full_avail = cfg.tier_thresholds_gib["full"]
    std_total, std_avail = cfg.tier_thresholds_gib["standard"]
    return {
        Tier.FULL: (int(full_total * GIB), int(full_avail * GIB)),
        Tier.STANDARD: (int(std_total * GIB), int(std_avail * GIB)),
    }


# --------------------------------------------------------------------------- #
# hardware probe
# --------------------------------------------------------------------------- #
def probe_ram() -> tuple[int, int]:
    """Return ``(total_bytes, available_bytes)``, best effort, stdlib-safe.

    Stdlib-only probing is implemented for Linux (``/proc/meminfo``) and
    macOS (``sysctl``/``vm_stat``). Windows and any other platform have no
    stdlib probe here and fall through to the "assume a mid-size machine"
    floor below (8/4 GiB) — install the ``system`` extra (``psutil``) for
    accurate probing on Windows; without it, tier classification on Windows
    is a guess, not a measurement.
    """
    try:
        import psutil  # optional [system] extra

        vm = psutil.virtual_memory()
        return int(vm.total), int(vm.available)
    except ImportError:
        pass

    if sys.platform.startswith("linux"):
        try:
            info: dict[str, int] = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    parts = line.split()
                    if parts and parts[0].rstrip(":") in ("MemTotal", "MemAvailable"):
                        info[parts[0].rstrip(":")] = int(parts[1]) * 1024
            return info.get("MemTotal", 8 * GIB), info.get("MemAvailable", 4 * GIB)
        except OSError:
            pass
    elif sys.platform == "darwin":
        try:
            total = int(
                subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True, text=True, timeout=2,
                ).stdout.strip()
            )
            available = _darwin_available()
            return total, available if available else total // 3
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    # Unknown platform / probe failed: assume a mid-size machine.
    return 8 * GIB, 4 * GIB


def _darwin_available() -> int:
    """Free + inactive pages from vm_stat (what macOS can hand out quickly)."""
    try:
        out = subprocess.run(
            ["vm_stat"], capture_output=True, text=True, timeout=2
        ).stdout
        page_size = 16384 if "page size of 16384" in out else 4096
        pages = 0
        for line in out.splitlines():
            if line.startswith(("Pages free", "Pages inactive")):
                pages += int(line.split(":")[1].strip().rstrip("."))
        return pages * page_size
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0


def _classify(total: int, available: int, thresholds: dict[Tier, tuple[int, int]]) -> Tier:
    for tier in (Tier.FULL, Tier.STANDARD):
        t_min, a_min = thresholds[tier]
        if total >= t_min and available >= a_min:
            return tier
    return Tier.LITE


# --------------------------------------------------------------------------- #
# governor
# --------------------------------------------------------------------------- #
class Governor:
    """Owns the current tier and derives the :class:`MemoryProfile`.

    ``probe_fn`` is injectable for tests (and for hosts with better
    information than psutil, e.g. a Jetson reading its own thermal state).
    ``on_tier_change(old, new)`` lets the host react — unload an embedder,
    print a notice.
    """

    def __init__(
        self,
        config: ElastimemConfig,
        *,
        probe_fn: Callable[[], tuple[int, int]] = probe_ram,
        on_tier_change: Callable[[Tier, Tier], None] | None = None,
    ) -> None:
        self.config = config
        self._probe = probe_fn
        self._on_tier_change = on_tier_change
        self._lock = threading.Lock()
        self._healthy_streak = 0
        self._last_pressure_report = 0.0
        self._thresholds = _tier_thresholds_bytes(config)

        total, available = self._probe()
        # Explicit None check: Tier.LITE is 0 and would be swallowed by `or`.
        self._startup_tier = (
            config.tier_override
            if config.tier_override is not None
            else _classify(total, available, self._thresholds)
        )
        self._tier = self._startup_tier
        self._profile = self._build_profile(self._tier)
        log.info("elastimem governor: startup tier %s (ram %.1f/%.1f GiB available)",
                 self._tier.name, available / GIB, total / GIB)

    # -- public --------------------------------------------------------- #
    @property
    def profile(self) -> MemoryProfile:
        return self._profile

    @property
    def tier(self) -> Tier:
        return self._tier

    def tick(self) -> MemoryProfile:
        """Cheap per-turn re-evaluation. Returns the (possibly new) profile."""
        if self.config.tier_override is not None:
            return self._profile
        total, available = self._probe()
        with self._lock:
            if available < _PRESSURE_AVAILABLE:
                self._set_tier(Tier.LITE)
            else:
                measured = _classify(total, available, self._thresholds)
                if measured < self._tier:
                    self._set_tier(measured)          # downgrade immediately
                elif measured > self._tier:
                    self._healthy_streak += 1          # upgrade cautiously
                    if self._healthy_streak >= self.config.upgrade_healthy_ticks:
                        self._set_tier(min(Tier(self._tier + 1), self._startup_tier))
                else:
                    self._healthy_streak = 0
        return self._profile

    def report_pressure(self) -> MemoryProfile:
        """Host signal: an OOM/decode failure happened. Downgrades one tier
        immediately - UNLESS this report arrives within
        _PRESSURE_REPORT_COOLDOWN_SECONDS of the last one, in which case it's
        coalesced into that same downgrade instead of dropping the tier
        again. Without this, a burst of several decode failures in one bad
        turn (a plausible retry pattern, or the same failure surfacing
        through more than one call site) could cascade the tier down
        multiple levels for what is really one underlying event, needing
        many more healthy ticks than warranted to recover."""
        with self._lock:
            now = time.monotonic()
            if now - self._last_pressure_report < _PRESSURE_REPORT_COOLDOWN_SECONDS:
                self._last_pressure_report = now
                return self._profile
            self._last_pressure_report = now
            if self._tier > Tier.LITE:
                self._set_tier(Tier(self._tier - 1))
        return self._profile

    def reconfigure(self, *, reprobe: bool = False) -> MemoryProfile:
        """Rebuild the current tier's budgets from ``self.config`` right now.

        Budgets are only recomputed on a tier change (see ``_set_tier``), so
        a host that mutates ``config.context_tokens`` (e.g. after switching
        to a model with a different context window) would otherwise see
        stale budgets until the tier happens to flip — which may never
        happen. Call this immediately after changing any config field that
        feeds budget math (``context_tokens``, ``static_prompt_tokens``,
        ``output_reserve``, ``tool_reserve``).

        ``reprobe=True`` also re-measures hardware and re-classifies the
        tier before rebuilding budgets, same as ``tick()``. The tier is
        RAM-based and blind to which model is active, so on its own it can't
        tell a 4K-context model from a 128K one apart - but startup order
        matters: if the host constructs its Elastimem store BEFORE loading
        its own (possibly large) model - as most hosts do, since the model
        card isn't known until the model is chosen - the constructor's
        one-time probe measures RAM the model hasn't claimed yet, and
        without reprobing the tier stays keyed to that pre-load reading
        until the next tick() (normally the first turn). Pass this when
        calling reconfigure() right after a model finishes loading, so the
        tier the host's first turn actually runs under reflects real
        post-load memory pressure instead of a stale pre-load guess.
        """
        with self._lock:
            self._thresholds = _tier_thresholds_bytes(self.config)
            if reprobe and self.config.tier_override is None:
                total, available = self._probe()
                self._set_tier(_classify(total, available, self._thresholds))
            self._profile = self._build_profile(self._tier)
        return self._profile

    # -- internals ------------------------------------------------------ #
    def _set_tier(self, tier: Tier) -> None:
        if tier == self._tier:
            return
        old, self._tier = self._tier, tier
        self._healthy_streak = 0
        self._profile = self._build_profile(tier)
        log.info("elastimem governor: tier %s -> %s", old.name, tier.name)
        if self._on_tier_change is not None:
            try:
                self._on_tier_change(old, tier)
            except Exception:
                log.exception("elastimem: on_tier_change callback failed")

    def _build_profile(self, tier: Tier) -> MemoryProfile:
        cfg = self.config
        dynamic = max(
            256,
            cfg.context_tokens - cfg.output_reserve - cfg.tool_reserve
            - cfg.static_prompt_tokens,
        )
        working = int(dynamic * cfg.working_share)
        memory_pool = dynamic - working
        split = dict(cfg.memory_split)

        if tier is Tier.LITE:
            # No episodic injection, half the session summaries; the freed
            # tokens go back to the working window (immediacy beats recall
            # on a starved machine).
            freed = split.pop("episodic") + split["sessions"] / 2
            split["sessions"] /= 2
            working += int(memory_pool * freed)
            split["episodic"] = 0.0

        budgets = Budgets(
            working=working,
            facts=int(memory_pool * split["facts"]),
            episodic=int(memory_pool * split["episodic"]),
            sessions=int(memory_pool * split["sessions"]),
            lessons=int(memory_pool * split["lessons"]),
        )
        return MemoryProfile(
            tier=tier,
            budgets=budgets,
            embeddings_enabled=tier is not Tier.LITE,
            llm_extraction_enabled=tier is not Tier.LITE,
            extraction_cadence={
                Tier.FULL: Cadence.PER_TURN,
                Tier.STANDARD: Cadence.BATCHED,
                Tier.LITE: Cadence.OFF,
            }[tier],
            rolling_summary_enabled=tier is not Tier.LITE,
            consolidation_level={
                Tier.FULL: ConsolidationLevel.FULL,
                Tier.STANDARD: ConsolidationLevel.DEDUPE_ONLY,
                Tier.LITE: ConsolidationLevel.OFF,
            }[tier],
            episodic_top_k={Tier.FULL: 4, Tier.STANDARD: 3, Tier.LITE: 0}[tier],
        )
