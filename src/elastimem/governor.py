"""The governor: what can this machine afford right now?

It picks a :class:`~elastimem.config.Tier` from three signals:

* **capacity** - total RAM: 16 GiB+ FULL, 8 GiB+ STANDARD, else LITE;
* **pressure** - the OS's own reading (macOS ``kern.memorystatus_vm_pressure_level``,
  Linux ``MemAvailable/MemTotal``, psutil elsewhere): "warn" caps at
  STANDARD, "critical" forces LITE. Free-page counts are not used, because
  macOS keeps most memory compressed or cached and reports little as free;
* **measured cost** - how long memory work took per turn. Over
  ``turn_budget_ms`` for several turns in a row steps the tier down.

Down is immediate; up waits for ``tiers.UPGRADE_TURNS`` healthy turns and
never exceeds what capacity and pressure allow. Hardware is read at most
every ``tiers.PROBE_SECONDS``.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable

from . import tiers
from .config import ElastimemConfig, Tier

log = logging.getLogger("elastimem")

GIB = 1024**3
ProbeFn = Callable[[], "tuple[int, str]"]


@dataclass(frozen=True)
class Budgets:
    """Token budget per prompt section. The same on every tier."""

    working: int
    facts: int
    moments: int
    topics: int
    sessions: int
    lessons: int


@dataclass(frozen=True)
class Status:
    """What memory will do this turn, and why."""

    tier: Tier
    budgets: Budgets
    keep_turns: int         # recent turns the host should keep verbatim
    vectors: bool           # is search by meaning on?
    llm: str                # background extraction: per_turn | batched | session_end | off
    graph_hops: int
    reasons: dict[str, str]

    @property
    def spec(self) -> tiers.TierSpec:
        return tiers.TIERS[self.tier]


def budgets(config: ElastimemConfig) -> Budgets:
    free = config.context_tokens - config.reserved_tokens
    working = int(free * tiers.WORKING_SHARE)
    pool = free - working
    moments = int(pool * tiers.SPLIT["moments"])
    topics = int(moments * tiers.TOPICS_SHARE)
    return Budgets(
        working=working,
        facts=int(pool * tiers.SPLIT["facts"]),
        moments=moments - topics,
        topics=topics,
        sessions=int(pool * tiers.SPLIT["sessions"]),
        lessons=int(pool * tiers.SPLIT["lessons"]),
    )


def keep_turns(b: Budgets) -> int:
    return max(tiers.MIN_KEEP_TURNS, b.working // tiers.TURN_TOKENS // 2)


# --- hardware ------------------------------------------------------------------
def probe() -> tuple[int, str]:
    """``(total_ram_bytes, pressure)`` with pressure normal | warn | critical."""
    try:
        if sys.platform == "darwin":
            out = subprocess.run(
                ["sysctl", "-n", "hw.memsize", "kern.memorystatus_vm_pressure_level"],
                capture_output=True, text=True, timeout=2,
            ).stdout.split()
            level = {"1": "normal", "2": "warn", "4": "critical"}.get(out[1], "normal")
            return int(out[0]), level
        if sys.platform.startswith("linux"):
            info = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    name, value, *_ = line.split()
                    info[name.rstrip(":")] = int(value) * 1024
            return info["MemTotal"], _level(info["MemAvailable"] / info["MemTotal"])
        import psutil
        vm = psutil.virtual_memory()
        return int(vm.total), _level(vm.available / vm.total)
    except Exception:
        return 8 * GIB, "normal"   # unknown: assume a mid-size machine


def _level(free_share: float) -> str:
    if free_share < tiers.CRITICAL_FREE_SHARE:
        return "critical"
    return "warn" if free_share < tiers.WARN_FREE_SHARE else "normal"


def on_battery() -> bool:
    """True when running on battery (needs psutil; False when unknown)."""
    try:
        import psutil
        info = psutil.sensors_battery()
        return bool(info and not info.power_plugged)
    except Exception:
        return False


def _allowed(total: int, pressure: str) -> Tier:
    capacity = (Tier.FULL if total >= tiers.FULL_RAM_GIB * GIB
                else Tier.STANDARD if total >= tiers.STANDARD_RAM_GIB * GIB else Tier.LITE)
    cap = {"normal": Tier.FULL, "warn": Tier.STANDARD, "critical": Tier.LITE}[pressure]
    return min(capacity, cap)


# --- governor ------------------------------------------------------------------
class Governor:
    def __init__(
        self,
        config: ElastimemConfig,
        *,
        probe_fn: ProbeFn | None = None,
        battery_fn: Callable[[], bool] | None = None,
        on_change: Callable[[Tier, Tier], None] | None = None,
    ) -> None:
        self.config = config
        self._probe = probe_fn or probe
        self._battery = battery_fn or on_battery
        self._on_change = on_change
        self._lock = threading.Lock()
        self._slow = self._healthy = 0
        self._last_probe = self._last_pressure_report = 0.0
        self.on_battery = False
        self.reason = ""
        self._tier = self._read()

    @property
    def tier(self) -> Tier:
        return self._tier

    def _read(self) -> Tier:
        """Re-read hardware; returns the tier it allows."""
        self._last_probe = time.monotonic()
        self.on_battery = self._battery()
        if self.config.tier is not None:
            self.reason = "fixed by configuration"
            return self.config.tier
        total, pressure = self._probe()
        allowed = _allowed(total, pressure)
        self.reason = f"{total / GIB:.0f} GiB RAM, memory pressure {pressure}"
        return allowed

    def check(self) -> Tier:
        """Cheap per-turn check: re-reads hardware at most every PROBE_SECONDS."""
        with self._lock:
            if time.monotonic() - self._last_probe >= tiers.PROBE_SECONDS:
                allowed = self._read()
                if self.config.tier is not None or allowed < self._tier:
                    self._set(allowed)
                elif allowed > self._tier and self._healthy >= tiers.UPGRADE_TURNS:
                    self._set(Tier(self._tier + 1))
            return self._tier

    def observe(self, ms: float) -> None:
        """Report how long this turn's memory work took."""
        with self._lock:
            if ms > self.config.turn_budget_ms:
                self._slow, self._healthy = self._slow + 1, 0
                if self._slow >= tiers.SLOW_TURNS and self._tier > Tier.LITE and self.config.tier is None:
                    self._set(Tier(self._tier - 1), f"memory work took {ms:.0f} ms per turn")
            else:
                self._slow, self._healthy = 0, self._healthy + 1

    def report_pressure(self) -> None:
        """Host signal (OOM, failed decode): step down one tier. Reports within
        PRESSURE_COOLDOWN_SECONDS of the last one count as the same event."""
        with self._lock:
            now = time.monotonic()
            first = now - self._last_pressure_report >= tiers.PRESSURE_COOLDOWN_SECONDS
            self._last_pressure_report = now
            if first and self._tier > Tier.LITE and self.config.tier is None:
                self._set(Tier(self._tier - 1), "host reported memory pressure")

    def reconfigure(self, config: ElastimemConfig) -> None:
        with self._lock:
            self.config = config
            self._set(self._read())

    def _set(self, tier: Tier, reason: str | None = None) -> None:
        self._slow = self._healthy = 0
        if reason:
            self.reason = reason
        if tier == self._tier:
            return
        old, self._tier = self._tier, tier
        log.info("elastimem: tier %s -> %s (%s)", old.name, tier.name, self.reason)
        if self._on_change:
            try:
                self._on_change(old, tier)
            except Exception:
                log.exception("elastimem: on_tier callback failed")
