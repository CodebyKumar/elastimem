"""Governor: tier classification, budgets, hysteresis, pressure response."""

import pytest

from engram import Engram, EngramConfig, Tier, Cadence
from engram.governor import Governor, GIB, probe_ram


def gov(total_gib, avail_gib, **cfg_kwargs):
    cfg = EngramConfig(**cfg_kwargs)
    return Governor(cfg, probe_fn=lambda: (int(total_gib * GIB), int(avail_gib * GIB)))


def test_tier_classification():
    assert gov(32, 20).tier is Tier.FULL
    assert gov(16, 8).tier is Tier.FULL
    assert gov(8, 4).tier is Tier.STANDARD
    assert gov(16, 3).tier is Tier.STANDARD   # big machine under load
    assert gov(4, 2).tier is Tier.LITE
    assert gov(8, 1).tier is Tier.LITE


def test_env_override(monkeypatch):
    monkeypatch.setenv("ENGRAM_TIER", "lite")
    g = Governor(EngramConfig(), probe_fn=lambda: (32 * GIB, 20 * GIB))
    assert g.tier is Tier.LITE
    assert g.tick().tier is Tier.LITE  # override pins the tier


def test_budgets_derive_from_context():
    g = gov(32, 20, context_tokens=4096, static_prompt_tokens=1200)
    b = g.profile.budgets
    dynamic = 4096 - 512 - 600 - 1200  # = 1784
    assert b.working == int(dynamic * 0.55)
    assert b.working + b.memory_total <= dynamic
    assert b.facts > b.sessions  # facts get the largest memory share


def test_lite_zeroes_episodic_and_boosts_working():
    full = gov(32, 20, context_tokens=8192)
    lite = gov(4, 2, context_tokens=8192)
    assert lite.profile.budgets.episodic == 0
    assert lite.profile.budgets.working > full.profile.budgets.working
    assert lite.profile.extraction_cadence is Cadence.OFF
    assert lite.profile.embeddings_enabled is False


def test_immediate_downgrade_and_cautious_upgrade():
    ram = {"avail": 20.0}
    cfg = EngramConfig(upgrade_healthy_ticks=3)
    g = Governor(cfg, probe_fn=lambda: (32 * GIB, int(ram["avail"] * GIB)))
    assert g.tier is Tier.FULL

    ram["avail"] = 2.0  # below both FULL and STANDARD availability thresholds
    assert g.tick().tier is Tier.LITE      # downgrade jumps straight to measured
    # recovery requires 3 consecutive healthy ticks, one tier per step
    ram["avail"] = 20.0
    tiers = [g.tick().tier for _ in range(8)]
    assert tiers[-1] is Tier.FULL
    assert tiers[0] is not Tier.FULL  # did not jump back instantly


def test_pressure_report_downgrades():
    g = gov(32, 20)
    assert g.tier is Tier.FULL
    assert g.report_pressure().tier is Tier.STANDARD
    assert g.report_pressure().tier is Tier.LITE
    assert g.report_pressure().tier is Tier.LITE  # floor


def test_tier_change_callback():
    calls = []
    ram = {"avail": 20.0}
    g = Governor(
        EngramConfig(),
        probe_fn=lambda: (32 * GIB, int(ram["avail"] * GIB)),
        on_tier_change=lambda old, new: calls.append((old.name, new.name)),
    )
    ram["avail"] = 0.5
    g.tick()
    assert calls == [("FULL", "LITE")]


def test_probe_ram_returns_sane_values():
    total, available = probe_ram()
    assert total >= 1 * GIB
    assert 0 < available <= total


def test_store_exposes_governor(tmp_path):
    s = Engram(str(tmp_path / "g.db"), probe_fn=lambda: (32 * GIB, 20 * GIB))
    assert s.profile.tier is Tier.FULL
    assert s.tick().tier is Tier.FULL
    s.close()
