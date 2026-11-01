import pytest

from elastimem import ElastimemConfig, Tier, governor, tiers
from elastimem.governor import GIB, Governor


@pytest.mark.parametrize("total_gib, pressure, tier", [
    (32, "normal", Tier.FULL),
    (16, "normal", Tier.FULL),
    (8, "normal", Tier.STANDARD),
    (8, "warn", Tier.STANDARD),       # this Mac: used to be LITE in 0.2
    (32, "warn", Tier.STANDARD),
    (32, "critical", Tier.LITE),
    (4, "normal", Tier.LITE),
])
def test_tier_from_capacity_and_pressure(total_gib, pressure, tier):
    assert Governor(ElastimemConfig(), probe_fn=lambda: (total_gib * GIB, pressure)).tier is tier


def test_fixed_tier_ignores_hardware():
    g = Governor(ElastimemConfig(tier=Tier.LITE), probe_fn=lambda: (64 * GIB, "normal"))
    g.observe(10_000)
    g.report_pressure()
    assert g.tier is Tier.LITE


def test_slow_turns_step_down_then_recover(monkeypatch):
    monkeypatch.setattr(tiers, "PROBE_SECONDS", 0)
    g = Governor(ElastimemConfig(turn_budget_ms=50), probe_fn=lambda: (32 * GIB, "normal"))
    for _ in range(tiers.SLOW_TURNS - 1):
        g.observe(200)
    assert g.tier is Tier.FULL                  # one bad stretch is not enough
    g.observe(200)
    assert g.tier is Tier.STANDARD and "ms per turn" in g.reason
    for _ in range(tiers.UPGRADE_TURNS):
        g.observe(5)
    assert g.check() is Tier.FULL


def test_pressure_drops_immediately(monkeypatch):
    monkeypatch.setattr(tiers, "PROBE_SECONDS", 0)
    readings = iter([(32 * GIB, "normal"), (32 * GIB, "critical")])
    g = Governor(ElastimemConfig(), probe_fn=lambda: next(readings))
    assert g.check() is Tier.LITE


def test_pressure_reports_in_a_burst_count_once():
    g = Governor(ElastimemConfig(), probe_fn=lambda: (32 * GIB, "normal"))
    g.report_pressure()
    g.report_pressure()
    assert g.tier is Tier.STANDARD


def test_env_override(monkeypatch):
    monkeypatch.setenv("ELASTIMEM_TIER", "lite")
    assert ElastimemConfig().tier is Tier.LITE


def test_probe_reads_this_machine():
    total, pressure = governor.probe()
    assert total > GIB and pressure in {"normal", "warn", "critical"}


def test_budgets_are_the_same_on_every_tier():
    b = governor.budgets(ElastimemConfig(context_tokens=8192, reserved_tokens=1024))
    assert b.working + b.facts + b.moments + b.topics + b.sessions + b.lessons <= 8192 - 1024
    assert governor.keep_turns(b) >= tiers.MIN_KEEP_TURNS
