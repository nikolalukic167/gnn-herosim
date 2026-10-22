"""HEROSIM_PG_FAIRSHARE_AWARE=1: the hand rule prices an exchange at the live fair-share rate (src/placement/fairshare_aware.py).

Priced cost with n active flows on the only link is (n+1) times the solo cost; with no active flow the price is bit-identical to the bar's;
the switch without HEROSIM_EXCHANGE_FAIRSHARE=1 raises; the gate carries the kind with the bar's checks plus the switch.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))
sys.path.insert(0, str(REPO_ROOT / "scripts_cosim"))

from src.placement import fairshare_aware as FA  # noqa: E402
from src.placement.infrastructure import Platform  # noqa: E402
from src.policy.peer_greedy_network.scheduler import PeerGreedyNetworkScheduler  # noqa: E402
from test_exchange_fairshare import world  # noqa: E402

MiB = 1024 ** 2
KIND = "cd_ib0_inflightfull_fsaware"


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.setenv("HEROSIM_PEER_EXCHANGE", "1")
    monkeypatch.setenv("HEROSIM_TRANSFER_MODEL", "pipelined")
    for k in ("HEROSIM_EXCHANGE_TCP", "HEROSIM_EXCHANGE_FAIRSHARE_PARTNERS", FA.AWARE_ENV):
        monkeypatch.delenv(k, raising=False)
    FA.reset()
    yield
    FA.reset()


def aware(monkeypatch):
    monkeypatch.setenv(FA.AWARE_ENV, "1")
    FA.reset()


def occupy(fabric, n):
    """n flows from p0 to a, huge enough to stay active through the pricing."""
    hops = fabric.directed_hops("p0", "a")
    for _ in range(n):
        fabric.fairshare.start(hops, 1e15)


def test_n_active_flows_on_the_only_link_price_n_plus_one_times_the_solo_cost(monkeypatch):
    aware(monkeypatch)
    env, fabric, platform, _t = world(monkeypatch, [50 * MiB])
    solo = platform._payload_transfer_time("p0", 50 * MiB)
    assert solo == pytest.approx(5.0)
    for n in (1, 2, 5):
        env2, fabric2, platform2, _ = world(monkeypatch, [50 * MiB])
        occupy(fabric2, n)
        assert FA.payload_seconds(platform2, "p0", 50 * MiB) == pytest.approx((n + 1) * solo)
    assert FA.STATS["nonzero"] == 3 and FA.STATS["priced"] == 3


def test_the_rule_prices_with_the_shared_rate_through_the_scheduler_estimate(monkeypatch):
    aware(monkeypatch)
    env, fabric, platform, _t = world(monkeypatch, [50 * MiB])
    occupy(fabric, 2)
    node_a = platform.node
    got = PeerGreedyNetworkScheduler._pg_exchange_seconds(node_a, platform, [("p0", 50 * MiB)])
    assert got == pytest.approx(3 * 5.0 + platform.peer_link_latency("p0"))


def test_with_zero_active_flows_the_price_is_bit_identical_to_the_bar(monkeypatch):
    env, fabric, platform, _t = world(monkeypatch, [50 * MiB, 7 * MiB])
    bar = [PeerGreedyNetworkScheduler._pg_exchange_seconds(platform.node, platform, [("p0", 50 * MiB), ("p1", 7 * MiB)]),
           platform._payload_transfer_time("p1", 7 * MiB)]
    aware(monkeypatch)
    again = [PeerGreedyNetworkScheduler._pg_exchange_seconds(platform.node, platform, [("p0", 50 * MiB), ("p1", 7 * MiB)]),
             FA.payload_seconds(platform, "p1", 7 * MiB)]
    assert again == bar and FA.STATS["nonzero"] == 0 and FA.STATS["priced"] == 3


def test_with_the_switch_off_active_flows_do_not_change_the_price(monkeypatch):
    env, fabric, platform, _t = world(monkeypatch, [50 * MiB])
    occupy(fabric, 3)
    assert FA.payload_seconds(platform, "p0", 50 * MiB) == platform._payload_transfer_time("p0", 50 * MiB)
    assert FA.STATS == {"priced": 0, "nonzero": 0}


def test_the_switch_without_fairshare_raises(monkeypatch):
    monkeypatch.setenv(FA.AWARE_ENV, "1")
    monkeypatch.delenv("HEROSIM_EXCHANGE_FAIRSHARE", raising=False)
    FA.reset()
    with pytest.raises(RuntimeError, match="HEROSIM_EXCHANGE_FAIRSHARE"):
        FA.fairshare_aware_enabled()
    monkeypatch.setenv("HEROSIM_EXCHANGE_FAIRSHARE", "0")
    FA.reset()
    with pytest.raises(RuntimeError, match="HEROSIM_EXCHANGE_FAIRSHARE"):
        FA.fairshare_aware_enabled()
    monkeypatch.setenv(FA.AWARE_ENV, "2")
    FA.reset()
    with pytest.raises(ValueError):
        FA.fairshare_aware_enabled()


def test_the_scheduler_refuses_the_switch_without_fairshare(monkeypatch):
    monkeypatch.setenv(FA.AWARE_ENV, "1")
    monkeypatch.delenv("HEROSIM_EXCHANGE_FAIRSHARE", raising=False)
    sched = PeerGreedyNetworkScheduler.__new__(PeerGreedyNetworkScheduler)
    with pytest.raises(RuntimeError, match="HEROSIM_EXCHANGE_FAIRSHARE"):
        sched._pg_init()


def test_the_drain_walk_prices_queued_peer_exchange_at_the_live_rate(monkeypatch):
    from types import SimpleNamespace
    from src.placement.live_audit import platform_queue_drain_seconds
    env, fabric, platform, task = world(monkeypatch, [50 * MiB])
    platform.type = {"shortName": "t0"}
    platform.id = 0
    platform.queue = SimpleNamespace(items=[task])
    platform.node.network_map = {"p0": {"latency": 0.01}}
    task.type = {"executionTime": {"t0": 0.0}, "stateSize": {}}
    task.id = 0
    task.node_name = None
    orch = platform.node.orchestrator_ref
    solo = platform_queue_drain_seconds(platform, orch, None)
    aware(monkeypatch)
    assert platform_queue_drain_seconds(platform, orch, None) == solo
    occupy(fabric, 1)
    shared = platform_queue_drain_seconds(platform, orch, None)
    assert shared == pytest.approx(solo + 5.0)


def test_the_gate_carries_the_kind_with_the_bars_checks_and_the_switch():
    import fresh_topo_burst_v1_gate as G
    k = KIND
    assert G.RULE_POLICY[k] == G.RULE_POLICY["cd_ib0_inflightfull"] == "peer_greedy_network_cd"
    assert G.ib0_alpha(k) == G.ib0_alpha("cd_ib0_inflightfull") and G.check_alias(k) == "cd"
    assert k in G.R1A_DIAG and k in G.R1A_ARMS and k not in G.R1A_CLASSICAL
    assert G.split_dc(k + "_k100") == (k, 1.0)
    assert "pgFairshareAware" in G.W3_TELEMETRY_KEYS
    src = (REPO_ROOT / "scripts_cosim/fresh_topo_burst_v1_gate.py").read_text()
    assert 'env["HEROSIM_PG_FAIRSHARE_AWARE"] = "1"' in src and '"HEROSIM_KEEP_ALIVE", "HEROSIM_PG_FAIRSHARE_AWARE"' in src
    assert '"HEROSIM_PG_FAIRSHARE_AWARE",' in (REPO_ROOT / "src/executesimulation.py").read_text()


def test_the_decision_cost_of_the_kind_is_the_bars_frozen_w_and_an_unknown_kind_still_fails():
    import json
    import fresh_topo_burst_v1_gate as G
    f = json.load(open(G.DC_W_FILE))
    assert f["w_ms_per_plan_alias"][KIND] == "cd_ib0_inflightfull" and "_source_w_ms_per_plan_alias" in f
    for rung in ("moderate", "heavy"):
        assert G.dc_cost_ms(KIND, 1.0, rung) == G.dc_cost_ms("cd_ib0_inflightfull", 1.0, rung) == (G.DC_ENV, repr(f["w_ms_per_plan"]["cd_ib0_inflightfull"][rung]))
    assert G.dc_cost_ms(KIND, 2.0, "moderate") == (G.DC_ENV, repr(2.0 * 0.0116))
    assert G.dc_cost_ms(KIND, None, "moderate") is None
    with pytest.raises(SystemExit, match="no calibrated w"):
        G.dc_cost_ms("cd_ib0_inflightfull_notakind", 1.0, "moderate")
