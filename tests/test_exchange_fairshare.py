"""HEROSIM_EXCHANGE_FAIRSHARE: peer-exchange flows that overlap in time share each link, processor sharing.

Off by default (unset or '0'; other values raise) and byte-identical when off: no engine, no summary block, the input stage unchanged.
On, a flow's rate is the minimum over its route's directed links of (capacity / flows active there), recomputed when a flow starts or
ends; a flow alone costs what it cost before; n equal flows on one link take n times as long; flows on disjoint links do not interact;
the end of a flow speeds up the others. It composes with HEROSIM_EXCHANGE_TCP (start-up per transfer, only the remaining bytes share).
Validation against the filed shaped-link check: scripts_cosim/validation/fairshare_netns_check.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import simpy

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from src.placement.infrastructure import Platform  # noqa: E402
from src.placement.network_fabric import (  # noqa: E402
    FairShare, NetworkFabric, exchange_fairshare_enabled, exchange_fairshare_partners, tcp_startup_delay,
    tcp_transfer,
)
from test_peer_exchange_cost import FakeNode, FakeOrchestrator, FakeTask  # noqa: E402

MiB = 1024 ** 2


def run_flows(specs):
    """specs: [(start_s, hops, nbytes, scale)] -> finish times (the event time of each flow's completion)."""
    env = simpy.Environment()
    fs = FairShare(env)
    done = [None] * len(specs)

    def one(i, start, hops, nbytes, scale):
        yield env.timeout(start)
        yield fs.start(hops, nbytes, scale)
        done[i] = env.now

    for i, (start, hops, nbytes, scale) in enumerate(specs):
        env.process(one(i, start, hops, nbytes, scale))
    env.run()
    return done, fs


L1 = [("a>b", 10.0)]


def test_one_flow_is_size_over_capacity():
    (t,), _ = run_flows([(0, L1, 100 * MiB, 1.0)])
    assert t == pytest.approx(10.0, rel=0, abs=1e-9)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_n_equal_flows_on_one_link_take_n_times(n):
    ts, fs = run_flows([(0, L1, 100 * MiB, 1.0)] * n)
    assert ts == pytest.approx([10.0 * n] * n, rel=0, abs=1e-9) and fs.peak_flows_on_a_link == n       # A1 R1: n x the single-flow time within 1e-9


def test_flows_on_disjoint_links_do_not_interact():
    ts, fs = run_flows([(0, [("a>b", 10.0)], 100 * MiB, 1.0), (0, [("c>d", 10.0)], 100 * MiB, 1.0)])
    assert ts == pytest.approx([10.0, 10.0]) and fs.peak_flows_on_a_link == 1


def test_the_end_of_a_flow_speeds_up_the_others():
    # 10 MiB and 30 MiB on a 10 MB/s link: the short one ends at 2 s (5 MB/s each), the long one has 20 left and runs alone: 4 s total
    ts, _ = run_flows([(0, L1, 10 * MiB, 1.0), (0, L1, 30 * MiB, 1.0)])
    assert ts == pytest.approx([2.0, 4.0])


def test_a_flow_that_starts_later_slows_the_running_one():
    # A: 100 MiB from 0; B: 100 MiB from 5. At 5 s A has 50 left; shared 5 MB/s each until A ends at 15 s; B has 50 left at 15, alone: 20 s
    ts, _ = run_flows([(0, L1, 100 * MiB, 1.0), (5, L1, 100 * MiB, 1.0)])
    assert ts == pytest.approx([15.0, 20.0])


def test_rate_is_the_minimum_over_the_route_of_capacity_over_flows():
    # F crosses l1 (cap 2) and l2 (cap 10); G only l2. F: min(2/1, 10/2) = 2 MB/s; G: 10/2 = 5 MB/s (the specified rule, not max-min)
    ts, _ = run_flows([(0, [("l1", 2.0), ("l2", 10.0)], 10 * MiB, 1.0), (0, [("l2", 10.0)], 10 * MiB, 1.0)])
    assert ts[0] == pytest.approx(5.0, rel=0, abs=1e-9) and ts[1] == pytest.approx(2.0, rel=0, abs=1e-9)


def test_opposite_directions_are_separate_capacities():
    ts, _ = run_flows([(0, [("a>b", 10.0)], 100 * MiB, 1.0), (0, [("b>a", 10.0)], 100 * MiB, 1.0)])
    assert ts == pytest.approx([10.0, 10.0])


def test_goodput_scale_and_determinism():
    (t,), _ = run_flows([(0, L1, 100 * MiB, 0.5)])
    assert t == pytest.approx(20.0)
    specs = [(0, L1, 7 * MiB, 1.0), (1.3, L1, 11 * MiB, 1.0), (2.1, [("x", 3.0), ("a>b", 10.0)], 5 * MiB, 0.9)]
    assert run_flows(specs)[0] == run_flows(specs)[0]


def test_tcp_startup_delay_is_exact_nonnegative_and_additive():
    # A1 amendment "ramp as an additive start-up delay; all bytes shared": delay + size / bottleneck == tcp_transfer, a lone flow pays exactly that
    for bn in (4.0, 7.2, 13.5, 75.0, 117.0, 1000.0):
        for lat in (0.0005, 0.01, 0.054, 0.09):
            for size in (0.01 * MiB, 0.5 * MiB, 4 * MiB, 50 * MiB, 500 * MiB):
                d = tcp_startup_delay(size, bn, lat)
                assert d >= 0 and d + size / (bn * MiB) == pytest.approx(tcp_transfer(size, bn, lat), abs=1e-9)


@pytest.mark.parametrize("bad", ["", "2", "yes", "true", "-1"])
def test_bad_values_raise(monkeypatch, bad):
    monkeypatch.setenv("HEROSIM_EXCHANGE_FAIRSHARE", bad)
    with pytest.raises(ValueError):
        exchange_fairshare_enabled()
    monkeypatch.delenv("HEROSIM_EXCHANGE_FAIRSHARE")
    monkeypatch.setenv("HEROSIM_EXCHANGE_FAIRSHARE_PARTNERS", bad or "nope")
    with pytest.raises(ValueError):
        exchange_fairshare_partners()


def test_unset_and_zero_are_off_and_partners_default_serial(monkeypatch):
    monkeypatch.delenv("HEROSIM_EXCHANGE_FAIRSHARE", raising=False)
    monkeypatch.delenv("HEROSIM_EXCHANGE_FAIRSHARE_PARTNERS", raising=False)
    assert exchange_fairshare_enabled() is False and exchange_fairshare_partners() == "serial"
    monkeypatch.setenv("HEROSIM_EXCHANGE_FAIRSHARE", "0")
    assert exchange_fairshare_enabled() is False
    monkeypatch.setenv("HEROSIM_EXCHANGE_FAIRSHARE_PARTNERS", "concurrent")
    assert exchange_fairshare_partners() == "concurrent"


# ------------------------------------------------------------------ platform level
@pytest.fixture
def peer_exchange_on(monkeypatch):
    monkeypatch.setenv("HEROSIM_PEER_EXCHANGE", "1")
    monkeypatch.setenv("HEROSIM_TRANSFER_MODEL", "pipelined")
    monkeypatch.delenv("HEROSIM_EXCHANGE_TCP", raising=False)
    monkeypatch.delenv("HEROSIM_EXCHANGE_FAIRSHARE_PARTNERS", raising=False)


def world(monkeypatch, payloads, fairshare=True, caps=(10.0, 10.0, 10.0), lat=0.05):
    """Task 0 runs on node `a`; partner i runs on node `p{i}`; every node hangs off core0 by its own link. All partners enter `a`
    through the one link core0>a."""
    if fairshare:
        monkeypatch.setenv("HEROSIM_EXCHANGE_FAIRSHARE", "1")
    else:
        monkeypatch.delenv("HEROSIM_EXCHANGE_FAIRSHARE", raising=False)
    env = simpy.Environment()
    links = {"a|core0": {"latency": 0.01, "bandwidth_mbps": caps[0]}}
    routes = {}
    for i in range(len(payloads)):
        links[f"core0|p{i}"] = {"latency": 0.01, "bandwidth_mbps": caps[1]}
        routes[f"p{i}"] = {"a": [f"p{i}", "core0", "a"]}
    fabric = NetworkFabric(env, {"links": links, "routes": routes,
                                 "access_classes": {n: {"class": "wired"} for n in ["a"] + [f"p{i}" for i in range(len(payloads))]}})
    from src.placement.orchestrator import build_peer_exchange_table
    tasks = {}
    orch = FakeOrchestrator(build_peer_exchange_table([[0, i + 1, p] for i, p in enumerate(payloads)]), tasks)
    a = FakeNode("a", {f"p{i}": lat for i in range(len(payloads))}, orch)
    a.fabric = fabric
    tasks[0] = FakeTask(0, a)
    for i in range(len(payloads)):
        n = FakeNode(f"p{i}", {"a": lat}, orch)
        n.fabric = fabric
        tasks[i + 1] = FakeTask(i + 1, n)
    platform = Platform.__new__(Platform)
    platform.node, platform.env = a, env
    return env, fabric, platform, tasks[0]


def exchange(env, platform, task):
    flows = Platform._fairshare_plan(platform, task)
    proc = env.process(Platform._fairshare_exchange(platform, task, flows))
    env.run(proc)
    return proc.value, flows


def test_one_flow_costs_today_s_charge(peer_exchange_on, monkeypatch):
    payload = 50 * MiB
    env, fabric, platform, t = world(monkeypatch, [payload])
    got, flows = exchange(env, platform, t)
    env2, f2, p2, t2 = world(monkeypatch, [payload], fairshare=False)
    assert got == pytest.approx(Platform._peer_exchange_time(p2, t2)) == pytest.approx(payload / (10 * MiB) + 0.05)
    assert flows[0]["solo"] == pytest.approx(got) and fabric.exchange_fairshare["extra_seconds"] == pytest.approx(0.0)


def test_n_equal_flows_into_one_link_take_n_times_plus_latency(peer_exchange_on, monkeypatch):
    monkeypatch.setenv("HEROSIM_EXCHANGE_FAIRSHARE_PARTNERS", "concurrent")
    for n in (2, 4):
        env, fabric, platform, t = world(monkeypatch, [50 * MiB] * n)
        got, _ = exchange(env, platform, t)
        assert got == pytest.approx(n * 5.0 + 0.05) and fabric.exchange_fairshare["peak_flows_on_a_link"] == n
        assert fabric.exchange_fairshare["transfers"] == n and fabric.exchange_fairshare["shared_transfers"] == n


def test_partners_serial_by_default_and_concurrent_on_request(peer_exchange_on, monkeypatch):
    # wide receiving link: two partners do not interact. Serial (the default, SCIENTIST's A1 ruling): the sum of both charges;
    # concurrent: the slower one
    env, fabric, platform, t = world(monkeypatch, [50 * MiB, 100 * MiB], caps=(100.0, 10.0))
    got, _ = exchange(env, platform, t)
    assert got == pytest.approx(5.05 + 10.05) and fabric.exchange_fairshare["partners"] == "serial"
    assert fabric.exchange_fairshare["shared_transfers"] == 0
    # serial partners into one narrow link do not share it with each other: each gets the whole link in turn
    env, fabric, platform, t = world(monkeypatch, [50 * MiB, 50 * MiB])
    got, _ = exchange(env, platform, t)
    assert got == pytest.approx(2 * 5.05) and fabric.exchange_fairshare["peak_flows_on_a_link"] == 1
    monkeypatch.setenv("HEROSIM_EXCHANGE_FAIRSHARE_PARTNERS", "concurrent")
    env, fabric, platform, t = world(monkeypatch, [50 * MiB, 100 * MiB], caps=(100.0, 10.0))
    got, _ = exchange(env, platform, t)
    assert got == pytest.approx(10.0 + 0.05) and fabric.exchange_fairshare["partners"] == "concurrent"


def test_flows_of_different_tasks_share_a_link_under_serial(peer_exchange_on, monkeypatch):
    # two tasks on node a, each fetching one partner over the one link into a: their flows overlap in time and share it
    env, fabric, platform, t0 = world(monkeypatch, [50 * MiB, 50 * MiB])
    orch = platform.node.orchestrator_ref
    orch.peer_exchange = {0: {1: 50 * MiB}, 7: {2: 50 * MiB}}
    orch.task_by_id[7] = FakeTask(7, platform.node)
    f0 = Platform._fairshare_plan(platform, t0)
    f7 = Platform._fairshare_plan(platform, orch.task_by_id[7])
    p0 = env.process(Platform._fairshare_exchange(platform, t0, f0))
    p7 = env.process(Platform._fairshare_exchange(platform, orch.task_by_id[7], f7))
    env.run(env.all_of([p0, p7]))
    assert p0.value == pytest.approx(10.05) and p7.value == pytest.approx(10.05)
    assert fabric.exchange_fairshare["peak_flows_on_a_link"] == 2 and fabric.exchange_fairshare["shared_transfers"] == 2


def test_composes_with_tcp_as_an_additive_startup_delay_and_all_bytes_shared(peer_exchange_on, monkeypatch):
    payload, lat = 50 * MiB, 0.05
    monkeypatch.setenv("HEROSIM_EXCHANGE_TCP", "1")
    env, fabric, platform, t = world(monkeypatch, [payload])
    got_tcp, flows = exchange(env, platform, t)
    assert got_tcp == pytest.approx(tcp_transfer(payload, 10.0, lat) + lat)   # one flow = the TCP charge
    assert flows[0]["b"] == pytest.approx(tcp_startup_delay(payload, 10.0, lat)) and flows[0]["rest"] == payload
    env, fabric, platform, t = world(monkeypatch, [payload] * 2)
    got_serial, _ = exchange(env, platform, t)
    assert got_serial == pytest.approx(2 * got_tcp)                           # serial partners: one after another, nothing shared
    monkeypatch.setenv("HEROSIM_EXCHANGE_FAIRSHARE_PARTNERS", "concurrent")
    env, fabric, platform, t = world(monkeypatch, [payload] * 2)
    got2, _ = exchange(env, platform, t)
    d = tcp_startup_delay(payload, 10.0, lat)
    assert got2 == pytest.approx(d + 2 * payload / (10 * MiB) + lat)           # delay once, then ALL bytes shared
    assert fabric.exchange_tcp["transfers"] == 2
    # A1 R1 (SCIENTIST): two simultaneous equal flows under TCP + fair share take at least as long as TCP alone, and at least as long as
    # fair share alone
    monkeypatch.delenv("HEROSIM_EXCHANGE_TCP")
    env, fabric, platform, t = world(monkeypatch, [payload] * 2)
    got_fs_only, _ = exchange(env, platform, t)
    assert got2 >= got_tcp and got2 >= got_fs_only - 1e-9 and got_fs_only == pytest.approx(2 * payload / (10 * MiB) + lat)


def test_co_located_partner_is_free_and_absent(peer_exchange_on, monkeypatch):
    env, fabric, platform, t = world(monkeypatch, [50 * MiB, 50 * MiB])
    peer = platform.node.orchestrator_ref.task_by_id[1]
    peer.platform = type("P", (), {"node": platform.node})()
    flows = Platform._fairshare_plan(platform, t)
    assert [f["peer_id"] for f in flows] == [2]


def test_off_is_inert(peer_exchange_on, monkeypatch):
    from src.placement.orchestrator import Orchestrator
    env, fabric, platform, t = world(monkeypatch, [50 * MiB], fairshare=False)
    assert fabric.fairshare is None and fabric.exchange_fairshare is None
    holder = type("H", (), {"nodes": type("N", (), {"items": [platform.node]})()})()
    assert "peerExchangeFairshare" not in Orchestrator._access_class_fields(holder)
    with pytest.raises(RuntimeError, match="fabric of the run"):
        Platform._fairshare_plan(platform, t)
    env, fabric, platform, t = world(monkeypatch, [50 * MiB], fairshare=True)
    holder = type("H", (), {"nodes": type("N", (), {"items": [platform.node]})()})()
    assert Orchestrator._access_class_fields(holder)["peerExchangeFairshare"]["model"] == "processor_sharing_v1"


def test_store_forward_model_is_refused(peer_exchange_on, monkeypatch):
    env, fabric, platform, t = world(monkeypatch, [50 * MiB])
    monkeypatch.setenv("HEROSIM_TRANSFER_MODEL", "store_forward")
    with pytest.raises(RuntimeError, match="pipelined"):
        Platform._fairshare_plan(platform, t)


def test_gate_whitelists_and_checks_the_block():
    sys.path.insert(0, str(REPO_ROOT / "scripts_cosim"))
    import fresh_topo_burst_v1_gate as G
    assert "peerExchangeFairshare" in G.W3_TELEMETRY_KEYS
    on = {"peerExchangeFairshare": {"partners": "serial"}, "env": {}}
    off = {"env": {}}
    f = G.exchange_fairshare_problems
    assert f(on, "1", None) == [] and f(on, "1", "serial") == [] and f(off, None, None) == [] and f(off, "0", None) == []
    assert f(off, "1", None) and f(on, None, None) and f(on, "0", None)
    assert f(on, "1", "concurrent") and f({"peerExchangeFairshare": {"partners": "concurrent"}}, "1", "concurrent") == []
    assert f({"env": {"HEROSIM_EXCHANGE_FAIRSHARE": "1"}}, "0", None)


def test_sbatch_passes_the_switch_through_the_wipe():
    text = (REPO_ROOT / "scripts_cosim/datalab/r1a_gate.sbatch").read_text()
    assert 'export HEROSIM_EXCHANGE_FAIRSHARE="$R1A_EXCHANGE_FAIRSHARE"' in text
    assert 'export HEROSIM_EXCHANGE_FAIRSHARE_PARTNERS="$R1A_EXCHANGE_FAIRSHARE_PARTNERS"' in text


def test_netns_validation_n2_and_n4():
    sys.path.insert(0, str(REPO_ROOT / "scripts_cosim/validation"))
    import statistics
    import fairshare_netns_check as V
    qs = [r for r in V.rows() if r["quotable"]]
    assert len(qs) == 16 and {r["n"] for r in qs} == {2, 4}
    e = [abs(r["tcp_err"]) for r in qs]
    assert statistics.median(e) <= 0.03 and max(e) <= 0.15 and sum(x <= 0.10 for x in e) == 15   # filed result: median 2.4 %, max 13.9 %, 15 of 16 within 10 %
    assert all(r["plain_err"] < 0 for r in qs)                    # without the start-up the model is faster than the measured flows


def test_a_flow_with_a_residue_below_the_clock_resolution_ends_instead_of_looping():
    """Late in a run (now ~ 4781 s, clock step ~ 9e-13 s) a flow whose remaining time is below half a clock step would wake at the
    same instant, with nothing advanced and the flow still above the 1e-9 byte threshold, and be rescheduled at that instant forever."""
    env = simpy.Environment()
    env.run(until=4781.135774386989)
    fs = FairShare(env)
    ev = fs.start([("a", 100.0)], 1000.0)
    fs.flows[0].rem = 3e-5  # 3e-5 bytes at 100 MiB/s is 2.9e-13 s, below half the clock step; the byte threshold is 1e-6
    fs._reschedule()  # the wake-up scheduled at start() is now stale
    t0 = env.now
    for _ in range(1000):
        if ev.processed or not env._queue:
            break
        env.step()
    assert ev.processed, "the flow never ended: zero-time reschedule loop"
    assert env.now - t0 < 1e-9
    assert fs.flows == []
