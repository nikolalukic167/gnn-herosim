"""HEROSIM_EXCHANGE_TCP: the peer-exchange transfer as one TCP connection (framing, handshake, slow start, goodput).

Off by default and bit-identical when unset or '0'; bad values raise. On, each non-co-located peer costs
`tcp_transfer(size, bottleneck, lat) + lat` (one handshake per partner, summed serially), co-located peers stay free, and the
fabric books `exchange_tcp` (the summary key `peerExchangeTcp`, present only when on).
R0: the implemented function reproduces the 20 n = 1 medians of the filed shaped-link check
(docs/lineages/uncut_groups_s0_v1/netns_check/netns_result.json, md5 9de512ea1f91f292edc86f56420fb0b6) within 0.15 s on at least
18 and 0.3 s on all 20, wired rows included (a check of the code against the measurements, not a quotable result).
"""
from __future__ import annotations

import hashlib
import json
import statistics
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.placement.network_fabric import exchange_tcp_enabled, tcp_transfer  # noqa: E402
sys.path.insert(0, str(REPO_ROOT / "tests"))
from test_peer_exchange_cost import (  # noqa: E402
    BANDWIDTH_MBPS, FakeNode, FakeOrchestrator, FakeTask, expected,
)


def cost(node, task, record=True):
    from simpy import Environment
    from src.placement.infrastructure import Platform
    platform = Platform.__new__(Platform)
    platform.node = node
    platform.env = Environment()
    return Platform._peer_exchange_time(platform, task, record=record)

NETNS_JSON = REPO_ROOT / "docs/lineages/uncut_groups_s0_v1/netns_check/netns_result.json"
NETNS_MD5 = "9de512ea1f91f292edc86f56420fb0b6"


@pytest.fixture
def peer_exchange_on(monkeypatch):
    monkeypatch.setenv("HEROSIM_PEER_EXCHANGE", "1")
    monkeypatch.setenv("HEROSIM_TRANSFER_MODEL", "pipelined")


def world(payloads, latencies):
    """Task 0 on node a; peers 1.. each on its own remote node; returns (node a, task 0)."""
    table = {0: {i + 1: p for i, p in enumerate(payloads)}}
    nodes, tasks = {}, {}
    orch = FakeOrchestrator(table, tasks)
    a = FakeNode("a", {f"n{i + 1}": lat for i, lat in enumerate(latencies)}, orch)
    tasks[0] = FakeTask(0, a)
    for i in range(len(payloads)):
        n = FakeNode(f"n{i + 1}", {"a": latencies[i]}, orch)
        nodes[i] = n
        tasks[i + 1] = FakeTask(i + 1, n)
    return a, tasks[0], tasks


def test_off_unset_and_zero_are_identical_and_plain(peer_exchange_on, monkeypatch):
    a, t, _ = world([50e6], [0.05])
    monkeypatch.delenv("HEROSIM_EXCHANGE_TCP", raising=False)
    unset = cost(a, t)
    monkeypatch.setenv("HEROSIM_EXCHANGE_TCP", "0")
    assert cost(a, t) == unset == expected(50e6, 0.05)


@pytest.mark.parametrize("bad", ["", "2", "true", "on", "-1"])
def test_bad_values_raise(monkeypatch, bad):
    monkeypatch.setenv("HEROSIM_EXCHANGE_TCP", bad)
    with pytest.raises(ValueError):
        exchange_tcp_enabled()


def test_on_charges_tcp_per_remote_partner_and_colocated_is_free(peer_exchange_on, monkeypatch):
    monkeypatch.setenv("HEROSIM_EXCHANGE_TCP", "1")
    a, t, tasks = world([50e6, 4e6], [0.05, 0.03])
    one = tcp_transfer(50e6, BANDWIDTH_MBPS, 0.05) + 0.05
    two = tcp_transfer(4e6, BANDWIDTH_MBPS, 0.03) + 0.03
    assert cost(a, t) == pytest.approx(one + two, rel=1e-12)
    assert one > expected(50e6, 0.05) and two > expected(4e6, 0.03)
    tasks[2].platform.node = a   # the second partner co-located with task 0
    assert cost(a, t) == pytest.approx(one, rel=1e-12)


def test_tcp_transfer_shape():
    # whole payload inside slow start: handshake + rounds - 1 RTT; large payload: handshake + rounds + remainder at goodput
    lat, rate = 0.05, 117.0
    g = rate * 1024 ** 2 / (1514 / 1448)
    assert tcp_transfer(1, rate, lat) == pytest.approx(0.1)               # one segment: t = rtt, loop adds rtt, minus rtt
    big = tcp_transfer(500 * 1024 ** 2, rate, lat)
    assert big > 500 * 1024 ** 2 / g
    assert tcp_transfer(2e6, rate, lat) < tcp_transfer(4e6, rate, lat)
    with pytest.raises(ValueError):
        tcp_transfer(1e6, 0.0, lat)


def test_r0_reproduces_the_filed_n1_medians():
    if not NETNS_JSON.exists():
        pytest.fail(f"{NETNS_JSON} missing: the filed shaped-link result is the R0 reference")
    assert hashlib.md5(NETNS_JSON.read_bytes()).hexdigest() == NETNS_MD5
    d = json.loads(NETNS_JSON.read_text())
    cases = {c["case"]: c for c in d["cases"]}
    errs = []
    for case, c in cases.items():
        for size in (4, 50, 250, 500):
            runs = [r["completions_s"][0] for r in d["runs"] if r["case"] == case and r["size_mb"] == size and r["flows"] == 1]
            assert len(runs) == 3
            model = tcp_transfer(size * 1024 ** 2, c["bottleneck_mbs"], c["latency_s"]) + c["latency_s"]
            errs.append(model - statistics.median(runs))
    assert len(errs) == 20
    assert sum(abs(e) <= 0.15 for e in errs) >= 18, [round(e, 3) for e in errs]
    assert max(abs(e) for e in errs) <= 0.30, [round(e, 3) for e in errs]


def _fabric_world():
    import simpy
    from src.placement.network_fabric import NetworkFabric
    from src.placement.orchestrator import build_peer_exchange_table

    topo = {
        "links": {"client_node0|core0": {"latency": 0.02, "bandwidth_mbps": 8.0},
                  "core0|core1": {"latency": 0.004, "bandwidth_mbps": 1000.0},
                  "client_node1|core1": {"latency": 0.02, "bandwidth_mbps": 1000.0}},
        "routes": {"client_node0": {"client_node1": ["client_node0", "core0", "core1", "client_node1"]}},
        "access_classes": {"client_node0": {"class": "wifi"}, "client_node1": {"class": "wired"}},
    }
    fabric = NetworkFabric(simpy.Environment(), topo)
    tasks = {}
    orch = FakeOrchestrator(build_peer_exchange_table([[0, 1, 5e6]]), tasks)
    a, b = FakeNode("client_node0", {}, orch), FakeNode("client_node1", {}, orch)
    a.fabric = b.fabric = fabric
    tasks[0], tasks[1] = FakeTask(0, a), FakeTask(1, b)
    return fabric, a, tasks[0]


def test_fabric_books_the_added_seconds_only_when_on(peer_exchange_on, monkeypatch):
    from src.placement.orchestrator import Orchestrator

    monkeypatch.delenv("HEROSIM_EXCHANGE_TCP", raising=False)
    fabric, a, t = _fabric_world()
    plain = cost(a, t)
    assert fabric.exchange_tcp is None
    holder = type("H", (), {"nodes": type("N", (), {"items": [a]})()})()
    assert "peerExchangeTcp" not in Orchestrator._access_class_fields(holder)

    monkeypatch.setenv("HEROSIM_EXCHANGE_TCP", "1")
    fabric, a, t = _fabric_world()
    on = cost(a, t)
    assert on == pytest.approx(tcp_transfer(5e6, 8.0, 0.044) + 0.044)
    assert fabric.exchange_tcp["transfers"] == 1
    assert fabric.exchange_tcp["extra_seconds"] == pytest.approx(on - plain)
    holder = type("H", (), {"nodes": type("N", (), {"items": [a]})()})()
    assert Orchestrator._access_class_fields(holder)["peerExchangeTcp"]["model"] == "slow_start_v1"
    assert cost(a, t, record=False) == pytest.approx(on) and fabric.exchange_tcp["transfers"] == 1   # record=False books nothing


def _gate():
    sys.path.insert(0, str(REPO_ROOT / "scripts_cosim"))
    import fresh_topo_burst_v1_gate as G
    return G


def test_gate_whitelists_the_block_and_checks_it_against_the_driver():
    G = _gate()
    assert "peerExchangeTcp" in G.W3_TELEMETRY_KEYS
    on = {"peerExchangeTcp": {"model": "slow_start_v1"}, "env": {"HEROSIM_EXCHANGE_TCP": "1"}}
    off = {"env": {}}
    assert G.exchange_tcp_problems(on, "1") == [] and G.exchange_tcp_problems(off, None) == [] and G.exchange_tcp_problems(off, "0") == []
    assert G.exchange_tcp_problems(off, "1")                                   # driven on, block missing
    assert G.exchange_tcp_problems({"peerExchangeTcp": {}, "env": {}}, "1")     # block without the switch in the env
    assert G.exchange_tcp_problems(on, None) and G.exchange_tcp_problems(on, "0")   # driven off, block present
