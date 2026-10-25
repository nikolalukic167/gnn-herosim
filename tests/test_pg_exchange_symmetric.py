"""HEROSIM_PG_EXCHANGE_SYMMETRIC=1: the rule's exchange term also charges each remote partner's pull from the candidate (n -> partner).

Off (the default) is the bar byte-for-byte; on adds, per known remote partner, the partner's own `_payload_transfer_time` from n plus its
`peer_link_latency`, the evaluations are counted, not metered (they sit inside the candidate's one plan eval). A cellular candidate (out 4 / in 75 MB/s) pays more than inbound alone; a wired pair adds
exactly its (symmetric) inbound time again.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import simpy

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from src.placement.infrastructure import Platform  # noqa: E402
from src.placement.network_fabric import NetworkFabric  # noqa: E402
from src.policy.peer_greedy_network.scheduler import PeerGreedyNetworkScheduler  # noqa: E402
from test_peer_exchange_cost import FakeNode, FakeOrchestrator  # noqa: E402

MiB = 1024 ** 2
PAYLOAD = 40 * MiB


@pytest.fixture(autouse=True)
def physics(monkeypatch):
    monkeypatch.setenv("HEROSIM_PEER_EXCHANGE", "1")
    monkeypatch.setenv("HEROSIM_TRANSFER_MODEL", "pipelined")
    for k in ("HEROSIM_EXCHANGE_TCP", "HEROSIM_EXCHANGE_FAIRSHARE", "HEROSIM_PG_FAIRSHARE_AWARE"):
        monkeypatch.delenv(k, raising=False)


def access(node, out, inn):
    return {"latency": 0.01, "access_node": node, "bandwidth_out_mbps": out, "bandwidth_in_mbps": inn, "bandwidth_mbps": min(out, inn)}


def world(rates):
    """Nodes `a` and `p0` hang off core0 by directional access links; `rates` = {node: (out, in)}."""
    env = simpy.Environment()
    links = {"a|core0": access("a", *rates["a"]), "core0|p0": access("p0", *rates["p0"])}
    routes = {"a": {"p0": ["a", "core0", "p0"]}, "p0": {"a": ["p0", "core0", "a"]}}
    fabric = NetworkFabric(env, {"links": links, "routes": routes, "access_classes": {n: {"class": "x"} for n in ("a", "p0")}})
    orch = FakeOrchestrator({}, {})
    nodes = {n: FakeNode(n, {m: 0.05 for m in ("a", "p0") if m != n}, orch) for n in ("a", "p0")}
    for n in nodes.values():
        n.fabric = fabric
    plats = {}
    for name, n in nodes.items():
        p = Platform.__new__(Platform)
        p.node, p.env = n, env
        plats[name] = p
    sched = PeerGreedyNetworkScheduler.__new__(PeerGreedyNetworkScheduler)
    sched.nodes = SimpleNamespace(items=list(nodes.values()))
    sched._pg_node_by_name = {}
    sched.pg_exchange_sym = False
    sched.pg_exchange_sym_evals = 0
    sched.sched_plan_evals = 0
    return sched, nodes, plats


def test_flag_off_scores_are_the_bars():
    sched, nodes, plats = world({"a": (4.0, 75.0), "p0": (117.0, 117.0)})
    peers = [("p0", PAYLOAD)]
    bar = PeerGreedyNetworkScheduler._pg_exchange_seconds(nodes["a"], plats["a"], peers)
    assert sched._pg_exch(nodes["a"], plats["a"], peers) == bar
    assert sched.sched_plan_evals == 0 and sched.pg_exchange_sym_evals == 0


def test_a_cellular_candidate_pays_exactly_the_reverse_transfer():
    sched, nodes, plats = world({"a": (4.0, 75.0), "p0": (117.0, 117.0)})
    peers = [("p0", PAYLOAD)]
    bar = PeerGreedyNetworkScheduler._pg_exchange_seconds(nodes["a"], plats["a"], peers)
    sched.pg_exchange_sym = True
    reverse = plats["p0"]._payload_transfer_time("a", PAYLOAD) + plats["p0"].peer_link_latency("a")
    got = sched._pg_exch(nodes["a"], plats["a"], peers)
    assert got == pytest.approx(bar + reverse)
    assert reverse > bar  # the 4 MB/s uplink, not the 75 MB/s downlink, bounds the reverse pull
    assert sched.sched_plan_evals == 0 and sched.pg_exchange_sym_evals == 1  # counted, not metered: inside the candidate's one plan eval


def test_wired_to_wired_adds_only_the_symmetric_term():
    sched, nodes, plats = world({"a": (117.0, 117.0), "p0": (117.0, 117.0)})
    peers = [("p0", PAYLOAD)]
    bar = PeerGreedyNetworkScheduler._pg_exchange_seconds(nodes["a"], plats["a"], peers)
    sched.pg_exchange_sym = True
    assert sched._pg_exch(nodes["a"], plats["a"], peers) == pytest.approx(2 * bar)


def test_a_colocated_or_absent_partner_adds_nothing():
    sched, nodes, plats = world({"a": (4.0, 75.0), "p0": (117.0, 117.0)})
    sched.pg_exchange_sym = True
    assert sched._pg_exch(nodes["a"], plats["a"], []) == 0.0
    assert sched._pg_exch(nodes["a"], plats["a"], [("a", PAYLOAD)]) == 0.0
    assert sched.sched_plan_evals == 0


@pytest.mark.parametrize("env", [{}, {"HEROSIM_PG_BATCH_ALPHA": "1"}, {"HEROSIM_PG_BATCH_ALPHA": "0", "HEROSIM_PG_CD_EXPANSION": "1"},
                                 {"HEROSIM_PG_BATCH_ALPHA": "0", "HEROSIM_PG_CD_EXACT": "1"}, {"HEROSIM_PG_BATCH_ALPHA": "0", "HEROSIM_PG_ILS_PER": "200"},
                                 {"HEROSIM_PG_BATCH_ALPHA": "0", "HEROSIM_PG_ORACLE": "1"}])
def test_the_flag_fails_loud_outside_ib0_without_search(monkeypatch, env):
    for k in ("HEROSIM_PG_BATCH_ALPHA", "HEROSIM_PG_CD_EXPANSION", "HEROSIM_PG_CD_EXACT", "HEROSIM_PG_ILS_PER", "HEROSIM_PG_ORACLE"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    with pytest.raises(RuntimeError, match="FAIL LOUD"):
        PeerGreedyNetworkScheduler._require_sym_scope()


def test_the_flag_is_accepted_at_alpha_zero_without_search(monkeypatch):
    for k in ("HEROSIM_PG_CD_EXPANSION", "HEROSIM_PG_CD_EXACT", "HEROSIM_PG_ILS_PER", "HEROSIM_PG_ORACLE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("HEROSIM_PG_BATCH_ALPHA", "0")
    PeerGreedyNetworkScheduler._require_sym_scope()
