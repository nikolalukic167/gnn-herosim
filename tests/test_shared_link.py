"""big_groups_s0_v1 M4: S' = S with each in-batch pair's transfer priced at the link bandwidth divided by the number of the batch's cross-node pairs on
that link (HEROSIM_PG_SHARED_LINK=1, `_pg_plan_cost` only). Off by default; with no shared link S' equals S to the digit."""
from types import SimpleNamespace

import pytest

from src.policy.peer_greedy_network import scheduler as S
from src.policy.peer_greedy_network.scheduler import PeerGreedyNetworkCDScheduler

MIB = 1024 * 1024
PAYLOAD = 100 * MIB   # 1 s on a 100 Mbps link


class _Fabric:
    """Star of three nodes: A-B, A-C, B-C are separate links (100 Mbps), a route is the direct link."""

    def __init__(self, bw=100.0):
        self.bw = bw

    def hops(self, src, dst):
        return [] if src == dst else [("-".join(sorted((src, dst))), self.bw)]


def _platform(pid, node, fabric):
    def transfer(peer_node_name, payload):
        hops = fabric.hops(peer_node_name, node.node_name)
        return payload / (min(bw for _k, bw in hops) * MIB) if hops else 0.0

    return SimpleNamespace(id=pid, type={"shortName": f"p{pid}"}, node=node, initialized=SimpleNamespace(triggered=True),
                           _payload_transfer_time=transfer, peer_link_latency=lambda peer, context="": 0.0)


def _shell(monkeypatch, flag, pairs, n_tasks=4, bw=100.0):
    monkeypatch.setenv("HEROSIM_PEER_EXCHANGE", "1")
    if flag:
        monkeypatch.setenv(S.PG_SHARED_LINK_ENV, "1")
    else:
        monkeypatch.delenv(S.PG_SHARED_LINK_ENV, raising=False)
    monkeypatch.setattr(S, "platform_queue_drain_seconds", lambda platform, orch, memo, exec_scale=1.0: 0.0)
    monkeypatch.setattr(S, "incoming_cold_start_time", lambda task, platform: 0.0)
    monkeypatch.setattr(S, "network_latency_between", lambda src, node, nodes: 0.0)
    monkeypatch.setattr(S, "_approx_comm", lambda task_type: 0.0)
    fabric = _Fabric(bw)
    nodes = {nm: SimpleNamespace(id=i, node_name=nm, fabric=fabric) for i, nm in enumerate("ABC")}
    plats = {nm: _platform(10 + i, nodes[nm], fabric) for i, nm in enumerate("ABC")}
    tasks = [SimpleNamespace(id=i, type={"name": "cnn", "executionTime": {p.type["shortName"]: 0.0 for p in plats.values()}}, node_name="client0")
             for i in range(n_tasks)]
    table = {}
    for a, b in pairs:
        table.setdefault(a, {})[b] = float(PAYLOAD)
        table.setdefault(b, {})[a] = float(PAYLOAD)
    s = object.__new__(PeerGreedyNetworkCDScheduler)
    s._pg_init()
    s.nodes = SimpleNamespace(items=list(nodes.values()))
    orch = SimpleNamespace(peer_exchange=table, task_by_id={})
    return s, tasks, nodes, plats, orch


def _service(s, tasks, plan, orch):
    return s._pg_plan_cost(tasks, plan, orch, {}, s.nodes.items)[1]


def test_two_pairs_on_one_link_each_get_half_the_bandwidth(monkeypatch):
    s, tasks, nodes, plats, orch = _shell(monkeypatch, True, [(0, 1), (2, 3)])
    plan = [(nodes["A"], plats["A"]), (nodes["B"], plats["B"]), (nodes["A"], plats["A"]), (nodes["B"], plats["B"])]
    assert _service(s, tasks, plan, orch) == pytest.approx([2.0, 2.0, 2.0, 2.0])   # 1 s alone, 2 pairs on A-B: 100/2 Mbps
    s0, tasks0, nodes0, plats0, orch0 = _shell(monkeypatch, False, [(0, 1), (2, 3)])
    plan0 = [(nodes0["A"], plats0["A"]), (nodes0["B"], plats0["B"]), (nodes0["A"], plats0["A"]), (nodes0["B"], plats0["B"])]
    assert _service(s0, tasks0, plan0, orch0) == pytest.approx([1.0, 1.0, 1.0, 1.0])   # S prices each pair alone


def test_pairs_on_different_links_equal_s_to_the_digit(monkeypatch):
    on = _shell(monkeypatch, True, [(0, 1), (2, 3)])
    s, tasks, nodes, plats, orch = on
    plan = [(nodes["A"], plats["A"]), (nodes["B"], plats["B"]), (nodes["A"], plats["A"]), (nodes["C"], plats["C"])]   # A-B and A-C
    shared = s._pg_plan_cost(tasks, plan, orch, {}, s.nodes.items)
    off = _shell(monkeypatch, False, [(0, 1), (2, 3)])
    s0, tasks0, nodes0, plats0, orch0 = off
    plan0 = [(nodes0["A"], plats0["A"]), (nodes0["B"], plats0["B"]), (nodes0["A"], plats0["A"]), (nodes0["C"], plats0["C"])]
    assert shared == s0._pg_plan_cost(tasks0, plan0, orch0, {}, s0.nodes.items)


def test_a_co_located_pair_loads_no_link(monkeypatch):
    s, tasks, nodes, plats, orch = _shell(monkeypatch, True, [(0, 1), (2, 3)])
    plan = [(nodes["A"], plats["A"]), (nodes["A"], plats["A"]), (nodes["A"], plats["A"]), (nodes["B"], plats["B"])]   # (0,1) co-located; (2,3) alone on A-B
    assert _service(s, tasks, plan, orch) == pytest.approx([0.0, 0.0, 1.0, 1.0])


def test_three_pairs_one_link_and_a_third_task_pair_keeps_the_count(monkeypatch):
    s, tasks, nodes, plats, orch = _shell(monkeypatch, True, [(0, 1), (2, 3), (4, 5)], n_tasks=6)
    plan = [(nodes["A"], plats["A"]), (nodes["B"], plats["B"])] * 3
    assert _service(s, tasks, plan, orch) == pytest.approx([3.0] * 6)


def test_partner_outside_the_batch_is_priced_as_in_s(monkeypatch):
    s, tasks, nodes, plats, orch = _shell(monkeypatch, True, [(0, 1), (0, 9)], n_tasks=2)
    orch.task_by_id = {9: SimpleNamespace(platform=SimpleNamespace(node=SimpleNamespace(node_name="C")))}
    plan = [(nodes["A"], plats["A"]), (nodes["B"], plats["B"])]
    # task 0: in-batch partner on B (1 s, one pair on A-B) + outside partner on C (1 s, S price); task 1: partner on A (1 s)
    assert _service(s, tasks, plan, orch) == pytest.approx([2.0, 1.0])


def test_flag_needs_a_batched_flavour(monkeypatch):
    monkeypatch.setenv(S.PG_SHARED_LINK_ENV, "1")
    monkeypatch.setenv("HEROSIM_PEER_EXCHANGE", "1")
    s = object.__new__(S.PeerGreedyNetworkScheduler)
    with pytest.raises(RuntimeError, match="batched flavour"):
        s._pg_init()
    monkeypatch.setenv(S.PG_SHARED_LINK_ENV, "2")
    with pytest.raises(ValueError, match=S.PG_SHARED_LINK_ENV):
        object.__new__(PeerGreedyNetworkCDScheduler)._pg_init()
