"""S_fs: ib0's surrogate S with every remote exchange transfer priced at the static max-min shared rate (in-flight FairShare flows at the captured state plus the plan's own transfers per
directed link). Set around a label search by the offline driver; off by default, and off or on with no shared link it equals S to the digit."""
from types import SimpleNamespace

import pytest
import simpy

from src.placement.network_fabric import FairShare
from src.policy.peer_greedy_network import scheduler as S
from src.policy.peer_greedy_network.scheduler import PeerGreedyNetworkCDScheduler

MIB = 1024 * 1024
PAYLOAD = 100 * MIB          # 1 s on a 100 MiB/s hop


class _Fabric:
    """Hub-and-spoke: node X reaches node Y over X>core, core>Y (100 MiB/s each). `flows` are the in-flight FairShare flows (each with .hops)."""

    def __init__(self, flows=()):
        self.fairshare = SimpleNamespace(flows=list(flows))

    def directed_hops(self, src, dst):
        return [] if src == dst else [(f"{src}>core", 100.0), (f"core>{dst}", 100.0)]

    def hops(self, src, dst):
        return [(k, bw) for k, bw in self.directed_hops(src, dst)]


def _platform(pid, node, fabric):
    def transfer(peer_node_name, payload):
        hops = fabric.directed_hops(peer_node_name, node.node_name)
        return payload / (min(bw for _k, bw in hops) * MIB) if hops else 0.0

    return SimpleNamespace(id=pid, type={"shortName": f"p{pid}"}, node=node, initialized=SimpleNamespace(triggered=True),
                           _payload_transfer_time=transfer, peer_link_latency=lambda peer, context="": 0.0)


def _shell(monkeypatch, pairs, fabric, names="ABCD", n_tasks=4):
    monkeypatch.setenv("HEROSIM_PEER_EXCHANGE", "1")
    monkeypatch.delenv(S.PG_SHARED_LINK_ENV, raising=False)
    monkeypatch.setattr(S, "platform_queue_drain_seconds", lambda platform, orch, memo, exec_scale=1.0: 0.0)
    monkeypatch.setattr(S, "incoming_cold_start_time", lambda task, platform: 0.0)
    monkeypatch.setattr(S, "network_latency_between", lambda src, node, nodes: 0.0)
    monkeypatch.setattr(S, "_approx_comm", lambda task_type: 0.0)
    nodes = {nm: SimpleNamespace(id=i, node_name=nm, fabric=fabric) for i, nm in enumerate(names)}
    plats = {nm: _platform(10 + i, nodes[nm], fabric) for i, nm in enumerate(names)}
    tasks = [SimpleNamespace(id=i, type={"name": "cnn", "executionTime": {p.type["shortName"]: 0.0 for p in plats.values()}}, node_name="client0") for i in range(n_tasks)]
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


def test_the_switch_is_off_by_default(monkeypatch):
    s, *_ = _shell(monkeypatch, [(0, 2)], _Fabric())
    assert s.pg_sfs is False


def test_with_no_shared_link_s_fs_equals_s(monkeypatch):
    s, tasks, nodes, plats, orch = _shell(monkeypatch, [(0, 2), (1, 3)], _Fabric())
    plan = [(nodes["C"], plats["C"]), (nodes["D"], plats["D"]), (nodes["A"], plats["A"]), (nodes["B"], plats["B"])]    # every directed hop used once
    base = s._pg_plan_cost(tasks, plan, orch, {}, s.nodes.items)
    s.pg_sfs = True
    assert s._pg_plan_cost(tasks, plan, orch, {}, s.nodes.items) == base


def test_two_equal_flows_on_one_link_are_each_priced_twice(monkeypatch):
    s, tasks, nodes, plats, orch = _shell(monkeypatch, [(0, 2), (1, 3)], _Fabric())
    plan = [(nodes["C"], plats["C"]), (nodes["C"], plats["C"]), (nodes["A"], plats["A"]), (nodes["B"], plats["B"])]    # A>C and B>C share core>C; C>A and C>B share C>core
    solo = _service(s, tasks, plan, orch)
    s.pg_sfs = True
    shared = _service(s, tasks, plan, orch)
    assert solo == pytest.approx([1.0, 1.0, 1.0, 1.0]) and shared == pytest.approx([2.0, 2.0, 2.0, 2.0])


def test_flows_in_flight_at_the_captured_state_count(monkeypatch):
    live = SimpleNamespace(hops=[("A>core", 100.0), ("core>C", 100.0)])
    s, tasks, nodes, plats, orch = _shell(monkeypatch, [(0, 2)], _Fabric([live]))
    plan = [(nodes["C"], plats["C"]), (nodes["D"], plats["D"]), (nodes["A"], plats["A"]), (nodes["B"], plats["B"])]
    s.pg_sfs = True
    assert _service(s, tasks, plan, orch)[0] == pytest.approx(2.0)       # 1 in flight + 1 own on A>core and core>C
    assert _service(s, tasks, plan, orch)[2] == pytest.approx(1.0)       # the reverse direction C>A is clear


def test_s_fs_ranks_two_plans_as_the_fairshare_engine_does(monkeypatch):
    fabric = _Fabric()
    s, tasks, nodes, plats, orch = _shell(monkeypatch, [(0, 2), (1, 3)], fabric)
    plan_x = [(nodes["C"], plats["C"]), (nodes["C"], plats["C"]), (nodes["A"], plats["A"]), (nodes["B"], plats["B"])]   # herded on C
    plan_y = [(nodes["C"], plats["C"]), (nodes["D"], plats["D"]), (nodes["A"], plats["A"]), (nodes["B"], plats["B"])]   # spread
    assert sum(_service(s, tasks, plan_x, orch)) == pytest.approx(sum(_service(s, tasks, plan_y, orch)))                    # S cannot tell them apart

    def engine_total(plan):
        env = simpy.Environment()
        share = FairShare(env)
        placed = [p[0].node_name for p in plan]
        evs = []
        for u, v in [(0, 2), (2, 0), (1, 3), (3, 1)]:
            hops = fabric.directed_hops(placed[v], placed[u])           # the transfer task u pulls from partner v
            evs.append(share.start(hops, PAYLOAD))
        done = []
        for ev in evs:
            ev.callbacks.append(lambda e: done.append(env.now))
        env.run()
        return sum(done)

    s.pg_sfs = True
    fs_x, fs_y = sum(_service(s, tasks, plan_x, orch)), sum(_service(s, tasks, plan_y, orch))
    assert engine_total(plan_x) > engine_total(plan_y) and fs_x > fs_y
