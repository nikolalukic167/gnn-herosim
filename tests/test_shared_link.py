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


def _shell(monkeypatch, flag, pairs, n_tasks=4, bw=100.0, fabric=None):
    monkeypatch.setenv("HEROSIM_PEER_EXCHANGE", "1")
    if flag:
        monkeypatch.setenv(S.PG_SHARED_LINK_ENV, "1")
    else:
        monkeypatch.delenv(S.PG_SHARED_LINK_ENV, raising=False)
    monkeypatch.setattr(S, "platform_queue_drain_seconds", lambda platform, orch, memo, exec_scale=1.0: 0.0)
    monkeypatch.setattr(S, "incoming_cold_start_time", lambda task, platform: 0.0)
    monkeypatch.setattr(S, "network_latency_between", lambda src, node, nodes: 0.0)
    monkeypatch.setattr(S, "_approx_comm", lambda task_type: 0.0)
    fabric = fabric or _Fabric(bw)
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


def _real_fabric(core_mbps, access_mbps):
    """A, B, C hang off core0 / core1 / core1; A->B crosses A|core0, core0|core1, B|core1. Real NetworkFabric, so hops() is what
    the live path and the surrogate both read."""
    from simpy import Environment
    from src.placement.network_fabric import NetworkFabric, link_key
    acc = {n: ("core0" if n == "A" else "core1") for n in "ABC"}
    links = {link_key(n, c): {"bandwidth_mbps": access_mbps} for n, c in acc.items()}
    links[link_key("core0", "core1")] = {"bandwidth_mbps": core_mbps}
    routes = {}
    for a in "ABC":
        for b in "ABC":
            if a == b:
                continue
            path = [a, acc[a], acc[b], b] if acc[a] != acc[b] else [a, acc[a], b]
            routes.setdefault(a, {})[b] = path
    return NetworkFabric(Environment(), {"links": links, "routes": routes})


def _two_pairs_ab(monkeypatch, fabric):
    monkeypatch.setenv("HEROSIM_TRANSFER_MODEL", "pipelined")
    s, tasks, nodes, plats, orch = _shell(monkeypatch, True, [(0, 1), (2, 3)], fabric=fabric)
    plan = [(nodes["A"], plats["A"]), (nodes["B"], plats["B"]), (nodes["A"], plats["A"]), (nodes["B"], plats["B"])]
    return _service(s, tasks, plan, orch)[0]


def test_s_prime_reads_the_fabric_bandwidth_not_a_constant(monkeypatch):
    """Halving the bandwidth of the bottleneck link doubles S''s pair transfer, on the real fabric (pipelined, as in the accel seat)."""
    base = _two_pairs_ab(monkeypatch, _real_fabric(core_mbps=100.0, access_mbps=100.0))
    half = _two_pairs_ab(monkeypatch, _real_fabric(core_mbps=50.0, access_mbps=50.0))
    quarter = _two_pairs_ab(monkeypatch, _real_fabric(core_mbps=25.0, access_mbps=25.0))
    assert base == pytest.approx(2.0)   # 2 pairs share every link of the route: 100/2 Mbps
    assert half == pytest.approx(2 * base)
    assert quarter == pytest.approx(4 * base)


def test_scaling_only_the_core_is_invisible_when_an_access_link_is_the_bottleneck(monkeypatch):
    """The surrogate prices the route's bottleneck, as the live pipelined path does: a core ladder that stays above the access rate moves nothing."""
    slow_access = [_two_pairs_ab(monkeypatch, _real_fabric(core_mbps=c, access_mbps=10.0)) for c in (1000.0, 500.0, 250.0, 125.0)]
    assert slow_access == pytest.approx([slow_access[0]] * 4)
    assert _two_pairs_ab(monkeypatch, _real_fabric(core_mbps=5.0, access_mbps=10.0)) == pytest.approx(2 * slow_access[0])


# ---- link_bound_s0_v1 L3, Amendment 1: INPUT routes sharing core links (HEROSIM_PG_INPUT_LINK) ----

def _input_fabric(core_mbps, access_mbps):
    """Clients X, Y on core0; servers A, B on core1; X->A and Y->B both cross core0|core1."""
    from simpy import Environment
    from src.placement.network_fabric import NetworkFabric, link_key
    att = {"X": "core0", "Y": "core0", "A": "core1", "B": "core1"}
    links = {link_key(n, c): {"bandwidth_mbps": access_mbps} for n, c in att.items()}
    links[link_key("core0", "core1")] = {"bandwidth_mbps": core_mbps}
    routes = {}
    for a in att:
        for b in att:
            if a != b:
                routes.setdefault(a, {})[b] = [a, att[a], att[b], b] if att[a] != att[b] else [a, att[a], b]
    return NetworkFabric(Environment(), {"links": links, "routes": routes})


def _input_wait(monkeypatch, mode, core_mbps, access_mbps, dests=("A", "B"), origins=("X", "Y"), input_bytes=100 * MIB):
    monkeypatch.setenv("HEROSIM_TRANSFER_MODEL", "pipelined")
    monkeypatch.setenv("HEROSIM_PEER_EXCHANGE", "1")
    monkeypatch.delenv(S.PG_SHARED_LINK_ENV, raising=False)
    monkeypatch.setenv(S.PG_INPUT_LINK_ENV, mode)
    fabric = _input_fabric(core_mbps, access_mbps)
    names = ["A", "B", "X", "Y"]
    nodes = {nm: SimpleNamespace(id=i, node_name=nm, fabric=fabric) for i, nm in enumerate(names)}
    plats = {nm: SimpleNamespace(node=nodes[nm]) for nm in names}
    app = SimpleNamespace(type={"name": "app"})
    tasks = [SimpleNamespace(id=i, node_name=o, application=app, type={"stateSize": {"app": {"input": input_bytes}}}) for i, o in enumerate(origins)]
    s = object.__new__(PeerGreedyNetworkCDScheduler)
    s._pg_init()
    plan = [(nodes[d], plats[d]) for d in dests]
    return s._pg_input_link_wait(tasks, plan)


def test_input_wait_is_the_other_inputs_hold_on_the_shared_core_link(monkeypatch):
    # core 100 MB/s, access 100 MB/s, 100 MB inputs: each holds core0|core1 for 1 s, so each waits behind the other's 1 s
    assert _input_wait(monkeypatch, "hold", 100.0, 100.0) == pytest.approx([1.0, 1.0])
    assert _input_wait(monkeypatch, "core", 100.0, 100.0) == pytest.approx([1.0, 1.0])


def test_hold_mode_uses_the_bottleneck_core_mode_the_core_bandwidth(monkeypatch):
    # slow access (10 MB/s): the live pipe holds the core link for input / 10 = 10 s; the literal core-bandwidth reading says 1 s
    assert _input_wait(monkeypatch, "hold", 100.0, 10.0) == pytest.approx([10.0, 10.0])
    assert _input_wait(monkeypatch, "core", 100.0, 10.0) == pytest.approx([1.0, 1.0])


def test_core_mode_scales_with_the_configured_core_bandwidth(monkeypatch):
    base = _input_wait(monkeypatch, "core", 100.0, 1000.0)[0]
    assert _input_wait(monkeypatch, "core", 50.0, 1000.0)[0] == pytest.approx(2 * base)
    assert _input_wait(monkeypatch, "core", 25.0, 1000.0)[0] == pytest.approx(4 * base)


def test_no_shared_core_link_is_zero_so_it_equals_s(monkeypatch):
    # both tasks go to A from different... same-side origin and destination: X->A uses core0|core1, a task from A to B never leaves core1
    assert _input_wait(monkeypatch, "core", 100.0, 100.0, dests=("A", "B"), origins=("X", "A")) == pytest.approx([0.0, 0.0])
    assert _input_wait(monkeypatch, "hold", 100.0, 100.0, dests=("X", "A"), origins=("X", "A")) == pytest.approx([0.0, 0.0])   # co-located inputs
