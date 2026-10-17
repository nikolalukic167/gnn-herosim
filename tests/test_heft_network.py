import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts_cosim"))

from src.policy.heft_network.scheduler import HeftNetworkBatchScheduler, rank_order, upward_ranks


def test_chain_ranks_accumulate_exec_and_comm():
    r = upward_ranks([1, 2, 3], {1: 2.0, 2: 3.0, 3: 5.0}, {1: [2], 2: [3]}, {(1, 2): 0.5, (2, 3): 1.0})
    assert r == {3: 5.0, 2: 3.0 + 1.0 + 5.0, 1: 2.0 + 0.5 + 9.0}
    assert rank_order([1, 2, 3], r) == [1, 2, 3]


def test_fork_join_takes_the_longest_path():
    mean_exec = {1: 1.0, 2: 4.0, 3: 1.0, 4: 2.0}
    succ = {1: [2, 3], 2: [4], 3: [4]}
    comm = {(1, 2): 0.0, (1, 3): 10.0, (2, 4): 0.0, (3, 4): 0.0}
    r = upward_ranks([1, 2, 3, 4], mean_exec, succ, comm)
    assert r[4] == 2.0 and r[2] == 6.0 and r[3] == 3.0
    assert r[1] == 1.0 + max(0.0 + 6.0, 10.0 + 3.0)


def test_independent_tasks_order_by_mean_exec_then_id():
    r = upward_ranks([5, 6, 7, 8], {5: 1.0, 6: 3.0, 7: 3.0, 8: 2.0}, {}, {})
    assert rank_order([5, 6, 7, 8], r) == [6, 7, 8, 5]


def test_a_successor_with_a_smaller_id_fails_loud():
    with pytest.raises(ValueError, match="larger id"):
        upward_ranks([1, 2], {1: 1.0, 2: 1.0}, {2: [1]}, {})


class _Plat:
    def __init__(self, pid, node, exec_s):
        self.id, self.node, self.exec_s = pid, node, exec_s
        self.type = {"shortName": f"t{pid}"}
        self.initialized = SimpleNamespace(triggered=True)

    def _payload_transfer_time(self, sender, payload):
        return payload / 10.0

    def peer_link_latency(self, sender, context=""):
        return 0.5


def _node(name, nid):
    return SimpleNamespace(node_name=name, id=nid)


def _task(tid, deps=()):
    return SimpleNamespace(id=tid, type={"name": "T"}, dependencies=list(deps))


def _sched(exec_by_task, table):
    s = object.__new__(HeftNetworkBatchScheduler)
    s.sched_plan_evals = 0
    s._pg_exec_table = lambda ttype, plat: exec_by_task[ttype["tid"]]
    s._pg_restrict = lambda task, valid: valid
    s._get_valid_replicas = lambda replicas, task: replicas
    s.nodes = SimpleNamespace(items=[])
    s._orch = SimpleNamespace(peer_exchange=table)
    return s


def test_ranks_use_the_dependency_and_exchange_edges_and_meter_the_estimates():
    n1, n2 = _node("a", 1), _node("b", 2)
    p1, p2 = _Plat(1, n1, 1.0), _Plat(2, n2, 1.0)
    p1.node, p2.node = n1, n2
    cands = {0: [(n1, p1), (n2, p2)], 1: [(n1, p1), (n2, p2)], 2: [(n1, p1)]}
    tasks = [_task(0), _task(1), _task(2)]
    for t, tid in zip(tasks, (0, 1, 2)):
        t.type = {"name": "T", "tid": tid}
    tasks[2].dependencies = [tasks[0]]
    s = _sched({0: 2.0, 1: 1.0, 2: 1.0}, {0: {1: 10.0}, 1: {0: 10.0}})
    r = s._heft_ranks(tasks, cands, s._orch)
    # 0 -> 1 by exchange (10 bytes: (10/10 + 0.5) on the cross-node pairs only, mean over 4 pairs) and 0 -> 2 by dependency (no payload)
    comm01 = (0.0 + 1.5 + 1.5 + 0.0) / 4
    assert r[2] == 1.0 and r[1] == 1.0 and r[0] == 2.0 + max(comm01 + 1.0, 0.0 + 1.0)
    assert s.sched_plan_evals == 4


def test_the_pass_places_in_rank_order_and_has_no_refine():
    n1 = _node("a", 1)
    p1 = _Plat(1, n1, 1.0)
    p1.node = n1
    tasks = [_task(0), _task(1), _task(2)]
    for t, tid in zip(tasks, (0, 1, 2)):
        t.type = {"name": "T", "tid": tid}
    s = _sched({0: 1.0, 1: 5.0, 2: 3.0}, {})
    seen = []

    def choose(task, candidates, orch, **kw):
        seen.append(int(task.id))
        return n1, p1, 1.0

    s._pg_choose = choose
    state = SimpleNamespace(replicas={"T": [(n1, p1)]})
    placements, planned, service_of, committed = {}, {}, {}, {}
    out = s._pg_batch_pass(tasks, state, s._orch, memo={}, committed_service=committed, planned=planned, placements=placements,
                           service_of=service_of, refine=False)
    assert out == 0 and seen == [1, 2, 0]
    assert placements == {0: (1, 1), 1: (1, 1), 2: (1, 1)} and planned == {0: "a", 1: "a", 2: "a"} and committed == {"a:1": 3.0}
    with pytest.raises(RuntimeError, match="no refine"):
        s._pg_batch_pass(tasks, state, s._orch, memo={}, committed_service={}, planned={}, placements={}, service_of={}, refine=True)


def test_the_policy_is_registered_everywhere_a_batched_rule_is():
    root = os.path.join(os.path.dirname(__file__), "..")
    sim = open(os.path.join(root, "src/placement/simulation.py")).read()
    assert '"heft_network_batch_heft_network_batch": (GNNOrchestrator, GNNAutoscaler, HeftNetworkBatchScheduler)' in sim
    ex = open(os.path.join(root, "src/executesimulation.py")).read()
    assert ex.count("'heft_network_batch',") == 2 and "scheduling_strategy = 'heft_network_batch_heft_network_batch'" in ex
    assert '"heft_network_batch_heft_network_batch": "HEFT-NETWORK-BATCH"' in open(os.path.join(root, "src/placement/model.py")).read()


def test_the_gate_knows_heft_outside_the_default_grid():
    from scripts_cosim import fresh_topo_burst_v1_gate as G
    assert G.RULE_POLICY["heft"] == "heft_network_batch"
    assert "heft" in G.R1A_DIAG and "heft" in G.R1A_ARMS and "heft" not in G.R1A_CLASSICAL
    assert G.RULE_POLICY["reactive"] == "knative_network" and G.RULE_POLICY["random"] == "random_network"
