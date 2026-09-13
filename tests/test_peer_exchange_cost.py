"""peer_affinity_v1: the pairwise-instance exchange term (HEROSIM_PEER_EXCHANGE=1).

Why this term exists. Every cost the simulator charges is indexed by a machine or by the
task's own route, so a pointwise scorer given per-machine counts expresses the label exactly
(docs/lineages/throughline.md, 2026-09-09). The one shape that argument does not cover is a
cost indexed by a PAIR of task instances with no commit order: at task i's input stage,
charge x_ij * transfer(node(i), node(j)) for every peer j. The Phase 0 paper screen
(docs/lineages/peer_affinity_v1.md) found it is the first joint structure count columns do
not repair; this is the physics that screen was a paper model of.

Guards, in the order the registration names them:

1. **Off by default and inert without a peer table** -- every existing corpus is
   bit-identical whether the flag is set or not (unit level here; corpus-level replay in
   tests/test_peer_exchange_replay.py).
2. **Co-located peers are free; remote peers pay hops x bytes / bottleneck + latency**, the
   same charge `_dependency_transfer_time` makes per remote parent -- and it is symmetric.
3. **A peer that is neither placed nor planned fails loud**, never charges 0.0.
4. **The orchestrator's table is symmetric and rejects self-pairs / conflicting duplicates.**
5. **The batch scheduler plans the whole batch before any member starts** under the flag.

Run: pipenv run python3 -m pytest tests/test_peer_exchange_cost.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.placement.infrastructure import Platform  # noqa: E402
from src.placement.orchestrator import build_peer_exchange_table  # noqa: E402

BANDWIDTH_MBPS = 100.0


class FakeOrchestrator:
    def __init__(self, table: Dict[int, Dict[int, float]], tasks: Dict[int, "FakeTask"]):
        self.peer_exchange = table
        self.task_by_id = tasks


class FakeNode:
    def __init__(self, name: str, network_map: Dict[str, float], orchestrator: Optional[FakeOrchestrator] = None):
        self.node_name = name
        self.network_map = network_map
        self.network = {"bandwidth": BANDWIDTH_MBPS}
        self.orchestrator_ref = orchestrator


class FakePlatform:
    def __init__(self, node: FakeNode):
        self.node = node


class FakeTask:
    def __init__(self, task_id: int, node: FakeNode | None = None, planned: Optional[str] = None):
        self.id = task_id
        self.platform = FakePlatform(node) if node is not None else None
        self.planned_node_name = planned

    def __repr__(self):
        return f"FakeTask({self.id})"


def cost(node: FakeNode, task: FakeTask) -> float:
    platform = Platform.__new__(Platform)
    platform.node = node
    return Platform._peer_exchange_time(platform, task)


def expected(payload: float, latency: float) -> float:
    return payload / (BANDWIDTH_MBPS * 1024 * 1024) + latency


@pytest.fixture
def peer_exchange_on(monkeypatch):
    monkeypatch.setenv("HEROSIM_PEER_EXCHANGE", "1")


def two_nodes(table: Dict[int, Dict[int, float]]):
    tasks: Dict[int, FakeTask] = {}
    orch = FakeOrchestrator(table, tasks)
    a = FakeNode("node1", {"node2": 0.05}, orch)
    b = FakeNode("node2", {"node1": 0.05}, orch)
    return a, b, tasks


# 1. default-off and inert ----------------------------------------------------------------

def test_disabled_by_default_costs_nothing(monkeypatch):
    monkeypatch.delenv("HEROSIM_PEER_EXCHANGE", raising=False)
    a, b, tasks = two_nodes(build_peer_exchange_table([[0, 1, 5e6]]))
    tasks[0] = FakeTask(0, a); tasks[1] = FakeTask(1, b)
    assert cost(a, tasks[0]) == 0.0


def test_no_peer_table_costs_nothing_even_when_enabled(peer_exchange_on):
    a, b, tasks = two_nodes({})
    tasks[0] = FakeTask(0, a)
    assert cost(a, tasks[0]) == 0.0


# 2. co-located free, remote priced, symmetric ---------------------------------------------

def test_colocated_peer_is_free(peer_exchange_on):
    a, b, tasks = two_nodes(build_peer_exchange_table([[0, 1, 5e6]]))
    tasks[0] = FakeTask(0, a); tasks[1] = FakeTask(1, a)
    assert cost(a, tasks[0]) == 0.0
    assert cost(a, tasks[1]) == 0.0


def test_remote_peer_costs_transfer_plus_latency_and_is_symmetric(peer_exchange_on):
    a, b, tasks = two_nodes(build_peer_exchange_table([[0, 1, 5e6]]))
    tasks[0] = FakeTask(0, a); tasks[1] = FakeTask(1, b)
    assert cost(a, tasks[0]) == pytest.approx(expected(5e6, 0.05))
    assert cost(b, tasks[1]) == pytest.approx(expected(5e6, 0.05))


def test_every_peer_is_charged_and_planned_node_is_accepted(peer_exchange_on):
    a, b, tasks = two_nodes(build_peer_exchange_table([[0, 1, 5e6], [0, 2, 1e6], [1, 2, 3e6]]))
    tasks[0] = FakeTask(0, a)
    tasks[1] = FakeTask(1, None, planned="node2")   # assigned by the batch pre-pass, not yet enqueued
    tasks[2] = FakeTask(2, b)
    assert cost(a, tasks[0]) == pytest.approx(expected(5e6, 0.05) + expected(1e6, 0.05))
    assert cost(b, tasks[2]) == pytest.approx(expected(1e6, 0.05))  # peer 1 co-located (planned), peer 0 remote


# 3. fail loud ------------------------------------------------------------------------------

def test_unplaced_and_unplanned_peer_fails_loud(peer_exchange_on):
    a, b, tasks = two_nodes(build_peer_exchange_table([[0, 1, 5e6]]))
    tasks[0] = FakeTask(0, a); tasks[1] = FakeTask(1, None)
    with pytest.raises(RuntimeError, match="neither a platform nor a planned node"):
        cost(a, tasks[0])


def test_unknown_peer_id_fails_loud(peer_exchange_on):
    a, b, tasks = two_nodes(build_peer_exchange_table([[0, 7, 5e6]]))
    tasks[0] = FakeTask(0, a)
    with pytest.raises(RuntimeError, match="not a task of this workload"):
        cost(a, tasks[0])


def test_unreachable_peer_node_fails_loud(peer_exchange_on):
    tasks: Dict[int, FakeTask] = {}
    orch = FakeOrchestrator(build_peer_exchange_table([[0, 1, 5e6]]), tasks)
    a = FakeNode("node1", {}, orch)
    b = FakeNode("node2", {}, orch)
    tasks[0] = FakeTask(0, a); tasks[1] = FakeTask(1, b)
    with pytest.raises(RuntimeError, match="no network_map entry"):
        cost(a, tasks[0])


# 4. the table ------------------------------------------------------------------------------

def test_table_is_symmetric_and_empty_without_triples():
    assert build_peer_exchange_table(None) == {}
    assert build_peer_exchange_table([]) == {}
    t = build_peer_exchange_table([[0, 1, 5e6], [2, 0, 1e6]])
    assert t == {0: {1: 5e6, 2: 1e6}, 1: {0: 5e6}, 2: {0: 1e6}}


@pytest.mark.parametrize("bad", [[[0, 0, 1e6]], [[0, 1, -1.0]], [[0, 1, 1e6], [1, 0, 2e6]], [[0, 1]]])
def test_table_rejects_malformed_triples(bad):
    with pytest.raises(ValueError):
        build_peer_exchange_table(bad)


# 5. the batch pre-pass ---------------------------------------------------------------------

def test_scheduler_plans_the_whole_batch_from_forced_placements(peer_exchange_on):
    from src.policy.determined.scheduler import DeterminedScheduler

    class N:
        def __init__(self, i, name):
            self.id, self.node_name = i, name

    class Store:
        def __init__(self, items):
            self.items = items

    sched = DeterminedScheduler.__new__(DeterminedScheduler)
    sched.nodes = Store([N(20, "node0"), N(21, "node1")])
    sched.forced_placements = {0: (20, 104), 1: (21, 111), 2: (-1, -1)}
    batch = [FakeTask(0), FakeTask(1), FakeTask(2)]
    DeterminedScheduler._plan_batch_nodes(sched, batch)
    assert [t.planned_node_name for t in batch] == ["node0", "node1", None]
    sched.forced_placements = {0: (99, 1)}
    with pytest.raises(RuntimeError, match="not a node of this run"):
        DeterminedScheduler._plan_batch_nodes(sched, [FakeTask(0)])
