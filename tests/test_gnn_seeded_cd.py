from types import SimpleNamespace

import pytest

from src.policy.gnn.scheduler import GNNScheduler
from src.policy.peer_greedy_network.gnn_seeded_cd import GNNSeededCDScheduler


def _fixture(monkeypatch, *, seed):
    scheduler = object.__new__(GNNSeededCDScheduler)
    scheduler.pg_batches = 0
    node_a = SimpleNamespace(id=1, node_name="a")
    node_b = SimpleNamespace(id=2, node_name="b")
    plat_a = SimpleNamespace(id=11, type={"shortName": "cpu"})
    plat_b = SimpleNamespace(id=22, type={"shortName": "cpu"})
    tasks = [
        SimpleNamespace(id=100 + i, type={"name": "task", "executionTime": {"cpu": 2.0}})
        for i in range(2)
    ]
    state = SimpleNamespace(replicas={"task": {(node_a.id, plat_a.id), (node_b.id, plat_b.id)}})
    monkeypatch.setattr(GNNScheduler, "_prefix_inference", lambda *args: dict(seed))
    monkeypatch.setattr(scheduler, "_pg_orchestrator", lambda: object())
    monkeypatch.setattr(scheduler, "_get_valid_replicas", lambda replicas, task: [(node_a, plat_a), (node_b, plat_b)])
    monkeypatch.setattr(scheduler, "_pg_peer_nodes", lambda task, orch, planned: [("a", 1.0)] if planned else [])
    monkeypatch.setattr(scheduler, "_pg_exchange_seconds", lambda node, platform, peers: 1.5 if node.id == 2 else 0.0)
    return scheduler, tasks, state


def test_gnn_seed_initializes_refinement_ledger(monkeypatch):
    scheduler, tasks, state = _fixture(monkeypatch, seed={0: (1, 11), 1: (2, 22)})
    seen = {}

    def refine(batch, state, orch, memo, service, planned, placements, service_of):
        seen.update(service=service.copy(), planned=planned.copy(),
                    placements=placements.copy(), service_of=service_of.copy())
        placements[0] = (2, 22)

    monkeypatch.setattr(scheduler, "_pg_refine", refine)
    result = scheduler._prefix_inference(tasks, state, {}, None)
    assert result == {0: (2, 22), 1: (2, 22)}
    assert seen["planned"] == {100: "a", 101: "b"}
    assert seen["placements"] == {0: (1, 11), 1: (2, 22)}
    assert seen["service_of"][100][0] == "a:11"
    assert seen["service_of"][101][0] == "b:22"
    assert seen["service"]["b:22"] > seen["service"]["a:11"]
    assert scheduler.pg_batches == 1


def test_invalid_gnn_seed_fails_before_refinement(monkeypatch):
    scheduler, tasks, state = _fixture(monkeypatch, seed={0: (1, 11), 1: (3, 33)})
    monkeypatch.setattr(scheduler, "_pg_refine", lambda *args: pytest.fail("refinement ran"))
    with pytest.raises(RuntimeError, match="not one valid replica"):
        scheduler._prefix_inference(tasks, state, {}, None)
