"""replica_guard_v1: the GNN_REPLICA_KEEPWARM serving guard (docs/lineages/replica_guard_v1.md)."""
from types import SimpleNamespace

import pytest

from src.policy.gnn.scheduler import GNNScheduler, _keepwarm_params


class _Node:
    def __init__(self, nid):
        self.id = nid


class _Plat:
    def __init__(self, pid, idle_since, queued=0, busy=False):
        self.id = pid
        self.idle_since = idle_since
        self.queue = SimpleNamespace(items=[object()] * queued)
        self.current_task = object() if busy else None

    def queue_length(self):
        return len(self.queue.items)


def _stub(now=100.0, keep_alive=30.0):
    stub = SimpleNamespace(env=SimpleNamespace(now=now),
                           autoscaler=SimpleNamespace(policy=SimpleNamespace(keep_alive=keep_alive)),
                           keepwarm_moves=0, keepwarm_batches_fired=0, keepwarm_expiring_seen=0)
    stub._get_valid_replicas = lambda pool, task: sorted(pool, key=lambda c: (c[0].id, c[1].id))
    return stub


def _task(tid, tname="dnn1"):
    return SimpleNamespace(id=tid, type={"name": tname})


def test_expiring_replica_gets_the_task_from_the_longest_queue():
    n1, n2, n3 = (_Node(i) for i in (1, 2, 3))
    busy_a = _Plat(10, 99.0, queued=5)
    busy_b = _Plat(11, 99.0, queued=1)
    expiring = _Plat(12, 70.0)          # idle 30 s >= 30 - 5
    pool = {(n1, busy_a), (n2, busy_b), (n3, expiring)}
    tasks = [_task(0), _task(1)]
    placements = {0: (2, 11), 1: (1, 10)}
    stub = _stub()
    out = GNNScheduler._keep_replicas_warm(stub, tasks, placements,
                                           SimpleNamespace(replicas={"dnn1": pool}), 4, 5.0)
    assert out == {0: (2, 11), 1: (3, 12)}
    assert (stub.keepwarm_moves, stub.keepwarm_batches_fired) == (1, 1)


def test_no_move_when_pool_is_large_or_replica_is_fresh_or_already_targeted():
    nodes = [_Node(i) for i in range(6)]
    fresh = _Plat(20, 90.0)             # idle 10 s < 25 s
    never_used = _Plat(21, float("inf"))
    chosen = _Plat(22, 60.0)            # expiring but already targeted by the batch
    pool = {(nodes[0], fresh), (nodes[1], never_used), (nodes[2], chosen)}
    tasks = [_task(0)]
    stub = _stub()
    out = GNNScheduler._keep_replicas_warm(stub, tasks, {0: (2, 22)},
                                           SimpleNamespace(replicas={"dnn1": pool}), 4, 5.0)
    assert out == {0: (2, 22)} and stub.keepwarm_moves == 0
    big = {(nodes[i], _Plat(30 + i, 0.0)) for i in range(5)}
    out = GNNScheduler._keep_replicas_warm(stub, tasks, {0: (0, 30)},
                                           SimpleNamespace(replicas={"dnn1": big}), 4, 5.0)
    assert out == {0: (0, 30)} and stub.keepwarm_moves == 0


def test_one_task_per_expiring_replica_and_other_types_untouched():
    n = [_Node(i) for i in range(4)]
    a, e1, e2 = _Plat(1, 99.0, queued=3), _Plat(2, 50.0), _Plat(3, 60.0)
    pool = {(n[0], a), (n[1], e1), (n[2], e2)}
    tasks = [_task(0), _task(1), _task(2, "dnn2")]
    other = {(n[3], _Plat(9, 99.0))}
    stub = _stub()
    out = GNNScheduler._keep_replicas_warm(stub, tasks, {0: (0, 1), 1: (0, 1), 2: (3, 9)},
                                           SimpleNamespace(replicas={"dnn1": pool, "dnn2": other}), 4, 5.0)
    assert sorted(out[i] for i in (0, 1)) == [(1, 2), (2, 3)] and out[2] == (3, 9)
    assert stub.keepwarm_moves == 2


def test_flag_parsing(monkeypatch):
    monkeypatch.delenv("GNN_REPLICA_KEEPWARM", raising=False)
    assert _keepwarm_params() is None
    monkeypatch.setenv("GNN_REPLICA_KEEPWARM", "1")
    assert _keepwarm_params() == (4, 5.0)
    monkeypatch.setenv("GNN_REPLICA_KEEPWARM", "yes")
    with pytest.raises(ValueError):
        _keepwarm_params()
