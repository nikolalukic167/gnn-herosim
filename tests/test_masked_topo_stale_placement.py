"""A decoded placement evicted before its turn is decoded again, alone; only a task with no replica left raises."""
import logging
from types import SimpleNamespace

import pytest

from src.policy.gnn.scheduler import GNNScheduler


def _scheduler(valid, decode_result):
    calls = []
    sched = SimpleNamespace(env=SimpleNamespace(now=1.0), nodes=SimpleNamespace(items=[SimpleNamespace(id=7, node_name="node3")]))
    sched._get_valid_replicas = lambda replicas, task: valid
    sched._capture_full_queue_snapshot = lambda: {}
    sched._capture_temporal_state_snapshot = lambda: None
    sched._prefix_inference = lambda tasks, state, q, t: calls.append(list(tasks)) or {0: decode_result}
    return sched, calls


def _task():
    return SimpleNamespace(id=5, type={"name": "rf"}, planned_node_name="old")


def _state():
    return SimpleNamespace(replicas={"rf": {("n", "p")}})


def test_stale_task_is_decoded_alone_and_replans_its_node(caplog):
    sched, calls = _scheduler(valid=[("n", "p")], decode_result=(7, 99))
    task = _task()
    with caplog.at_level(logging.WARNING):
        out = GNNScheduler._redecode_stale_placement(sched, task, _state())
    assert out == (7, 99) and calls == [[task]] and task.planned_node_name == "node3"
    assert "stale placement re-decoded for task 5" in caplog.text


def test_no_valid_replica_left_is_not_redecoded():
    sched, calls = _scheduler(valid=[], decode_result=(7, 99))
    task = _task()
    assert GNNScheduler._redecode_stale_placement(sched, task, _state()) is None
    assert calls == [] and task.planned_node_name == "old"


def _prefix_batch_harness(evict_after_decode):
    """A scheduler stub that runs the real ``_process_task_batch_prefix`` up to the commit loop."""
    import simpy

    env = simpy.Environment()
    class Ref:  # hashable: replicas are (node, platform) pairs in a set
        def __init__(self, **kw):
            self.__dict__.update(kw)

    node = Ref(id=40, node_name="node0")
    plat = Ref(id=199)
    replicas = {"rf": {(node, plat)}}
    state = SimpleNamespace(replicas=replicas)
    sched = object.__new__(GNNScheduler)
    sched.env = env
    sched.mutex = simpy.Store(env)
    sched.mutex.items.append(state)
    sched.nodes = SimpleNamespace(items=[node])
    sched._live_audit_policy_name = "test"
    sched.prefix_batches = sched.prefix_tasks_decoded = sched.prefix_tasks_deferred = 0
    sched._capture_full_queue_snapshot = lambda: {}
    sched._capture_temporal_state_snapshot = lambda: None
    sched._get_valid_replicas = lambda reps, task: sorted(reps, key=lambda r: r[0].id)
    deferred = []

    def inference(tasks, st, q, t):
        if evict_after_decode:
            replicas["rf"].clear()  # the mutex is released after the decode; a scale-up evicts the replica
        return {i: (node.id, plat.id) for i in range(len(tasks))}

    def defer(task, st):
        deferred.append(task.id)
        if False:
            yield

    sched._prefix_inference = inference
    sched._defer = defer
    task = SimpleNamespace(id=11, type={"name": "rf"}, planned_node_name=None)
    return env, sched, state, task, deferred


def test_task_whose_every_replica_was_evicted_is_deferred_not_raised(monkeypatch):
    monkeypatch.delenv("GNN_SERVE_CORPUS_SLATE", raising=False)
    env, sched, state, task, deferred = _prefix_batch_harness(evict_after_decode=True)
    env.process(sched._process_task_batch_prefix([task]))
    env.run()
    assert deferred == [11]
    assert sched.prefix_tasks_deferred == 1
    assert state in sched.mutex.items  # the state is handed back for the next waiter


def test_no_eviction_never_takes_the_new_branch(monkeypatch):
    env, sched, state, task, deferred = _prefix_batch_harness(evict_after_decode=False)
    calls = []
    sched._redecode_stale_placement = lambda *a: calls.append(a)
    sched.gnn_pure_decisions = 0
    sched.autoscaler = None

    class Stop(Exception):
        pass

    def boom(*a, **k):  # the commit proceeds past the match check; stop there
        raise Stop

    import src.placement.replica_seeding as rs
    monkeypatch.setattr(rs, "start_deferred_cold_init", boom)
    env.process(sched._process_task_batch_prefix([task]))
    with pytest.raises(Stop):
        env.run()
    assert calls == [] and deferred == []
