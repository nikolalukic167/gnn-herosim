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
