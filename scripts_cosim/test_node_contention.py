#!/usr/bin/env python3
"""Physics tests for node_contention_v3 — the shared node execution-slot pool.

The 2026-08-17 separability measurement showed the co-sim target was additive to
98.8-100%, because nothing on a node is shared: each platform has its own FIFO queue and
co-located platforms contend for nothing. That is why same-node message-passing edges were
falsified — they carried a coupling the physics did not have.

These tests guard the two properties that change:

1. **Default-off is bit-identical.** With no slot pool the node behaves exactly as
   node_disk_v2, so every existing corpus reproduces unchanged.
2. **Co-location actually costs something.** Two tasks on one node with a single shared
   slot must finish strictly later than the same two tasks on two nodes — this is the
   non-additivity a GNN can exploit.

Run: pipenv run python3 -m pytest scripts_cosim/test_node_contention.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import simpy
from simpy.resources.store import FilterStore

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.placement.infrastructure import Node  # noqa: E402
from src.placement.scheduling_cost import node_contention_wait  # noqa: E402

EXEC_TIME = 2.0
PLATFORM_TYPE = {"shortName": "xavierCpu", "name": "test-cpu"}
TASK_TYPE = {"name": "dnn2", "executionTime": {"xavierCpu": EXEC_TIME}}


class _FakeTask:
    def __init__(self) -> None:
        self.type = TASK_TYPE


class _FakePlatform:
    """Minimal stand-in exposing only what node_contention_wait reads."""

    def __init__(
        self, env: simpy.Environment, queued: int, virtual_backlog: float = 0.0
    ) -> None:
        self.type = PLATFORM_TYPE
        self.queue = simpy.Store(env)
        self.virtual_warmup_total_time = virtual_backlog
        for _ in range(queued):
            self.queue.put(_FakeTask())


def _node(env: simpy.Environment, compute_slots, queue_depths, virtual_backlogs=None):
    platforms = FilterStore(env)
    node = Node(
        env=env,
        node_id=0,
        memory=8.0,
        platforms=platforms,
        storage=FilterStore(env),
        network_map={},
        network={"bandwidth": 100.0},
        policy=None,
        data=None,
        node_type="xavier",
        node_name="node0",
        compute_slots=compute_slots,
    )
    backlogs = virtual_backlogs or [0.0] * len(queue_depths)
    for depth, backlog in zip(queue_depths, backlogs):
        platforms.put(_FakePlatform(env, depth, backlog))
    return node


def test_no_slot_pool_is_node_disk_v2():
    """Default (None) must add exactly zero, so existing corpora reproduce unchanged."""
    env = simpy.Environment()
    node = _node(env, None, [5, 5, 5])
    target, *_ = node.platforms.items
    assert node.compute_slots is None
    assert node_contention_wait(node, target) == 0.0


def test_sibling_backlog_creates_coupling():
    """The cost of a placement must depend on what else sits on the node."""
    env = simpy.Environment()
    idle = _node(env, 1, [0, 0, 0])
    busy = _node(env, 1, [0, 4, 3])
    idle_target, *_ = idle.platforms.items
    busy_target, *_ = busy.platforms.items

    assert node_contention_wait(idle, idle_target) == 0.0
    # 7 sibling-queued tasks x 2.0s over 1 slot
    assert node_contention_wait(busy, busy_target) == pytest.approx(14.0)


def test_more_slots_reduce_the_wait():
    env = simpy.Environment()
    node = _node(env, 4, [0, 4, 4])
    target, *_ = node.platforms.items
    # 8 sibling tasks x 2.0s over 4 slots
    assert node_contention_wait(node, target) == pytest.approx(4.0)


def test_target_platform_own_queue_is_not_double_counted():
    """queue_work already charges the target's own queue; contention must not re-add it."""
    env = simpy.Environment()
    node = _node(env, 1, [6, 0, 0])
    target, *_ = node.platforms.items
    assert node_contention_wait(node, target) == 0.0


def test_batch_mates_on_the_node_are_charged():
    env = simpy.Environment()
    node = _node(env, 1, [0, 0])
    target, *_ = node.platforms.items
    assert node_contention_wait(node, target) == 0.0
    assert node_contention_wait(
        node, target, added_on_node=2, added_unit_exec=EXEC_TIME
    ) == pytest.approx(4.0)


def test_seeded_backlog_counts_toward_contention():
    """Regression: queue depth is seeded as a compressed warmup backlog, not as
    queue.items. Counting only queue.items made contention silently zero -- and the
    backlog is ~95% of RTT, so the target stayed additive."""
    env = simpy.Environment()
    node = _node(env, 1, [0, 0, 0], virtual_backlogs=[0.0, 1.5, 2.5])
    target, *_ = node.platforms.items
    assert node_contention_wait(node, target) == pytest.approx(4.0)


def test_invalid_slot_count_fails_loudly():
    env = simpy.Environment()
    with pytest.raises(ValueError, match="compute_slots must be >= 1"):
        _node(env, 0, [1])


def test_shared_slots_serialize_co_located_execution():
    """End-to-end SimPy check: one slot forces two co-located runs to serialize."""
    env = simpy.Environment()
    node = _node(env, 1, [])
    finished: list[float] = []

    def run():
        with node.compute_slots.request() as slot:
            yield slot
            yield env.timeout(EXEC_TIME)
            finished.append(env.now)

    env.process(run())
    env.process(run())
    env.run()

    assert finished == pytest.approx([EXEC_TIME, 2 * EXEC_TIME])


def test_two_slots_allow_parallel_execution():
    env = simpy.Environment()
    node = _node(env, 2, [])
    finished: list[float] = []

    def run():
        with node.compute_slots.request() as slot:
            yield slot
            yield env.timeout(EXEC_TIME)
            finished.append(env.now)

    env.process(run())
    env.process(run())
    env.run()

    assert finished == pytest.approx([EXEC_TIME, EXEC_TIME])
