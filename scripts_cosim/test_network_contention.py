#!/usr/bin/env python3
"""Physics tests for network_contention_v1 — the shared node ingress pipe.

The 2026-08-17 separability measurement showed the co-sim target was additive to
98.8-100%. Network latency is the largest single term in the shallow-queue regime
(~40% of per-task RTT) and is a static table lookup paid once per task, so it is
perfectly additive: routing N tasks over a link costs exactly what routing 1 costs.

network_contention_v1 splits that term the way the physics actually splits:

* **propagation latency** stays an un-serialized timeout — additive, unchanged;
* **input transmission** (stateSize / bandwidth) is served through the destination
  node's single shared pipe, so concurrent inbound transfers queue behind each other.

The transmission itself is still a function of (task, target node) and stays additive.
The *wait* for the pipe is the non-additive part — it depends on how many batch-mates
were placed on the same node, which is exactly what a pointwise scorer cannot express.

node_contention_v3 failed because placed tasks held its resource for only ~0.024 s and
never overlapped. The guard against repeating that is test_transfer_is_long_enough_to_
overlap, which pins the hold time against the measured per-task RTT budget.

These tests guard three properties:

1. **Default-off is bit-identical.** With no bandwidth set there is no pipe and no
   transmission time at all, so every existing corpus reproduces unchanged.
2. **Co-location on a node actually costs something.** Two tasks whose transfers overlap
   on one node must serialize; on two nodes they must not.
3. **The ECT mirror agrees with the simulation**, so Knative/ECT baselines stay fair.

Run: pipenv run python3 -m pytest scripts_cosim/test_network_contention.py -q
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
from src.placement.scheduling_cost import (  # noqa: E402
    ingress_transfer_time,
    ingress_wait,
)

# The real nofs-* input payload and the bandwidth the pilot targets.
INPUT_BYTES = 153_600
BANDWIDTH_MBPS = 1.5
MIB = 1024.0 * 1024.0
EXPECTED_TRANSFER = INPUT_BYTES / (BANDWIDTH_MBPS * MIB)  # ~0.0977 s

APP_NAME = "nofs-dnn2"
TASK_TYPE = {
    "name": "dnn2",
    "executionTime": {"xavierCpu": 0.0239175},
    "stateSize": {APP_NAME: {"input": INPUT_BYTES, "output": 8000}},
}


class _FakeApplication:
    def __init__(self, name: str = APP_NAME) -> None:
        self.type = {"name": name}


class _FakeTask:
    """Minimal stand-in exposing only what the ingress helpers read."""

    def __init__(self, application=None, node_name: str = "client_node0") -> None:
        self.type = TASK_TYPE
        self.application = _FakeApplication() if application is None else application
        self.node_name = node_name


def _node(env: simpy.Environment, ingress_bandwidth_mbps, node_name: str = "node0"):
    return Node(
        env=env,
        node_id=0,
        memory=8.0,
        platforms=FilterStore(env),
        storage=FilterStore(env),
        network_map={},
        network={"bandwidth": 100.0},
        policy=None,
        data=None,
        node_type="xavier",
        node_name=node_name,
        ingress_bandwidth_mbps=ingress_bandwidth_mbps,
    )


# --- 1. default-off is node_disk_v2 -----------------------------------------------


def test_no_pipe_is_node_disk_v2():
    """Unset bandwidth means no resource, no transmission time, no wait."""
    env = simpy.Environment()
    node = _node(env, None)
    assert node.ingress_pipe is None
    assert node.ingress_bandwidth_mbps is None
    assert ingress_transfer_time(_FakeTask(), node.ingress_bandwidth_mbps) == 0.0
    assert ingress_wait(node, _FakeTask(), added_on_node=3) == 0.0


def test_invalid_bandwidth_fails_loudly():
    env = simpy.Environment()
    with pytest.raises(ValueError, match="ingress_bandwidth_mbps must be > 0"):
        _node(env, 0.0)
    with pytest.raises(ValueError, match="ingress_bandwidth_mbps must be > 0"):
        _node(env, -1.5)


# --- 2. the transfer term ----------------------------------------------------------


def test_transfer_time_matches_size_over_bandwidth():
    env = simpy.Environment()
    node = _node(env, BANDWIDTH_MBPS)
    got = ingress_transfer_time(_FakeTask(), node.ingress_bandwidth_mbps)
    assert got == pytest.approx(EXPECTED_TRANSFER)
    assert got == pytest.approx(0.09766, abs=1e-4)


def test_transfer_scales_inversely_with_bandwidth():
    env = simpy.Environment()
    slow = ingress_transfer_time(_FakeTask(), 0.5)
    mid = ingress_transfer_time(_FakeTask(), 1.5)
    fast = ingress_transfer_time(_FakeTask(), 5.0)
    assert slow > mid > fast
    assert slow == pytest.approx(3.0 * mid)


def test_transfer_is_long_enough_to_overlap():
    """The guard against repeating the node_contention_v3 failure.

    That lineage added a shared resource held for only the ~0.024 s execution, so placed
    tasks never overlapped and nodeContentionTime was exactly 0.0 everywhere. A transfer
    must be a meaningful fraction of the ~0.24 s per-task RTT budget to have any chance
    of overlapping a batch-mate's.
    """
    per_task_rtt_budget = 0.24
    exec_time = TASK_TYPE["executionTime"]["xavierCpu"]
    transfer = ingress_transfer_time(_FakeTask(), BANDWIDTH_MBPS)
    assert transfer > 4 * exec_time
    assert transfer > 0.25 * per_task_rtt_budget


def test_task_without_matching_state_size_pays_nothing():
    """An app the task type has no stateSize entry for must not silently cost time."""
    env = simpy.Environment()
    node = _node(env, BANDWIDTH_MBPS)
    task = _FakeTask(application=_FakeApplication("nofs-unknown"))
    assert ingress_transfer_time(task, node.ingress_bandwidth_mbps) == 0.0


# --- 3. the wait term (the non-additive part) --------------------------------------


def test_batch_mates_on_the_node_are_charged():
    """This is the coupling: cost depends on where the OTHER tasks went."""
    env = simpy.Environment()
    node = _node(env, BANDWIDTH_MBPS)
    task = _FakeTask()
    assert ingress_wait(node, task, added_on_node=0) == 0.0
    assert ingress_wait(node, task, added_on_node=1) == pytest.approx(EXPECTED_TRANSFER)
    assert ingress_wait(node, task, added_on_node=3) == pytest.approx(
        3 * EXPECTED_TRANSFER
    )


# --- 4. end-to-end SimPy: transfers actually serialize ------------------------------


def test_shared_pipe_serializes_concurrent_transfers():
    """Two simultaneous inbound transfers to one node must serialize."""
    env = simpy.Environment()
    node = _node(env, BANDWIDTH_MBPS)
    transfer = ingress_transfer_time(_FakeTask(), node.ingress_bandwidth_mbps)
    finished = []

    def run():
        with node.ingress_pipe.request() as pipe:
            yield pipe
            yield env.timeout(transfer)
            finished.append(env.now)

    env.process(run())
    env.process(run())
    env.run()

    assert finished == pytest.approx([transfer, 2 * transfer])


def test_transfers_to_different_nodes_do_not_serialize():
    """Coupling must be node-local: separate nodes have separate pipes."""
    env = simpy.Environment()
    node_a = _node(env, BANDWIDTH_MBPS, node_name="node0")
    node_b = _node(env, BANDWIDTH_MBPS, node_name="node1")
    transfer = ingress_transfer_time(_FakeTask(), BANDWIDTH_MBPS)
    finished = []

    def run(node):
        with node.ingress_pipe.request() as pipe:
            yield pipe
            yield env.timeout(transfer)
            finished.append(env.now)

    env.process(run(node_a))
    env.process(run(node_b))
    env.run()

    assert finished == pytest.approx([transfer, transfer])


def test_staggered_transfers_still_overlap():
    """The realistic case: platforms dequeue at slightly different times.

    Batch-mates do not request the pipe simultaneously — they are dequeued after
    differing platform backlogs. Coupling survives as long as the stagger is smaller
    than the transfer, which is why shallow queues matter: they keep the spread in
    dequeue times small relative to the transfer.
    """
    env = simpy.Environment()
    node = _node(env, BANDWIDTH_MBPS)
    transfer = ingress_transfer_time(_FakeTask(), BANDWIDTH_MBPS)
    stagger = transfer / 2.0
    waits = []

    def run(delay):
        yield env.timeout(delay)
        start = env.now
        with node.ingress_pipe.request() as pipe:
            yield pipe
            waits.append(env.now - start)
            yield env.timeout(transfer)

    env.process(run(0.0))
    env.process(run(stagger))
    env.run()

    assert waits[0] == pytest.approx(0.0)
    # The late arrival waits out the remainder of the first transfer.
    assert waits[1] == pytest.approx(transfer - stagger)
    assert waits[1] > 0


def test_node_accumulates_total_ingress_wait():
    env = simpy.Environment()
    node = _node(env, BANDWIDTH_MBPS)
    transfer = ingress_transfer_time(_FakeTask(), BANDWIDTH_MBPS)

    def run():
        start = env.now
        with node.ingress_pipe.request() as pipe:
            yield pipe
            node.ingress_wait_total += env.now - start
            yield env.timeout(transfer)

    env.process(run())
    env.process(run())
    env.run()

    assert node.ingress_wait_total == pytest.approx(transfer)


# --- 5. the ECT mirror agrees with the simulation -----------------------------------


def test_ect_mirror_matches_simulated_wait():
    """ECT must predict the wait the simulation actually charges, or baselines go stale."""
    env = simpy.Environment()
    node = _node(env, BANDWIDTH_MBPS)
    transfer = ingress_transfer_time(_FakeTask(), BANDWIDTH_MBPS)
    observed = []

    def run():
        start = env.now
        with node.ingress_pipe.request() as pipe:
            yield pipe
            observed.append(env.now - start)
            yield env.timeout(transfer)

    # Three simultaneous arrivals: waits are 0, 1x, 2x the transfer.
    for _ in range(3):
        env.process(run())
    env.run()

    for already_committed, actual in enumerate(observed):
        predicted = ingress_wait(node, _FakeTask(), added_on_node=already_committed)
        assert predicted == pytest.approx(actual)
