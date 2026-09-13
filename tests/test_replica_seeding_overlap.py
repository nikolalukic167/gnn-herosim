"""`integrate_initial_replicas` under overlapping (non-unique) replica placements.

peer_affinity_v1, 2026-09-10. A corpus generated with `--allow-non-unique-replicas` hosts
replicas of several task types on ONE platform. The shared helper used to claim each seeded
slot unconditionally, so the second task type to reach a shared slot found it gone and raised
`missing from available_resources (double-book or create_nodes mismatch)` -- which made every
reactive and GNN live arm refuse such a corpus while `determined_determined` replayed it fine.

The four properties pinned here:

1. **Overlap is accepted** and the physical slot is charged exactly once (availability, memory).
2. **Contention is seeded for every hosting type**, not just the first -- the autoscaler indexes
   `average_contention[task_type][(node.id, platform.id)]` and would KeyError otherwise.
3. **A genuinely absent slot still fails loud**, with the original message.
4. **Non-overlap behaviour is unchanged** (the regression guard for the program's baseline).

Run: pipenv run python3 -m pytest tests/test_replica_seeding_overlap.py -q
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.placement.replica_seeding import integrate_initial_replicas  # noqa: E402

TASK_TYPES = {t: {"memoryRequirements": {"cpu": 2.0}} for t in ("a", "b", "c")}


class FakePlatform:
    def __init__(self, pid: int):
        self.id = pid
        self.type = {"shortName": "cpu"}

    def __repr__(self):
        return f"P{self.id}"


class FakeNode:
    def __init__(self, name: str, platforms):
        self.node_name = name
        self.id = int(name[-1])
        self.platforms = list(platforms)
        self.available_platforms = len(self.platforms)
        self.available_memory = 100.0

    def __repr__(self):
        return self.node_name


def rig():
    p1, p2 = FakePlatform(1), FakePlatform(2)
    node = FakeNode("node0", [p1, p2])
    available = {node: {p1, p2}}
    replicas = {t: set() for t in TASK_TYPES}
    contention = defaultdict(dict)
    return node, p1, p2, available, replicas, contention


def call(replicas, available, initial, contention):
    return integrate_initial_replicas(
        replicas=replicas, available_resources=available, initial_replicas=initial,
        task_types=TASK_TYPES, average_contention=contention, label="Test")


# 1 + 2 ---------------------------------------------------------------------------------

def test_shared_platform_is_accepted_and_charged_once():
    node, p1, _p2, available, replicas, contention = rig()
    total = call(replicas, available, {"a": {(node, p1)}, "b": {(node, p1)}}, contention)
    assert total == 2                                   # both replica tuples integrated
    assert available[node] == {_p2}                     # the shared slot left the pool once
    assert node.available_platforms == 1                # charged once, not twice
    assert node.available_memory == pytest.approx(98.0)  # 100 - one 2.0 charge
    assert replicas["a"] == {(node, p1)} and replicas["b"] == {(node, p1)}


def test_contention_is_seeded_for_every_hosting_type():
    node, p1, _p2, available, replicas, contention = rig()
    call(replicas, available, {"a": {(node, p1)}, "b": {(node, p1)}, "c": {(node, p1)}}, contention)
    for t in ("a", "b", "c"):
        assert contention[t][(node.id, p1.id)] == 0.0


def test_three_types_on_one_slot_charge_the_slot_once():
    node, p1, p2, available, replicas, contention = rig()
    call(replicas, available, {"a": {(node, p1), (node, p2)}, "b": {(node, p1)}, "c": {(node, p1)}}, contention)
    assert available[node] == set()
    assert node.available_platforms == 0
    assert node.available_memory == pytest.approx(96.0)   # two distinct slots, 2.0 each


# 3 -------------------------------------------------------------------------------------

def test_platform_that_never_existed_still_fails_loud():
    node, p1, _p2, available, replicas, contention = rig()
    ghost = FakePlatform(99)
    with pytest.raises(RuntimeError, match="missing from available_resources"):
        call(replicas, available, {"a": {(node, ghost)}}, contention)


def test_unknown_task_type_still_fails_loud():
    node, p1, _p2, available, replicas, contention = rig()
    with pytest.raises(KeyError, match="unknown task_type"):
        call(replicas, available, {"zz": {(node, p1)}}, contention)


# 4 -- the regression guard for every existing unique-replica corpus --------------------

def test_disjoint_seeding_is_bit_identical_to_the_old_behaviour():
    node, p1, p2, available, replicas, contention = rig()
    total = call(replicas, available, {"a": {(node, p1)}, "b": {(node, p2)}}, contention)
    assert total == 2
    assert available[node] == set()
    assert node.available_platforms == 0
    assert node.available_memory == pytest.approx(96.0)   # both slots charged
    assert contention["a"][(node.id, p1.id)] == 0.0
    assert contention["b"][(node.id, p2.id)] == 0.0


def test_empty_and_absent_initial_replicas_are_noops():
    node, _p1, _p2, available, replicas, contention = rig()
    assert call(replicas, available, {}, contention) == 0
    assert call(replicas, available, {"a": set()}, contention) == 0
    assert available[node] == {_p1, _p2} and node.available_platforms == 2
