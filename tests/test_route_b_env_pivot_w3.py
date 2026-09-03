"""route_b env pivot W3 — generator overlapping scarce eligibility.

Unit coverage for src/generate_infrastructure.py's preinit.replica_overlap key
(:610-698 region, relaxing the disjoint assigned_platforms invariant). Full sweep-
generation / decoder-mask smoke is exercised separately as a throwaway generator run
(see the session report), not committed here — this file covers the pure placement
function in isolation, which is what W4's presets and any future rung actually depend on.
"""

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generate_infrastructure import (  # noqa: E402
    ReplicaStarvationError,
    generate_replica_placements_deterministic,
)

NODES = [
    {"node_name": "server0", "platforms": ["cpu", "gpu"]},
    {"node_name": "server1", "platforms": ["cpu"]},
]
SIM_INPUTS = {"task_types": {
    "typeA": {"platforms": ["cpu"]},
    "typeB": {"platforms": ["cpu"]},
}}


def _config(overlap=None, per_server=1):
    preinit = {"servers": ["server0", "server1"], "clients": []}
    if overlap is not None:
        preinit["replica_overlap"] = overlap
    return {
        "preinit": preinit,
        "replicas": {
            "typeA": {"per_server": per_server},
            "typeB": {"per_server": per_server},
        },
    }


def _keys(placements):
    return {(p["node_name"], p["platform_id"]) for p in placements}


def test_default_starves_the_second_type_and_now_says_so():
    """Absent replica_overlap key -> the original FCFS behavior: typeA claims both
    server cpu platforms and typeB gets none. Until 2026-09-03 that returned
    `typeB: []` silently and the dataset died one stage later as an unlabelled
    `System state capture FAILED`; it now raises at the point of cause, with counts."""
    rng = random.Random(1)
    with pytest.raises(ReplicaStarvationError) as exc:
        generate_replica_placements_deterministic(NODES, _config(), SIM_INPUTS, rng)
    msg = str(exc.value)
    assert "['typeB']" in msg
    assert "'typeA': 2" in msg and "'typeB': 0" in msg
    assert "replica_overlap=False" in msg


# A third server with TWO cpu platforms gives typeB something to claim after typeA's
# FCFS pass (one per server), so the disjointness / byte-identity properties can be
# asserted on a non-starving fixture. Platform ids: server0 {0 cpu, 1 gpu},
# server1 {2 cpu}, server2 {3 cpu, 4 cpu}.
NODES_3 = NODES + [{"node_name": "server2", "platforms": ["cpu", "cpu"]}]


def _config3(overlap=None, per_server=1):
    cfg = _config(overlap=overlap, per_server=per_server)
    cfg["preinit"]["servers"] = ["server0", "server1", "server2"]
    return cfg


def test_default_is_disjoint_and_unchanged():
    rng = random.Random(1)
    out = generate_replica_placements_deterministic(NODES_3, _config3(), SIM_INPUTS, rng)
    assert _keys(out["typeA"]) == {("server0", 0), ("server1", 2), ("server2", 3)}
    assert _keys(out["typeB"]) == {("server2", 4)}
    assert _keys(out["typeA"]).isdisjoint(_keys(out["typeB"]))


def test_replica_overlap_false_is_byte_identical_to_absent():
    rng1 = random.Random(1)
    rng2 = random.Random(1)
    out_absent = generate_replica_placements_deterministic(
        NODES_3, _config3(overlap=None), SIM_INPUTS, rng1)
    out_false = generate_replica_placements_deterministic(
        NODES_3, _config3(overlap=False), SIM_INPUTS, rng2)
    assert out_absent == out_false


def test_type_that_requested_nothing_is_not_starvation():
    """A type present in `replicas` with per_server=0 and per_client=0 legitimately
    gets no replicas; the guard must not fire on it."""
    cfg = _config()
    cfg["replicas"]["typeB"] = {"per_server": 0}
    rng = random.Random(1)
    out = generate_replica_placements_deterministic(NODES, cfg, SIM_INPUTS, rng)
    assert out["typeB"] == []


def test_replica_overlap_true_shares_platforms_across_task_types():
    rng = random.Random(1)
    out = generate_replica_placements_deterministic(
        NODES, _config(overlap=True), SIM_INPUTS, rng)
    assert _keys(out["typeA"]) == {("server0", 0), ("server1", 2)}
    assert _keys(out["typeB"]) == {("server0", 0), ("server1", 2)}
    assert _keys(out["typeA"]) == _keys(out["typeB"]), (
        "overlap must let typeB claim the SAME platforms typeA did")


def test_replica_overlap_still_forbids_a_type_double_booking_its_own_platform():
    """Even under overlap, ONE task type can never claim the same (node, platform_id)
    twice for itself — per_server capped at the number of suitable platforms on a
    node prevents this mechanically, verified directly rather than assumed."""
    rng = random.Random(1)
    # per_server=5 on a node offering only 1 cpu platform -> can create at most 1
    # replica there regardless of overlap, never a duplicate of itself.
    cfg = _config(overlap=True, per_server=5)
    out = generate_replica_placements_deterministic(NODES, cfg, SIM_INPUTS, rng)
    for task_type, placements in out.items():
        keys = [(p["node_name"], p["platform_id"]) for p in placements]
        assert len(keys) == len(set(keys)), f"{task_type} double-booked itself"


def test_replica_overlap_true_never_crashes_the_duplicate_check():
    """The original RuntimeError guarded against a real bug (self-duplication); under
    overlap, legitimate CROSS-type sharing must never trip it."""
    rng = random.Random(3)
    out = generate_replica_placements_deterministic(
        NODES, _config(overlap=True, per_server=2), SIM_INPUTS, rng)
    assert out["typeA"] and out["typeB"]  # both got replicas, no exception raised
