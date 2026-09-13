"""peer_affinity_warm_v1: a live-audit snapshot becomes a brute-force-labelled dataset.

Four pieces, each additive and off unless a snapshot / env flag asks for it:

  * live_audit.platform_queue_drain_seconds -- what a queued backlog will actually cost
    to serve (execution + I/O + latency + the peer transfers under HEROSIM_PEER_EXCHANGE=1),
    carried in every snapshot so the replay drains on the live clock, not the exec+comm one.
  * live_snapshot_seed -- `candidate: false` replicas are replayed as occupied platforms
    but kept out of `replicas`; a measured `queue_drain_seconds` overrides the formula.
  * knative_network_batch.KNATIVE_BATCH_BY_PEER_GROUP -- the reactive arm can batch by
    peer group so its snapshots are labellable.
  * make_warm_corpus -- alignment check, batch workload, candidate subsampling.
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.placement import live_audit  # noqa: E402
from src.placement.live_snapshot_seed import (  # noqa: E402
    _seed_platform_state,
    apply_live_snapshot_seed,
    build_live_snapshot_seed,
)


# ---------------------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------------------

TASK_TYPES = {
    "dnn1": {
        "executionTime": {"xavierCpu": 0.5, "rpiCpu": 2.0},
        "coldStartDuration": {"xavierCpu": 0.0, "rpiCpu": 0.0},
        "stateSize": {"nofs-dnn1": {"input": 0, "output": 0}},
    },
}


class _Event:
    def __init__(self):
        self.triggered = False

    def succeed(self):
        self.triggered = True


class _Store:
    def __init__(self, items):
        self.items = list(items)


class _Node:
    def __init__(self, name, network_map=None):
        self.node_name = name
        self.network_map = network_map or {}
        self.storage = _Store([])
        self.network = {"bandwidth": 1000.0}
        self.platforms = _Store([])
        self.available_platforms = 0


class _Platform:
    def __init__(self, pid, node, short="xavierCpu"):
        self.id = pid
        self.node = node
        self.type = {"shortName": short, "name": short}
        self.queue = _Store([])
        self.initialized = _Event()
        self.previous_task = None
        self.virtual_warmup_count = 0
        self.virtual_warmup_total_time = 0.0
        self.virtual_warmup_task_type = None
        self.transfer_calls = []

    def queue_length(self):
        return len(self.queue.items) + self.virtual_warmup_count

    def seed_virtual_warmup(self, task_type, task_type_name, count):
        self.virtual_warmup_count += count
        self.virtual_warmup_total_time += count * float(task_type["executionTime"][self.type["shortName"]])
        self.virtual_warmup_task_type = task_type_name

    def _payload_transfer_time(self, other_node, payload):
        self.transfer_calls.append((other_node, payload))
        return payload / 1e8  # 100 MB/s


def _task(tid, ttype="dnn1", source="client_node0", platform=None):
    return SimpleNamespace(id=tid, type=TASK_TYPES[ttype], node_name=source, platform=platform,
                           planned_node_name=None, scheduled=_Event())


# ---------------------------------------------------------------------------------------
# platform_queue_drain_seconds
# ---------------------------------------------------------------------------------------

@pytest.fixture
def peer_env(monkeypatch):
    monkeypatch.setenv("HEROSIM_PEER_EXCHANGE", "1")


def test_drain_charges_execution_latency_and_known_peer_transfers(peer_env):
    node_a = _Node("node0", {"node1": {"latency": 0.05}, "client_node0": {"latency": 0.01}})
    node_b = _Node("node1")
    plat = _Platform(1, node_a)
    peer_plat = _Platform(2, node_b)
    t1 = _task(1)
    t2 = _task(2)
    peer_of_t1 = _task(7, platform=peer_plat)          # placed on the other node
    unplaced_peer = _task(8)                             # neither placed nor planned
    plat.queue.items = [t1, t2]
    orch = SimpleNamespace(
        peer_exchange={1: {7: 1e8, 8: 5e8}, 2: {}},
        task_by_id={1: t1, 2: t2, 7: peer_of_t1, 8: unplaced_peer},
    )
    drain = live_audit.platform_queue_drain_seconds(plat, orch)
    # two tasks x (0.5 exec + 0.002 storage I/O + 0.01 source latency) + one known peer:
    # 1.0 transfer + 0.05 latency
    assert drain == pytest.approx(2 * (0.5 + 0.002 + 0.01) + 1.0 + 0.05)
    # priced once per peer node (per byte), and the unplaced peer is not charged
    assert plat.transfer_calls == [("node1", 1.0)]


def test_drain_ignores_peers_when_exchange_physics_is_off(monkeypatch):
    monkeypatch.setenv("HEROSIM_PEER_EXCHANGE", "0")
    node_a = _Node("node0", {"node1": {"latency": 0.05}})
    plat = _Platform(1, node_a)
    t1 = _task(1, source="node0")
    plat.queue.items = [t1]
    orch = SimpleNamespace(peer_exchange={1: {7: 1e8}}, task_by_id={1: t1, 7: _task(7, platform=_Platform(2, _Node("node1")))})
    assert live_audit.platform_queue_drain_seconds(plat, orch) == pytest.approx(0.5 + 0.002)


def test_drain_memo_walks_each_queue_once(peer_env):
    plat = _Platform(1, _Node("node0"))
    plat.queue.items = [_task(1, source="node0"), _task(2, source="node0")]
    memo = {}
    first = live_audit.platform_queue_drain_seconds(plat, None, memo)
    plat.queue.items.append(_task(3, source="node0"))   # a later change is NOT re-walked
    assert live_audit.platform_queue_drain_seconds(plat, None, memo) == first
    assert live_audit.platform_queue_drain_seconds(plat, None, None) == pytest.approx(3 * 0.502)


def test_drain_carries_an_existing_virtual_backlog(peer_env):
    plat = _Platform(1, _Node("node0"))
    plat.virtual_warmup_total_time = 123.0
    assert live_audit.platform_queue_drain_seconds(plat, None) == pytest.approx(123.0)


# ---------------------------------------------------------------------------------------
# live_snapshot_seed
# ---------------------------------------------------------------------------------------

def _snapshot(candidate_flags=(True, False), drains=(0.0, 0.0)):
    return {
        "full_queue_snapshot": {"node0:1": 3, "node0:2": 5},
        "replicas_by_type": {
            "dnn1": [
                {"node_name": "node0", "platform_id": 1, "initialized": True, "queue_length": 3,
                 "platform_type": "xavierCpu", "candidate": candidate_flags[0],
                 "queue_drain_seconds": drains[0]},
                {"node_name": "node0", "platform_id": 2, "initialized": True, "queue_length": 5,
                 "platform_type": "xavierCpu", "candidate": candidate_flags[1],
                 "queue_drain_seconds": drains[1]},
            ]
        },
        "tasks": [],
    }


def test_seed_keeps_candidate_flag_and_drain_and_defaults_true():
    seed = build_live_snapshot_seed(_snapshot((True, False), (10.0, 20.0)))
    specs = {int(s["platform_id"]): s for s in seed["replicas_by_type"]["dnn1"]}
    assert specs[1]["candidate"] is True and specs[2]["candidate"] is False
    assert specs[2]["queue_drain_seconds"] == 20.0
    legacy = build_live_snapshot_seed({"replicas_by_type": {"dnn1": [{"node_name": "node0", "platform_id": 1}]}})
    assert legacy["replicas_by_type"]["dnn1"][0]["candidate"] is True


def test_apply_seed_reserves_non_candidates_and_uses_measured_drain():
    node = _Node("node0")
    p1, p2 = _Platform(1, node), _Platform(2, node)
    node.platforms.items = [p1, p2]
    nodes = _Store([node])
    sim_data = SimpleNamespace(task_types=TASK_TYPES)
    seed = build_live_snapshot_seed(_snapshot((True, False), (0.0, 99.0)))
    replicas = apply_live_snapshot_seed(nodes, sim_data, None, None, seed)
    assert replicas["dnn1"] == {(node, p1)}
    assert getattr(p2, "snapshot_reserved", False) is True
    assert not getattr(p1, "snapshot_reserved", False)
    # both are seeded: p1 by the formula (3 x (0.5 exec + 0.002 I/O)), p2 by the measured drain
    assert p1.virtual_warmup_count == 3 and p1.virtual_warmup_total_time == pytest.approx(3 * 0.502)
    assert p2.virtual_warmup_count == 5 and p2.virtual_warmup_total_time == pytest.approx(99.0)
    assert p1.queue_length() == 3 and p2.queue_length() == 5


def test_seed_platform_state_formula_unchanged_without_measured_drain():
    node = _Node("node0")
    p = _Platform(4, node)
    _seed_platform_state({("node0", 4): (node, p)}, SimpleNamespace(task_types=TASK_TYPES),
                         {"node_name": "node0", "platform_id": 4, "queue_length": 2, "task_type_hint": "dnn1"})
    assert p.virtual_warmup_total_time == pytest.approx(2 * 0.502)


# ---------------------------------------------------------------------------------------
# knative batch: peer-group batching flag + closure
# ---------------------------------------------------------------------------------------

def test_knative_batch_peer_group_flag_and_closure(monkeypatch):
    from src.policy.knative_network_batch import scheduler as knb

    monkeypatch.delenv("KNATIVE_BATCH_BY_PEER_GROUP", raising=False)
    assert knb._read_batch_by_peer_group() is False
    monkeypatch.setenv("KNATIVE_BATCH_BY_PEER_GROUP", "1")
    assert knb._read_batch_by_peer_group() is True
    monkeypatch.setenv("KNATIVE_BATCH_BY_PEER_GROUP", "maybe")
    with pytest.raises(ValueError):
        knb._read_batch_by_peer_group()
    orch = SimpleNamespace(peer_exchange={0: {1: 1.0}, 1: {0: 1.0, 2: 1.0}, 2: {1: 1.0}, 5: {6: 1.0}}, task_by_id={})
    assert knb._peer_group(orch, 0) == {0, 1, 2}
    assert knb._peer_group(orch, 6) == {6}   # table is per-source; 6 has no row of its own


# ---------------------------------------------------------------------------------------
# make_warm_corpus
# ---------------------------------------------------------------------------------------

def _live_snapshot(first_id=20, n=10):
    cands_a = [{"queue_key": f"node0:{i}", "node_name": "node0", "platform_id": i, "platform_type": "xavierCpu"} for i in (1, 2, 3, 4)]
    cands_b = [{"queue_key": f"node1:{i}", "node_name": "node1", "platform_id": i, "platform_type": "xavierCpu"} for i in (5, 6, 7)]
    tasks = []
    for k in range(n):
        ttype = "dnn1" if k % 2 == 0 else "dnn2"
        tasks.append({"task_id": first_id + k, "task_type": ttype, "source_node": "client_node0",
                      "candidates": list(cands_a if ttype == "dnn1" else cands_b)})
    reps = {"dnn1": [{"node_name": "node0", "platform_id": i} for i in (1, 2, 3, 4)],
            "dnn2": [{"node_name": "node1", "platform_id": i} for i in (5, 6, 7)]}
    return {"snapshot_id": 3, "time": 12.0, "tasks": tasks, "replicas_by_type": reps, "full_queue_snapshot": {}}


def _trace(n_events=40):
    events = []
    for gid in range(n_events):
        ttype = "dnn1" if gid % 2 == 0 else "dnn2"
        events.append({"timestamp": gid * 0.001, "node_name": "client_node0",
                       "application": {"name": f"nofs-{ttype}", "dag": {ttype: []}, "demand_scale": {ttype: 1.2}},
                       "qos": {"name": "medium", "maxDurationDeviation": 15}})
    pairs = [[g, g + 1, 2e8] for g in range(0, n_events, 10) for _ in [0]] + [[g + 2, g + 5, 5e7] for g in range(0, n_events, 10)]
    return {"events": events, "peer_exchange": pairs}


def test_alignment_check_rejects_partial_and_shifted_batches():
    from scripts_cosim.make_warm_corpus import SnapshotRejected, check_aligned_peer_group

    assert check_aligned_peer_group(_live_snapshot(20), 10) == list(range(20, 30))
    with pytest.raises(SnapshotRejected):
        check_aligned_peer_group(_live_snapshot(23), 10)
    with pytest.raises(SnapshotRejected):
        check_aligned_peer_group(_live_snapshot(20, n=7), 10)


def test_batch_workload_reindexes_pairs_and_rejects_outside_pairs():
    from scripts_cosim.make_warm_corpus import SnapshotRejected, build_batch_workload

    order = ["nofs-dnn1", "nofs-dnn2"]
    wl = build_batch_workload(_live_snapshot(20), _trace(), list(range(20, 30)), order)
    assert len(wl["events"]) == 10 and all(e["timestamp"] == 0.0 for e in wl["events"])
    # grouped by application in wsc order: the five dnn1 events (trace ids 20,22,..,28)
    # come first, then the five dnn2 ones -- the order the sweep assigns task ids in
    assert [e["application"]["name"] for e in wl["events"]] == ["nofs-dnn1"] * 5 + ["nofs-dnn2"] * 5
    assert wl["trace_task_ids"] == [20, 22, 24, 26, 28, 21, 23, 25, 27, 29]
    assert wl["events"][0]["application"]["demand_scale"] == {"dnn1": 1.2}
    # pairs (20,21) and (22,25) in trace ids -> (0,5) and (1,7) in co-sim ids
    assert sorted(wl["peer_exchange"]) == [[0, 5, 2e8], [1, 7, 5e7]]
    bad = _trace()
    bad["peer_exchange"].append([25, 31, 1e6])
    with pytest.raises(SnapshotRejected):
        build_batch_workload(_live_snapshot(20), bad, list(range(20, 30)), order)
    with pytest.raises(SnapshotRejected):
        build_batch_workload(_live_snapshot(20), _trace(), list(range(20, 30)), ["nofs-dnn1"])


def test_choose_candidates_fits_target_and_flags_the_rest():
    from scripts_cosim.make_warm_corpus import choose_candidates, flag_candidates

    snap = _live_snapshot(20)
    subset, record = choose_candidates(snap, random.Random(1), target_combos=3 ** 10, max_combos=10 ** 6)
    assert record["num_combos"] <= 3 ** 10 and min(record["per_task_candidates"]) >= 1
    assert record["full_live_product"] == 4 ** 5 * 3 ** 5
    flagged = flag_candidates(snap, subset)
    for ttype, specs in flagged["replicas_by_type"].items():
        chosen = {s["platform_id"] for s in specs if s["candidate"]}
        assert {f"{s['node_name']}:{s['platform_id']}" for s in specs if s["candidate"]} == subset[ttype]
        assert len(chosen) == record["R"] or len(chosen) == len(specs)
    for task in flagged["tasks"]:
        assert all(c["queue_key"] in subset[task["task_type"]] for c in task["candidates"])
    # the same seed draws the same subset (the draw is part of the dataset's provenance)
    subset2, record2 = choose_candidates(snap, random.Random(1), target_combos=3 ** 10, max_combos=10 ** 6)
    assert subset2 == subset and record2 == record


def test_infrastructure_carries_the_seed_and_empty_tables(tmp_path):
    from scripts_cosim.make_warm_corpus import build_infrastructure, choose_candidates, flag_candidates
    from src.executecosimulation import load_deterministic_infrastructure_data

    snap = _live_snapshot(20)
    subset, _ = choose_candidates(snap, random.Random(2), 3 ** 10, 10 ** 6)
    base = {"network_maps": {"node0": {"client_node0": 0.01}}, "metadata": {"seed": 9}, "link_topology": None}
    infra = build_infrastructure(base, flag_candidates(snap, subset), {"snapshot_id": 3})
    # the classic tables list exactly the candidate slate, typed, for the scoring tools
    listed = {t: {f"{r['node_name']}:{r['platform_id']}" for r in reps} for t, reps in infra["replica_placements"].items()}
    assert listed == {t: set(s) for t, s in subset.items()}
    assert all(r["platform_type"] == "xavierCpu" for reps in infra["replica_placements"].values() for r in reps)
    assert set(infra["queue_distributions"]) == set(subset)
    assert infra["metadata"]["warm_snapshot"] == {"snapshot_id": 3}
    flags = [s["candidate"] for specs in infra["live_snapshot_seed"]["replicas_by_type"].values() for s in specs]
    assert True in flags and False in flags
    path = tmp_path / "infrastructure.json"
    path.write_text(json.dumps(infra))
    cfg = {"nodes": {"client_nodes": {"count": 1}, "server_nodes": {"count": 2}},
           "pci": {"xavier": {"specs": {"platforms": ["xavierCpu"], "storage": [], "memory": 8}}}}
    loaded = load_deterministic_infrastructure_data(cfg, path)
    assert loaded["live_snapshot_seed"] == infra["live_snapshot_seed"]
