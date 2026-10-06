import json

import pytest

from scripts_cosim.important.generate_sampled_slate_snapshots import (
    _peer_workload,
    group_events,
    group_peer_exchange,
    sampled_plans,
    snapshot_replicas,
)


def _fixture():
    tasks = [
        {
            "task_id": 40 + i,
            "task_type": "dnn1",
            "source_node": "client_node0",
            "candidates": [
                {"node_id": 1, "platform_id": 7, "execution_time": 0.1},
                {"node_id": 2, "platform_id": 8, "execution_time": 0.2},
            ],
        }
        for i in range(3)
    ]
    snapshot = {"snapshot_id": 4, "tasks": tasks}
    events = [
        {"application": {"dag": {"dnn1": []}, "demand_scale": {"dnn1": float(i + 1)}}, "node_name": "client_node0"}
        for i in range(43)
    ]
    workload = {"events": events, "peer_exchange": [[40, 42, 123.0], [41, 42, 456.0], [0, 1, 1.0]]}
    return snapshot, workload


def test_peer_pairs_remap_global_ids_and_keep_payloads():
    snapshot, workload = _fixture()
    assert group_peer_exchange(snapshot, workload) == [[0, 2, 123.0], [1, 2, 456.0]]


def test_peer_pair_crossing_group_fails_loud():
    snapshot, workload = _fixture()
    workload["peer_exchange"].append([40, 39, 1.0])
    with pytest.raises(ValueError, match="crosses snapshot boundary"):
        group_peer_exchange(snapshot, workload)


def test_demand_scales_restored_from_global_workload():
    snapshot, workload = _fixture()
    events = group_events(snapshot, workload)
    mini = _peer_workload(snapshot["tasks"], [[0, 2, 123.0]], events)
    assert [e["application"]["demand_scale"]["dnn1"] for e in mini["events"]] == [41.0, 42.0, 43.0]
    assert mini["peer_exchange"] == [[0, 2, 123.0]]
    del workload["events"][40]["application"]["demand_scale"]
    with pytest.raises(ValueError, match="demand_scale"):
        group_events(snapshot, workload)


def test_sampled_slate_is_replayable_and_contains_split_peer_plan():
    snapshot, _ = _fixture()
    first = sampled_plans(snapshot["tasks"], 8, 101)
    assert first == sampled_plans(snapshot["tasks"], 8, 101)
    assert len({json.dumps(plan, sort_keys=True) for plan in first}) == 8
    assert any(plan[0][0] != plan[2][0] for plan in first)


def test_scheduling_time_replicas_preserve_all_task_types():
    snapshot, _ = _fixture()
    snapshot["replicas_by_type"] = {
        "dnn1": [{"node_name": "node1", "platform_id": 7}],
        "rf": [{"node_name": "node2", "platform_id": 8}],
    }
    assert snapshot_replicas(snapshot) == {
        "dnn1": [["node1", 7]],
        "rf": [["node2", 8]],
    }
    with pytest.raises(ValueError, match="scheduling-time replicas"):
        snapshot_replicas({"tasks": snapshot["tasks"]})


def test_proposal_slate_keeps_three_exact_seeds_and_mutates_locally():
    snapshot, _ = _fixture()
    proposals = {
        "gnn": {str(i): [1, 7] for i in range(3)},
        "mpoff": {str(i): [2, 8] for i in range(3)},
        "hand": {str(i): [1, 7] if i != 1 else [2, 8] for i in range(3)},
    }
    plans = sampled_plans(snapshot["tasks"], 7, 101, proposals)
    for proposal in proposals.values():
        assert any(all(list(plan[i]) == proposal[str(i)] for i in range(3)) for plan in plans)
    assert plans == sampled_plans(snapshot["tasks"], 7, 101, proposals)
    proposals["gnn"]["0"] = [99, 99]
    with pytest.raises(ValueError, match="infeasible"):
        sampled_plans(snapshot["tasks"], 7, 101, proposals)


def test_proposal_slate_covers_every_candidate_edge_or_rejects_budget():
    snapshot, _ = _fixture()
    snapshot["tasks"][0]["candidates"].append(
        {"node_id": 3, "platform_id": 9, "execution_time": 0.3}
    )
    proposals = {
        "gnn": {str(i): [1, 7] for i in range(3)},
        "mpoff": {str(i): [2, 8] for i in range(3)},
        "hand": {str(i): [1, 7] if i != 1 else [2, 8] for i in range(3)},
    }
    plans = sampled_plans(snapshot["tasks"], 7, 101, proposals)
    for i, task in enumerate(snapshot["tasks"]):
        expected = {(c["node_id"], c["platform_id"]) for c in task["candidates"]}
        assert {plan[i] for plan in plans} == expected
    with pytest.raises(ValueError, match="mandatory"):
        sampled_plans(snapshot["tasks"], 3, 101, proposals)
