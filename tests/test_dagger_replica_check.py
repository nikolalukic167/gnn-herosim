from scripts_cosim import dagger_replica_check as R


def _snap(rep_nodes, cand_nodes=(), types=("rf", "dnn2")):
    return {"tasks": [{"task_id": 2, "task_type": "rf", "candidates": [{"node_name": n, "platform_id": p} for n, p in cand_nodes]},
                      {"task_id": 3, "task_type": "dnn2", "candidates": []}],
            "replicas_by_type": {"rf": [{"node_name": n, "node_id": i, "platform_id": p} for n, i, p in rep_nodes], "dnn2": [{"node_name": "node1", "node_id": 161, "platform_id": 803}]},
            "full_queue_snapshot": {}}


def test_a_served_replica_in_the_replay_set_passes_by_id():
    s = _snap([("node7", 167, 841)])
    assert R.check_state(s, {"chosen": [[2, 167, 841], [3, 161, 803]]}) == []


def test_a_served_replica_missing_from_the_replay_set_fails():
    s = _snap([("node7", 167, 841)])
    p = R.check_state(s, {"chosen": [[2, 173, 864]]})
    assert len(p) == 1 and "not in the capture's replica payload" in p[0]
    p = R.check_state(s, {"chosen": [[2, 167, 999]]})
    assert len(p) == 1 and "not in the replay's replica set" in p[0]


def test_a_replica_offered_as_a_batch_candidate_is_in_the_replay_set():
    s = _snap([("node7", 167, 841)], cand_nodes=[("node9", 864)])
    assert R.check_state(s, {"chosen": [[2, "node9", 864]]}) == []
    assert R.check_state(s, {"chosen": []}) == [] and R.check_state(s, {"chosen": None}) == []          # deferred tasks are not in the served plan
