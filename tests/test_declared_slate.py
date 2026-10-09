"""The node's declared candidate pruning (src/placement/declared_slate.py): top 5 per task by standalone cost with a deterministic
tie-break, sub-batches of <= 4 above 100,000 plans, and the serving mode recorded in the checkpoint and refused on mismatch."""
import pytest

from src.placement import declared_slate as D


def cand(node, plat, drain=0.0, exec_=1.0, lat=0.0, init=True, cold=5.0):
    return {"node_id": node, "platform_id": plat, "queue_drain_seconds": drain, "execution_time": exec_,
            "network_latency": lat, "initialized": init, "cold_start_time": cold}


def test_standalone_cost_is_the_peer_greedy_base_without_exchange():
    assert D.standalone_cost(cand(0, 0, drain=2.0, exec_=3.0, lat=0.5)) == pytest.approx(5.5)
    assert D.standalone_cost(cand(0, 0, drain=2.0, exec_=3.0, lat=0.5, init=False, cold=7.0)) == pytest.approx(12.5)
    assert D.standalone_cost({}) == 0.0


def test_top_five_by_cost_with_node_platform_tie_break():
    cs = [cand(n, 0, drain=float(n % 3)) for n in range(9)]  # costs 1+drain: ties across nodes
    kept = D.prune_candidates(cs)
    assert len(kept) == 5
    costs = [D.standalone_cost(c) for c in kept]
    assert costs == sorted(costs)
    assert [c["node_id"] for c in kept] == [0, 3, 6, 1, 4]       # cost 1: nodes 0,3,6; cost 2: nodes 1,4,(7)
    assert D.prune_candidates(list(reversed(cs))) == kept          # order of the input does not matter
    assert D.prune_candidates(cs[:3]) == sorted(cs[:3], key=lambda c: (D.standalone_cost(c), c["node_id"], c["platform_id"]))


def test_plan_space_and_one_group_below_the_cap():
    tasks = [{"task_id": i, "candidates": [cand(n, 0) for n in range(8)]} for i in range(6)]
    sl = D.slate(tasks, [(0, 1), (1, 2)])
    assert sl.plans == 5 ** 6 == 15_625 and sl.groups == [list(range(6))]
    assert sl.pruned and not sl.sub_batched and sl.full_sizes == [8] * 6


def test_a_large_batch_is_split_into_sub_batches_of_at_most_four():
    n = 9                                                           # 5**9 = 1,953,125 > 100,000
    tasks = [{"task_id": 100 + i, "candidates": [cand(k, 0) for k in range(6)]} for i in range(n)]
    pairs = [(100 + i, 101 + i) for i in range(n - 1)]             # a chain
    sl = D.slate(tasks, pairs)
    assert sl.sub_batched and sl.plans == 5 ** 9
    assert sorted(p for g in sl.groups for p in g) == list(range(n))
    assert all(len(g) <= D.SUB_BATCH for g in sl.groups)
    assert sl.groups == [[0, 1, 2, 3], [4, 5, 6, 7], [8]]            # a chain is walked in order and cut every 4
    assert D.slate(tasks, pairs).groups == sl.groups                 # deterministic


def test_sub_batches_follow_the_peer_graph_not_the_id_order():
    # two stars: ids 0,2,4,6 around 0 and ids 1,3,5,7 around 1; ids interleave
    pairs = [(0, 2), (0, 4), (0, 6), (1, 3), (1, 5), (1, 7)]
    assert D.sub_batches(range(8), pairs) == [[0, 2, 4, 6], [1, 3, 5, 7]]


def test_max_plans_test_hook(monkeypatch):
    tasks = [{"task_id": i, "candidates": [cand(k, 0) for k in range(5)]} for i in range(6)]
    assert not D.slate(tasks, [(0, 1)]).sub_batched
    monkeypatch.setenv(D.MAX_PLANS_ENV, "400")
    assert D.slate(tasks, [(0, 1)]).sub_batched


def test_serving_mode_is_refused_on_mismatch(monkeypatch):
    monkeypatch.delenv(D.ENV, raising=False)
    D.require_matching_slate(None, what="m")                          # unpruned checkpoint, unpruned run
    with pytest.raises(ValueError, match="candidate_slate='declared_pruning_v1'"):
        D.require_matching_slate(D.RULE, what="m")                   # pruned checkpoint, plain run
    monkeypatch.setenv(D.ENV, D.RULE)
    D.require_matching_slate(D.RULE, what="m")
    with pytest.raises(ValueError, match="candidate_slate=None"):
        D.require_matching_slate(None, what="m")                     # plain checkpoint, pruned run
    monkeypatch.setenv(D.ENV, "something_else")
    with pytest.raises(ValueError, match="expected unset"):
        D.serving_slate()


def test_loaders_refuse_a_checkpoint_trained_on_another_slate(tmp_path, monkeypatch):
    import json

    import torch

    from src.executesimulation import load_gnn_model
    from src.policy.gnn.gnn_model import TaskPlacementGNN

    model = TaskPlacementGNN(task_feature_dim=3, platform_feature_dim=14, embedding_dim=64, hidden_dim=64, num_layers=3, edge_dim=5)
    path = tmp_path / "m.pt"
    torch.save(model.state_dict(), path)
    path.with_suffix(".contract.json").write_text(json.dumps({
        "inference_feature_layout": "dim22", "queue_feature_contract": "legacy_v0", "queue_norm_mode": "scheduler_adaptive",
        "topology_feature_contract": "src_index_v0", "candidate_slate": D.RULE}))
    monkeypatch.delenv(D.ENV, raising=False)
    with pytest.raises(ValueError, match="candidate_slate"):
        load_gnn_model(path)
    monkeypatch.setenv(D.ENV, D.RULE)
    model2, _ = load_gnn_model(path)
    assert model2 is not None


def test_restrict_to_gids_makes_a_sub_batch_snapshot():
    from src.placement.fidelity_replay import restrict_to_gids

    snap = {"tasks": [{"task_id": g} for g in (10, 11, 12)],
            "fidelity": {"batch": [{"gid": 10}, {"gid": 11}, {"gid": 12}],
                         "pairs": [[10, 11, 1.0], [11, 12, 1.0], [3, 12, 1.0], [3, 10, 1.0]],
                         "peers": {"10": [[11, "n0", 1.0], [99, "n1", 2.0]], "12": [[11, "n0", 1.0]], "3": [[12, None, 1.0]]},
                         "queued": [{"gid": 3}]}}
    out = restrict_to_gids(snap, [10, 11])
    assert [t["task_id"] for t in out["tasks"]] == [10, 11]
    f = out["fidelity"]
    assert [r["gid"] for r in f["batch"]] == [10, 11]
    assert f["pairs"] == [[10, 11, 1.0], [3, 10, 1.0]]                # pairs to the sibling 12 are invisible
    assert f["peers"] == {"10": [[11, "n0", 1.0], [99, "n1", 2.0]], "3": []}
    assert len(snap["fidelity"]["batch"]) == 3                         # the input is untouched
