from scripts_cosim import uncut_xarm as X


def _snap(pairs, batch):
    return {"fidelity": {"batch": [{"gid": g} for g in batch], "pairs": pairs}}


def test_pair_counts_split_same_and_cross_node_pairs_in_gid_order():
    snap = _snap([[1, 2, 5.0], [2, 3, 5.0], [3, 9, 5.0]], [3, 1, 2])          # 9 is outside the batch; plan is in gid order 1, 2, 3
    r = X.pair_counts(snap, [[7, 0], [7, 1], [8, 0]])
    assert r == {"same_node": 1, "cross_node": 1, "nodes_used": 2}


def test_summarize_task_weighted_and_slate_shares():
    def row(sid, h, i, b, ils_ranks):
        return {"state_id": sid, "bucket": "11_50", "n_batch": len(ils_ranks), "ranks": {"ils": ils_ranks, "hand": [[1, 5]] * len(ils_ranks)}, "rtt_hand": h, "rtt_ils": i, "rtt_b": b,
                "b_cap_hit": False, "b_equals_hand": b == h, "b_equals_ils": b == i,
                "pairs": {"hand": {"cross_node": 2}, "ils": {"cross_node": 0}, "cd_exactS_ib0": {"cross_node": 1}}}
    out = X.summarize([row("a", 200, 100, 150, [[1, 6], [7, 9]]), row("b", 100, 100, 100, [[2, 4], [1, 4]])], "cd_exactS_ib0")["pooled"]
    assert out["ils_outside_top5_task_share"] == 0.25 and out["states_with_any_ils_task_outside_top5"] == 0.5
    assert abs(out["hand_task_weighted_pct"] - 50.0) < 1e-9 and abs(out["cd_exactS_ib0_task_weighted_pct"] - 25.0) < 1e-9
    assert out["cross_node_pairs_mean"] == {"hand": 2, "ils": 0, "cd_exactS_ib0": 1}
