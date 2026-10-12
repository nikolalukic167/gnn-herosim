"""surrogate_gap_screen: the table logic (arms, budgeted search, aggregation) on synthetic plan tables; the replay passes run on datalab."""
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts_cosim"))

import surrogate_gap_screen as G  # noqa: E402

C = [(0, 10), (1, 11), (2, 12)]


def _table(fn, n=3, cands=None):
    cands = cands or [C] * n
    combos = sorted(itertools.product(*cands))
    return combos, np.array([fn(c) for c in combos], float)


def test_candidates_of_requires_the_full_product():
    combos, _ = _table(lambda c: 1.0)
    assert G.candidates_of(combos) == [C, C, C]
    with pytest.raises(RuntimeError, match="full product"):
        G.candidates_of(combos[:-1])


def test_load_table_sorts_and_keeps_the_lowest_duplicate(tmp_path):
    d = tmp_path / "ds_0"
    (d / "placements").mkdir(parents=True)
    rows = [{"placement_plan": {"1": [1, 11], "0": [0, 10]}, "rtt": 5.0}, {"placement_plan": {"0": [0, 10], "1": [1, 11]}, "rtt": 4.0},
            {"placement_plan": {"0": [0, 10], "1": [0, 10]}, "rtt": 3.0}, {"note": "x"}]
    (d / "placements" / "placements.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    combos, rtt = G.load_table(d)
    assert combos == [((0, 10), (0, 10)), ((0, 10), (1, 11))] and rtt.tolist() == [3.0, 4.0]


def test_label_shape_ties_gap_and_rank():
    combos, rtt = _table(lambda c: 2.0 if c[0] == c[1] == c[2] == C[0] else (2.0 if c == (C[1],) * 3 else 3.0 + c[0][0]))
    s = np.arange(len(rtt), dtype=float)[::-1].copy()   # best-S is the last row
    sh = G.label_shape(combos, rtt, s_cd=s, s_ib0=s)
    assert sh["opt"] == 2.0 and sh["opt_ties"] == 2 and sh["multi_node_opt"] is False
    assert sh["top2_gap_rel"] == pytest.approx(0.5)      # the next distinct latency is 3.0
    assert sh["opt_rank_cd"] == len(rtt)                  # the canonical optimum is row 0, S-ranked last


def test_ranked_curve_is_the_best_of_the_top_k_and_k0_is_the_pick():
    rtt = np.array([5.0, 1.0, 3.0, 2.0, 4.0])
    s = np.array([0.0, 1.0, 2.0, 3.0, 4.0])              # S puts the 5.0 plan first
    c = G.ranked_curve(rtt, s, ks=(1, 2, 4))
    assert c[0] == pytest.approx(4.0) and c[1] == pytest.approx(4.0) and c[2] == pytest.approx(0.0) and c[4] == 0.0
    g = G.ranked_curve(rtt, -s, ks=(1, 2), descending=True)   # a higher score is better: same order
    assert g == c | {4: 0.0} or g[0] == c[0]


def test_random_curve_matches_enumeration():
    rtt = np.array([4.0, 1.0, 3.0, 2.0, 6.0, 5.0])
    for k in (1, 2, 3):
        exp = np.mean([min(rtt[list(s)]) for s in itertools.combinations(range(len(rtt)), k)])
        assert G.random_curve(rtt, ks=(k,))[k] == pytest.approx((exp - 1.0) / 1.0)
    assert G.random_curve(rtt, ks=(6,))[6] == 0.0


def test_search_finds_the_separable_optimum_and_counts_evaluations():
    lab = lambda c: 1.0 + sum(abs(x[0] - 1) for x in c)       # optimum at node 1 for every task, separable
    combos, rtt = _table(lab)
    table = dict(zip(combos, rtt.tolist()))
    start = (C[0], C[2], C[0])
    traj, used, conv = G.budgeted_search(table, [C] * 3, start, 5000)
    assert conv and traj[-1] == 1.0 and used == len(traj)
    assert all(a >= b for a, b in zip(traj, traj[1:]))         # best-so-far never rises
    t1, u1, c1 = G.budgeted_search(table, [C] * 3, start, 1)
    assert u1 == 1 and t1 == [lab(start)] and not c1
    curve, info = G.search_curve(table, [C] * 3, start, 1.0)
    assert curve[1] == pytest.approx((lab(start) - 1.0)) and curve[32] <= curve[8] <= curve[1] and info["regret"] == 0.0


def test_expansion_leaves_a_state_icm_cannot():
    # moving ANY single task off node 0 costs 5, moving all three to node 1 is the optimum: only an expansion move reaches it
    def lab(c):
        k = sum(x == C[1] for x in c)
        return 1.0 if k == 3 else (10.0 if k == 0 else 15.0)
    combos, rtt = _table(lab, cands=[[C[0], C[1]]] * 3)
    table = dict(zip(combos, rtt.tolist()))
    traj, _, conv = G.budgeted_search(table, [[C[0], C[1]]] * 3, (C[0],) * 3, 5000)
    assert conv and traj[-1] == 1.0


def test_state_row_has_every_arm_and_the_gnn_only_when_scored():
    combos, rtt = _table(lambda c: 1.0 + sum(x[0] for x in c) + 0.01 * hash(c) % 7 * 0)
    s = rtt + np.linspace(0, 0.5, len(rtt))
    row = G.state_row(combos, rtt, s, s)
    assert set(row["arms"]) == {"ib0_s", "cd_s", "exp_ib0", "exp_cd", "random"} and row["n"] == 3
    row = G.state_row(combos, rtt, s, s, gnn=-rtt)
    assert row["arms"]["gnn"][0] == 0.0 and row["arms"]["ib0_s"][0] == 0.0


def test_k_needed_and_aggregate():
    curves = [{k: (0.0 if k >= 4 else 0.2) for k in (0,) + G.KS} for _ in range(10)]
    assert G.k_needed(curves, 0.01, 0.9) == 4 and G.k_needed(curves, 0.01, 0.9, ks=(0, 1, 2)) is None
    combos, rtt = _table(lambda c: 1.0 + sum(x[0] for x in c))
    rows = [dict(G.state_row(combos, rtt, rtt.copy(), rtt.copy()), rung="heavy", ds=f"d{i}", source="t") for i in range(3)]
    rows.append({"ds": "bad", "rung": "heavy", "error": "x"})
    agg = G.aggregate(rows)
    cell = agg[("heavy", 3)]
    assert cell["states"] == 3 and cell["arms"]["ib0_s"][0]["median"] == 0.0 and cell["needed"]["ib0_s"]["1%/90"] == 0
    assert "gnn" not in cell["arms"] and cell["conv"]["exp_ib0"]["converged"] == 1.0


def test_usable_rejects_incomplete_over_cap_and_pruned(tmp_path):
    def mk(name, meta, draw, n=2):
        d = tmp_path / name
        d.mkdir()
        (d / "placement_metadata.json").write_text(json.dumps(meta))
        (d / "warm_snapshot.json").write_text(json.dumps({"provenance": {"candidate_draw": draw, "cell_config": "/x/inputs/wf1_heavy/cfg.json"}}))
        (d / "workload.json").write_text(json.dumps({"trace_task_ids": list(range(n))}))
        return d
    ok = mk("ok", {"sweep_complete": True, "num_placements": 9}, {"rule": "declared_pruning_v1_uncapped", "pruned": False, "full_plans": 9})
    assert G.usable(ok, 100) == ("heavy", 2, 9)
    assert G.usable(mk("big", {"sweep_complete": True, "num_placements": 9}, {"pruned": False}), 5) is None
    assert G.usable(mk("inc", {"sweep_complete": False, "num_placements": 9}, {"pruned": False}), 100) is None
    assert G.usable(mk("pr", {"sweep_complete": True, "num_placements": 9}, {"rule": "declared_pruning_v1", "pruned": True, "full_plans": 40}), 100) is None
