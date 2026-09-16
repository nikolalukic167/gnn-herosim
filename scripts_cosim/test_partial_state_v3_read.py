"""Tests for partial_state_v3's registered read."""
import numpy as np
import pytest

from scripts_cosim.partial_state_v3_read import (
    P0_MAX_COLUMN_DIFF, P0_MIN_DATASETS, P0_MIN_RANK_RECOVERY, P1_ALPHA, P1_MIN_SEEDS,
    P1_TIE_PP, P2_CELLS, P2_MIN_CELLS, P2_MIN_SEEDS, P2_TIE_PCT, P2_V2_CONTROL_MEDIANS,
    P3_CHECKPOINT_SEEDS, P3_EXPECTED, P3_MIN_CELLS, P3_RUNGS, P3_TOL, P3_TOPOLOGY_SEEDS,
    P4_MIN_IMPROVEMENT, V_COSTS, V_DEGRADES, V_GENERALISES, V_P0_FAIL, V_P0_PASS,
    V_P2_CONTROL_FAIL, V_SCALE_HELPS, V_SCALE_NO, V_STILL_PINNED, V_TIE, V_UNREADABLE,
    cell_deficit, p0_dataset, paired_tie, read_p0, read_p1, read_p2, read_p3, read_p4,
)

BASE, K2, K3, T, L = 10, 24, 8, 4, 4


# --- constants are the signed bars ------------------------------------------------------

def test_registered_constants():
    assert (P0_MAX_COLUMN_DIFF, P0_MIN_RANK_RECOVERY, P0_MIN_DATASETS) == (0.0, 1.0, 516)
    assert (P1_TIE_PP, P1_ALPHA, P1_MIN_SEEDS) == (1.0, 0.05, 12)
    assert (P2_TIE_PCT, P2_MIN_SEEDS, P2_MIN_CELLS) == (5.0, 12, 2)
    assert len(P2_CELLS) == 3 and set(P2_V2_CONTROL_MEDIANS) == set(P2_CELLS)
    assert [r[1] for r in P3_RUNGS] == [6, 12, 24, 80]
    assert 9004 not in P3_TOPOLOGY_SEEDS and 3 not in P3_CHECKPOINT_SEEDS
    assert (P3_MIN_CELLS, P3_TOL, P4_MIN_IMPROVEMENT) == (3, 0.10, 0.05)
    assert P3_EXPECTED == "DEGRADES"


# --- P0 ----------------------------------------------------------------------------------

def _blocks(n_nodes, ranks, types, base=None, link=None, corrupt_v3_base=False):
    """Build matched v2/v3 blocks for edges with given (rank, type) on an n_nodes cluster."""
    n = len(ranks)
    base = np.random.RandomState(0).rand(n, BASE) if base is None else base
    link = np.random.RandomState(1).rand(n, L) if link is None else link
    k2 = np.zeros((n, K2)); k3 = np.zeros((n, K3))
    for i, (r, k) in enumerate(zip(ranks, types)):
        k2[i, r * T + k] = 1.0
        k3[i, k * 2] = 0.0 if n_nodes <= 1 else r / (n_nodes - 1)
        k3[i, k * 2 + 1] = 1.0 / n_nodes
    v2 = np.hstack([base, k2, link])
    v3 = np.hstack([base + (1e-9 if corrupt_v3_base else 0.0), k3, link])
    return v2, v3


def _p0(v2, v3):
    return p0_dataset(v2, v3, base_dim=BASE, v2_krank_dim=K2, v3_krank_dim=K3, types=T, link_dim=L)


def test_p0_recovers_every_rank_on_a_six_node_cluster():
    r = _p0(*_blocks(6, ranks=[0, 1, 2, 3, 4, 5, 2], types=[0, 1, 2, 3, 0, 1, 3]))
    assert r["rank_recovery"] == 1.0 and r["base_diff"] == 0.0 and r["link_diff"] == 0.0


def test_p0_single_node_cluster_has_rank_zero():
    r = _p0(*_blocks(1, ranks=[0, 0], types=[2, 3]))
    assert r["rank_recovery"] == 1.0


def test_p0_catches_a_moved_base_column():
    r = _p0(*_blocks(6, ranks=[0, 1], types=[0, 0], corrupt_v3_base=True))
    assert r["base_diff"] > 0.0


def test_p0_catches_a_wrong_rank():
    v2, v3 = _blocks(6, ranks=[0, 3], types=[1, 1])
    v3[1, BASE + 1 * 2] = 0.2   # rank 1 encoded where v2 says rank 3
    assert _p0(v2, v3)["rank_recovery"] == 0.5


def test_p0_refuses_a_non_one_hot_v2_block():
    v2, v3 = _blocks(6, ranks=[0], types=[0])
    v2[0, BASE + 5] = 1.0
    with pytest.raises(ValueError, match="not one-hot"):
        _p0(v2, v3)


def test_read_p0_needs_the_whole_corpus():
    rows = [{"base_diff": 0.0, "link_diff": 0.0, "rank_recovery": 1.0}] * 515
    assert read_p0(rows)["verdict"] == V_UNREADABLE
    rows.append(rows[0])
    assert read_p0(rows)["verdict"] == V_P0_PASS
    rows[3] = {"base_diff": 0.0, "link_diff": 0.0, "rank_recovery": 0.999}
    assert read_p0(rows)["verdict"] == V_P0_FAIL


# --- P1 / P2 -------------------------------------------------------------------------------

def test_paired_tie_reads_a_tie_and_a_cost():
    v2 = {s: 20.0 + s for s in range(1, 17)}
    assert paired_tie({s: v + 0.2 for s, v in v2.items()}, v2, tol=1.0, alpha=0.05,
                      min_seeds=12, relative=False)["verdict"] == V_TIE
    r = paired_tie({s: v + 3.0 for s, v in v2.items()}, v2, tol=1.0, alpha=0.05,
                   min_seeds=12, relative=False)
    assert r["verdict"] == V_COSTS and r["p"] < 0.05


def test_paired_tie_needs_min_seeds():
    v2 = {s: 20.0 for s in range(1, 9)}
    assert read_p1({s: 21.0 for s in v2}, v2)["verdict"] == V_UNREADABLE


def _p2_inputs(delta_pct):
    v2 = {c: {s: 50.0 + s for s in range(1, 17)} for c in P2_CELLS}
    v3 = {c: {s: v * (1 + delta_pct / 100.0) for s, v in d.items()} for c, d in v2.items()}
    return v3, v2


def test_p2_control_must_reproduce_queue_range_medians():
    v3, v2 = _p2_inputs(0.0)
    bad = dict(P2_V2_CONTROL_MEDIANS); bad["cell_s9001_f4000_pg16"] += 0.01
    assert read_p2(v3, v2, bad)["verdict"] == V_P2_CONTROL_FAIL
    assert read_p2(v3, v2, P2_V2_CONTROL_MEDIANS)["verdict"] == V_TIE


def test_p2_reads_a_live_cost_on_all_three_cells():
    v3, v2 = _p2_inputs(+12.0)
    r = read_p2(v3, v2, P2_V2_CONTROL_MEDIANS)
    assert r["verdict"] == "REPRESENTATION-COSTS-LIVE" and r["ties"] == 0


# --- P3 / P4 -------------------------------------------------------------------------------

def test_cell_deficit_is_relative_to_reactive():
    assert cell_deficit({1: 11.0, 2: 12.0, 4: 13.0, 5: 14.0}, 10.0) == pytest.approx(0.25)


def _rung(d_gnn, d_mpoff=0.5, n_cells=4, incomplete=False, wall=5.0):
    cells = {}
    for i in range(n_cells):
        react = 20.0 + i
        cells[f"cs{i}"] = {
            "reactive": react,
            "gnn": {s: react * (1 + d_gnn) for s in P3_CHECKPOINT_SEEDS},
            "mpoff": {s: react * (1 + d_mpoff) for s in P3_CHECKPOINT_SEEDS},
            "completed": {"gnn": 3 if incomplete else 4, "mpoff": 4},
            "expected": {"gnn": 4, "mpoff": 4},
        }
    return {"cells": cells, "wallclock_min": [wall] * n_cells}


def test_p3_generalises_when_every_scaled_rung_stays_within_tol():
    r = read_p3({"R0": _rung(0.30), "R1": _rung(0.35), "R2": _rung(0.38), "R3": _rung(0.39)})
    assert r["verdict"] == V_GENERALISES and r["first_break"] is None


def test_p3_degrades_and_names_the_first_breaking_rung():
    r = read_p3({"R0": _rung(0.30), "R1": _rung(0.35), "R2": _rung(0.45), "R3": _rung(0.9)})
    assert r["verdict"] == V_DEGRADES and r["first_break"] == "R2"


def test_p3_a_still_pinned_beats_any_latency_read():
    r = read_p3({"R0": _rung(0.30), "R1": _rung(0.30), "R2": _rung(0.30, incomplete=True),
                 "R3": _rung(0.30)})
    assert r["verdict"] == V_STILL_PINNED and r["pinned"][0][0] == "R2"


def test_p3_rung_below_min_cells_is_skipped_not_read():
    r = read_p3({"R0": _rung(0.30), "R1": _rung(0.9, n_cells=2), "R2": _rung(0.32),
                 "R3": _rung(0.33)})
    assert r["verdict"] == V_GENERALISES and "R1" not in r["readable_scaled"]


def test_p3_unaffordable_r3_is_flagged_not_failed():
    r = read_p3({"R0": _rung(0.30), "R1": _rung(0.31), "R2": _rung(0.32), "R3": _rung(0.33, wall=90)})
    assert r["verdict"] == V_GENERALISES and r["r3_affordable"] is False


def test_p4_only_reads_after_generalises():
    p3 = read_p3({"R0": _rung(0.30), "R1": _rung(0.35), "R2": _rung(0.45), "R3": _rung(0.9)})
    assert read_p4(p3)["verdict"] == "NOT-READ"


def test_p4_scale_helps_needs_monotone_and_five_points():
    p3 = read_p3({"R0": _rung(0.30), "R1": _rung(0.25), "R2": _rung(0.20), "R3": _rung(0.15)})
    assert read_p4(p3)["verdict"] == V_SCALE_HELPS
    p3 = read_p3({"R0": _rung(0.30), "R1": _rung(0.32), "R2": _rung(0.20), "R3": _rung(0.15)})
    assert read_p4(p3)["verdict"] == V_SCALE_NO
    p3 = read_p3({"R0": _rung(0.30), "R1": _rung(0.29), "R2": _rung(0.28), "R3": _rung(0.27)})
    assert read_p4(p3)["verdict"] == V_SCALE_NO
