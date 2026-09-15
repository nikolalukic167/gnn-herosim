#!/usr/bin/env python3
"""scheduler_residence_v1 R3 -- the bars.

The correlation is the instrument, so its arithmetic is pinned against scipy (ties included),
and so is the control that can void it: drainable_objective_v1's C0 and peer_affinity_v1's H5
both died of a control that moved with the treatment, and this one is registered in advance.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts_cosim.scheduler_residence_v1_r3_read import (  # noqa: E402
    R3_ALPHA,
    R3_CONTROL_RATIO,
    R3_EXPECTED_SIGN,
    R3_HOLM_N,
    R3_MIN_ABS_RHO,
    R3_MIN_CELLS_READ,
    R3_MIN_SEEDS_PER_CELL,
    R3_N_CELLS,
    R3_PRIMARY_KEY,
    R3_SECOND_KEY,
    R3ReadError,
    holm,
    read,
    spearman,
)


def test_bars_are_the_registered_constants():
    assert (R3_MIN_ABS_RHO, R3_ALPHA, R3_HOLM_N, R3_MIN_CELLS_READ) == (0.60, 0.05, 2, 10)
    assert (R3_N_CELLS, R3_MIN_SEEDS_PER_CELL, R3_EXPECTED_SIGN) == (12, 3, -1)
    assert R3_CONTROL_RATIO == 0.75
    assert (R3_PRIMARY_KEY, R3_SECOND_KEY) == ("min_reachable_servers", "hosting_node_spread")


def test_spearman_matches_scipy_including_ties():
    ss = pytest.importorskip("scipy.stats")
    cases = [
        ([1, 2, 3, 4, 5, 6, 7, 8], [2, 1, 4, 3, 6, 5, 8, 7]),
        ([1, 1, 2, 2, 3, 3, 4, 4], [9, 8, 7, 6, 5, 4, 3, 2]),
        ([1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1, -2]),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [5, 3, 8, 1, 9, 2, 7, 4, 11, 6, 12, 10]),
    ]
    for xs, ys in cases:
        rho, p = spearman(xs, ys)
        ref = ss.spearmanr(xs, ys)
        assert rho == pytest.approx(ref.statistic, rel=1e-9)
        assert p == pytest.approx(ref.pvalue, rel=1e-6, abs=1e-12)


def test_spearman_is_none_below_four_points():
    assert spearman([1, 2, 3], [3, 2, 1]) == (None, None)


def test_spearman_is_none_on_a_constant_column():
    assert spearman([1, 1, 1, 1, 1], [1, 2, 3, 4, 5]) == (None, None)


def test_holm_uses_the_registered_family():
    assert holm([0.03], 2, 0.05) == [False]     # 0.03 >= 0.05/2
    assert holm([0.02], 2, 0.05) == [True]
    with pytest.raises(R3ReadError):
        holm([0.01, 0.02, 0.03], 2, 0.05)


# ------------------------------------------------------------------------- the read
def _manifest(n=R3_N_CELLS, min_reach=None, spread=None):
    min_reach = min_reach or [1 + (i % 3) for i in range(n)]
    spread = spread or [float(i) for i in range(n)]
    return {"selected": [
        {"cell": f"cell_r3s{9100 + i}_f4000_pg16", "seed": 9100 + i,
         "structure": {R3_PRIMARY_KEY: float(min_reach[i]), R3_SECOND_KEY: spread[i]}}
        for i in range(n)]}


def _arms(manifest, reactive, gnn, mpoff=None, seeds=(1, 2, 4, 5)):
    out = {}
    for i, e in enumerate(manifest["selected"]):
        c = e["cell"]
        out[f"{c}__reactive"] = {"arm": f"{c}__reactive", "averageQueueTime": reactive[i]}
        for arm, vals in (("gnn", gnn), ("mpoff", mpoff if mpoff is not None else gnn)):
            for s in seeds:
                k = f"{c}__{arm}_s{s}"
                out[k] = {"arm": k, "averageQueueTime": vals[i]}
    return out


def test_read_fires_when_fewer_reachable_servers_means_more_excess():
    m = _manifest(min_reach=[1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
    reactive = [10.0] * 12
    # excess falls as min_reach rises -> negative rho, the registered sign
    gnn = [40, 41, 42, 43, 25, 26, 27, 28, 12, 13, 14, 15]
    got = read(m, _arms(m, reactive, [float(g) for g in gnn]))
    assert got["verdict"] == "LOPSIDEDNESS-PREDICTS"
    assert got["tests"][R3_PRIMARY_KEY]["rho"] < -R3_MIN_ABS_RHO


def test_read_does_not_fire_on_the_wrong_sign():
    """More reachable servers, more excess, is not the registered claim."""
    m = _manifest(min_reach=[1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
    gnn = [12, 13, 14, 15, 25, 26, 27, 28, 40, 41, 42, 43]
    got = read(m, _arms(m, [10.0] * 12, [float(g) for g in gnn]))
    assert got["tests"][R3_PRIMARY_KEY]["rho"] > 0
    assert got["verdict"] == "LOPSIDEDNESS-DOES-NOT-PREDICT"


def test_control_voids_the_reading_when_reactive_moves_with_it():
    """If lopsidedness hurts reactive just as much, it is the environment, not the arm."""
    m = _manifest(min_reach=[1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
    reactive = [40, 41, 42, 43, 25, 26, 27, 28, 12, 13, 14, 15]
    gnn = [r + 30 - i for i, r in enumerate(reactive)]
    got = read(m, _arms(m, [float(r) for r in reactive], [float(g) for g in gnn]))
    assert got["tests"][R3_PRIMARY_KEY]["confounded"] is True
    assert got["verdict"] == "CONFOUNDED-ENVIRONMENT"


def test_control_does_not_void_when_reactive_is_flat():
    m = _manifest(min_reach=[1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
    gnn = [40, 41, 42, 43, 25, 26, 27, 28, 12, 13, 14, 15]
    got = read(m, _arms(m, [10.0] * 12, [float(g) for g in gnn]))
    assert got["tests"][R3_PRIMARY_KEY]["confounded"] is False


def test_read_voids_below_the_minimum_cell_count():
    m = _manifest()
    arms = _arms(m, [10.0] * 12, [20.0 + i for i in range(12)])
    for e in m["selected"][:4]:                       # kill four cells' reactive arms
        del arms[f"{e['cell']}__reactive"]
    got = read(m, arms)
    assert got["verdict"] == "VOID-TOO-FEW-CELLS"
    assert got["n_read"] == 8 and len(got["dropped"]) == 4


def test_a_cell_with_too_few_seeds_is_dropped_loudly_not_silently():
    m = _manifest()
    arms = _arms(m, [10.0] * 12, [20.0 + i for i in range(12)])
    cell = m["selected"][0]["cell"]
    for s in (2, 4, 5):
        del arms[f"{cell}__gnn_s{s}"]                 # leaves 1, below the bar of 3
    got = read(m, arms)
    assert got["n_read"] == 11
    assert any(d["cell"] == cell for d in got["dropped"])


def test_read_refuses_a_manifest_with_the_wrong_cell_count():
    with pytest.raises(R3ReadError):
        read(_manifest(n=9), {})


def test_excess_is_measured_against_each_cell_s_OWN_reactive_arm():
    m = _manifest(n=12)
    reactive = [float(5 * i) for i in range(12)]
    gnn = [r + 7.0 for r in reactive]                 # constant excess, varying reactive
    got = read(m, _arms(m, reactive, gnn))
    for row in got["cells"]:
        assert row["gnn"]["excess_queue_s"] == pytest.approx(7.0)
    assert got["verdict"] == "LOPSIDEDNESS-DOES-NOT-PREDICT"
