#!/usr/bin/env python3
"""queue_range_v1 -- the bars themselves.

The read is the instrument, so its own arithmetic is pinned here: the PAIRED test against
scipy, Holm against its registered family size, and the decile bucketing against a hand
worked example. serving_stability_v1's S3-d shipped an unpaired test on seed-paired arms and
it moved a verdict; that is why the pinning test exists.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts_cosim.queue_range_v1_read import (  # noqa: E402
    CORPUS_DIM7_CANDIDATE_MAX,
    EARLY_DECILES,
    LATE_DECILES,
    Q0_BLIND_FRAC,
    Q0_EARLY_LATE_RATIO,
    Q0_MIN_CELLS,
    Q0_OUT_OF_RANGE_FRAC,
    Q1_MAX_OUT_OF_RANGE_FRAC,
    Q1_MIN_PINNED_FRAC,
    Q2_ALPHA,
    Q2_HOLM_N,
    Q2_MIN_CELLS,
    Q2_MIN_SEEDS,
    Q3_MIN_CELLS,
    Q4_CONTROL_DECILES,
    Q4_MIN_CELLS,
    Q4_MIN_DECILES,
    QueueRangeReadError,
    holm,
    queue_range_decile_summary,
    read_q0,
    read_q1,
    wilcoxon_p,
)


# ------------------------------------------------------------------ bars are what was signed
def test_bars_are_the_registered_constants():
    assert (Q0_OUT_OF_RANGE_FRAC, Q0_BLIND_FRAC, Q0_EARLY_LATE_RATIO, Q0_MIN_CELLS) == (
        0.20, 0.20, 3.0, 2)
    assert (Q1_MIN_PINNED_FRAC, Q1_MAX_OUT_OF_RANGE_FRAC) == (0.99, 0.0)
    assert (Q2_ALPHA, Q2_HOLM_N, Q2_MIN_SEEDS, Q2_MIN_CELLS) == (0.05, 6, 12, 2)
    assert Q3_MIN_CELLS == 2
    assert (Q4_MIN_DECILES, Q4_MIN_CELLS, Q4_CONTROL_DECILES) == (4, 2, 2)
    assert EARLY_DECILES == (1, 2) and LATE_DECILES == (9, 10)


# ----------------------------------------------------------------------------- the test
def test_wilcoxon_matches_scipy():
    scipy_stats = pytest.importorskip("scipy.stats")
    cases = [
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        [-1.0, 2.0, -3.0, 4.0, 5.0, -6.0, 7.0, 0.5, 9.0, -0.25],
        [0.4, 0.4, -0.4, 1.1, 1.1, 2.0, -2.0, 3.0, 3.0, 3.0, -0.1, 0.2],
        [-0.05] * 7 + [0.9] * 6,
    ]
    for xs in cases:
        mine = wilcoxon_p(xs)
        theirs = scipy_stats.wilcoxon(
            xs, zero_method="wilcox", correction=False, mode="approx"
        ).pvalue
        assert mine == pytest.approx(theirs, rel=1e-6, abs=1e-9), xs


def test_wilcoxon_is_none_below_six_pairs():
    assert wilcoxon_p([1.0, 2.0, 3.0, 4.0, 5.0]) is None


def test_wilcoxon_ignores_exact_zeros_like_scipy():
    assert wilcoxon_p([0.0] * 20) is None


# ------------------------------------------------------------------------------ holm
def test_holm_uses_the_registered_family_not_the_computed_one():
    # One test at p = 0.02. Against a family of 6 that is 0.02 >= 0.05/6 = 0.0083 -> not
    # significant. Shrinking the family to 1 would call it significant; that is the error.
    assert holm([0.02], 6, 0.05) == [False]
    assert holm([0.02], 1, 0.05) == [True]


def test_holm_stops_at_the_first_failure():
    # Sorted: 0.0001 < 0.05/6 = 0.00833 (sig), 0.001 < 0.05/5 = 0.01 (sig),
    # 0.02 >= 0.05/4 = 0.0125 (not) -- and everything after it stays not.
    assert holm([0.001, 0.02, 0.0001], 6, 0.05) == [True, False, True]
    assert holm([0.001, 0.02, 0.0001, 0.001], 6, 0.05) == [True, False, True, True]
    assert holm([0.009, 0.0099, 0.02], 6, 0.05) == [False, False, False]


def test_holm_refuses_a_family_smaller_than_the_tests_run():
    with pytest.raises(QueueRangeReadError):
        holm([0.01, 0.02, 0.03], 2, 0.05)


def test_holm_skips_none():
    assert holm([None, 0.0001], 6, 0.05) == [False, True]


# --------------------------------------------------------------------- decile bucketing
def _rec(t, divisor=1.0, raw=0.0, d7spread=0.0, blind=0.0, d7max=0.0):
    return [t, divisor, raw, d7spread, blind, d7max]


def test_bucketing_uses_the_task_decile_edges():
    bounds = [10.0, 20.0]        # three buckets: <10, <20, rest
    recs = [_rec(1.0), _rec(9.9), _rec(10.0), _rec(19.9), _rec(20.0), _rec(99.0)]
    got = queue_range_decile_summary(recs, bounds)
    assert [r["n"] for r in got["deciles"]] == [2, 2, 2]


def test_empty_decile_is_reported_not_invented():
    got = queue_range_decile_summary([_rec(1.0), _rec(2.0)], [10.0, 20.0])
    rows = got["deciles"]
    assert rows[0]["n"] == 2 and rows[1]["n"] == 0 and rows[2]["n"] == 0
    assert "blind_frac" not in rows[1]        # no share invented for an empty bucket


def test_out_of_range_uses_the_corpus_maximum():
    recs = [_rec(1.0, d7max=CORPUS_DIM7_CANDIDATE_MAX),
            _rec(2.0, d7max=CORPUS_DIM7_CANDIDATE_MAX + 0.1)]
    got = queue_range_decile_summary(recs, [])
    assert got["out_of_range_frac"] == 0.5    # equal to the max is inside


def test_no_records_fails_loud():
    with pytest.raises(QueueRangeReadError):
        queue_range_decile_summary([], [10.0])


def test_pinned_frac_is_exact_equality_not_a_tolerance():
    got = queue_range_decile_summary([_rec(1.0, divisor=1.0), _rec(2.0, divisor=1.0001)], [])
    assert got["pinned_frac"] == 0.5


# --------------------------------------------------------------------------- Q0 and Q1
def _arm(cell, arm, policy, seed, *, deciles, n_batches=100, pinned=1.0, oor=0.0,
         elapsed=30.0):
    qr = {
        "deciles": deciles, "n_batches": n_batches, "pinned_frac": pinned,
        "blind_frac": 0.0, "out_of_range_frac": oor, "max_dim7": 100.0, "max_divisor": 1.0,
    }
    return {"arm": f"{cell}__{arm}_{policy}_s{seed}", "averageElapsedTime": elapsed,
            "queue_range_summary": qr}


def _deciles(early_oor, late_oor):
    rows = []
    for d in range(1, 11):
        frac = early_oor if d in EARLY_DECILES else (late_oor if d in LATE_DECILES else 0.0)
        rows.append({"decile": d, "n": 100, "out_of_range_frac": frac, "blind_frac": 0.0,
                     "median_divisor": 1.0, "pinned_frac": 1.0, "median_dim7_max": 0.0,
                     "median_dim7_spread": 0.0})
    return rows


def _corpus(early_oor, late_oor, cells):
    from scripts_cosim.queue_range_v1_read import CELLS
    arms = {}
    for cell in CELLS:
        e, l = (early_oor, late_oor) if cell in cells else (0.0, 0.0)
        for seed in range(1, 17):
            if (("gnn", seed) in (("gnn", 3),)
                    or (cell == "cell_s7901_f4000_pg16" and seed in (8, 14))):
                continue
            doc = _arm(cell, "plain", "gnn", seed, deciles=_deciles(e, l))
            arms[doc["arm"]] = doc
    return arms


def test_q0_fires_when_the_column_leaves_late_and_not_early():
    from scripts_cosim.queue_range_v1_read import CELLS
    got = read_q0(_corpus(0.02, 0.90, CELLS[:2]))
    assert got["verdict"] == "COLUMN-LEAVES-CONTRACT"
    assert got["cells_firing"] == 2


def test_q0_does_not_fire_on_one_cell():
    from scripts_cosim.queue_range_v1_read import CELLS
    got = read_q0(_corpus(0.02, 0.90, CELLS[:1]))
    assert got["verdict"] == "COLUMN-IN-CONTRACT"


def test_q0_does_not_fire_when_the_column_is_equally_out_early_and_late():
    """A column that is out of range from the first decision is not what destroys an
    advantage that exists in the first decision."""
    from scripts_cosim.queue_range_v1_read import CELLS
    got = read_q0(_corpus(0.90, 0.90, CELLS))
    assert got["verdict"] == "COLUMN-IN-CONTRACT"


def test_q0_excludes_the_burned_and_unservable_seeds():
    from scripts_cosim.queue_range_v1_read import CELLS
    got = read_q0(_corpus(0.02, 0.90, CELLS))
    assert got["per_cell"][CELLS[0]]["n"] == 13     # 16 - seed 3 - seeds 8 and 14
    assert got["per_cell"][CELLS[1]]["n"] == 15     # 16 - seed 3


def test_q1_fails_an_intervention_whose_divisor_never_pinned():
    from scripts_cosim.queue_range_v1_read import CELLS
    arms = {}
    for cell in CELLS:
        for policy in ("gnn", "mpoff"):
            for arm, pinned in (("plain", 0.0), ("pinned", 0.10), ("inrange", 1.0)):
                for seed in range(1, 17):
                    if (policy, seed) == ("gnn", 3):
                        continue
                    if cell == CELLS[0] and policy == "gnn" and seed in (8, 14):
                        continue
                    doc = _arm(cell, arm, policy, seed, deciles=_deciles(0.0, 0.0),
                               pinned=pinned)
                    arms[doc["arm"]] = doc
    got = read_q1(arms)
    assert got["verdict"] == "KNOB-NOT-BOUND"
    assert got["per_arm"][f"{CELLS[0]}|gnn|pinned"]["ok"] is False
    assert got["per_arm"][f"{CELLS[0]}|gnn|inrange"]["ok"] is True
    assert got["per_arm"][f"{CELLS[0]}|gnn|plain"]["ok"] is True


def test_q1_fails_inrange_if_a_single_batch_is_above_the_clamp():
    from scripts_cosim.queue_range_v1_read import CELLS
    arms = {}
    for cell in CELLS:
        for policy in ("gnn", "mpoff"):
            for arm in ("plain", "pinned", "inrange"):
                for seed in range(1, 17):
                    if (policy, seed) == ("gnn", 3):
                        continue
                    if cell == CELLS[0] and policy == "gnn" and seed in (8, 14):
                        continue
                    doc = _arm(cell, arm, policy, seed, deciles=_deciles(0.0, 0.0),
                               pinned=1.0, oor=0.0001 if arm == "inrange" else 0.0)
                    arms[doc["arm"]] = doc
    got = read_q1(arms)
    assert got["per_arm"][f"{CELLS[0]}|gnn|inrange"]["ok"] is False


def test_q1_rejects_an_arm_with_no_instrument():
    from scripts_cosim.queue_range_v1_read import CELLS, _qr
    with pytest.raises(QueueRangeReadError):
        _qr({"arm": f"{CELLS[0]}__plain_gnn_s1"})
