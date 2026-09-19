"""Tests for cluster_scale_v1's registered read (S0.b)."""
import json
import os
import tempfile

import pytest

from scripts_cosim.cluster_scale_v1_read import (
    S0_COLLECTION_MAX_S, S0_CORPUS_CANDIDATE_MAX, S0_MAX_WALLCLOCK_MIN, S0_MIN_CELLS,
    S0_RECONSTRUCTION_TOL, S0_RHO_BAND, S0B_RUNGS, VERDICT_DOES_NOT, VERDICT_SCALES,
    VERDICT_UNREADABLE, format_s0b, load_arms, median, read_s0b, rung_summary,
)


def _arm(seed, rung, collection, hol=0.0, place=0.0, wall=90, batch=9.0, err=0.0):
    wait = collection + hol + place
    return {
        "arm": f"{seed}__gnn__{rung}", "cell": seed, "arm_kind": "gnn",
        "wallclock_s": wall,
        "residence_summary": {
            "collection_s": collection, "head_of_line_s": hol, "placement_s": place,
            "average_wait_time_s": wait, "sum_of_parts_s": wait,
            "reconstruction_error": err, "mean_batch_size": batch,
            "n_batches": 5000, "n_tasks": 50000, "unstamped": 0,
        },
    }


def _dir(arms):
    d = tempfile.mkdtemp()
    for a in arms:
        json.dump(a, open(os.path.join(d, a["arm"] + ".summary.json"), "w"))
    return d


# --- the registered constants are the bars, and they do not drift -----------------------

def test_registered_constants_are_the_ones_signed_off():
    assert S0_COLLECTION_MAX_S == 2.0
    assert S0_RHO_BAND == (0.33, 3.0)
    assert S0_MAX_WALLCLOCK_MIN == 45
    assert S0_CORPUS_CANDIDATE_MAX == 5.0
    assert S0_MIN_CELLS == 3
    assert S0_RECONSTRUCTION_TOL == 0.01


def test_rung_ladder_is_ordered_fastest_last():
    rates = [r[1] for r in S0B_RUNGS]
    assert rates == sorted(rates)
    assert [r[0] for r in S0B_RUNGS] == ["f4000", "f1000", "f300"]


# --- median -----------------------------------------------------------------------------

def test_median_odd_and_even():
    assert median([3.0, 1.0, 2.0]) == 2.0
    assert median([1.0, 2.0, 3.0, 4.0]) == 2.5
    assert median([]) is None


# --- load_arms: the defect that cost job 769390 -----------------------------------------

def test_arm_without_a_rung_tag_is_refused():
    """Job 769390's arms were named cell__kind, so all three rungs shared a summary path."""
    d = _dir([_arm("cs6s9001", "f300", 0.7)])
    p = os.path.join(d, "cs6s9001__gnn__f300.summary.json")
    doc = json.load(open(p))
    doc["arm"] = "cs6s9001__gnn"          # the old, rung-less spelling
    json.dump(doc, open(p, "w"))
    with pytest.raises(ValueError, match="no rung tag"):
        load_arms(d)


def test_load_arms_splits_seed_and_rung():
    arms = load_arms(_dir([_arm("cs6s9001", "f300", 0.7), _arm("cs6s9002", "f4000", 6.1)]))
    assert {(a["_seed"], a["_rung"]) for a in arms} == {
        ("cs6s9001", "f300"), ("cs6s9002", "f4000")}


# --- rung_summary ------------------------------------------------------------------------

def test_rung_summary_medians_over_cells():
    arms = load_arms(_dir([_arm(f"cs6s900{i}", "f300", c)
                           for i, c in enumerate([0.70, 0.73, 0.76], start=1)]))
    r = rung_summary(arms, "f300")
    assert r["n_cells"] == 3 and r["readable"] is True
    assert r["collection_s"] == pytest.approx(0.73)


def test_rung_below_min_cells_is_unreadable():
    arms = load_arms(_dir([_arm("cs6s9001", "f300", 0.7), _arm("cs6s9002", "f300", 0.8)]))
    r = rung_summary(arms, "f300")
    assert r["readable"] is False and "S0_MIN_CELLS" in r["reason"]


def test_missing_rung_reports_rather_than_raises():
    r = rung_summary(load_arms(_dir([_arm("cs6s9001", "f300", 0.7)])), "f1000")
    assert r["n_cells"] == 0 and r["readable"] is False


def test_a_decomposition_that_does_not_reconcile_fails_loud():
    arms = load_arms(_dir([_arm(f"cs6s900{i}", "f300", 0.7, err=0.5) for i in (1, 2, 3)]))
    with pytest.raises(ValueError, match="does not reconcile"):
        rung_summary(arms, "f300")


def test_a_missing_residence_summary_fails_loud():
    d = _dir([_arm("cs6s9001", "f300", 0.7)])
    p = os.path.join(d, "cs6s9001__gnn__f300.summary.json")
    doc = json.load(open(p)); doc.pop("residence_summary")
    json.dump(doc, open(p, "w"))
    with pytest.raises(ValueError, match="no residence_summary"):
        rung_summary(load_arms(d), "f300")


# --- read_s0b: the primary ----------------------------------------------------------------

def _ladder(cols, wall=90):
    arms = []
    for (tag, _), c in zip(S0B_RUNGS, cols):
        arms += [_arm(f"cs6s900{i}", tag, c, wall=wall) for i in (1, 2, 3)]
    return _dir(arms)


def test_monotone_decline_below_the_bar_fires():
    res = read_s0b(_ladder([6.1, 2.2, 0.73]))
    assert res["verdict"] == VERDICT_SCALES
    assert res["monotone"] is True and res["meets_bar"] is True
    assert res["fold_reduction"] == pytest.approx(6.1 / 0.73)


def test_decline_that_stalls_above_the_bar_does_not_fire():
    res = read_s0b(_ladder([6.1, 4.0, 2.5]))
    assert res["verdict"] == VERDICT_DOES_NOT and res["meets_bar"] is False


def test_non_monotone_does_not_fire_even_under_the_bar():
    """Reaching the bar by luck at the fastest rung is not the registered claim."""
    res = read_s0b(_ladder([1.0, 3.0, 0.5]))
    assert res["monotone"] is False and res["verdict"] == VERDICT_DOES_NOT


def test_a_single_readable_rung_cannot_show_a_trend():
    arms = [_arm(f"cs6s900{i}", "f300", 0.73) for i in (1, 2, 3)]
    arms += [_arm("cs6s9001", "f4000", 6.1)]          # 1 cell -> unreadable
    res = read_s0b(_dir(arms))
    assert res["verdict"] == VERDICT_UNREADABLE and res["n_readable"] == 1


def test_wallclock_gates_s1_not_s0():
    """S0.c is reported beside the primary and never changes the S0.b verdict."""
    res = read_s0b(_ladder([6.1, 2.2, 0.73], wall=60 * 60))
    assert res["wallclock_ok"] is False
    assert res["verdict"] == VERDICT_SCALES


def test_format_prints_every_rung_and_the_verdict():
    out = format_s0b(read_s0b(_ladder([6.1, 2.2, 0.73])))
    for tag, _ in S0B_RUNGS:
        assert tag in out
    assert VERDICT_SCALES in out and str(S0_COLLECTION_MAX_S) in out


def test_format_survives_an_unreadable_rung():
    arms = [_arm(f"cs6s900{i}", "f300", 0.73) for i in (1, 2, 3)]
    arms += [_arm(f"cs6s900{i}", "f1000", 2.2) for i in (1, 2, 3)]
    out = format_s0b(read_s0b(_dir(arms)))
    assert "f4000" in out and "UNREADABLE" not in out.split("f4000")[0]


# --- the landed numbers -------------------------------------------------------------------

def test_the_landed_s0b_read_is_reproducible():
    """Guards the published verdict against a silent change in the read."""
    d = "simulation_data/cluster_scale_v1/s0b"
    if not os.path.isdir(d):
        pytest.skip("S0.b summaries not present locally")
    res = read_s0b(d)
    assert res["verdict"] == VERDICT_SCALES
    assert res["n_readable"] == 3
    assert res["fastest_collection_s"] == pytest.approx(0.729, abs=5e-4)
    assert res["baseline_collection_s"] == pytest.approx(6.099, abs=5e-4)
