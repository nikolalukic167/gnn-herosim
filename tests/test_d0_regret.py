"""D0 (scripts_cosim/uncut_d0_regret.py): the pure statistics and the NO-SIGNAL rule."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts_cosim"))
import uncut_d0_regret as D  # noqa: E402


def test_move_cost_is_in_units_of_the_mean_per_task_cost_and_clipped():
    assert D.move_cost(5.0, 100.0, 10) == (0.5, 0.5)
    assert D.move_cost(-3.0, 100.0, 10) == (0.0, 0.0)
    c, raw = D.move_cost(5000.0, 100.0, 10)
    assert c == D.D_MAX and raw == 500.0


def test_c20_is_the_top_fifth_share_of_cost():
    assert D.c20([1.0] * 10) == pytest.approx(0.2)                       # uniform costs: exactly the fifth
    assert D.c20([10.0, 1, 1, 1, 1, 1, 1, 1, 1, 1]) == pytest.approx(11 / 19)       # the top two of ten
    assert D.c20([]) is None and D.c20([0.0, 0.0]) is None
    assert D.c20([5.0]) == pytest.approx(1.0)                            # one miss: it is its own top fifth


def _tables():
    reg = {"g1": {"n": 4, "bucket": "51_100", "S0": 40.0, "label": [[0, 0]] * 4,
                  "moves": {"0": [[[1, 0], 10.0], [[2, 0], 0.5]], "1": [[[1, 0], 30.0]], "2": []}}}
    tf = {"runs": ["a"], "groups": {"g1": {"a": {"argmax": {"0": [1, 0], "1": [0, 0], "2": [0, 0]}, "hit": 2, "n_choice": 3}}}}
    return reg, tf


def test_group_rows_marks_misses_scores_and_clear_steps():
    reg, tf = _tables()
    rows = {r["task"]: r for r in D.group_rows(reg, tf)}
    assert rows[0]["miss"] and rows[0]["scored"] and rows[0]["d"] == pytest.approx(1.0)      # 10 / (40 / 4)
    assert not rows[1]["miss"] and rows[1]["clear"]                                        # the only alternative costs 3 units
    assert not rows[0]["clear"]                                                            # an alternative costs 0.05 units
    assert not rows[2]["clear"] and rows[2]["alts"] == 0


def test_a_miss_the_engine_never_scored_is_counted_not_costed():
    reg, tf = _tables()
    tf["groups"]["g1"]["a"]["argmax"]["0"] = [7, 7]
    r = [x for x in D.group_rows(reg, tf) if x["task"] == 0][0]
    assert r["miss"] and not r["scored"] and r["d"] is None


def _bucket(c20v, scored):
    return {"C20": c20v, "scored_share": scored}


def test_verdict_rule():
    assert D.verdict({"51_100": _bucket(0.40, 0.9), "101_256": _bucket(0.45, 0.9)})["verdict"] == "NO-SIGNAL"
    assert D.verdict({"51_100": _bucket(0.40, 0.9), "101_256": _bucket(0.55, 0.9)})["verdict"] == "SIGNAL-PRESENT"     # both buckets must fall under 50 %
    assert D.verdict({"51_100": _bucket(0.70, 0.9), "101_256": _bucket(0.55, 0.9)})["verdict"] == "SIGNAL-PRESENT"
    assert D.verdict({"51_100": _bucket(0.40, 0.9), "101_256": _bucket(0.45, 0.5)})["verdict"] == "UNREADABLE"          # too few misses scored
    assert D.verdict({"51_100": _bucket(None, None), "101_256": _bucket(0.45, 0.9)})["verdict"] == "UNREADABLE"


def test_test_id_and_topology_assertion():
    D.assert_not_test("gnn_datasets_wf1_heldout/ds_10954", 16210, ["gnn_datasets_wf1_train/ds_40000"])
    for bad in (16226, 16251, 16270, 19537, 19548):
        with pytest.raises(SystemExit):
            D.assert_not_test("x", bad, [])
    with pytest.raises(SystemExit):
        D.assert_not_test("gnn_datasets_wf1_train/ds_40000", 16210, ["gnn_datasets_wf1_train/ds_40000"])
