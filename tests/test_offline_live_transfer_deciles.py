"""Tests for offline_live_transfer_v1 R3 -- when does a bad seed become bad?"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts_cosim import offline_live_transfer_v1_decile_read as R3  # noqa: E402


def _arms(gap_by_decile, n_per_decile=600, base=10.0):
    """Two synthetic captures: the worst arm carries `gap_by_decile[d]` extra queue."""
    best, worst = {}, {}
    tid = 0
    for d, gap in enumerate(gap_by_decile):
        for i in range(n_per_decile):
            order = float(d * n_per_decile + i)
            best[tid] = (order, base, base)
            worst[tid] = (order, base + gap, base + gap)
            tid += 1
    return best, worst


def test_a_gap_present_only_in_the_first_decile_reads_born_bad():
    best, worst = _arms([100.0] + [0.0] * 9)
    out = R3.read(best, worst)
    assert out["verdict"] == "PRESENT-FROM-THE-START"
    assert out["first_decile_frac"] == pytest.approx(1.0)


def test_a_gap_that_only_appears_late_reads_compounds():
    best, worst = _arms([0.0] * 5 + [10.0, 20.0, 30.0, 40.0, 50.0])
    out = R3.read(best, worst)
    assert out["verdict"] == "COMPOUNDS"
    assert out["first_decile_frac"] == pytest.approx(0.0)


def test_a_uniform_gap_clears_the_compounds_bar_but_is_flat():
    """The registered COMPOUNDS bar (decile-1 share <= 0.20) is also cleared by a FLAT gap.

    That ambiguity is in the bar as signed, so the tool reports the curve's shape beside the
    verdict: COMPOUNDS is a closed-loop claim only when the shape is RISING.
    """
    best, worst = _arms([10.0] * 10)
    out = R3.read(best, worst)
    assert out["first_decile_frac"] == pytest.approx(0.1)
    assert out["verdict"] == "COMPOUNDS"
    assert out["shape"] == "FLAT"


def test_a_growing_gap_is_reported_as_rising():
    best, worst = _arms([0.0] * 5 + [10.0, 20.0, 30.0, 40.0, 50.0])
    out = R3.read(best, worst)
    assert out["verdict"] == "COMPOUNDS"
    assert out["shape"] == "RISING"
    assert out["second_half_share"] == pytest.approx(1.0)


def test_the_bar_boundaries_are_the_registered_ones():
    assert R3.R3_START_MIN_FRAC == 0.50
    assert R3.R3_COMPOUND_MAX_FRAC == 0.20
    assert R3.R3_DECILES == 10
    assert R3.R3_MIN_TASKS_PER_DECILE == 500
    assert R3.R3_MIN_FULL_GAP_S == 1.0


def test_exactly_at_the_born_bad_boundary_fires_it():
    """Decile 1 carries 9 of 18 units of gap mass: exactly the 0.50 bar."""
    best, worst = _arms([9.0] + [1.0] * 9)
    out = R3.read(best, worst)
    assert out["first_decile_frac"] == pytest.approx(0.5)
    assert out["verdict"] == "PRESENT-FROM-THE-START"


def test_two_seeds_that_serve_alike_are_void_not_compounding():
    """A near-zero gap would otherwise read COMPOUNDS on a noise ratio."""
    best, worst = _arms([0.0] * 9 + [0.5])
    out = R3.read(best, worst)
    assert out["verdict"] == "VOID-NO-GAP"
    assert "no spread to decompose" in out["why"]


def test_captures_covering_different_tasks_are_refused():
    best, worst = _arms([1.0] * 10)
    worst.pop(max(worst))
    with pytest.raises(R3.DecileReadError, match="different tasks"):
        R3.read(best, worst)


def test_disjoint_captures_are_refused():
    best, _ = _arms([1.0] * 10)
    other = {k + 10 ** 7: v for k, v in best.items()}
    with pytest.raises(R3.DecileReadError, match="share no taskId"):
        R3.read(best, other)


def test_a_thin_decile_is_refused_rather_than_read():
    best, worst = _arms([1.0] * 10, n_per_decile=R3.R3_MIN_TASKS_PER_DECILE - 100)
    with pytest.raises(R3.DecileReadError, match="below"):
        R3.read(best, worst)


def test_a_missing_field_is_not_read_as_zero():
    with pytest.raises(R3.DecileReadError, match="has no queueTime"):
        R3._num({"taskId": 3, "dispatchedTime": 1.0}, "queueTime")


def test_a_zero_field_is_a_real_value():
    assert R3._num({"taskId": 3, "queueTime": 0.0}, "queueTime") == 0.0


def test_decile_bounds_come_from_one_arm_and_apply_to_both():
    """Same trace, same task set: a task must land in the same decile for both arms."""
    best, worst = _arms([1.0] * 10)
    bounds = R3.decile_bounds([v[0] for v in best.values()])
    for tid in list(best)[::137]:
        assert R3.decile_of(best[tid][0], bounds) == R3.decile_of(worst[tid][0], bounds)


def test_decile_bounds_refuse_fewer_tasks_than_deciles():
    with pytest.raises(R3.DecileReadError, match="cannot be split"):
        R3.decile_bounds([1.0, 2.0, 3.0])


def test_load_arm_refuses_a_capture_without_task_records(tmp_path):
    import json
    p = tmp_path / "empty.json"
    p.write_text(json.dumps({"stats": {"taskResults": [], "statsSchemaVersion": 3}}))
    with pytest.raises(R3.DecileReadError, match="KEEP_RAW"):
        R3.load_arm(p)
