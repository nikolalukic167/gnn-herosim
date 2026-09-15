"""Tests for serving_stability_v1 S3, the live gate. Bars are constants; these pin machinery."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts_cosim import serving_stability_v1_s3_read as S3  # noqa: E402
from scripts_cosim.serving_stability_v1_s1_s2_read import (  # noqa: E402
    ARMS, BURNED, CELLS, KNOWN_UNSERVABLE, StabilityReadError,
)

scipy_stats = pytest.importorskip("scipy.stats")


def _arm(name, elapsed, bind_pct=90.0, decisions=100000):
    return {"arm": name, "averageElapsedTime": elapsed, "schedulerCounters": {
        "queue_guard_decisions": decisions,
        "queue_guard_steps_active": int(decisions * bind_pct / 100.0),
        "queue_guard_masked": 1000}}


def _world(guarded_s, unguarded_s, reactive_s=26.0, bind=90.0, drop=()):
    g, u = {}, {}
    for cell in CELLS:
        u[f"{cell}__reactive"] = {"arm": f"{cell}__reactive",
                                  "averageElapsedTime": reactive_s}
        for arm in ARMS:
            for s in range(1, 17):
                if (arm, s) in KNOWN_UNSERVABLE or (arm, s) in BURNED.get(cell, ()):
                    continue
                if (cell, arm, s) in drop:
                    continue
                g[f"{cell}__guarded_{arm}_s{s}"] = _arm(
                    f"{cell}__guarded_{arm}_s{s}", guarded_s, bind)
                u[f"{cell}__{arm}_s{s}"] = {"arm": f"{cell}__{arm}_s{s}",
                                            "averageElapsedTime": unguarded_s}
    return g, u


def test_bars_are_the_values_signed_in_the_node():
    assert S3.S3_K == 3.0
    assert S3.S3_MIN_SEEDS == 12
    assert S3.S3_MIN_CELLS == 2
    assert S3.S3_ALPHA == 0.05
    assert S3.S3B_MIN_BIND_PCT == 5.0
    assert S3.S3A_MAX_HANGS == 0


def test_mann_whitney_matches_scipy():
    xs = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0]
    ys = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 30.0]
    ours = S3.mann_whitney_u_p(xs, ys)
    ref = scipy_stats.mannwhitneyu(xs, ys, alternative="two-sided",
                                   use_continuity=True).pvalue
    assert ours == pytest.approx(ref, rel=0.05)


def test_a_guardrail_that_never_bound_voids_the_gate():
    g, u = _world(guarded_s=10.0, unguarded_s=50.0, bind=1.0)
    out = S3.read(g, u)
    assert out["S3b"]["verdict"] == "GUARDRAIL-DID-NOTHING-VOID"
    assert out["S3c"]["verdict"] == "VOID-S3B"
    assert out["S3d"]["verdict"] == "VOID-S3B"


def test_missing_counters_fail_loud_rather_than_reading_as_zero():
    g, u = _world(10.0, 50.0)
    key = next(iter(g))
    g[key]["schedulerCounters"] = {}
    with pytest.raises(StabilityReadError, match="whitelist"):
        S3.read(g, u)


def test_a_hang_confounds_the_primary_rather_than_being_ignored():
    dropped = tuple((CELLS[0], "gnn", s) for s in (5, 6))
    g, u = _world(10.0, 50.0, drop=dropped)
    out = S3.read(g, u)
    assert out["S3a"]["hangs"] == 2
    assert out["S3a"]["verdict"] == "DEADLOCKS-NOT-A-SERVING-DEFAULT"
    assert out["S3c"]["verdict"].startswith("CONFOUNDED-S3A")


def test_beating_reactive_everywhere_fires_the_primary():
    g, u = _world(guarded_s=10.0, unguarded_s=50.0, reactive_s=26.0)
    out = S3.read(g, u)
    assert out["S3a"]["verdict"] == "NO-DEADLOCK"
    assert out["S3b"]["verdict"] == "GUARDRAIL-BOUND"
    assert out["S3c"]["verdict"] == "LEARNED-BEATS-REACTIVE"
    assert out["S3d"]["verdict"] == "GUARDRAIL-HELPS"
    assert out["outcome"] == "STABILITY-WAS-THE-LEVER"


def test_helping_without_beating_reactive_is_helps_not_enough():
    g, u = _world(guarded_s=40.0, unguarded_s=60.0, reactive_s=26.0)
    out = S3.read(g, u)
    assert out["S3c"]["verdict"] == "REACTIVE-STILL-WINS"
    assert out["S3d"]["verdict"] == "GUARDRAIL-HELPS"
    assert out["outcome"] == "HELPS-NOT-ENOUGH"


def test_a_guardrail_that_makes_things_worse_is_not_the_lever():
    g, u = _world(guarded_s=70.0, unguarded_s=50.0, reactive_s=26.0)
    out = S3.read(g, u)
    assert out["S3d"]["verdict"] == "GUARDRAIL-DOES-NOT-HELP"
    assert out["outcome"] == "STABILITY-NOT-THE-LEVER"


def test_one_cell_is_not_enough_for_the_primary():
    g, u = _world(guarded_s=40.0, unguarded_s=50.0, reactive_s=26.0)
    for s in range(1, 17):
        k = f"{CELLS[0]}__guarded_gnn_s{s}"
        if k in g:
            g[k]["averageElapsedTime"] = 10.0
    out = S3.read(g, u)
    assert out["S3c"]["gnn_cells_firing"] == 1
    assert out["S3c"]["verdict"] == "REACTIVE-STILL-WINS"


def test_the_registered_pointwise_prediction_is_carried_in_the_output():
    g, u = _world(10.0, 50.0)
    out = S3.read(g, u)
    assert "count theorem" in out["registered_prediction"]
    assert "mpoff_cells_firing" in out["S3c"]
