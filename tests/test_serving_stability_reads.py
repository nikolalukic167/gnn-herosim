"""Tests for serving_stability_v1 S1/S2. Bars are constants; these pin the machinery."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts_cosim import serving_stability_v1_s1_s2_read as S  # noqa: E402


def _summary(arm, deciles, n_per=5000):
    return {"arm": arm, "decile_summary": {
        "deciles": [{"decile": i + 1, "n": n_per, "mean_queue_s": q,
                     "mean_elapsed_s": q + 8.0} for i, q in enumerate(deciles)],
        "n_tasks": n_per * len(deciles),
        "mean_queue_s": sum(deciles) / len(deciles),
        "queue_max_over_min": max(deciles) / min(deciles) if min(deciles) > 0 else None}}


def _world(gnn_early, mpoff_early, gnn_ratio_tail, reactive=(20.0,) * 10, cells=S.CELLS):
    """Build a full arm set. `*_early` is the queue in deciles 1-2 for that arm."""
    arms = {}
    for cell in cells:
        arms[f"{cell}__reactive"] = _summary(f"{cell}__reactive", list(reactive))
        for arm, early in (("gnn", gnn_early), ("mpoff", mpoff_early)):
            for seed in range(1, 17):
                if (arm, seed) in S.KNOWN_UNSERVABLE:
                    continue
                if (arm, seed) in S.BURNED.get(cell, ()):
                    continue
                tail = gnn_ratio_tail if arm == "gnn" else 30.0
                arms[f"{cell}__{arm}_s{seed}"] = _summary(
                    f"{cell}__{arm}_s{seed}", [early, early] + [tail] * 8)
    return arms


def test_bars_are_the_values_signed_in_the_node():
    assert S.S1_MIN_SEEDS == 12
    assert S.S1_MIN_CELLS == 2
    assert S.S2_REACTIVE_MAX_RATIO == 2.5
    assert S.S2_ARM_MULTIPLE == 2.0
    assert S.S2_MIN_CELLS == 2
    assert S.EARLY_DECILES == 2
    assert S.S1_REGISTERED_EXPECTATION == "UNCERTAIN"


def test_the_scoping_seeds_are_burned_on_their_cell_only():
    assert S.BURNED["cell_s7901_f4000_pg16"] == (("gnn", 8), ("gnn", 14))
    assert "cell_s9001_f4000_pg16" not in S.BURNED


def test_burned_and_unservable_seeds_are_dropped():
    arms = _world(10.0, 10.0, 40.0)
    c1 = S.seeds_for(arms, "cell_s7901_f4000_pg16", "gnn")
    assert 3 not in c1 and 8 not in c1 and 14 not in c1
    assert len(c1) == 13
    c2 = S.seeds_for(arms, "cell_s9001_f4000_pg16", "gnn")
    assert 3 not in c2 and 8 in c2 and 14 in c2
    assert len(c2) == 15


def test_early_queue_is_weighted_by_decile_size():
    s = {"arm": "x", "decile_summary": {"deciles": [
        {"decile": 1, "n": 1000, "mean_queue_s": 10.0},
        {"decile": 2, "n": 3000, "mean_queue_s": 20.0}]}}
    assert S.early_queue(s) == pytest.approx(17.5)


def test_early_queue_refuses_a_missing_decile_mean():
    s = {"arm": "x", "decile_summary": {"deciles": [
        {"decile": 1, "n": 10, "mean_queue_s": None},
        {"decile": 2, "n": 10, "mean_queue_s": 1.0}]}}
    with pytest.raises(S.StabilityReadError, match="no mean queue"):
        S.early_queue(s)


def test_s1_fires_when_the_arms_are_ahead_early_on_every_cell():
    out = S.read(_world(gnn_early=10.0, mpoff_early=10.0, gnn_ratio_tail=40.0))
    assert out["S1"]["verdict"] == "EARLY-ADVANTAGE-REAL"
    assert out["S1"]["gnn_cells_firing"] == 3


def test_s1_does_not_fire_when_the_arms_are_behind_early():
    out = S.read(_world(gnn_early=30.0, mpoff_early=30.0, gnn_ratio_tail=40.0))
    assert out["S1"]["verdict"] == "NO-EARLY-ADVANTAGE"
    assert out["S1"]["gnn_cells_firing"] == 0


def test_s1_needs_two_cells_not_one():
    arms = _world(gnn_early=10.0, mpoff_early=10.0, gnn_ratio_tail=40.0)
    # make cells 2 and 3 slower early than reactive
    for cell in S.CELLS[1:]:
        for seed in S.seeds_for(arms, cell, "gnn"):
            arms[f"{cell}__gnn_s{seed}"] = _summary(
                f"{cell}__gnn_s{seed}", [30.0, 30.0] + [40.0] * 8)
    out = S.read(arms)
    assert out["S1"]["gnn_cells_firing"] == 1
    assert out["S1"]["verdict"] == "NO-EARLY-ADVANTAGE"


def test_s1_reports_mpoff_but_the_verdict_is_read_on_gnn():
    arms = _world(gnn_early=30.0, mpoff_early=10.0, gnn_ratio_tail=40.0)
    out = S.read(arms)
    assert out["S1"]["mpoff_cells_firing"] == 3
    assert out["S1"]["verdict"] == "NO-EARLY-ADVANTAGE"


def test_s2_fires_when_reactive_is_bounded_and_the_arms_are_not():
    out = S.read(_world(gnn_early=10.0, mpoff_early=10.0, gnn_ratio_tail=40.0,
                        reactive=(15.0, 15.0) + (20.0,) * 8))
    assert out["S2"]["per_cell"][S.CELLS[0]]["reactive_ratio"] == pytest.approx(20 / 15)
    assert out["S2"]["verdict"] == "ARMS-DO-NOT-STABILISE"


def test_s2_does_not_fire_when_reactive_is_itself_unstable():
    """If the reference wanders too, the contrast is not about stabilisation."""
    out = S.read(_world(gnn_early=10.0, mpoff_early=10.0, gnn_ratio_tail=40.0,
                        reactive=(2.0,) + (100.0,) * 9))
    assert out["S2"]["per_cell"][S.CELLS[0]]["reactive_clears"] is False
    assert out["S2"]["verdict"] == "STABILITY-NOT-SEPARATED"


def test_s2_does_not_fire_when_the_arms_are_as_stable_as_reactive():
    out = S.read(_world(gnn_early=18.0, mpoff_early=18.0, gnn_ratio_tail=20.0,
                        reactive=(18.0,) * 2 + (20.0,) * 8))
    assert out["S2"]["verdict"] == "STABILITY-NOT-SEPARATED"


def test_a_missing_reactive_arm_is_refused():
    arms = _world(10.0, 10.0, 40.0)
    del arms[f"{S.CELLS[0]}__reactive"]
    with pytest.raises(S.StabilityReadError, match="no reactive arm"):
        S.read(arms)


def test_a_thin_cell_is_voided_not_read():
    arms = _world(10.0, 10.0, 40.0)
    for seed in list(range(1, 16)):
        arms.pop(f"{S.CELLS[1]}__gnn_s{seed}", None)
    out = S.read(arms)
    assert out["S1"]["per_cell"][S.CELLS[1]]["gnn"]["verdict"] == "VOID-TOO-FEW-SEEDS"
