"""Tests for offline_live_transfer_v1 R2 -- the surrogate search.

Bars are constants in the module and were committed before any candidate was correlated
with live latency; these pin the machinery, not the bars' values.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts_cosim import offline_live_transfer_v1_surrogate_read as S  # noqa: E402
from scripts_cosim.offline_live_transfer_v1_pairing_read import (  # noqa: E402
    ARMS, FAMILIES, TransferReadError,
)


# ------------------------------------------------------------------ the statistics
def test_concentration_is_one_when_everything_lands_on_one_platform():
    assert S.concentration({0: (1, 2), 1: (1, 2), 2: (1, 2)}) == pytest.approx(1.0)


def test_concentration_is_minimal_when_the_batch_is_spread():
    assert S.concentration({0: (1, 1), 1: (2, 2), 2: (3, 3)}) == pytest.approx(1 / 3)


def test_concentration_refuses_an_empty_plan():
    with pytest.raises(TransferReadError):
        S.concentration({})


def _snapshot(depths):
    """One task per entry; depths maps (node, platform) -> queue_length."""
    return {"tasks": [{"task_id": 100, "candidates": [
        {"node_id": n, "platform_id": p, "queue_length": q, "queue_key": f"{n}:{p}"}
        for (n, p), q in depths.items()]}]}


def test_depth_is_zero_when_the_shallowest_replica_is_chosen():
    snap = _snapshot({(1, 1): 0.0, (2, 2): 9.0})
    assert S.depth_above_shallowest({0: (1, 1)}, snap, {100: 0}) == pytest.approx(0.0)


def test_depth_is_the_gap_when_a_deeper_replica_is_chosen():
    snap = _snapshot({(1, 1): 2.0, (2, 2): 9.0})
    assert S.depth_above_shallowest({0: (2, 2)}, snap, {100: 0}) == pytest.approx(7.0)


def test_depth_refuses_a_placement_that_is_not_a_candidate():
    snap = _snapshot({(1, 1): 0.0})
    with pytest.raises(TransferReadError, match="not among this task's candidates"):
        S.depth_above_shallowest({0: (7, 7)}, snap, {100: 0})


def test_depth_refuses_when_no_task_matches_the_snapshot():
    snap = _snapshot({(1, 1): 0.0})
    with pytest.raises(TransferReadError, match="could be matched"):
        S.depth_above_shallowest({5: (1, 1)}, snap, {100: 9})


def test_range_sensitivity_is_zero_when_the_plan_does_not_move():
    plan = {0: (1, 1), 1: (2, 2)}
    assert S.range_sensitivity(plan, dict(plan)) == pytest.approx(0.0)


def test_range_sensitivity_is_one_when_every_placement_moves():
    assert S.range_sensitivity({0: (1, 1), 1: (2, 2)}, {0: (3, 3), 1: (4, 4)}) == 1.0


def test_range_sensitivity_refuses_disjoint_plans():
    with pytest.raises(TransferReadError, match="share no tasks"):
        S.range_sensitivity({0: (1, 1)}, {9: (1, 1)})


# ----------------------------------------------------------------------- Holm
def test_holm_matches_the_textbook_on_a_known_family():
    adj = S.holm({"a": 0.01, "b": 0.02, "c": 0.04}, 4)
    assert adj["a"] == pytest.approx(0.04)   # 4 * 0.01
    assert adj["b"] == pytest.approx(0.06)   # 3 * 0.02
    assert adj["c"] == pytest.approx(0.08)   # 2 * 0.04


def test_holm_is_monotone_non_decreasing():
    adj = S.holm({"a": 0.001, "b": 0.30, "c": 0.02}, 4)
    assert adj["a"] <= adj["c"] <= adj["b"]


def test_holm_caps_at_one():
    assert S.holm({"a": 0.9}, 4)["a"] == pytest.approx(1.0)


def test_holm_family_size_smaller_than_the_tests_is_refused():
    with pytest.raises(TransferReadError, match="family size"):
        S.holm({"a": 0.1, "b": 0.2}, 1)


def test_a_not_computed_candidate_still_divides_by_four():
    """The registered family size is 4; shrinking it would weaken the correction."""
    three = S.holm({"a": 0.01, "b": 0.02, "c": 0.04}, S.R2_HOLM_N)
    assert three["a"] == pytest.approx(0.04)


# ----------------------------------------------------------------------- the read
def _grid(fn):
    """One value per (family, arm, seed) from a callable of the seed."""
    return {(f, a, s): fn(f, a, s)
            for f in FAMILIES for a in ARMS for s in range(1, 17)}


def _live():
    return _grid(lambda f, a, s: 50.0 + s)


def test_a_perfect_predictor_is_promoted():
    vals = {"depth": _grid(lambda f, a, s: float(s))}
    out = S.read(vals, _live())
    assert out["verdict"] == "SURROGATE:depth"
    assert out["promoted"] == ["depth"]
    assert out["not_computed"] == ["concentration", "one_step_regret", "range_sensitivity"]


def test_noise_is_not_promoted():
    noise = [0.31, -0.22, 0.07, 0.44, -0.38, 0.12, -0.05, 0.27,
             -0.41, 0.19, 0.02, -0.33, 0.36, -0.14, 0.23, -0.29]
    vals = {"depth": _grid(lambda f, a, s: noise[s - 1])}
    out = S.read(vals, _live())
    assert out["verdict"] == "NO-SURROGATE"
    assert out["promoted"] == []


def test_at_most_one_candidate_is_promoted_and_the_other_is_reported():
    vals = {
        "depth": _grid(lambda f, a, s: float(s)),
        # slightly noisier but still a strong predictor
        "concentration": _grid(lambda f, a, s: float(s) + (0.4 if s % 2 else -0.4)),
    }
    out = S.read(vals, _live())
    assert len(out["promoted"]) == S.R2_MAX_PROMOTED == 1
    assert out["promoted"] == ["depth"]
    assert out["reported_not_used"] == ["concentration"]


def test_an_unregistered_candidate_is_refused():
    with pytest.raises(TransferReadError, match="unregistered"):
        S.read({"made_up": _grid(lambda f, a, s: float(s))}, _live())


def test_a_candidate_clearing_rho_but_failing_holm_is_not_promoted():
    """Strong in two families, absent in the third, so the pooled p is weak."""
    vals = {"depth": _grid(lambda f, a, s: float(s) if f != "F3_warm" else 0.5 * (s % 3))}
    out = S.read(vals, _live())
    row = out["candidates"]["depth"]
    # whatever the correlation, promotion requires all three sub-bars
    assert row["is_surrogate"] == (
        row["clears_rho_bar"] and row["clears_p_bar"] and row["clears_family_bar"])


def test_bars_are_the_values_signed_in_the_node():
    assert S.CANDIDATES == ("depth", "concentration", "one_step_regret", "range_sensitivity")
    assert S.R2_POOLED_Z_MIN_ABS_RHO == 0.50
    assert S.R2_WITHIN_MIN_ABS_RHO == 0.40
    assert S.R2_WITHIN_MIN_FAMILIES == 2
    assert S.R2_ALPHA == 0.05
    assert S.R2_HOLM_N == 4
    assert S.R2_MAX_PROMOTED == 1


# ------------------------------------------------- the queue-scale decode (candidate 4)
def test_queue_scale_of_one_is_a_strict_no_op():
    torch = pytest.importorskip("torch")
    from scripts_cosim.drainable_debug_d1_decode import scale_queue_depth

    class G:
        pass
    g = G()
    g.platform_features = torch.arange(28, dtype=torch.float32).reshape(2, 14)
    before = g.platform_features.clone()
    assert scale_queue_depth([g], 1.0) == 0
    assert torch.equal(g.platform_features, before)


def test_queue_scale_touches_only_the_queue_column():
    torch = pytest.importorskip("torch")
    from scripts_cosim.drainable_debug_d1_decode import QUEUE_DEPTH_DIM, scale_queue_depth

    class G:
        pass
    g = G()
    g.platform_features = torch.ones((3, 14), dtype=torch.float32)
    assert scale_queue_depth([g], 5.0) == 1
    for col in range(14):
        expected = 5.0 if col == QUEUE_DEPTH_DIM else 1.0
        assert torch.allclose(g.platform_features[:, col],
                              torch.full((3,), expected)), f"column {col} changed"


def test_queue_scale_refuses_an_unexpected_platform_width():
    torch = pytest.importorskip("torch")
    from scripts_cosim.drainable_debug_d1_decode import scale_queue_depth

    class G:
        pass
    g = G()
    g.platform_features = torch.ones((2, 9), dtype=torch.float32)
    with pytest.raises(SystemExit, match="expected 14"):
        scale_queue_depth([g], 5.0)


def test_queue_scale_refuses_a_non_positive_scale():
    from scripts_cosim.drainable_debug_d1_decode import scale_queue_depth
    with pytest.raises(SystemExit, match="must be > 0"):
        scale_queue_depth([], 0.0)
