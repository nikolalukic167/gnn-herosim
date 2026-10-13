"""uncut_groups_s0_v1 w(n): decision wall and plan evaluations per served batch size, beside the unchanged totals."""
import pytest

from src.placement.decision_timing import DecisionLog


def test_by_bucket_splits_calls_wall_and_plans_by_served_size():
    log = DecisionLog()
    log.add(4, 0.010, 1, plan_evals=600)
    log.add(10, 0.020, 0, plan_evals=100_000)
    log.add(11, 0.500, 2, plan_evals=50_000)
    log.add(226, 2.000, 5, plan_evals=400_000)
    log.add(1, 0.001)  # a per-arrival call: no plan meter
    s = log.summary()
    b = s["by_bucket"]
    assert b["le10"] == {"calls": 3, "total_s": pytest.approx(0.031), "plan_evals": 100_600}
    assert b["11_50"] == {"calls": 1, "total_s": pytest.approx(0.5), "plan_evals": 50_000}
    assert b["gt50"] == {"calls": 1, "total_s": pytest.approx(2.0), "plan_evals": 400_000}
    assert s["calls"] == 5 and s["total_s"] == pytest.approx(2.531) and s["refine_moves_total"] == 8
    assert sum(v["calls"] for v in b.values()) == s["calls"]
