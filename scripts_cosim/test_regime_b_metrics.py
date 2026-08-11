#!/usr/bin/env python3
"""Unit tests for Regime B metrics harness."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts_cosim.regime_b_metrics import (  # noqa: E402
    attach_burst_ids_from_workload,
    burst_regime_summary,
    total_rtt_trap_stats,
)


def test_last_task_is_by_task_id_not_max_elapsed():
    rows = [
        {"task_id": 0, "elapsed_time": 100.0, "queue_time": 0.0, "burst_id": "b0"},
        {"task_id": 1, "elapsed_time": 50.0, "queue_time": 0.0, "burst_id": "b0"},
        {"task_id": 2, "elapsed_time": 75.0, "queue_time": 10.0, "burst_id": "b0"},
    ]
    summary = burst_regime_summary(rows)
    assert summary["regime_b_primary_score_s"] == 100.0
    assert summary["last_task_rtt_s"] == 75.0
    assert summary["burst_summaries"][0]["last_task_elapsed_s"] == 75.0
    assert summary["burst_summaries"][0]["max_elapsed_s"] == 100.0


def test_oracle_regret_and_total_rtt_trap():
    rows = [
        {"task_id": i, "elapsed_time": 31.0 + i * 31.0, "queue_time": float(i * 31), "burst_id": "cold"}
        for i in range(4)
    ]
    summary = burst_regime_summary(rows, oracle_rtt=31.65)
    assert summary["regime_b_primary_score_s"] == pytest.approx(124.0)
    assert summary["oracle_regret_ratio"] == pytest.approx(124.0 / 31.65)
    trap = total_rtt_trap_stats(rows)
    assert trap["total_rtt_s"] == pytest.approx(sum(31.0 + i * 31.0 for i in range(4)))
    assert trap["total_rtt_over_primary_ratio"] > 2.0


def test_attach_burst_ids_from_workload():
    task_rows = [
        {"task_id": 0, "elapsed_time": 1.0, "queue_time": 0.0},
        {"task_id": 1, "elapsed_time": 2.0, "queue_time": 0.0},
    ]
    events = [{"burst_id": "burst_a"}, {"burst_id": "burst_b"}]
    tagged = attach_burst_ids_from_workload(task_rows, events)
    assert [r["burst_id"] for r in tagged] == ["burst_a", "burst_b"]
    multi = burst_regime_summary(tagged)
    assert multi["regime_b_primary_score_s"] == 2.0
    assert {b["burst_id"] for b in multi["burst_summaries"]} == {"burst_a", "burst_b"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
