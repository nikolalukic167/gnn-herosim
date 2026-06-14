#!/usr/bin/env python3
"""
Regime B metrics calibration — prove the apparatus on existing cold-start physics.

Runs N=4 contended vs parallel cold bursts (platform_reuse_v1) using the same
infrastructure as test_cold_start_queue_last_task_ab.py, scores with
burst_regime_summary, and checks oracle regret ≈ 4× (125s / 31s).

Usage:
    pipenv run python3 scripts_cosim/calibrate_regime_b.py
    pipenv run python3 scripts_cosim/calibrate_regime_b.py --warmth-physics node_disk_v2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts_cosim.regime_b_metrics import (  # noqa: E402
    attach_burst_ids_from_workload,
    burst_regime_summary,
    total_rtt_trap_stats,
)
from src.executecosimulation import (  # noqa: E402
    KEEP_ALIVE,
    QUEUE_LENGTH,
    execute_simulation,
    extract_task_metrics,
    load_simulation_inputs,
    rtt_from_stats,
)

SIM_INPUT = PROJECT_ROOT / "data/nofs-ids"
N_TASKS = 4
TOLERANCE_S = 1.0
OUT_DIR = PROJECT_ROOT / "simulation_data/regime_b_calibration"

# Expected last-task RTT at N=4 (from storage_contention.md / cold-start AB audit)
EXPECTED = {
    "platform_reuse_v1": {"contended": 125.57, "parallel": 31.65, "min_regret_ratio": 3.5},
    "node_disk_v2": {"contended": 31.65, "parallel": 31.65, "min_regret_ratio": 0.9},
}


def build_burst_workload(n: int, *, burst_id: str) -> Dict[str, Any]:
    events = [
        {
            "timestamp": 0.0,
            "application": {"name": "nofs-dnn1", "dag": {"dnn1": []}},
            "qos": {"name": "medium", "maxDurationDeviation": 15},
            "node_name": "client_node0",
            "burst_id": burst_id,
        }
        for _ in range(n)
    ]
    return {"rps": max(n, 1), "duration": 1, "events": events}


def run_burst_sim(
    sim_inputs: Dict[str, Any],
    nodes: List[Dict[str, Any]],
    forced: Dict[int, Tuple[int, int]],
    det_placements: List[Dict[str, Any]],
    n: int,
    *,
    burst_id: str,
    warmth_physics: str,
) -> Dict[str, Any]:
    workload = build_burst_workload(n, burst_id=burst_id)
    infrastructure: Dict[str, Any] = {
        "network": {"bandwidth": 100.0},
        "nodes": nodes,
        "preinitialize_platforms": True,
        "defer_cold_replica_init": True,
        "warmth_physics": warmth_physics,
        "deterministic_replica_placements": {"dnn1": det_placements},
        "replica_plan": {
            "preinit_clients": [],
            "preinit_servers": [f"node{i}" for i in range(n)],
            "preinit_task_types": ["dnn1"],
            "replicas_config": {"dnn1": {"per_client": 0, "per_server": 0}},
            "prewarm_config": {},
        },
        "forced_placements": forced,
        "scheduler": {"batch_size": n, "batch_timeout": 0.02},
        "fast_forward_warmup": True,
        "fast_forward_threshold": 1,
    }
    config = {"infrastructure": infrastructure, "workload": workload}

    prev_capture = os.environ.get("GNN_CAPTURE_DATASET_STATE")
    os.environ["GNN_CAPTURE_DATASET_STATE"] = "0"
    os.environ["SIM_FORCE_FULL_STATS"] = "1"
    try:
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            result = execute_simulation(
                config,
                sim_inputs,
                scheduling_strategy="determined_determined",
                cache_policy="fifo",
                task_priority="fifo",
                keep_alive=KEEP_ALIVE,
                queue_length=QUEUE_LENGTH,
            )
    finally:
        if prev_capture is None:
            os.environ.pop("GNN_CAPTURE_DATASET_STATE", None)
        else:
            os.environ["GNN_CAPTURE_DATASET_STATE"] = prev_capture

    stats = result.get("stats", {})
    task_rows = attach_burst_ids_from_workload(
        extract_task_metrics(stats),
        workload["events"],
    )
    return {
        "stats": stats,
        "workload": workload,
        "task_rows": task_rows,
        "total_rtt": rtt_from_stats(stats),
        "regime_b": burst_regime_summary(task_rows),
        "trap": total_rtt_trap_stats(task_rows),
    }


def _import_ab_helpers():
    from scripts_cosim.test_cold_start_queue_last_task_ab import (  # noqa: WPS433
        build_test_nodes,
        contended_placements,
        parallel_placements,
        load_theory_constants,
    )

    return build_test_nodes, contended_placements, parallel_placements, load_theory_constants


def calibrate(*, warmth_physics: str) -> Dict[str, Any]:
    build_test_nodes, contended_placements, parallel_placements, load_theory_constants = (
        _import_ab_helpers()
    )
    sim_inputs = load_simulation_inputs(SIM_INPUT)
    theory = load_theory_constants(sim_inputs)
    nodes, node_id_by_name, plat_id_by_node_local = build_test_nodes(N_TASKS)

    fc, dc = contended_placements(N_TASKS, node_id_by_name, plat_id_by_node_local)
    fp, dp = parallel_placements(N_TASKS, node_id_by_name, plat_id_by_node_local)

    oracle_run = run_burst_sim(
        sim_inputs,
        nodes,
        fp,
        dp,
        N_TASKS,
        burst_id="parallel_oracle",
        warmth_physics=warmth_physics,
    )
    oracle_score = float(oracle_run["regime_b"]["regime_b_primary_score_s"])

    contended_run = run_burst_sim(
        sim_inputs,
        nodes,
        fc,
        dc,
        N_TASKS,
        burst_id="contended_greedy",
        warmth_physics=warmth_physics,
    )
    contended_scored = burst_regime_summary(
        contended_run["task_rows"],
        oracle_rtt=oracle_score,
    )

    expected = EXPECTED.get(warmth_physics, EXPECTED["platform_reuse_v1"])
    contended_score = float(contended_scored["regime_b_primary_score_s"])
    regret_ratio = float(contended_scored.get("oracle_regret_ratio", float("nan")))

    checks = {
        "oracle_near_expected": abs(oracle_score - expected["parallel"]) <= TOLERANCE_S,
        "contended_near_expected": abs(contended_score - expected["contended"]) <= TOLERANCE_S,
        "regret_ratio_min": regret_ratio >= expected["min_regret_ratio"],
        "total_rtt_trap_visible": contended_run["trap"]["total_rtt_over_primary_ratio"] >= 3.0,
    }

    return {
        "warmth_physics": warmth_physics,
        "theory": theory,
        "n_tasks": N_TASKS,
        "oracle": {
            "burst_id": "parallel_oracle",
            "regime_b_primary_score_s": oracle_score,
            "expected_s": expected["parallel"],
            "total_rtt_s": oracle_run["total_rtt"],
            "burst_summaries": oracle_run["regime_b"]["burst_summaries"],
        },
        "contended": {
            "burst_id": "contended_greedy",
            "regime_b_primary_score_s": contended_score,
            "expected_s": expected["contended"],
            "total_rtt_s": contended_run["total_rtt"],
            "burst_summaries": contended_scored["burst_summaries"],
            "oracle_rtt_s": contended_scored.get("oracle_rtt_s"),
            "oracle_regret_s": contended_scored.get("oracle_regret_s"),
            "oracle_regret_ratio": regret_ratio,
            "total_rtt_trap": contended_run["trap"],
        },
        "checks": checks,
        "pass": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate Regime B metrics harness")
    parser.add_argument(
        "--warmth-physics",
        default="platform_reuse_v1",
        choices=["platform_reuse_v1", "node_disk_v2"],
        help="Warmth physics profile (platform_reuse_v1 shows ~4× regret)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUT_DIR,
        help="Directory for calibration JSON output",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("Regime B Metrics Calibration")
    print(f"warmth_physics={args.warmth_physics}  N={N_TASKS}  SIM_FORCE_FULL_STATS=1")
    print("=" * 72)

    result = calibrate(warmth_physics=args.warmth_physics)

    oracle = result["oracle"]
    contended = result["contended"]
    print(f"\nOracle (parallel cold burst)")
    print(f"  regime_b_primary_score_s: {oracle['regime_b_primary_score_s']:.2f}s  (expected ≈{oracle['expected_s']:.2f}s)")
    print(f"  total_rtt (misleading):   {oracle['total_rtt_s']:.2f}s")

    print(f"\nContended (co-located cold burst — FilterStore trap)")
    print(f"  regime_b_primary_score_s: {contended['regime_b_primary_score_s']:.2f}s  (expected ≈{contended['expected_s']:.2f}s)")
    print(f"  total_rtt (misleading):   {contended['total_rtt_s']:.2f}s")
    print(f"  total_rtt / primary:      {contended['total_rtt_trap']['total_rtt_over_primary_ratio']:.2f}x")

    if contended.get("oracle_regret_ratio") is not None:
        print(f"\nOracle regret")
        print(f"  oracle_rtt_s:             {contended['oracle_rtt_s']:.2f}s")
        print(f"  oracle_regret_s:          {contended['oracle_regret_s']:.2f}s")
        print(f"  oracle_regret_ratio:      {contended['oracle_regret_ratio']:.2f}x")

    print(f"\nPer-burst queueTime (contended):")
    for bs in contended["burst_summaries"]:
        print(
            f"  burst={bs['burst_id']}  n={bs['n_tasks']}  "
            f"max_queue={bs['max_queue_time_s']:.2f}s  mean_queue={bs['mean_queue_time_s']:.2f}s"
        )

    print(f"\nChecks:")
    for name, ok in result["checks"].items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")

    args.output.mkdir(parents=True, exist_ok=True)
    out_path = args.output / f"calibration_{args.warmth_physics}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nResults written to {out_path}")

    print("\n" + "=" * 72)
    if result["pass"]:
        print("VERDICT: PASS — Regime B metrics harness calibrated")
    else:
        print("VERDICT: FAIL — see checks above")
    print("=" * 72)

    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
