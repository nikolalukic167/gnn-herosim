#!/usr/bin/env python3
"""
A/B test: refresh_optimal_full_stats.py --rewrite-ssc vs --repair --force

Questions answered:
  1. Does --rewrite-ssc faithfully rebuild SSC from optimal_result.json stats?
  2. Does --repair (re-sim) change RTT labels vs stored co-sim export?
  3. Are rewrite and repair SSC payloads equivalent for graph-cache fields?

Arms per dataset (uses temp copies; does not mutate corpus):
  A  existing SSC on disk
  B  build_system_state_captured(optimal stats)  [= --rewrite-ssc logic]
  C  execute_simulation(optimal placement, SIM_FORCE_FULL_STATS=1)  [= --repair core]

Pass:
  - B matches A on graph-relevant SSC fields (replica set order ignored)
  - C RTT within 1e-3 of stored optimal total_rtt / per-task elapsedTime
  - B vs C graph fields match after repair-scale normalization
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.executecosimulation import (  # noqa: E402
    KEEP_ALIVE,
    QUEUE_LENGTH,
    build_system_state_captured,
    execute_simulation,
    rtt_from_stats,
)
from scripts_cosim.refresh_optimal_full_stats import (  # noqa: E402
    rewrite_ssc_from_optimal,
    system_state_complete,
    ssc_file_complete,
)

RTT_TOL = 1e-3
GRAPH_FIELDS = (
    "scheduler_state",
    "initialized_snapshot",
    "available_resources",
)


def _n_tasks_from_workload(workload: Any) -> int:
    if isinstance(workload, dict) and "events" in workload:
        return len(workload["events"])
    if isinstance(workload, list):
        return len(workload)
    return 0


def _normalize_replicas(replicas: Any) -> Dict[str, Set[str]]:
    if not isinstance(replicas, dict):
        return {}
    out: Dict[str, Set[str]] = {}
    for node, plats in replicas.items():
        if isinstance(plats, dict):
            out[str(node)] = {str(k) for k in plats.keys()}
        elif isinstance(plats, list):
            out[str(node)] = {str(p) for p in plats}
        else:
            out[str(node)] = set()
    return out


def _placement_graph_slice(ssc: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for tp in ssc.get("task_placements") or []:
        rows.append(
            {
                "task_id": tp.get("task_id"),
                "execution_node": tp.get("execution_node"),
                "execution_platform": tp.get("execution_platform"),
                "queue_time": round(float(tp.get("queue_time", 0)), 6),
                "elapsed_time": round(float(tp.get("elapsed_time", 0)), 6),
                "queue_snapshot_keys": sorted(
                    (tp.get("queue_snapshot_at_scheduling") or {}).keys()
                ),
                "full_queue_nonempty": bool(tp.get("full_queue_snapshot")),
                "temporal_nonempty": bool(tp.get("temporal_state_at_scheduling")),
            }
        )
    return sorted(rows, key=lambda r: int(r["task_id"] or -1))


def compare_ssc_graph_fields(
    a: Dict[str, Any], b: Dict[str, Any]
) -> Tuple[bool, List[str]]:
    diffs: List[str] = []

    rep_a = _normalize_replicas(a.get("replicas"))
    rep_b = _normalize_replicas(b.get("replicas"))
    if rep_a != rep_b:
        diffs.append(f"replicas: {rep_a} != {rep_b}")

    for field in GRAPH_FIELDS:
        if a.get(field) != b.get(field):
            diffs.append(f"{field}: mismatch")

    pa = _placement_graph_slice(a)
    pb = _placement_graph_slice(b)
    if pa != pb:
        diffs.append(f"task_placements graph slice: {pa} != {pb}")

    rtt_a = round(float(a.get("total_rtt", 0)), 6)
    rtt_b = round(float(b.get("total_rtt", 0)), 6)
    if abs(rtt_a - rtt_b) > RTT_TOL:
        diffs.append(f"total_rtt: {rtt_a} vs {rtt_b}")

    return len(diffs) == 0, diffs


def run_repair_sim(optimal_path: Path) -> Dict[str, Any]:
    with open(optimal_path) as f:
        old = json.load(f)
    placement_plan = {
        int(k): (int(v[0]), int(v[1]))
        for k, v in old["sample"]["placement_plan"].items()
    }
    config = copy.deepcopy(old["config"])
    sim_inputs = old["sim_inputs"]
    config.setdefault("infrastructure", {})["forced_placements"] = placement_plan
    os.environ["SIM_FORCE_FULL_STATS"] = "1"
    os.environ["COSIM_SUPPRESS_SIM_PRINTS"] = "1"
    result = execute_simulation(
        config,
        sim_inputs,
        "determined_determined",
        cache_policy="fifo",
        task_priority="fifo",
        keep_alive=KEEP_ALIVE,
        queue_length=QUEUE_LENGTH,
    )
    stats = result.get("stats")
    if not stats:
        raise RuntimeError(f"execute_simulation returned no stats for {optimal_path.parent.name}")
    return stats


def per_task_rtts(stats: Dict[str, Any]) -> List[float]:
    rows = []
    for tr in sorted(stats.get("taskResults") or [], key=lambda t: int(t.get("taskId", -1))):
        rows.append(float(tr.get("elapsedTime", 0)))
    return rows


def audit_dataset(dataset_dir: Path, run_repair: bool) -> Dict[str, Any]:
    optimal_path = dataset_dir / "optimal_result.json"
    ssc_path = dataset_dir / "system_state_captured_unique.json"
    result: Dict[str, Any] = {"dataset": dataset_dir.name}

    if not optimal_path.exists():
        result["status"] = "skip_no_optimal"
        return result

    with open(optimal_path) as f:
        optimal = json.load(f)
    stats = optimal.get("stats") or {}
    n_tasks = _n_tasks_from_workload(optimal.get("config", {}).get("workload", {}))
    result["n_tasks"] = n_tasks
    result["stored_total_rtt"] = round(rtt_from_stats(stats), 6)
    result["stored_per_task_rtt"] = per_task_rtts(stats)
    result["stored_queue_times"] = [
        round(float(tr.get("queueTime", 0)), 6)
        for tr in sorted(stats.get("taskResults") or [], key=lambda t: int(t.get("taskId", -1)))
    ]
    result["stored_pull_times"] = [
        round(float(tr.get("pullTime", 0)), 6)
        for tr in sorted(stats.get("taskResults") or [], key=lambda t: int(t.get("taskId", -1)))
    ]
    result["optimal_state_complete"] = system_state_complete(optimal, n_tasks)
    result["ssc_on_disk_complete"] = ssc_file_complete(ssc_path, n_tasks) if ssc_path.exists() else False

    ssc_existing: Dict[str, Any] = {}
    if ssc_path.exists():
        with open(ssc_path) as f:
            ssc_existing = json.load(f)

    ssc_rewrite = build_system_state_captured(stats)
    rewrite_ok, rewrite_diffs = compare_ssc_graph_fields(ssc_existing, ssc_rewrite)
    result["rewrite_vs_existing_ok"] = rewrite_ok
    result["rewrite_vs_existing_diffs"] = rewrite_diffs

    with tempfile.TemporaryDirectory(prefix="rewrite_ssc_ab_") as tmp:
        tmp_dir = Path(tmp) / dataset_dir.name
        tmp_dir.mkdir()
        shutil.copy2(optimal_path, tmp_dir / "optimal_result.json")
        if ssc_path.exists():
            shutil.copy2(ssc_path, tmp_dir / "system_state_captured_unique.json")

        status = rewrite_ssc_from_optimal(tmp_dir / "optimal_result.json")
        result["rewrite_cli_status"] = status
        with open(tmp_dir / "system_state_captured_unique.json") as f:
            ssc_after_cli = json.load(f)
        cli_ok, cli_diffs = compare_ssc_graph_fields(ssc_rewrite, ssc_after_cli)
        result["rewrite_cli_matches_build"] = cli_ok
        result["rewrite_cli_diffs"] = cli_diffs

    if not run_repair:
        result["status"] = "rewrite_only"
        return result

    repair_stats = run_repair_sim(optimal_path)
    repair_rtt = rtt_from_stats(repair_stats)
    repair_per_task = per_task_rtts(repair_stats)
    result["repair_total_rtt"] = round(repair_rtt, 6)
    result["repair_per_task_rtt"] = repair_per_task
    result["repair_queue_times"] = [
        round(float(tr.get("queueTime", 0)), 6)
        for tr in sorted(repair_stats.get("taskResults") or [], key=lambda t: int(t.get("taskId", -1)))
    ]
    result["repair_pull_times"] = [
        round(float(tr.get("pullTime", 0)), 6)
        for tr in sorted(repair_stats.get("taskResults") or [], key=lambda t: int(t.get("taskId", -1)))
    ]

    rtt_delta = abs(result["stored_total_rtt"] - result["repair_total_rtt"])
    result["repair_rtt_delta"] = round(rtt_delta, 9)
    result["repair_rtt_ok"] = rtt_delta <= RTT_TOL

    per_task_ok = all(
        abs(a - b) <= RTT_TOL
        for a, b in zip(result["stored_per_task_rtt"], repair_per_task)
    )
    result["repair_per_task_ok"] = per_task_ok

    ssc_repair = build_system_state_captured(repair_stats)
    repair_ssc_ok, repair_ssc_diffs = compare_ssc_graph_fields(ssc_rewrite, ssc_repair)
    result["rewrite_vs_repair_ssc_ok"] = repair_ssc_ok
    result["rewrite_vs_repair_ssc_diffs"] = repair_ssc_diffs

    result["status"] = "full"
    result["pass"] = (
        rewrite_ok
        and cli_ok
        and result["repair_rtt_ok"]
        and per_task_ok
        and repair_ssc_ok
    )
    return result


def pick_datasets(base_dir: Path, n: int, indices: List[int]) -> List[Path]:
    if indices:
        return [base_dir / f"ds_{i:05d}" for i in indices]
    dirs = sorted(base_dir.glob("ds_*"))
    if n <= 0:
        return dirs
    # spread across corpus
    if len(dirs) <= n:
        return dirs
    step = max(1, len(dirs) // n)
    return [dirs[i * step] for i in range(n)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=PROJECT_ROOT / "simulation_data/gnn_datasets_4tasks_1060_warmth_v2",
    )
    parser.add_argument("--datasets", type=int, default=5)
    parser.add_argument("--indices", type=int, nargs="*", default=[])
    parser.add_argument(
        "--skip-repair",
        action="store_true",
        help="Only test rewrite path (fast; no re-simulation)",
    )
    args = parser.parse_args()

    dataset_dirs = pick_datasets(args.base_dir, args.datasets, args.indices)
    if not dataset_dirs:
        print(f"No datasets under {args.base_dir}")
        sys.exit(1)

    print(f"Base dir: {args.base_dir}")
    print(f"Testing {len(dataset_dirs)} dataset(s), repair={'no' if args.skip_repair else 'yes'}")
    print("-" * 72)

    results = []
    for d in dataset_dirs:
        if not d.exists():
            print(f"{d.name}: MISSING")
            continue
        r = audit_dataset(d, run_repair=not args.skip_repair)
        results.append(r)
        print(f"\n=== {r['dataset']} ===")
        print(f"  optimal stats complete: {r.get('optimal_state_complete')}")
        print(f"  SSC on disk complete:   {r.get('ssc_on_disk_complete')}")
        print(f"  stored total RTT:       {r.get('stored_total_rtt')}s")
        print(f"  stored queueTimes:      {r.get('stored_queue_times')}")
        print(f"  stored pullTimes:       {r.get('stored_pull_times')}")
        print(f"  rewrite vs existing:    {'PASS' if r.get('rewrite_vs_existing_ok') else 'FAIL'}")
        if r.get("rewrite_vs_existing_diffs"):
            for diff in r["rewrite_vs_existing_diffs"][:5]:
                print(f"    - {diff}")
        print(f"  rewrite CLI status:     {r.get('rewrite_cli_status')}")
        print(f"  rewrite CLI = build():  {'PASS' if r.get('rewrite_cli_matches_build') else 'FAIL'}")
        if not args.skip_repair:
            print(f"  repair total RTT:       {r.get('repair_total_rtt')}s (delta {r.get('repair_rtt_delta')})")
            print(f"  repair RTT match:       {'PASS' if r.get('repair_rtt_ok') else 'FAIL'}")
            print(f"  repair per-task RTT:    {'PASS' if r.get('repair_per_task_ok') else 'FAIL'}")
            print(f"  rewrite vs repair SSC:  {'PASS' if r.get('rewrite_vs_repair_ssc_ok') else 'FAIL'}")
            if r.get("rewrite_vs_repair_ssc_diffs"):
                for diff in r["rewrite_vs_repair_ssc_diffs"][:5]:
                    print(f"    - {diff}")
            print(f"  OVERALL:                {'PASS' if r.get('pass') else 'FAIL'}")

    print("\n" + "=" * 72)
    print("SUMMARY")
    rewrite_pass = sum(1 for r in results if r.get("rewrite_vs_existing_ok"))
    print(f"  rewrite vs existing SSC: {rewrite_pass}/{len(results)} PASS")
    if not args.skip_repair:
        repair_rtt_pass = sum(1 for r in results if r.get("repair_rtt_ok"))
        repair_ssc_pass = sum(1 for r in results if r.get("rewrite_vs_repair_ssc_ok"))
        overall_pass = sum(1 for r in results if r.get("pass"))
        print(f"  repair RTT vs stored:  {repair_rtt_pass}/{len(results)} PASS")
        print(f"  rewrite vs repair SSC: {repair_ssc_pass}/{len(results)} PASS")
        print(f"  overall:                 {overall_pass}/{len(results)} PASS")

        pull_nonzero = sum(
            1 for r in results if any(p > 0 for p in r.get("stored_pull_times") or [])
        )
        print(f"  datasets with pullTime>0: {pull_nonzero}/{len(results)}")

    incomplete = sum(1 for r in results if not r.get("ssc_on_disk_complete"))
    if incomplete:
        print(f"  WARNING: {incomplete} dataset(s) had incomplete SSC on disk")


if __name__ == "__main__":
    main()
