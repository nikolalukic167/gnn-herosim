#!/usr/bin/env python3
"""
Audit or repair optimal_result.json exports for per-task queueSnapshotAtScheduling.

Repair re-runs the stored optimal placement with SIM_FORCE_FULL_STATS=1 using
config/sim_inputs embedded in optimal_result.json (no brute-force re-search).

Existing run_queue_big 4-task corpora are typically already complete; use --audit-only
to confirm before a cache rebuild.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.executecosimulation import KEEP_ALIVE, QUEUE_LENGTH, execute_simulation
from src.placement.model import DataclassJSONEncoder

logger = logging.getLogger(__name__)


def _n_tasks_from_workload(workload: Any) -> int:
    if isinstance(workload, dict) and "events" in workload:
        return len(workload["events"])
    if isinstance(workload, list):
        return len(workload)
    return 0


def per_task_snapshots_complete(data: Dict[str, Any], n_tasks: int) -> bool:
    task_results = data.get("stats", {}).get("taskResults", [])
    if len(task_results) < n_tasks:
        return False
    for tr in sorted(task_results, key=lambda t: int(t.get("taskId", -1))):
        q = tr.get("queueSnapshotAtScheduling") or tr.get("queue_snapshot_at_scheduling")
        if not isinstance(q, dict) or not q:
            return False
    return True


def audit_dataset(optimal_path: Path) -> str:
    with open(optimal_path, "r") as f:
        data = json.load(f)
    n_tasks = _n_tasks_from_workload(data.get("config", {}).get("workload", {}))
    if n_tasks == 0:
        return "skip_no_workload"
    if not data.get("stats", {}).get("taskResults"):
        return "missing_task_results"
    if per_task_snapshots_complete(data, n_tasks):
        return "ok"
    return "needs_refresh"


def repair_dataset(optimal_path: Path, dry_run: bool) -> str:
    status = audit_dataset(optimal_path)
    if status == "ok":
        return "skip_ok"
    if status.startswith("skip"):
        return status

    with open(optimal_path, "r") as f:
        old = json.load(f)

    placement_plan = old.get("sample", {}).get("placement_plan")
    config = old.get("config")
    sim_inputs = old.get("sim_inputs")
    if not placement_plan or not config or not sim_inputs:
        return "skip_incomplete_export"

    if dry_run:
        return "would_refresh"

    os.environ["SIM_FORCE_FULL_STATS"] = "1"
    full_config = copy.deepcopy(config)
    result = execute_simulation(
        full_config,
        sim_inputs,
        "determined_determined",
        cache_policy="fifo",
        task_priority="fifo",
        keep_alive=KEEP_ALIVE,
        queue_length=QUEUE_LENGTH,
    )
    result["sample"] = old.get("sample", {"placement_plan": placement_plan})
    result["config"] = config
    result["sim_inputs"] = sim_inputs

    backup = optimal_path.with_suffix(".json.bak")
    if not backup.exists():
        optimal_path.rename(backup)
    with open(optimal_path, "w") as f:
        json.dump(result, f, cls=DataclassJSONEncoder, indent=2)

    with open(optimal_path, "r") as f:
        refreshed = json.load(f)
    n_tasks = _n_tasks_from_workload(config.get("workload", {}))
    if per_task_snapshots_complete(refreshed, n_tasks):
        return "refreshed"
    return "refresh_incomplete"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=PROJECT_ROOT
        / "simulation_data"
        / "artifacts"
        / "run_queue_big"
        / "gnn_datasets_4tasks",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--repair",
        action="store_true",
        help="Re-run optimal sim with SIM_FORCE_FULL_STATS=1 when snapshots missing",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    counts: Dict[str, int] = {}
    dirs = sorted(args.base_dir.glob("ds_*"))
    if args.limit > 0:
        dirs = dirs[: args.limit]

    for dataset_dir in dirs:
        optimal_path = dataset_dir / "optimal_result.json"
        if not optimal_path.exists():
            counts["skip_no_optimal"] = counts.get("skip_no_optimal", 0) + 1
            continue
        if args.repair:
            status = repair_dataset(optimal_path, args.dry_run)
        else:
            status = audit_dataset(optimal_path)
        counts[status] = counts.get(status, 0) + 1

    logger.info("Summary: %s", counts)


if __name__ == "__main__":
    main()
