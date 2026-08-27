#!/usr/bin/env python3
"""
Audit or repair optimal_result.json exports for inference-aligned system state.

Repair re-runs the stored optimal placement with SIM_FORCE_FULL_STATS=1 using
config/sim_inputs embedded in optimal_result.json (no brute-force re-search).
Writes system_state_captured_unique.json with scheduling-time top-level state
(replicas, available_resources, scheduler_state) plus per-task queue/temporal
snapshots matching live GNN inference capture.

DOES NOT write or replace placements/placements.jsonl.
Repair + recache is NOT sufficient for near-RTT training: rtt_chunk_*.pkl needs
the full (placement_plan, rtt) sweep from brute-force co-sim. See
docs/notes/placements_jsonl_required.md.

Use --rewrite-ssc to rebuild SSC files from already-refreshed optimal_result.json
without re-running simulation.
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

from src.executecosimulation import (
    QUEUE_LENGTH,
    build_system_state_captured,
    cosim_keep_alive,
    execute_simulation,
)
from src.placement.model import DataclassJSONEncoder

logger = logging.getLogger(__name__)


def _n_tasks_from_workload(workload: Any) -> int:
    if isinstance(workload, dict) and "events" in workload:
        return len(workload["events"])
    if isinstance(workload, list):
        return len(workload)
    return 0


def _normalize_placement_plan(
    placement_plan: Dict[Any, Any],
) -> Dict[int, tuple[int, int]]:
    return {int(k): (int(v[0]), int(v[1])) for k, v in placement_plan.items()}


def write_system_state_captured(dataset_dir: Path, stats: Dict[str, Any]) -> None:
    captured_state = build_system_state_captured(stats)
    out_path = dataset_dir / "system_state_captured_unique.json"
    with open(out_path, "w") as f:
        json.dump(captured_state, f, indent=2, cls=DataclassJSONEncoder)


def system_state_complete(data: Dict[str, Any], n_tasks: int) -> bool:
    stats = data.get("stats", {})
    task_results = stats.get("taskResults", [])
    if len(task_results) < n_tasks:
        return False

    for tr in sorted(task_results, key=lambda t: int(t.get("taskId", -1))):
        q = tr.get("queueSnapshotAtScheduling") or tr.get("queue_snapshot_at_scheduling")
        fqs = tr.get("fullQueueSnapshot") or tr.get("full_queue_snapshot")
        temporal = tr.get("temporalStateAtScheduling") or tr.get(
            "temporal_state_at_scheduling"
        )
        if not isinstance(q, dict) or not q:
            return False
        if not isinstance(fqs, dict) or not fqs:
            return False
        if not isinstance(temporal, dict) or not temporal:
            return False

    scheduling = stats.get("schedulingStateCapture") or {}
    system_state_results = stats.get("systemStateResults") or []
    top = scheduling or (system_state_results[-1] if system_state_results else {})
    if not top.get("replicas"):
        return False
    return True


def ssc_file_complete(ssc_path: Path, n_tasks: int) -> bool:
    if not ssc_path.exists():
        return False
    with open(ssc_path, "r") as f:
        ssc = json.load(f)
    if not ssc.get("replicas"):
        return False
    if not ssc.get("scheduler_state"):
        return False
    placements = ssc.get("task_placements") or []
    if len(placements) < n_tasks:
        return False
    for tp in placements[:n_tasks]:
        if not tp.get("queue_snapshot_at_scheduling"):
            return False
        if not tp.get("full_queue_snapshot"):
            return False
        if not tp.get("temporal_state_at_scheduling"):
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
    if system_state_complete(data, n_tasks) and ssc_file_complete(
        optimal_path.parent / "system_state_captured_unique.json", n_tasks
    ):
        return "ok"
    return "needs_refresh"


def rewrite_ssc_from_optimal(optimal_path: Path) -> str:
    with open(optimal_path, "r") as f:
        data = json.load(f)
    stats = data.get("stats")
    if not stats or not stats.get("taskResults"):
        return "skip_no_stats"
    n_tasks = _n_tasks_from_workload(data.get("config", {}).get("workload", {}))
    write_system_state_captured(optimal_path.parent, stats)
    if ssc_file_complete(optimal_path.parent / "system_state_captured_unique.json", n_tasks):
        return "ssc_rewritten"
    return "ssc_incomplete"


def repair_dataset(
    optimal_path: Path,
    dry_run: bool,
    write_ssc: bool = True,
    force: bool = False,
) -> str:
    if not force:
        audit_status = audit_dataset(optimal_path)
        if audit_status == "ok":
            return "skip_ok"
        if audit_status.startswith("skip"):
            return audit_status

    with open(optimal_path, "r") as f:
        old = json.load(f)

    placement_plan = old.get("sample", {}).get("placement_plan")
    config = old.get("config")
    sim_inputs = old.get("sim_inputs")
    if not placement_plan or not config or not sim_inputs:
        return "skip_incomplete_export"

    if dry_run:
        return "would_refresh"

    placement_plan = _normalize_placement_plan(placement_plan)
    os.environ["SIM_FORCE_FULL_STATS"] = "1"
    full_config = copy.deepcopy(config)
    full_config.setdefault("infrastructure", {})["forced_placements"] = placement_plan
    # keep_alive goes through cosim_keep_alive() (HEROSIM_COSIM_KEEP_ALIVE, unset =
    # constants.KEEP_ALIVE, bit-identical to prior behavior). The route_b DAG corpora
    # were generated with HEROSIM_COSIM_KEEP_ALIVE=1000000 — replaying them at the
    # default 30 s evicts idle forced replicas mid-episode ("Invalid forced placement
    # ... not in replicas"), exactly the failure cosim_keep_alive()'s docstring
    # records. The repair env must match the collection's generation env.
    result = execute_simulation(
        full_config,
        sim_inputs,
        "determined_determined",
        cache_policy="fifo",
        task_priority="fifo",
        keep_alive=cosim_keep_alive(),
        queue_length=QUEUE_LENGTH,
    )
    # Faithfulness guard: the replay re-runs the SAME forced plan under the SAME
    # config/sim_inputs, so when the old export carries a total_rtt the replay must
    # reproduce it — a divergence means the replay env does not match the generation
    # env (keep-alive, HEROSIM_DATA_LOCALITY, ...) and the rewritten export would
    # silently describe different physics than the corpus. Fail before touching the
    # file.
    old_rtt = (old.get("stats") or {}).get("total_rtt")
    new_rtt = (result.get("stats") or {}).get("total_rtt")
    if old_rtt is not None and new_rtt is not None:
        if abs(new_rtt - old_rtt) > 1e-9 * max(1.0, abs(old_rtt)):
            raise RuntimeError(
                f"{optimal_path.parent.name}: replay total_rtt {new_rtt!r} != stored "
                f"{old_rtt!r} — replay env does not reproduce the generation env; "
                "refusing to overwrite. Set the collection's generation env "
                "(HEROSIM_COSIM_KEEP_ALIVE / HEROSIM_DATA_LOCALITY / ...) and re-run."
            )
    result["sample"] = {
        **(old.get("sample") or {}),
        "placement_plan": {str(k): [v[0], v[1]] for k, v in placement_plan.items()},
    }
    result["config"] = config
    result["sim_inputs"] = sim_inputs

    backup = optimal_path.with_suffix(".json.bak")
    if not backup.exists():
        optimal_path.rename(backup)
    with open(optimal_path, "w") as f:
        json.dump(result, f, cls=DataclassJSONEncoder, indent=2)

    if write_ssc:
        write_system_state_captured(optimal_path.parent, result["stats"])

    with open(optimal_path, "r") as f:
        refreshed = json.load(f)
    n_tasks = _n_tasks_from_workload(config.get("workload", {}))
    if system_state_complete(refreshed, n_tasks):
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
        "--start-from",
        type=int,
        default=0,
        help="Skip first N ds_* directories (sorted) before processing",
    )
    parser.add_argument(
        "--max-datasets",
        type=int,
        default=0,
        help="Process at most N datasets after --start-from (0 = all remaining)",
    )
    parser.add_argument(
        "--repair",
        action="store_true",
        help="Re-run optimal sim with SIM_FORCE_FULL_STATS=1 when state is incomplete",
    )
    parser.add_argument(
        "--rewrite-ssc",
        action="store_true",
        help="Rebuild system_state_captured_unique.json from optimal_result.json only",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-repair even when queue snapshots exist but SSC/top-level/temporal is incomplete",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    counts: Dict[str, int] = {}
    dirs = sorted(args.base_dir.glob("ds_*"))
    if args.start_from > 0:
        dirs = dirs[args.start_from:]
    if args.max_datasets > 0:
        dirs = dirs[: args.max_datasets]
    if args.limit > 0:
        dirs = dirs[: args.limit]

    for dataset_dir in dirs:
        optimal_path = dataset_dir / "optimal_result.json"
        if not optimal_path.exists():
            counts["skip_no_optimal"] = counts.get("skip_no_optimal", 0) + 1
            continue

        if args.rewrite_ssc:
            status = rewrite_ssc_from_optimal(optimal_path)
        elif args.repair:
            if args.force or audit_dataset(optimal_path) != "ok":
                status = repair_dataset(optimal_path, args.dry_run, force=args.force)
            else:
                status = "skip_ok"
        else:
            status = audit_dataset(optimal_path)
        counts[status] = counts.get(status, 0) + 1

    logger.info("Summary: %s", counts)


if __name__ == "__main__":
    main()
