#!/usr/bin/env python3
"""Label live audit snapshots with co-sim RTT combos for GNN training."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts_cosim.live_snapshot_cosim_oracle import (
    CosimOracleContext,
    combo_to_placement_plan,
    enumerate_combos,
    oracle_choice_cosim,
    snapshot_tasks,
)


def _live_task_placements(snapshot: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    full_q = snapshot.get("full_queue_snapshot") or {}
    for task_idx, task in enumerate(snapshot.get("tasks", [])):
        qsnap = {
            c["queue_key"]: int(c.get("queue_length", 0) or 0)
            for c in task.get("candidates", [])
        }
        temporal = {
            c["queue_key"]: {
                "current_task_remaining": float(c.get("current_task_remaining", 0) or 0),
                "cold_start_remaining": float(c.get("cold_start_remaining", 0) or 0),
                "comm_remaining": float(c.get("comm_remaining", 0) or 0),
            }
            for c in task.get("candidates", [])
        }
        rows.append(
            {
                "task_id": task_idx,
                "task_type": task.get("task_type"),
                "source_node": task.get("source_node"),
                "queue_snapshot_at_scheduling": qsnap,
                "full_queue_snapshot": full_q,
                "temporal_state_at_scheduling": temporal,
            }
        )
    return rows


def _inject_live_snapshots_into_task_results(
    result: Dict[str, Any],
    snapshot: Mapping[str, Any],
) -> None:
    task_rows = _live_task_placements(snapshot)
    task_results = result.get("stats", {}).get("taskResults") or []
    by_id = {int(tr.get("taskId", -1)): tr for tr in task_results}
    for row in task_rows:
        tr = by_id.get(int(row["task_id"]))
        if tr is None:
            continue
        tr["queueSnapshotAtScheduling"] = row["queue_snapshot_at_scheduling"]
        tr["fullQueueSnapshot"] = row["full_queue_snapshot"]
        tr["temporalStateAtScheduling"] = row["temporal_state_at_scheduling"]


def _combo_count(snapshot: Mapping[str, Any]) -> int:
    tasks = snapshot.get("tasks") or []
    total = 1
    for task in tasks:
        total *= max(1, len(task.get("candidates") or []))
    return total


def export_snapshot(
    ctx: CosimOracleContext,
    snapshot: Mapping[str, Any],
    out_dir: Path,
    max_combos: int,
) -> bool:
    tasks = snapshot_tasks(snapshot, None)
    if not tasks:
        return False

    combos, combo_count = enumerate_combos(tasks, max_combos)
    if not combos:
        return False

    placements_path = out_dir / "placements" / "placements.jsonl"
    placements_path.parent.mkdir(parents=True, exist_ok=True)

    with placements_path.open("w") as f:
        for combo in combos:
            plan = combo_to_placement_plan(combo)
            rtt = ctx.run_placement_plan(snapshot, tasks, plan)
            if not math.isfinite(rtt):
                continue
            payload = {
                "placement_plan": {str(k): [int(v[0]), int(v[1])] for k, v in plan.items()},
                "rtt": float(rtt),
            }
            f.write(json.dumps(payload, separators=(",", ":")) + "\n")

    oracle = oracle_choice_cosim(ctx, snapshot, tasks, max_combos)
    plan = combo_to_placement_plan(oracle.combo)
    optimal_plan = {str(k): [int(v[0]), int(v[1])] for k, v in plan.items()}

    # Re-run optimal to capture full stats payload for prepare_graphs_cache_seq.
    from copy import deepcopy
    from contextlib import redirect_stderr, redirect_stdout
    from io import StringIO
    import os

    from src.executesimulation import execute_simulation
    from src.placement.live_snapshot_seed import build_live_snapshot_seed
    from src.placement.constants import KEEP_ALIVE, QUEUE_LENGTH

    infrastructure = deepcopy(ctx._base_infrastructure)
    infrastructure["live_snapshot_seed"] = build_live_snapshot_seed({**snapshot, "tasks": list(tasks)})
    infrastructure["forced_placements"] = {
        int(k): (int(v[0]), int(v[1])) for k, v in plan.items()
    }
    infrastructure["fast_forward_warmup"] = True
    infrastructure["fast_forward_threshold"] = 1
    infrastructure["scheduler"] = {"batch_size": len(tasks), "batch_timeout": 0.02}

    from scripts_cosim.live_snapshot_cosim_oracle import build_workload_from_snapshot

    config = {"infrastructure": infrastructure, "workload": build_workload_from_snapshot(tasks)}
    prev = os.environ.get("GNN_CAPTURE_DATASET_STATE")
    os.environ["GNN_CAPTURE_DATASET_STATE"] = "0"
    try:
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            optimal_result = execute_simulation(
                config,
                ctx._sim_inputs,
                scheduling_strategy="determined_determined",
                cache_policy="fifo",
                task_priority="fifo",
                keep_alive=KEEP_ALIVE,
                queue_length=QUEUE_LENGTH,
            )
    finally:
        if prev is None:
            os.environ.pop("GNN_CAPTURE_DATASET_STATE", None)
        else:
            os.environ["GNN_CAPTURE_DATASET_STATE"] = prev

    _inject_live_snapshots_into_task_results(optimal_result, snapshot)
    optimal_result["sample"] = {"placement_plan": optimal_plan}

    with (out_dir / "optimal_result.json").open("w") as f:
        json.dump(optimal_result, f)
    with (out_dir / "best.json").open("w") as f:
        json.dump({"rtt": float(oracle.rtt)}, f)
    with (out_dir / "system_state_captured_unique.json").open("w") as f:
        json.dump({"task_placements": _live_task_placements(snapshot)}, f)
    with (out_dir / "live_snapshot.json").open("w") as f:
        json.dump(snapshot, f)

    meta = {
        "snapshot_id": snapshot.get("snapshot_id"),
        "combo_count": combo_count,
        "labeled_combos": sum(1 for _ in placements_path.open()),
        "optimal_rtt": float(oracle.rtt),
    }
    with (out_dir / "placement_metadata.json").open("w") as f:
        json.dump(meta, f, indent=2)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "simulation_data" / "space_with_network.json")
    parser.add_argument("--sim-input", type=Path, default=PROJECT_ROOT / "data" / "nofs-ids")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-combos", type=int, default=200000)
    parser.add_argument("--max-runtime-s", type=int, default=None)
    parser.add_argument("--sort-by-combos", action="store_true", help="Label smallest combo spaces first")
    args = parser.parse_args()

    ctx = CosimOracleContext(args.config, args.sim_input, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    snapshots: List[Dict[str, Any]] = []
    with args.snapshots.open() as f:
        for line in f:
            if line.strip():
                snapshots.append(json.loads(line))
    if args.sort_by_combos:
        snapshots.sort(key=_combo_count)

    deadline = time.monotonic() + args.max_runtime_s if args.max_runtime_s else None
    exported = 0
    for snapshot in snapshots:
        if args.limit is not None and exported >= args.limit:
            break
        if deadline is not None and time.monotonic() >= deadline:
            print(f"[TIME] stopping after {exported} exports", flush=True)
            break
        sid = snapshot.get("snapshot_id", exported)
        out = args.output_dir / f"ds_{int(sid):05d}"
        if out.exists() and (out / "placements" / "placements.jsonl").exists():
            exported += 1
            continue
        print(f"[export] snapshot {sid} combos={_combo_count(snapshot)}", flush=True)
        try:
            if export_snapshot(ctx, snapshot, out, args.max_combos):
                exported += 1
        except ValueError as exc:
            print(f"[WARN] snapshot {sid}: {exc}", flush=True)
    print(f"Exported {exported} live training datasets to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
