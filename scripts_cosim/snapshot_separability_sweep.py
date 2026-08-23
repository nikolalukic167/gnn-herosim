#!/usr/bin/env python3
"""Bounded placement sweeps over live-audit snapshots, written as pseudo-datasets.

WS3 of the co-sim deep-dive campaign: the question is whether the co-sim RTT target is
still pointwise-additive on LIVE-regime states (queue-loaded, autoscaled, mid-trace),
or whether the synthetic t=0 snapshot regime manufactured the additivity. Each snapshot
from a live-audit capture (`LIVE_AUDIT_SNAPSHOT_PATH` JSONL, batch>=4 with per-task
candidate lists) is swept through the co-sim oracle physics
(`live_snapshot_cosim_oracle.CosimOracleContext`) over a bounded Cartesian product of
placements, and the (plan, rtt) rows are written in the standard dataset-dir schema so
`separability_diagnostic.py` runs on the output unmodified:

    <out-root>/ds_snap_<NNNNN>/
        placements/placements.jsonl     {"placement_plan": {...}, "rtt": r} per plan
        workload.json                   events with application.dag + node_name
        infrastructure.json             copied from the cell (link context, optional)
        space_with_network.json         copied from the cell config (link context)
        snapshot_meta.json              provenance: snapshot id/time, pruning stats

Bounding (pre-registered): each task's candidate list is pruned to --top-k by
`expected_completion_from_snapshot_candidate` (the same schedule-time cost the audit
pipeline uses), so the sweep is <= K^n_tasks combos. Unlike the co-sim corpus
generator there is NO unique-replica constraint — collisions are allowed, exactly the
interaction the additivity question is about. Pruned/skipped counts are first-class
outputs: silently dropping snapshots would reproduce the sampling-bias bug this
campaign is cataloguing.

Usage (J2 calibration):
  python3 scripts_cosim/snapshot_separability_sweep.py \
      --snapshots <cell>.jsonl --config <cell_config.json> --sim-input data/nofs-ids \
      --out-root simulation_data/snapshot_sweeps/<cell> \
      --top-k 6 --max-snapshots 2 --workers 8 --calibrate
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts_cosim.live_snapshot_cosim_oracle import (  # noqa: E402
    CosimOracleContext,
    build_workload_from_snapshot,
    combo_to_placement_plan,
    snapshot_tasks,
)
from src.placement.scheduling_cost import (  # noqa: E402
    expected_completion_from_snapshot_candidate,
)

_CTX: Optional[CosimOracleContext] = None
_SNAPSHOT: Optional[Dict[str, Any]] = None
_TASKS: Optional[List[Dict[str, Any]]] = None


def _init_worker(config_path: str, sim_input: str, seed: int, snapshot_json: str) -> None:
    global _CTX, _SNAPSHOT, _TASKS
    _CTX = CosimOracleContext(Path(config_path), Path(sim_input), seed=seed)
    _SNAPSHOT = json.loads(snapshot_json)
    _TASKS = list(_SNAPSHOT["tasks"])


def _run_combo(plan_items: Sequence[Tuple[int, Tuple[int, int]]]) -> Tuple[Dict[int, Tuple[int, int]], float]:
    assert _CTX is not None and _SNAPSHOT is not None and _TASKS is not None
    plan = dict(plan_items)
    rtt = _CTX.run_placement_plan(_SNAPSHOT, _TASKS, plan)
    return plan, rtt


def prune_candidates(
    tasks: Sequence[Dict[str, Any]], top_k: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, int]]]:
    """Per task, keep the top_k candidates by schedule-time ECT (pointwise)."""
    pruned_tasks: List[Dict[str, Any]] = []
    stats: List[Dict[str, int]] = []
    for task in tasks:
        candidates = list(task.get("candidates", []))
        if not candidates:
            raise ValueError(f"task {task.get('task_id')} has no candidates")
        scored = sorted(
            candidates,
            key=lambda c: expected_completion_from_snapshot_candidate(
                c, int(c.get("queue_length", 0)), 0
            ),
        )
        kept = scored[:top_k]
        stats.append({"candidates": len(candidates), "kept": len(kept)})
        t = dict(task)
        t["candidates"] = kept
        pruned_tasks.append(t)
    return pruned_tasks, stats


def stratified_subsample(snapshots: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    """Uniform stride over time order, so early and late trace windows both survive."""
    if len(snapshots) <= n:
        return snapshots
    ordered = sorted(snapshots, key=lambda s: float(s.get("time", 0.0)))
    idx = [round(i * (len(ordered) - 1) / (n - 1)) for i in range(n)]
    return [ordered[i] for i in sorted(set(idx))]


def sweep_snapshot(
    snap: Dict[str, Any],
    args: argparse.Namespace,
    ds_dir: Path,
) -> Dict[str, Any]:
    tasks = snapshot_tasks(snap, args.horizon)
    full_product = math.prod(len(t.get("candidates", [])) for t in tasks)
    tasks, prune_stats = prune_candidates(tasks, args.top_k)
    combos = list(
        itertools.product(*[
            [(int(c["node_id"]), int(c["platform_id"])) for c in t["candidates"]]
            for t in tasks
        ])
    )
    plans = [tuple(enumerate(combo)) for combo in combos]

    snap_for_worker = {**snap, "tasks": tasks}
    started = time.perf_counter()
    rows: List[Tuple[Dict[int, Tuple[int, int]], float]] = []
    failed = 0
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(str(args.config), str(args.sim_input), args.seed, json.dumps(snap_for_worker)),
    ) as pool:
        for plan, rtt in pool.map(_run_combo, plans, chunksize=8):
            if math.isfinite(rtt):
                rows.append((plan, rtt))
            else:
                failed += 1
    elapsed = time.perf_counter() - started

    if not rows:
        raise RuntimeError(
            f"snapshot {snap.get('snapshot_id')}: all {len(plans)} combos failed — "
            "refusing to write an empty sweep"
        )

    placements_dir = ds_dir / "placements"
    placements_dir.mkdir(parents=True, exist_ok=True)
    with open(placements_dir / "placements.jsonl", "w") as f:
        for plan, rtt in rows:
            rec = {
                "placement_plan": {str(k): list(v) for k, v in plan.items()},
                "rtt": rtt,
            }
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")

    workload = build_workload_from_snapshot(tasks)
    (ds_dir / "workload.json").write_text(json.dumps(workload, indent=1))

    # Link context for the diagnostic's link-repair controls (backbone cells only).
    if args.cell_infra and Path(args.cell_infra).exists():
        shutil.copy2(args.cell_infra, ds_dir / "infrastructure.json")
        shutil.copy2(args.config, ds_dir / "space_with_network.json")

    meta = {
        "snapshot_id": snap.get("snapshot_id"),
        "time": snap.get("time"),
        "policy": snap.get("policy"),
        "trigger_task_id": snap.get("trigger_task_id"),
        "source_snapshots": str(args.snapshots),
        "config": str(args.config),
        "seed": args.seed,
        "top_k": args.top_k,
        "horizon": args.horizon,
        "full_candidate_product": full_product,
        "swept_combos": len(plans),
        "rows_written": len(rows),
        "failed_combos": failed,
        "prune_stats": prune_stats,
        "elapsed_s": round(elapsed, 3),
        "per_combo_ms": round(1000.0 * elapsed / max(len(plans), 1), 2),
    }
    (ds_dir / "snapshot_meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshots", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True, help="Cell config (space_with_network schema)")
    ap.add_argument("--sim-input", type=Path, default=Path("data/nofs-ids"))
    ap.add_argument("--cell-infra", type=Path, default=None,
                    help="Cell infrastructure.json — copied in for link-repair context")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--max-snapshots", type=int, default=50)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    ap.add_argument("--calibrate", action="store_true",
                    help="Stop after the first snapshot and print timing")
    args = ap.parse_args()

    snapshots: List[Dict[str, Any]] = []
    with open(args.snapshots) as f:
        for line in f:
            line = line.strip()
            if line:
                snapshots.append(json.loads(line))
    if not snapshots:
        raise SystemExit(f"FAIL LOUD: no snapshots in {args.snapshots}")

    picked = stratified_subsample(snapshots, args.max_snapshots)
    print(f"[sweep] {len(snapshots)} snapshots captured, sweeping {len(picked)} "
          f"(top_k={args.top_k}, horizon={args.horizon}, workers={args.workers})", flush=True)

    args.out_root.mkdir(parents=True, exist_ok=True)
    summary: List[Dict[str, Any]] = []
    for i, snap in enumerate(picked):
        ds_dir = args.out_root / f"ds_snap_{int(snap.get('snapshot_id', i)):05d}"
        done_meta = ds_dir / "snapshot_meta.json"
        if done_meta.exists():
            meta = json.loads(done_meta.read_text())
            if meta.get("rows_written", 0) > 0:
                print(f"[sweep] SKIP (exists): {ds_dir.name} rows={meta['rows_written']}", flush=True)
                summary.append(meta)
                continue
        meta = sweep_snapshot(snap, args, ds_dir)
        summary.append(meta)
        print(
            f"[sweep] {ds_dir.name}: t={meta['time']:.1f}s combos={meta['swept_combos']} "
            f"rows={meta['rows_written']} failed={meta['failed_combos']} "
            f"({meta['per_combo_ms']}ms/combo, {meta['elapsed_s']}s)",
            flush=True,
        )
        if args.calibrate:
            print("[sweep] calibration run complete", flush=True)
            break

    (args.out_root / "sweep_summary.json").write_text(json.dumps(summary, indent=1))
    total_failed = sum(m["failed_combos"] for m in summary)
    total_rows = sum(m["rows_written"] for m in summary)
    print(f"[sweep] DONE: {len(summary)} snapshots, {total_rows} rows, {total_failed} failed combos",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
