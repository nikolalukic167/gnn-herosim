#!/usr/bin/env python3
"""
Backfill initialized_snapshot into existing SSC files.

For each dataset that already has a system_state_captured_unique.json, this script
re-runs ONLY Phase 1 of the co-simulation pipeline (capture_system_state_from_first_task)
using the dataset's own config/workload/infrastructure.  The resulting SSC is written
back to the dataset directory with `initialized_snapshot` at the top level.

Why Phase 1 only:
  - Phase 2 (brute-force placement) and Phase 3 (parallel sim sweep) are expensive and
    produce no additional information for the initialized_snapshot feature.
  - Phase 1 runs a single simulation with the first workload task, captures the system
    state at scheduling time (including platform.initialized.triggered for every platform),
    and writes system_state_captured_unique.json.

Usage:
    pipenv run python3 scripts_cosim/backfill_initialized_snapshot.py [options]

Options:
    --dataset-dir DIR       Root directory of the dataset collection (default: auto-detected)
    --max-datasets N        Stop after processing N datasets (default: all)
    --skip-existing         Skip datasets whose SSC already contains initialized_snapshot
    --start-from IDX        Start from ds_IDX (e.g. --start-from 50 → ds_00050)
    --workers N             Number of *sequential* re-runs (Phase 1 is single-process)
    --quiet / --no-quiet    Suppress per-dataset logging
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.executecosimulation import (
    build_system_state_captured,
    capture_system_state_from_first_task,
    determine_replica_placement,
    flatten_workloads,
    load_simulation_inputs,
    prepare_simulation_config,
    prepare_workloads,
)
from src.placement.model import DataclassJSONEncoder
from src.sample_loader import load_primary_sample_and_mapping


def log(msg: str, quiet: bool = False, force: bool = False):
    if not quiet or force:
        print(msg)


def backfill_dataset(
    dataset_dir: Path,
    sim_input_path: Path,
    sample_json_file: Path,
    samples_file: Path,
    mapping_file: Path,
    quiet: bool = False,
) -> str:
    """
    Re-run Phase 1 for a single dataset and rewrite its SSC file.

    Returns one of: 'updated', 'skipped', 'failed'
    """
    ssc_path = dataset_dir / "system_state_captured_unique.json"
    config_path = dataset_dir / "space_with_network.json"
    workload_path = dataset_dir / "workload.json"
    infra_path = dataset_dir / "infrastructure.json"

    for required in (config_path, workload_path, infra_path):
        if not required.exists():
            log(f"  SKIP: missing {required.name}", quiet)
            return "skipped"

    try:
        with open(config_path) as f:
            infra_config = json.load(f)
        with open(workload_path) as f:
            workload_base = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        log(f"  FAIL: could not load config/workload: {exc}", quiet, force=True)
        return "failed"

    try:
        sample, mapping, _ = load_primary_sample_and_mapping(
            sample_json_path=sample_json_file,
            samples_npy_path=samples_file,
            mapping_pkl_path=mapping_file,
        )
    except Exception as exc:
        log(f"  FAIL: could not load sample/mapping: {exc}", quiet, force=True)
        return "failed"

    try:
        sim_inputs = load_simulation_inputs(sim_input_path)
    except Exception as exc:
        log(f"  FAIL: load_simulation_inputs: {exc}", quiet, force=True)
        return "failed"

    apps = list(infra_config.get("wsc", {}).keys())
    if not apps:
        log("  FAIL: no apps in wsc config", quiet, force=True)
        return "failed"

    try:
        sim_config = prepare_simulation_config(
            sample, mapping, infra_config, infrastructure_file=infra_path
        )
        replica_plan = determine_replica_placement(sim_config, sim_inputs)
        workloads = prepare_workloads(sample, mapping, workload_base, apps)
        flattened = flatten_workloads(workloads)
    except Exception as exc:
        log(f"  FAIL: infrastructure prep: {exc}", quiet, force=True)
        return "failed"

    # Propagate fast-forward flag from config if present (matches original run)
    if "fast_forward_warmup" in infra_config:
        pass  # already in infra_config; capture_system_state_from_first_task reads it

    log(f"  Running Phase 1 …", quiet)
    try:
        active_replicas = capture_system_state_from_first_task(
            sample=sample,
            mapping=mapping,
            infra_config=infra_config,
            sim_inputs=sim_inputs,
            workload_events=flattened["events"],
            replica_plan=replica_plan,
            output_dir=dataset_dir,
            infrastructure_file=infra_path,
        )
    except Exception as exc:
        log(f"  FAIL: Phase 1 error: {exc}", quiet, force=True)
        return "failed"

    if active_replicas is None:
        log("  FAIL: Phase 1 returned None", quiet, force=True)
        return "failed"

    # Build SSC from phase-1 simulation result
    capture_sim_path = dataset_dir / "first_task_state_capture_simulation.json"
    if not capture_sim_path.exists():
        log(f"  FAIL: {capture_sim_path.name} not written by Phase 1", quiet, force=True)
        return "failed"

    try:
        with open(capture_sim_path) as f:
            capture_result = json.load(f)
        stats = capture_result.get("stats", {})
        captured_state = build_system_state_captured(stats)
    except Exception as exc:
        log(f"  FAIL: build_system_state_captured: {exc}", quiet, force=True)
        return "failed"

    if not captured_state.get("initialized_snapshot"):
        log(
            "  WARN: initialized_snapshot empty after Phase 1 "
            "(preinit=0 config may see all-False; this is correct behaviour)",
            quiet,
        )

    try:
        with open(ssc_path, "w") as f:
            json.dump(captured_state, f, indent=2, cls=DataclassJSONEncoder)
        log(f"  ✓ wrote {ssc_path.name} "
            f"({len(captured_state.get('initialized_snapshot', {}))} platforms snapshotted)", quiet)
    except Exception as exc:
        log(f"  FAIL: writing SSC: {exc}", quiet, force=True)
        return "failed"

    return "updated"


def main():
    parser = argparse.ArgumentParser(
        description="Backfill initialized_snapshot into existing SSC files via Phase-1 re-run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset-dir", type=Path, default=None,
        help="Root gnn_datasets_* directory (default: auto-detected from simulation_data/)",
    )
    parser.add_argument("--max-datasets", "-n", type=int, default=None,
                        help="Stop after processing N datasets")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip datasets whose SSC already has initialized_snapshot")
    parser.add_argument("--start-from", type=int, default=0,
                        help="Start from dataset index (e.g. 50 → ds_00050)")
    parser.add_argument("--quiet", "-q", action="store_true", default=False)
    parser.add_argument("--no-quiet", action="store_false", dest="quiet")
    args = parser.parse_args()

    quiet = args.quiet
    base_dir = PROJECT_ROOT / "simulation_data"

    # Auto-detect dataset directory (most recently modified gnn_datasets_* dir)
    if args.dataset_dir:
        output_base = args.dataset_dir
    else:
        candidates = sorted(base_dir.glob("gnn_datasets_*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            print("ERROR: no gnn_datasets_* directory found under simulation_data/")
            sys.exit(1)
        output_base = candidates[0]
        print(f"Auto-detected dataset directory: {output_base.name}")

    sim_input_path = PROJECT_ROOT / "data" / "nofs-ids"
    sample_json_file = base_dir / "sample_simple.json"
    samples_file = base_dir / "lhs_samples_simple.npy"
    mapping_file = base_dir / "lhs_samples_simple_mapping.pkl"

    dataset_dirs = sorted(output_base.glob("ds_*"))
    if args.start_from:
        dataset_dirs = [d for d in dataset_dirs if int(d.name.split("_")[1]) >= args.start_from]
    if args.max_datasets:
        dataset_dirs = dataset_dirs[: args.max_datasets]

    log(f"\n=== Backfill initialized_snapshot ===", quiet, force=True)
    log(f"Dataset directory : {output_base}", quiet, force=True)
    log(f"Datasets to process: {len(dataset_dirs)}", quiet, force=True)
    log(f"Skip-existing mode : {args.skip_existing}", quiet, force=True)

    progress_log = PROJECT_ROOT / "logs" / f"backfill_initialized_{output_base.name}.txt"
    (PROJECT_ROOT / "logs").mkdir(parents=True, exist_ok=True)

    total = updated = skipped = failed = 0
    start_wall = time.time()

    for dataset_dir in dataset_dirs:
        total += 1
        ds_name = dataset_dir.name
        ssc_path = dataset_dir / "system_state_captured_unique.json"

        # Optional: skip if SSC already has initialized_snapshot
        if args.skip_existing and ssc_path.exists():
            try:
                with open(ssc_path) as f:
                    existing = json.load(f)
                if existing.get("initialized_snapshot"):
                    log(f"[{ds_name}] already has initialized_snapshot — skip", quiet)
                    skipped += 1
                    continue
            except Exception:
                pass  # Re-run if we can't read the existing file

        log(f"\n[{ds_name}]", quiet)
        t0 = time.time()
        status = backfill_dataset(
            dataset_dir=dataset_dir,
            sim_input_path=sim_input_path,
            sample_json_file=sample_json_file,
            samples_file=samples_file,
            mapping_file=mapping_file,
            quiet=quiet,
        )
        elapsed = time.time() - t0

        with open(progress_log, "a") as pf:
            pf.write(f"{ds_name} {status.upper()} {datetime.now().isoformat()} {elapsed:.1f}s\n")

        if status == "updated":
            updated += 1
        elif status == "skipped":
            skipped += 1
        else:
            failed += 1

        if total % 25 == 0:
            wall = time.time() - start_wall
            rate = total / wall if wall > 0 else 0
            log(
                f"\n--- Progress {total}/{len(dataset_dirs)} "
                f"({100*total/len(dataset_dirs):.1f}%) "
                f"updated={updated} failed={failed} — {rate:.2f} ds/s ---",
                quiet, force=True,
            )

    wall = time.time() - start_wall
    log(f"\n=== Done ===", quiet, force=True)
    log(f"Total   : {total}", quiet, force=True)
    log(f"Updated : {updated}", quiet, force=True)
    log(f"Skipped : {skipped}", quiet, force=True)
    log(f"Failed  : {failed}", quiet, force=True)
    log(f"Time    : {wall:.1f}s ({wall/60:.1f} min)", quiet, force=True)
    log(f"Log     : {progress_log}", quiet, force=True)


if __name__ == "__main__":
    main()
