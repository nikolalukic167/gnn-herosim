#!/usr/bin/env python3
"""
A/B Alignment Test: Phase-1 SSC initialized_snapshot vs Phase-3 scheduling-time state.

HYPOTHESIS UNDER TEST
---------------------
The initialized_snapshot captured in the SSC (from Phase 1 — one task, auto-resolve)
accurately represents the platform initialization state that Phase 3 brute-force
simulations see at scheduling time.

If TRUE  → dim-8 shared_fate_signal is a valid training feature; it reflects the
           actual state the simulator uses when computing RTT labels.

If FALSE → dim-8 is a proxy for preinit-config identity, not per-placement cold
           signal; the training signal is structurally disconnected from the labels.

METHOD
------
For each tested dataset:
  A) Read initialized_snapshot from system_state_captured_unique.json  (Phase 1 result)
  B) Re-run N Phase 3 simulations (different placement plans) using execute_simulation
     directly, then read schedulingStateCapture.initialized_snapshot from each result
  C) Compare A vs B per placement:
     - exact_match_rate  : fraction of platforms with identical bool value
     - warm_agreement    : IoU on warm-platform sets  (A_warm ∩ B_warm) / (A_warm ∪ B_warm)
     - cold_agreement    : IoU on cold-platform sets
     - false_warm_count  : platforms SSC says warm but Phase 3 sees cold (dangerous)
     - false_cold_count  : platforms SSC says cold but Phase 3 sees warm (conservative)

PASS criterion: exact_match_rate ≥ 0.95 across all tested placements.
FAIL criterion: exact_match_rate < 0.90 on any placement, or systematic bias.

Usage:
    pipenv run python3 scripts_cosim/test_ssc_phase_alignment.py [options]

Options:
    --dataset-dir DIR   Path to gnn_datasets_* directory (default: auto-detect)
    --datasets N        Number of datasets to test (default: 3)
    --placements N      Number of Phase 3 placements to test per dataset (default: 5)
    --start-from IDX    First dataset index to test
"""

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.executecosimulation import (
    build_system_state_captured,
    determine_replica_placement,
    execute_simulation,
    flatten_workloads,
    load_simulation_inputs,
    prepare_simulation_config,
    prepare_workloads,
    KEEP_ALIVE,
    QUEUE_LENGTH,
)
from src.sample_loader import load_primary_sample_and_mapping


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_placements(dataset_dir: Path, n: int) -> List[Dict[int, Tuple[int, int]]]:
    """Read up to n placement plans from placements.jsonl."""
    path = dataset_dir / "placements" / "placements.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"placements.jsonl missing: {path}")
    plans = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            raw = entry.get("placement_plan", {})
            # keys may be str; convert to int
            plan = {int(k): tuple(v) for k, v in raw.items()}
            plans.append(plan)
            if len(plans) >= n:
                break
    if not plans:
        raise ValueError(f"No placements found in {path}")
    return plans


def compare_snapshots(
    snap_a: Dict[str, bool],
    snap_b: Dict[str, bool],
) -> Dict[str, Any]:
    """
    Compare two initialized_snapshot dicts.
    snap_a = Phase 1 (SSC), snap_b = Phase 3 (live simulation).
    """
    keys_a = set(snap_a)
    keys_b = set(snap_b)
    common = keys_a & keys_b

    if not common:
        return {
            "error": "no common platform keys",
            "keys_only_in_a": len(keys_a - keys_b),
            "keys_only_in_b": len(keys_b - keys_a),
        }

    matches = sum(1 for k in common if snap_a[k] == snap_b[k])
    exact_match_rate = matches / len(common)

    warm_a = {k for k in common if snap_a[k]}
    warm_b = {k for k in common if snap_b[k]}
    cold_a = {k for k in common if not snap_a[k]}
    cold_b = {k for k in common if not snap_b[k]}

    warm_iou = (
        len(warm_a & warm_b) / len(warm_a | warm_b)
        if (warm_a | warm_b) else 1.0
    )
    cold_iou = (
        len(cold_a & cold_b) / len(cold_a | cold_b)
        if (cold_a | cold_b) else 1.0
    )

    # Dangerous direction: SSC says warm but Phase 3 sees cold → GNN trained
    # with is_warm=1 but sim fires cold start
    false_warm = warm_a - warm_b
    # Conservative direction: SSC says cold but Phase 3 sees warm → GNN undertrades
    false_cold = cold_a - cold_b

    return {
        "common_platforms": len(common),
        "keys_only_in_a": len(keys_a - keys_b),
        "keys_only_in_b": len(keys_b - keys_a),
        "exact_match_rate": exact_match_rate,
        "warm_iou": warm_iou,
        "cold_iou": cold_iou,
        "false_warm_count": len(false_warm),
        "false_cold_count": len(false_cold),
        "false_warm_examples": sorted(false_warm)[:5],
        "false_cold_examples": sorted(false_cold)[:5],
        "warm_in_a": len(warm_a),
        "warm_in_b": len(warm_b),
        "cold_in_a": len(cold_a),
        "cold_in_b": len(cold_b),
    }


def run_phase3_sim(
    placement_plan: Dict[int, Tuple[int, int]],
    infra_config: Dict[str, Any],
    sim_inputs: Dict[str, Any],
    sample,
    mapping,
    infra_file: Path,
    workload: Dict[str, Any],
    replica_plan: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, bool]]:
    """
    Run a single Phase 3 simulation and return initialized_snapshot
    from schedulingStateCapture (captured at batch-scheduling time).
    replica_plan must be provided to match original Phase 3 conditions
    (all pre-placed replicas available, not just one autoscaled replica).
    """
    sim_config = prepare_simulation_config(
        sample, mapping, infra_config,
        placement_plan=placement_plan,
        replica_plan=replica_plan,
        infrastructure_file=infra_file,
    )
    # Propagate fast-forward flag
    if "fast_forward_warmup" in infra_config:
        sim_config["fast_forward_warmup"] = infra_config["fast_forward_warmup"]
        sim_config["fast_forward_threshold"] = infra_config.get("fast_forward_threshold", 100)

    full_config = {"infrastructure": sim_config, "workload": workload}

    try:
        result = execute_simulation(
            full_config, sim_inputs,
            "determined_determined",
            model_locations={}, models={},
            cache_policy="fifo", task_priority="fifo",
            keep_alive=KEEP_ALIVE, queue_length=QUEUE_LENGTH,
        )
    except Exception as exc:
        print(f"    ⚠  simulation error: {exc}")
        return None

    stats = result.get("stats", {})
    sched_cap = stats.get("schedulingStateCapture") or {}
    snap = sched_cap.get("initialized_snapshot")

    if snap is None:
        # Fallback: build_system_state_captured path
        captured = build_system_state_captured(stats)
        snap = captured.get("initialized_snapshot")

    if not snap:
        print("    ⚠  no initialized_snapshot in Phase 3 result")
        return None

    return {k: bool(v) for k, v in snap.items()}


# ---------------------------------------------------------------------------
# Per-dataset test
# ---------------------------------------------------------------------------

def test_dataset(
    dataset_dir: Path,
    sim_input_path: Path,
    sample_json_file: Path,
    samples_file: Path,
    mapping_file: Path,
    n_placements: int,
) -> Dict[str, Any]:
    ds_name = dataset_dir.name
    print(f"\n{'='*60}")
    print(f"Dataset: {ds_name}")
    print(f"{'='*60}")

    # --- Load SSC (Phase 1 result) ---
    ssc_path = dataset_dir / "system_state_captured_unique.json"
    if not ssc_path.exists():
        return {"status": "skip", "reason": "no SSC"}
    with open(ssc_path) as f:
        ssc = json.load(f)
    snap_phase1: Dict[str, bool] = {
        k: bool(v) for k, v in (ssc.get("initialized_snapshot") or {}).items()
    }
    if not snap_phase1:
        return {"status": "skip", "reason": "empty initialized_snapshot in SSC"}

    warm_count = sum(1 for v in snap_phase1.values() if v)
    cold_count = len(snap_phase1) - warm_count
    print(f"Phase 1 SSC: {len(snap_phase1)} platforms — {warm_count} warm / {cold_count} cold")

    # --- Load config, workload, infra ---
    config_path = dataset_dir / "space_with_network.json"
    workload_path = dataset_dir / "workload.json"
    infra_path = dataset_dir / "infrastructure.json"

    for p in (config_path, workload_path, infra_path):
        if not p.exists():
            return {"status": "skip", "reason": f"missing {p.name}"}

    with open(config_path) as f:
        infra_config = json.load(f)
    with open(workload_path) as f:
        workload_base = json.load(f)

    try:
        sample, mapping, _ = load_primary_sample_and_mapping(
            sample_json_path=sample_json_file,
            samples_npy_path=samples_file,
            mapping_pkl_path=mapping_file,
        )
        sim_inputs = load_simulation_inputs(sim_input_path)
    except Exception as exc:
        return {"status": "skip", "reason": str(exc)}

    apps = list(infra_config.get("wsc", {}).keys())
    if not apps:
        return {"status": "skip", "reason": "no apps in wsc"}

    try:
        base_sim_config = prepare_simulation_config(
            sample, mapping, infra_config,
            infrastructure_file=infra_path,
        )
        replica_plan = determine_replica_placement(base_sim_config, sim_inputs)
    except Exception as exc:
        return {"status": "skip", "reason": f"replica_plan setup: {exc}"}

    workloads = prepare_workloads(sample, mapping, workload_base, apps)
    flattened = flatten_workloads(workloads)

    # --- Load Phase 3 placement plans ---
    try:
        plans = load_placements(dataset_dir, n_placements)
    except Exception as exc:
        return {"status": "skip", "reason": str(exc)}
    print(f"Testing {len(plans)} Phase 3 placement(s)")

    # --- Run Phase 3 sims and compare ---
    comparison_results = []
    for i, plan in enumerate(plans):
        print(f"  Placement {i+1}/{len(plans)}: {dict(list(plan.items())[:2])}…  ", end="", flush=True)
        t0 = time.time()
        snap_p3 = run_phase3_sim(
            placement_plan=plan,
            infra_config=infra_config,
            sim_inputs=sim_inputs,
            sample=sample,
            mapping=mapping,
            infra_file=infra_path,
            workload=flattened,
            replica_plan=replica_plan,
        )
        elapsed = time.time() - t0

        if snap_p3 is None:
            print(f"SKIP (no snapshot) [{elapsed:.1f}s]")
            continue

        cmp = compare_snapshots(snap_phase1, snap_p3)
        match_rate = cmp.get("exact_match_rate", 0.0)
        warm_iou = cmp.get("warm_iou", 0.0)
        fw = cmp.get("false_warm_count", 0)
        fc = cmp.get("false_cold_count", 0)

        status_icon = "✓" if match_rate >= 0.95 else ("⚠" if match_rate >= 0.90 else "✗")
        print(
            f"{status_icon} match={match_rate:.3f}  warm_iou={warm_iou:.3f}  "
            f"false_warm={fw}  false_cold={fc}  [{elapsed:.1f}s]"
        )
        cmp["placement_idx"] = i
        comparison_results.append(cmp)

    if not comparison_results:
        return {"status": "skip", "reason": "no valid Phase 3 results"}

    # --- Aggregate ---
    match_rates = [c["exact_match_rate"] for c in comparison_results]
    warm_ious = [c["warm_iou"] for c in comparison_results]
    false_warms = [c["false_warm_count"] for c in comparison_results]
    false_colds = [c["false_cold_count"] for c in comparison_results]

    min_match = min(match_rates)
    avg_match = sum(match_rates) / len(match_rates)
    avg_warm_iou = sum(warm_ious) / len(warm_ious)
    total_fw = sum(false_warms)
    total_fc = sum(false_colds)

    verdict = "PASS" if min_match >= 0.95 else ("MARGINAL" if min_match >= 0.90 else "FAIL")
    print(f"\n  Summary: min_match={min_match:.3f}  avg_match={avg_match:.3f}  "
          f"avg_warm_iou={avg_warm_iou:.3f}  total_false_warm={total_fw}  "
          f"total_false_cold={total_fc}  → {verdict}")

    return {
        "status": verdict,
        "dataset": ds_name,
        "phase1_warm": warm_count,
        "phase1_cold": cold_count,
        "n_placements_tested": len(comparison_results),
        "min_match_rate": min_match,
        "avg_match_rate": avg_match,
        "avg_warm_iou": avg_warm_iou,
        "total_false_warm": total_fw,
        "total_false_cold": total_fc,
        "per_placement": comparison_results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="A/B test: Phase-1 SSC initialized_snapshot vs Phase-3 scheduling state",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dataset-dir", type=Path, default=None,
                        help="Root gnn_datasets_* directory (default: auto-detect)")
    parser.add_argument("--datasets", "-d", type=int, default=3,
                        help="Number of datasets to test (default: 3)")
    parser.add_argument("--placements", "-p", type=int, default=5,
                        help="Phase 3 placements to test per dataset (default: 5)")
    parser.add_argument("--start-from", type=int, default=0,
                        help="First dataset index to test")
    args = parser.parse_args()

    base_dir = PROJECT_ROOT / "simulation_data"
    sim_input_path = PROJECT_ROOT / "data" / "nofs-ids"
    sample_json_file = base_dir / "sample_simple.json"
    samples_file = base_dir / "lhs_samples_simple.npy"
    mapping_file = base_dir / "lhs_samples_simple_mapping.pkl"

    if args.dataset_dir:
        output_base = args.dataset_dir
    else:
        candidates = sorted(
            base_dir.glob("gnn_datasets_*"),
            key=lambda p: p.stat().st_mtime, reverse=True
        )
        # Prefer the large 1060 dir if present
        for c in candidates:
            if "1060" in c.name:
                output_base = c
                break
        else:
            if not candidates:
                print("ERROR: no gnn_datasets_* dir found")
                sys.exit(1)
            output_base = candidates[0]

    print(f"Dataset dir : {output_base}")
    print(f"Datasets    : {args.datasets}")
    print(f"Placements  : {args.placements} per dataset")
    print(f"Start from  : ds_{args.start_from:05d}")

    dataset_dirs = sorted(output_base.glob("ds_*"))
    if args.start_from:
        dataset_dirs = [d for d in dataset_dirs if int(d.name.split("_")[1]) >= args.start_from]
    dataset_dirs = dataset_dirs[: args.datasets]

    all_results = []
    for dataset_dir in dataset_dirs:
        r = test_dataset(
            dataset_dir=dataset_dir,
            sim_input_path=sim_input_path,
            sample_json_file=sample_json_file,
            samples_file=samples_file,
            mapping_file=mapping_file,
            n_placements=args.placements,
        )
        all_results.append(r)

    # --- Final verdict ---
    print(f"\n{'='*60}")
    print("OVERALL RESULT")
    print(f"{'='*60}")

    tested = [r for r in all_results if r["status"] not in ("skip",)]
    if not tested:
        print("No datasets successfully tested.")
        sys.exit(1)

    pass_count = sum(1 for r in tested if r["status"] == "PASS")
    marginal_count = sum(1 for r in tested if r["status"] == "MARGINAL")
    fail_count = sum(1 for r in tested if r["status"] == "FAIL")

    all_min_matches = [r["min_match_rate"] for r in tested]
    all_avg_matches = [r["avg_match_rate"] for r in tested]
    all_false_warms = [r["total_false_warm"] for r in tested]
    global_min = min(all_min_matches)
    global_avg = sum(all_avg_matches) / len(all_avg_matches)
    total_false_warm = sum(all_false_warms)

    print(f"Datasets tested : {len(tested)}  (PASS={pass_count}  MARGINAL={marginal_count}  FAIL={fail_count})")
    print(f"Global min match: {global_min:.4f}")
    print(f"Global avg match: {global_avg:.4f}")
    print(f"Total false_warm: {total_false_warm}  (SSC=warm but Phase3=cold — dangerous for training)")

    print()
    if global_min >= 0.95 and total_false_warm == 0:
        print("✓ HYPOTHESIS CONFIRMED: Phase-1 SSC initialized_snapshot accurately reflects")
        print("  Phase-3 scheduling-time state. dim-8 is a valid per-platform training signal.")
    elif global_min >= 0.90:
        print("⚠ MARGINAL: High but imperfect alignment. Investigate false_warm cases.")
        print("  dim-8 signal is mostly valid but may introduce occasional label noise.")
    else:
        print("✗ HYPOTHESIS REJECTED: Significant drift between Phase-1 SSC and Phase-3 state.")
        print("  dim-8 may be a preinit-config proxy, not a per-platform cold signal.")
        print("  Do NOT retrain until the source of drift is understood and corrected.")


if __name__ == "__main__":
    main()
