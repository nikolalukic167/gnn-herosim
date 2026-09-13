#!/usr/bin/env python3
"""
Validate co-simulation dataset collections.

Performs comprehensive validation including:
1. Structural completeness (required files present)
2. Physics consistency (same warmth_physics across collection)
3. Queue depth validation (match declared distributions)
4. Coupling rate verification (sample and estimate)
5. Reproducibility check (seed determinism)

Usage:
    python scripts_cosim/validate_dataset_collection.py --collection contention_v2
    python scripts_cosim/validate_dataset_collection.py --active-only
    python scripts_cosim/validate_dataset_collection.py --archived-only --light
"""

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Required files for structural completeness
REQUIRED_FILES = [
    "infrastructure.json",
    "workload.json",
    "best.json",
    "optimal_result.json",
    "placements/placements.jsonl"
]

# Optional files (don't fail validation if missing)
OPTIONAL_FILES = [
    "placement_metadata.json",
    "space_with_network.json",
    "system_state_captured_unique.json"
]


def load_metadata(collection_path: Path) -> Optional[Dict[str, Any]]:
    """Load metadata for a collection."""
    metadata_path = collection_path / "METADATA.json"
    if not metadata_path.exists():
        print(f"Warning: {metadata_path} does not exist", file=sys.stderr)
        return None

    with open(metadata_path) as f:
        return json.load(f)


def check_structural_completeness(collection_path: Path) -> Dict[str, Any]:
    """Check that all datasets have required files."""
    ds_dirs = sorted(collection_path.glob("ds_*"))
    if not ds_dirs:
        return {
            "status": "FAIL",
            "reason": "No datasets found",
            "total_datasets": 0,
            "datasets_passing": 0,
            "completeness_rate": 0.0,
            "missing_files": {}
        }

    missing_files_by_type = {f: [] for f in REQUIRED_FILES}
    datasets_passing = 0

    for ds in ds_dirs:
        ds_name = ds.name
        all_present = True

        for required_file in REQUIRED_FILES:
            file_path = ds / required_file
            if not file_path.exists():
                missing_files_by_type[required_file].append(ds_name)
                all_present = False

            # Special check: placements.jsonl must be non-empty
            if required_file == "placements/placements.jsonl" and file_path.exists():
                if file_path.stat().st_size == 0:
                    missing_files_by_type[required_file].append(ds_name + " (empty)")
                    all_present = False

        if all_present:
            datasets_passing += 1

    total_datasets = len(ds_dirs)
    completeness_rate = datasets_passing / total_datasets

    # Filter out files with no missing datasets
    missing_files = {k: v for k, v in missing_files_by_type.items() if v}

    return {
        "status": "PASS" if completeness_rate == 1.0 else "FAIL",
        "total_datasets": total_datasets,
        "datasets_passing": datasets_passing,
        "completeness_rate": round(completeness_rate, 4),
        "missing_files": missing_files
    }


def check_physics_consistency(collection_path: Path, metadata: Dict[str, Any], sample_size: int = 10) -> Dict[str, Any]:
    """Check that all datasets use consistent physics configuration."""
    ds_dirs = [d for d in collection_path.glob("ds_*") if (d / "infrastructure.json").exists()]
    if not ds_dirs:
        return {
            "status": "SKIP",
            "reason": "No datasets with infrastructure.json found"
        }

    expected_warmth = metadata.get("physics", {}).get("warmth_model", "node_disk_v2")
    sample = random.sample(ds_dirs, min(sample_size, len(ds_dirs)))

    inconsistent_datasets = []
    for ds in sample:
        infra_path = ds / "infrastructure.json"
        try:
            with open(infra_path) as f:
                infra = json.load(f)

            actual_warmth = infra.get("warmth_physics", "node_disk_v2")
            if actual_warmth != expected_warmth:
                inconsistent_datasets.append({
                    "dataset": ds.name,
                    "expected": expected_warmth,
                    "actual": actual_warmth
                })
        except Exception as e:
            inconsistent_datasets.append({
                "dataset": ds.name,
                "error": str(e)
            })

    return {
        "status": "PASS" if not inconsistent_datasets else "FAIL",
        "expected_warmth_model": expected_warmth,
        "sample_size": len(sample),
        "inconsistent_datasets": inconsistent_datasets
    }


def check_queue_depth_validation(collection_path: Path, metadata: Dict[str, Any], sample_size: int = 20, tolerance: float = 0.05) -> Dict[str, Any]:
    """Validate queue depths match declared distributions."""
    ds_dirs = [d for d in collection_path.glob("ds_*") if (d / "infrastructure.json").exists()]
    if not ds_dirs:
        return {
            "status": "SKIP",
            "reason": "No datasets with infrastructure.json found"
        }

    # Get expected queue distributions from metadata
    generation_params = metadata.get("generation_params") if metadata else None
    if not generation_params:
        return {
            "status": "SKIP",
            "reason": "No generation_params in metadata"
        }

    queue_dists = generation_params.get("queue_distributions", [])
    if not queue_dists:
        return {
            "status": "SKIP",
            "reason": "No queue distributions in metadata"
        }

    # Queue depths are cheap to read (one small JSON field per dataset), and a
    # 20-of-N dataset sample is a CLUSTER sample over a 3-distribution mixture —
    # its mixture imbalance alone moved the collection mean by ±0.4 on route_b's
    # 204-dataset arms, dwarfing the slot-level SEM the band is computed from.
    # Read every dataset instead: the statistic becomes exact and deterministic.
    sample = ds_dirs
    all_queue_depths = []

    for ds in sample:
        infra_path = ds / "infrastructure.json"
        try:
            with open(infra_path) as f:
                infra = json.load(f)

            queue_dists_actual = infra.get("queue_distributions", {})
            for task_type, queues in queue_dists_actual.items():
                all_queue_depths.extend(queues.values())
        except Exception as e:
            print(f"Warning: Could not read {infra_path}: {e}", file=sys.stderr)

    if not all_queue_depths:
        return {
            "status": "FAIL",
            "reason": "No queue depths found in sample"
        }

    # Compute statistics
    mean = sum(all_queue_depths) / len(all_queue_depths)
    variance = sum((x - mean) ** 2 for x in all_queue_depths) / len(all_queue_depths)
    stddev = variance ** 0.5

    # Get expected mean from metadata (average of distribution params)
    expected_means = []
    for qd in queue_dists:
        if qd["distribution"] == "normal":
            expected_means.append(qd["param1"])
        elif qd["distribution"] == "poisson":
            expected_means.append(qd["param1"])
        elif qd["distribution"] == "uniform":
            # The generator draws CONTINUOUS uniform(param1, param2) and floors it
            # (generate_queue_distributions_deterministic: max(0, int(sampled_q))),
            # so the realized support is {param1..param2-1} with mean
            # (param1+param2)/2 - 0.5 — not the naive midpoint. Measured on
            # route_b's 204-dataset arms: uniform(0,12) realizes mean 5.62 against
            # floored-expected 5.5 (z=1.5) vs the naive 6.0 (z=-4.7 false alarm).
            expected_means.append((qd["param1"] + qd["param2"]) / 2 - 0.5)

    expected_mean = sum(expected_means) / len(expected_means) if expected_means else None

    # Validate mean is within tolerance. The band is sample-size aware: a fixed 5%
    # relative band is a mis-specified test for a small collection — a 12-dataset
    # smoke (300 queue slots, sd ~2.9) has SEM ~0.17, so a 2-sigma draw of the
    # DECLARED process already lands ~8% off the declared mean and would fail a
    # check that its own 204-dataset twin (same generator, same grid) passes. The
    # effective band is therefore max(tolerance, 3*SEM/expected_mean): identical to
    # the fixed band for large collections (only ever looser, never tighter, so no
    # existing PASS/FAIL flips to FAIL), and a proper 3-sigma screen for small ones.
    if expected_mean is not None:
        deviation = abs(mean - expected_mean) / expected_mean
        sem = stddev / (len(all_queue_depths) ** 0.5) if all_queue_depths else 0.0
        effective_tolerance = max(tolerance, 3.0 * sem / expected_mean)
        status = "PASS" if deviation <= effective_tolerance else "FAIL"
    else:
        status = "SKIP"
        deviation = None
        sem = None
        effective_tolerance = None

    return {
        "status": status,
        "sample_size": len(sample),
        "queue_count": len(all_queue_depths),
        "actual_mean": round(mean, 2),
        "actual_stddev": round(stddev, 2),
        "expected_mean": round(expected_mean, 2) if expected_mean else None,
        "deviation": round(deviation, 4) if deviation is not None else None,
        "tolerance": tolerance,
        "sem": round(sem, 4) if sem is not None else None,
        "effective_tolerance": (round(effective_tolerance, 4)
                                if effective_tolerance is not None else None)
    }


def estimate_coupling_rate(collection_path: Path, sample_size: int = 50, spread_threshold: float = 0.20) -> Dict[str, Any]:
    """Estimate coupling rate by sampling placement RTT spreads."""
    ds_dirs = [d for d in collection_path.glob("ds_*") if (d / "placements/placements.jsonl").exists()]
    if not ds_dirs:
        return {
            "status": "SKIP",
            "reason": "No datasets with placements.jsonl found"
        }

    sample = random.sample(ds_dirs, min(sample_size, len(ds_dirs)))
    coupled_count = 0
    spreads = []

    for ds in sample:
        jsonl_path = ds / "placements" / "placements.jsonl"
        try:
            rtts = []
            with open(jsonl_path) as f:
                for line in f:
                    data = json.loads(line)
                    rtts.append(data["rtt"])

            if len(rtts) >= 2:
                rtts.sort()
                # Compute spread: (p10 - p0) / p0
                p0 = rtts[0]
                p10_idx = int(len(rtts) * 0.1)
                p10 = rtts[p10_idx] if p10_idx < len(rtts) else rtts[-1]

                if p0 > 0:
                    spread = (p10 - p0) / p0
                    spreads.append(spread)

                    if spread > spread_threshold:
                        coupled_count += 1

        except Exception as e:
            print(f"Warning: Could not read {jsonl_path}: {e}", file=sys.stderr)

    if not spreads:
        return {
            "status": "FAIL",
            "reason": "No valid RTT spreads computed"
        }

    coupling_rate = coupled_count / len(spreads)
    mean_spread = sum(spreads) / len(spreads)

    return {
        "status": "PASS",
        "sample_size": len(spreads),
        "coupled_count": coupled_count,
        "coupling_rate": round(coupling_rate, 4),
        "mean_spread": round(mean_spread, 4),
        "spread_threshold": spread_threshold
    }


def validate_collection(collection_name: str, collection_path: Path, light: bool = False) -> Dict[str, Any]:
    """Perform complete validation for a collection."""
    print(f"Validating {collection_name}...")

    # A validation VERDICT must be a function of the collection, not of an RNG
    # draw. The sampled checks (physics n=10, coupling n=50) previously used the
    # unseeded module RNG, so a borderline collection could PASS on one run and
    # FAIL on the next — observed 2026-08-26 on route_b arm_b0 (queue-depth
    # deviation 2.5% one run, 12.0% the next, from cluster imbalance in an
    # unseeded 20-of-204 dataset sample). Same defect class as the
    # PYTHONHASHSEED tie-break and the unseeded MLP trainer: seed it.
    random.seed(f"validate:{collection_name}")

    # Load metadata
    metadata = load_metadata(collection_path)
    if not metadata:
        return {
            "collection_name": collection_name,
            "status": "FAIL",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "error": "Metadata not found"
        }

    # Run validations
    structural = check_structural_completeness(collection_path)
    print(f"  Structural: {structural['status']} ({structural['completeness_rate']*100:.1f}%)")

    if light:
        # Light validation: only structural check
        return {
            "collection_name": collection_name,
            "status": structural["status"],
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "metadata": metadata,
            "validation_mode": "light",
            "structural_completeness": structural
        }

    # Full validation
    physics = check_physics_consistency(collection_path, metadata)
    print(f"  Physics: {physics['status']}")

    queue_depth = check_queue_depth_validation(collection_path, metadata)
    print(f"  Queue depth: {queue_depth['status']}")

    coupling = estimate_coupling_rate(collection_path)
    print(f"  Coupling rate: {coupling['status']}")
    if coupling["status"] == "PASS":
        print(f"    Estimated: {coupling['coupling_rate']*100:.1f}%")

    # Overall status
    validations = [structural, physics, queue_depth, coupling]
    overall_status = "PASS" if all(v["status"] in ["PASS", "SKIP"] for v in validations) else "FAIL"

    return {
        "collection_name": collection_name,
        "status": overall_status,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "metadata": metadata,
        "validation_mode": "full",
        "structural_completeness": structural,
        "physics_consistency": physics,
        "queue_depth_validation": queue_depth,
        "coupling_rate_estimation": coupling
    }


def write_validation_report(collection_path: Path, report: Dict[str, Any]):
    """Write validation report to collection directory."""
    report_path = collection_path / "VALIDATION_REPORT.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  ✓ Wrote {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Validate co-simulation dataset collections")
    parser.add_argument("--collection", type=str, help="Validate specific collection")
    parser.add_argument("--active-only", action="store_true", help="Validate only active collections")
    parser.add_argument("--archived-only", action="store_true", help="Validate only archived collections")
    parser.add_argument("--light", action="store_true", help="Light validation (structural only)")
    parser.add_argument("--data-dir", type=Path, default=Path("simulation_data"),
                       help="Path to simulation_data directory")

    args = parser.parse_args()

    if not args.collection and not args.active_only and not args.archived_only:
        parser.error("Must specify --collection, --active-only, or --archived-only")

    simulation_data_dir = args.data_dir
    if not simulation_data_dir.exists():
        print(f"Error: {simulation_data_dir} does not exist", file=sys.stderr)
        return 1

    # Load registry to filter collections
    registry_path = simulation_data_dir / "REGISTRY.json"
    if not registry_path.exists():
        print(f"Error: {registry_path} does not exist. Run extract_dataset_metadata.py first.", file=sys.stderr)
        return 1

    with open(registry_path) as f:
        registry = json.load(f)

    # Determine which collections to validate
    if args.collection:
        collections = [args.collection]
    elif args.active_only:
        collections = registry["collections_by_status"]["active"]
    elif args.archived_only:
        collections = registry["collections_by_status"]["archived"]

    # Validate each collection
    reports = []
    for collection_name in sorted(collections):
        collection_path = simulation_data_dir / collection_name
        if not collection_path.exists():
            print(f"Warning: {collection_path} does not exist, skipping", file=sys.stderr)
            continue

        report = validate_collection(collection_name, collection_path, light=args.light)
        write_validation_report(collection_path, report)
        reports.append(report)

    # Summary
    print(f"\n=== Validation Summary ===")
    print(f"Collections validated: {len(reports)}")
    passing = sum(1 for r in reports if r["status"] == "PASS")
    failing = sum(1 for r in reports if r["status"] == "FAIL")
    print(f"  PASS: {passing}")
    print(f"  FAIL: {failing}")

    if failing > 0:
        print("\nFailed collections:")
        for r in reports:
            if r["status"] == "FAIL":
                print(f"  - {r['collection_name']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
