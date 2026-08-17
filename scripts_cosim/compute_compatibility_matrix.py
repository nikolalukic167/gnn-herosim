#!/usr/bin/env python3
"""
Compute compatibility matrix for co-simulation dataset collections.

Analyzes which collections can be combined for training based on:
- Warmth physics model (node_disk_v2 vs platform_reuse_v1)
- Queue feature contract (legacy_v0 vs scale_invariant_v1)
- Task structure (4-task vs 1-task)
- Status (active vs deprecated/archived)

Generates:
- Pairwise compatibility matrix
- Training group recommendations

Usage:
    python scripts_cosim/compute_compatibility_matrix.py
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


def load_registry(simulation_data_dir: Path) -> Dict[str, Any]:
    """Load the global registry."""
    registry_path = simulation_data_dir / "REGISTRY.json"
    if not registry_path.exists():
        raise FileNotFoundError(f"{registry_path} does not exist. Run extract_dataset_metadata.py first.")

    with open(registry_path) as f:
        return json.load(f)


def check_compatibility(collection_a: Dict[str, Any], collection_b: Dict[str, Any]) -> Tuple[str, str]:
    """
    Check if two collections are compatible for training.

    Returns:
        (status, reason) where status is "compatible", "combinable_with_care", or "incompatible"
    """
    # Skip if either is deprecated
    if collection_a["status"] == "deprecated" or collection_b["status"] == "deprecated":
        return ("incompatible", "One or both collections are deprecated")

    # Check warmth physics
    warmth_a = collection_a["physics"]["warmth_model"]
    warmth_b = collection_b["physics"]["warmth_model"]
    if warmth_a != warmth_b:
        return ("incompatible", f"Different warmth_physics: {warmth_a} vs {warmth_b}")

    # Check queue feature contract
    contract_a = collection_a["physics"]["queue_feature_contract"]
    contract_b = collection_b["physics"]["queue_feature_contract"]
    if contract_a != contract_b:
        return ("incompatible", f"Different queue_feature_contract: {contract_a} vs {contract_b}")

    # Check task structure (infer from collection name)
    is_1task_a = "1task" in collection_a["collection_name"]
    is_1task_b = "1task" in collection_b["collection_name"]
    if is_1task_a != is_1task_b:
        return ("incompatible", "Different task counts (1-task vs 4-task)")

    # Check if same collection
    if collection_a["collection_name"] == collection_b["collection_name"]:
        return ("compatible", "Same collection")

    # Check queue depths - if very different, requires normalization
    if "queue_depth_mean" in collection_a.get("results", {}) and "queue_depth_mean" in collection_b.get("results", {}):
        mean_a = collection_a["results"]["queue_depth_mean"]
        mean_b = collection_b["results"]["queue_depth_mean"]

        # If queue depths differ by >50%, flag as requiring care
        if mean_a > 0 and mean_b > 0:
            ratio = max(mean_a, mean_b) / min(mean_a, mean_b)
            if ratio > 1.5:
                return ("combinable_with_care", f"Queue depths differ significantly: {mean_a:.1f} vs {mean_b:.1f}")

    # All checks passed
    return ("compatible", "Compatible physics and structure")


def build_compatibility_matrix(registry: Dict[str, Any]) -> Dict[str, Any]:
    """Build pairwise compatibility matrix."""
    collections = registry["collections"]
    collection_names = sorted(collections.keys())

    matrix = {}
    for name_a in collection_names:
        matrix[name_a] = {}
        for name_b in collection_names:
            if name_a == name_b:
                continue

            status, reason = check_compatibility(collections[name_a], collections[name_b])
            matrix[name_a][name_b] = {
                "status": status,
                "reason": reason
            }

    return matrix


def identify_training_groups(registry: Dict[str, Any], compatibility_matrix: Dict[str, Any]) -> Dict[str, Any]:
    """Identify groups of collections that can be trained together."""
    collections = registry["collections"]
    active_collections = [
        name for name, meta in collections.items()
        if meta["status"] == "active"
    ]

    # Group by physics + contract
    groups = {}

    for name in active_collections:
        meta = collections[name]
        warmth = meta["physics"]["warmth_model"]
        contract = meta["physics"]["queue_feature_contract"]
        is_1task = "1task" in name

        # Create group key
        task_type = "1task" if is_1task else "4task"
        group_key = f"{contract}_{warmth}_{task_type}"

        if group_key not in groups:
            groups[group_key] = {
                "collections": [],
                "physics": {
                    "warmth_model": warmth,
                    "queue_feature_contract": contract
                },
                "task_structure": task_type
            }

        groups[group_key]["collections"].append(name)

    # Compute total datasets for each group
    for group_key, group_data in groups.items():
        total_datasets = sum(
            collections[name]["results"]["total_datasets"]
            for name in group_data["collections"]
        )
        completed_datasets = sum(
            collections[name]["results"]["completed_datasets"]
            for name in group_data["collections"]
        )
        group_data["total_datasets"] = total_datasets
        group_data["completed_datasets"] = completed_datasets

        # Add purpose
        if "regime_b" in group_key:
            group_data["purpose"] = "Regime B distillation"
        elif "1task" in group_key:
            group_data["purpose"] = "Single-task baseline (deprecated)"
        else:
            group_data["purpose"] = "General GNN training"

    return groups


def main():
    parser = argparse.ArgumentParser(description="Compute compatibility matrix for dataset collections")
    parser.add_argument("--data-dir", type=Path, default=Path("simulation_data"),
                       help="Path to simulation_data directory")

    args = parser.parse_args()

    simulation_data_dir = args.data_dir
    if not simulation_data_dir.exists():
        print(f"Error: {simulation_data_dir} does not exist", file=sys.stderr)
        return 1

    # Load registry
    print("Loading registry...")
    registry = load_registry(simulation_data_dir)

    # Build compatibility matrix
    print("Building compatibility matrix...")
    matrix = build_compatibility_matrix(registry)

    # Identify training groups
    print("Identifying training groups...")
    training_groups = identify_training_groups(registry, matrix)

    # Build output
    output = {
        "schema_version": "1.0.0",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "compatibility_matrix": matrix,
        "training_groups": training_groups
    }

    # Write to file
    output_path = simulation_data_dir / "COMPATIBILITY_MATRIX.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"✓ Wrote {output_path}")

    # Print summary
    print(f"\n=== Training Groups ===")
    for group_key, group_data in sorted(training_groups.items()):
        print(f"\n{group_key}:")
        print(f"  Purpose: {group_data['purpose']}")
        print(f"  Physics: {group_data['physics']['warmth_model']} + {group_data['physics']['queue_feature_contract']}")
        print(f"  Collections: {len(group_data['collections'])}")
        print(f"  Total datasets: {group_data['total_datasets']}")
        print(f"  Completed: {group_data['completed_datasets']}")
        for name in group_data["collections"]:
            meta = registry["collections"][name]
            print(f"    - {name} ({meta['results']['completed_datasets']} datasets)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
