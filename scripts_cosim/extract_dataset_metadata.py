#!/usr/bin/env python3
"""
Extract metadata from co-simulation dataset collections.

Scans simulation_data/ directory, extracts generation parameters from grid presets,
computes statistics from datasets, and generates:
- Per-collection METADATA.json files
- Global REGISTRY.json index

Usage:
    python scripts_cosim/extract_dataset_metadata.py --all
    python scripts_cosim/extract_dataset_metadata.py --collection contention_v2
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import random

# Import grid presets
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts_cosim.generate_gnn_datasets_fast import GRID_PRESETS

# Collection metadata (manually curated)
COLLECTION_INFO = {
    # Shallow-queue series — inverts the contention series' core lever
    "gnn_datasets_4tasks_shallow_v1": {
        "version": "v1",
        "status": "active",
        "problem_category": "separability",
        "purpose": "Shallow queues (depth ~0-8) so the collision term dominates instead of being diluted",
        "hypothesis": (
            "Queue depth predicts separability: measured monotonic across all 899 "
            "contention_v2 sweeps (shallowest quartile additive R2 0.978 / +1.986pp "
            "collision gain, deepest 0.998 / +0.181pp). The additive term is "
            "depth x exec_time and grows with depth while the interaction term "
            "added_in_batch x exec_time does not, so deep queues dilute the only coupling "
            "the corpus has. SUCCESS: coupled(>1%) 31.0% vs contention_v2's 7.1%, "
            "coupled(>5%) 19.6% vs 2.4%, pointwise worst-case regret 48.4% vs 3.6%. "
            "Note additive R2 mean 0.961 but median 0.9999 -- the target is bimodal, so "
            "use the coupled fraction as the gate, not mean R2."
        ),
        "created_date": "2026-08-17"
    },

    # Contention series
    "gnn_datasets_4tasks_contention_v1": {
        "version": "v1",
        "status": "active",
        "problem_category": "contention",
        "purpose": "Test collision frequency with all-cold replicas",
        "hypothesis": "High collision frequency breaks greedy placement (FAILED: greedy still optimal)",
        "created_date": "2026-01-01"
    },
    "gnn_datasets_4tasks_contention_v1_probe": {
        "version": "v1",
        "status": "active",
        "problem_category": "contention",
        "purpose": "Initial probe for contention_v1",
        "created_date": "2026-01-01"
    },
    "gnn_datasets_4tasks_contention_v2": {
        "version": "v2",
        "status": "active",
        "problem_category": "contention",
        "purpose": "Scarce warm resources force task competition",
        "hypothesis": "Warm replicas + heavy queues create anti-correlated preferences (SUCCESS: 7.1% coupling)",
        "created_date": "2026-01-10"
    },
    "gnn_datasets_4tasks_contention_v2_verify": {
        "version": "v2",
        "status": "active",
        "problem_category": "contention",
        "purpose": "Single dataset verification for contention_v2",
        "created_date": "2026-01-10"
    },
    "gnn_datasets_4tasks_contention_v3": {
        "version": "v3",
        "status": "active",
        "problem_category": "contention",
        "purpose": "Push coupling higher with sparser topology + heavier queues",
        "hypothesis": "Target >25% coupling rate (vs v2's 7.1%)",
        "created_date": "2026-01-15"
    },
    "gnn_datasets_4tasks_contention_v4_pilot": {
        "version": "v4",
        "status": "active",
        "problem_category": "contention",
        "purpose": "Pilot for deep queues matching live failure regime",
        "hypothesis": "Queues deep enough to make co-location catastrophic (target 10-15% coupling)",
        "created_date": "2026-08-10"
    },
    "gnn_datasets_4tasks_contention_v5_quick_test": {
        "version": "v5",
        "status": "active",
        "problem_category": "contention",
        "purpose": "Quick test of deepest queues for GNN advantage",
        "hypothesis": "norm450/pois500 (12.9x deeper than v2) push coupling to 15-25%",
        "created_date": "2026-08-14"
    },

    # Warmth series
    "gnn_datasets_4tasks_1060_warmth_v2": {
        "version": "v2",
        "status": "active",
        "problem_category": "warmth",
        "purpose": "1060-scale warmth study with dense topology",
        "hypothesis": "Initial replica warmth impacts placement quality",
        "created_date": "2026-01-05"
    },
    "gnn_datasets_4tasks_sparse_warmth_v2": {
        "version": "v2",
        "status": "active",
        "problem_category": "warmth",
        "purpose": "Warmth study with sparse topology",
        "hypothesis": "Sparse networks emphasize warmth advantages",
        "created_date": "2026-01-05"
    },
    "gnn_datasets_4tasks_skew_warmth_v2": {
        "version": "v2",
        "status": "active",
        "problem_category": "warmth",
        "purpose": "Warmth with hub topology and asymmetric latency",
        "hypothesis": "Degree-skewed cores create distinct placement tiers",
        "created_date": "2026-01-05"
    },
    "gnn_datasets_4tasks_skew_warmth_v2_test": {
        "version": "v2",
        "status": "active",
        "problem_category": "warmth",
        "purpose": "Single test case for skew_warmth_v2",
        "created_date": "2026-01-05"
    },

    # Regime B
    "gnn_datasets_4tasks_regime_b_cold_burst_v1": {
        "version": "v1",
        "status": "active",
        "problem_category": "regime_b",
        "purpose": "Frozen Regime B problem baseline",
        "hypothesis": "12-task cold burst on 12-server cluster under platform_reuse_v1",
        "created_date": "2026-08-01"
    },
    "gnn_datasets_4tasks_regime_b_cold_burst_v1_oracle_split_cosim": {
        "version": "v1",
        "status": "active",
        "problem_category": "regime_b",
        "purpose": "Oracle-split variant for Regime B trajectory harvest",
        "created_date": "2026-08-05"
    },

    # Hetero baselines
    "hetero_small_knative_eval_20260606": {
        "version": "v1",
        "status": "active",
        "problem_category": "baseline",
        "purpose": "Small-scale Knative industry baseline evaluation",
        "created_date": "2026-06-06"
    },
    "hetero_large_knative_eval_20260606": {
        "version": "v1",
        "status": "active",
        "problem_category": "baseline",
        "purpose": "Large-scale Knative industry baseline evaluation",
        "created_date": "2026-06-06"
    },
    "hetero_small_knative_eval_5tasks_20260606": {
        "version": "v1",
        "status": "active",
        "problem_category": "baseline",
        "purpose": "5-task variant of Knative baseline",
        "created_date": "2026-06-06"
    },

    # Production
    "gnn_datasets_4tasks_highq_safe_20260606": {
        "version": "v1",
        "status": "active",
        "problem_category": "production",
        "purpose": "Production-quality stable release with high queue depths",
        "created_date": "2026-06-06"
    },

    # Deprecated
    "gnn_datasets_1task": {
        "version": "v1",
        "status": "deprecated",
        "problem_category": "baseline",
        "purpose": "Single-task baseline (insufficient graph structure for GNN)",
        "archive_reason": "Single-task scenarios don't provide network heterogeneity signal needed for GNN advantage",
        "created_date": "2025-12-01"
    },

    # route_b_v1 series — diamond4 DAG (4 tasks, parent-coupled dispatch) on the
    # ROUTE_B_PILOT_V1_GRID scarce-placement substrate. Registered in
    # docs/lineages/route_b_v1/stage2-preregistration.md and the route_b_v1 LINEAGES entries.
    # Generation env (needed verbatim by any replay, e.g. the ssc repair):
    # HEROSIM_COSIM_KEEP_ALIVE=1000000 HEROSIM_RETAIN_TASK_TIMES=1, Arm S corpora
    # adding HEROSIM_DATA_LOCALITY=1 HEROSIM_OUTPUT_SIZE_BYTES=800000000.
    "gnn_datasets_dag4_route_b_smoke_s": {
        "version": "v1",
        "status": "active",
        "problem_category": "route_b",
        "purpose": (
            "Route B smoke corpus, Arm S (coupling + competition: data locality on, "
            "800MB parent->child output, DAG dispatch). Calibrated the frozen alpha "
            "ladder {inf, 3.0, 2.0} (stage-1 amendment A2); stage-2 B5 overfit "
            "pre-probe + sigma calibration corpus. Replay env: "
            "HEROSIM_COSIM_KEEP_ALIVE=1000000 HEROSIM_DATA_LOCALITY=1"
        ),
        "created_date": "2026-08-25"
    },
    "gnn_datasets_dag4_route_b_smoke_b0": {
        "version": "v1",
        "status": "active",
        "problem_category": "route_b",
        "purpose": (
            "Route B smoke corpus, Arm B0 (competition without coupling: same grid "
            "and scarcity, no locality/payload physics). Specificity-control twin "
            "of the smoke_s corpus; alpha-ladder calibration. Replay env: "
            "HEROSIM_COSIM_KEEP_ALIVE=1000000 (no HEROSIM_DATA_LOCALITY)"
        ),
        "created_date": "2026-08-25"
    },
    "gnn_datasets_dag4_route_b_pilot_v1_arm_s": {
        "version": "v1",
        "status": "active",
        "problem_category": "route_b",
        "purpose": (
            "Route B stage-1 gated corpus, Arm S (coupling + competition), 204 "
            "datasets, seeds 901-917. Stage-1 PASS: best pointwise surrogate wrong "
            ">5% on 17.2% of datasets at alpha=2.0. Stage 2 demotes this to the "
            "SECONDARY (statistics-known) replication set — never the primary gate "
            "(docs/lineages/route_b_v1/stage2-preregistration.md par.5). Replay env: "
            "HEROSIM_COSIM_KEEP_ALIVE=1000000 HEROSIM_DATA_LOCALITY=1"
        ),
        "hypothesis": (
            "Node-memory contention stacked on route A's pairwise transfer term "
            "breaks additive-argmin optimality where competition binds (alpha=2.0); "
            "confirmed at n=204 (Wilson CI [0.126, 0.229]), verifier-agreed to 1e-9"
        ),
        "created_date": "2026-08-25"
    },
    "gnn_datasets_dag4_route_b_pilot_v1_arm_b0": {
        "version": "v1",
        "status": "active",
        "problem_category": "route_b",
        "purpose": (
            "Route B stage-1 corpus, Arm B0 (competition without coupling), 204 "
            "datasets, seeds 901-917. Specificity control: firing fraction at the "
            "5% bar is 0 with residue <=2.49%, so Arm S firing is attributable to "
            "the coupling, not the decoder or the scarcity alone. Replay env: "
            "HEROSIM_COSIM_KEEP_ALIVE=1000000 (no HEROSIM_DATA_LOCALITY)"
        ),
        "created_date": "2026-08-25"
    },
}

# Map collection names to grid presets
COLLECTION_TO_GRID = {
    "gnn_datasets_4tasks_contention_v1": "contention_v1",
    "gnn_datasets_4tasks_contention_v1_probe": "contention_v1",
    "gnn_datasets_4tasks_contention_v2": "contention_v2",
    "gnn_datasets_4tasks_contention_v2_verify": "contention_v2",
    "gnn_datasets_4tasks_contention_v3": "contention_v3",
    "gnn_datasets_4tasks_contention_v4_pilot": "contention_v4_deepq",
    "gnn_datasets_4tasks_contention_v5_quick_test": "contention_v5_quick_test",
    "gnn_datasets_4tasks_1060_warmth_v2": "warmth_v2",
    "gnn_datasets_4tasks_sparse_warmth_v2": "sparse_warmth_v2",
    "gnn_datasets_4tasks_skew_warmth_v2": "skew_warmth_v2",
    "gnn_datasets_4tasks_skew_warmth_v2_test": "skew_warmth_v2",
    "gnn_datasets_4tasks_regime_b_cold_burst_v1": "regime_b_cold_burst_v1",
    "gnn_datasets_4tasks_regime_b_cold_burst_v1_oracle_split_cosim": "regime_b_cold_burst_v1",
    "gnn_datasets_dag4_route_b_smoke_s": "route_b_pilot_v1",
    "gnn_datasets_dag4_route_b_smoke_b0": "route_b_pilot_v1",
    "gnn_datasets_dag4_route_b_pilot_v1_arm_s": "route_b_pilot_v1",
    "gnn_datasets_dag4_route_b_pilot_v1_arm_b0": "route_b_pilot_v1",
}


def count_datasets(collection_path: Path) -> Tuple[int, int]:
    """Count total and completed datasets in a collection."""
    ds_dirs = list(collection_path.glob("ds_*"))
    total = len(ds_dirs)
    completed = sum(1 for ds in ds_dirs if (ds / "best.json").exists())
    return total, completed


LEGACY_CONTRACT_DEFAULT = "legacy_v0"


def detect_queue_feature_contract(collection_path: Path) -> str:
    """Read the queue feature contract from this collection's graph cache.

    The contract governs platform dims 7/13 and lives in the cache, not the datasets.
    ``compute_compatibility_matrix`` refuses to mix collections whose contracts differ, so
    a wrong value here silently permits an invalid training merge. Falls back to
    ``legacy_v0`` only when no cache exists — every pre-2026-08 cache was built that way.
    """
    collection_name = collection_path.name
    suffix = collection_name.removeprefix("gnn_datasets_4tasks_")
    data_dir = collection_path.parent

    # Exact-name caches first: an unambiguous 1:1 mapping to this collection.
    for cache_dir in (
        data_dir / f"graphs_cache_{suffix}",
        data_dir / f"graphs_cache_{collection_name}",
    ):
        contract = _read_cache_contract(cache_dir / "metadata.json")
        if contract:
            return contract

    # Otherwise a collection may have several caches. contention_v2 has both a legacy_v0
    # cache (_873_v5.5) and a scale_invariant_v1 one (_873_v5.7_siv1_dim14), so the
    # contract is NOT single-valued per collection. Only report one when every candidate
    # agrees; a conflict means the schema cannot express the truth and picking either
    # would license an invalid merge.
    found = {
        contract
        for cache_dir in sorted(data_dir.glob(f"graphs_cache_{suffix}*"))
        if (contract := _read_cache_contract(cache_dir / "metadata.json"))
    }
    if len(found) == 1:
        return found.pop()
    if len(found) > 1:
        print(
            f"Warning: {collection_name} has caches with conflicting queue feature "
            f"contracts {sorted(found)}; recording {LEGACY_CONTRACT_DEFAULT}. Verify the "
            f"contract against the specific cache before training on this collection.",
            file=sys.stderr,
        )
    return LEGACY_CONTRACT_DEFAULT


def _read_cache_contract(meta_path: Path) -> Optional[str]:
    if not meta_path.exists():
        return None
    try:
        contract = json.loads(meta_path.read_text()).get("queue_feature_contract")
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Warning: could not read {meta_path}: {exc}", file=sys.stderr)
        return None
    # A cache that declares no contract predates the field, which per CLAUDE.md means
    # legacy_v0. Skipping it instead would hide a real conflict: contention_v2's deployed
    # _873_v5.5 cache has no field, so ignoring it made the collection look like pure
    # scale_invariant_v1.
    return str(contract) if contract else LEGACY_CONTRACT_DEFAULT


def sample_infrastructure_physics(collection_path: Path, sample_size: int = 3) -> Optional[Dict[str, str]]:
    """Sample infrastructure.json files to extract physics configuration."""
    ds_dirs = [d for d in collection_path.glob("ds_*") if (d / "infrastructure.json").exists()]
    if not ds_dirs:
        return None

    sample = random.sample(ds_dirs, min(sample_size, len(ds_dirs)))
    physics_configs = []

    for ds in sample:
        infra_path = ds / "infrastructure.json"
        try:
            with open(infra_path) as f:
                infra = json.load(f)

            # Extract warmth physics (from config or infer from structure)
            warmth_model = infra.get("warmth_physics", "node_disk_v2")  # Default

            # The queue feature contract is a property of the CACHE, not the raw
            # dataset -- the datasets only carry queue depths. Read it from the
            # associated graph cache's metadata.json, which records it authoritatively.
            # Guessing here is a correctness hazard because compute_compatibility_matrix
            # decides what may be trained together from this field.
            queue_contract = detect_queue_feature_contract(collection_path)

            physics_configs.append({
                "warmth_model": warmth_model,
                "queue_feature_contract": queue_contract
            })
        except Exception as e:
            print(f"Warning: Could not read {infra_path}: {e}", file=sys.stderr)

    if not physics_configs:
        return None

    # Return most common physics config
    return physics_configs[0]  # For now, assume consistent


def compute_queue_statistics(collection_path: Path, sample_size: int = 10) -> Optional[Dict[str, float]]:
    """Sample datasets to compute queue depth statistics."""
    ds_dirs = [d for d in collection_path.glob("ds_*") if (d / "infrastructure.json").exists()]
    if not ds_dirs:
        return None

    sample = random.sample(ds_dirs, min(sample_size, len(ds_dirs)))
    all_queue_depths = []

    for ds in sample:
        infra_path = ds / "infrastructure.json"
        try:
            with open(infra_path) as f:
                infra = json.load(f)

            queue_dists = infra.get("queue_distributions", {})
            for task_type, queues in queue_dists.items():
                all_queue_depths.extend(queues.values())
        except Exception as e:
            print(f"Warning: Could not read {infra_path}: {e}", file=sys.stderr)

    if not all_queue_depths:
        return None

    mean = sum(all_queue_depths) / len(all_queue_depths)
    variance = sum((x - mean) ** 2 for x in all_queue_depths) / len(all_queue_depths)
    stddev = variance ** 0.5

    return {
        "queue_depth_mean": round(mean, 2),
        "queue_depth_stddev": round(stddev, 2)
    }


def compute_placement_statistics(collection_path: Path, sample_size: int = 20) -> Optional[Dict[str, Any]]:
    """Sample datasets to compute placement statistics."""
    ds_dirs = [d for d in collection_path.glob("ds_*") if (d / "placements/placements.jsonl").exists()]
    if not ds_dirs:
        return None

    sample = random.sample(ds_dirs, min(sample_size, len(ds_dirs)))
    placement_counts = []

    for ds in sample:
        jsonl_path = ds / "placements" / "placements.jsonl"
        try:
            with open(jsonl_path) as f:
                count = sum(1 for _ in f)
            placement_counts.append(count)
        except Exception as e:
            print(f"Warning: Could not read {jsonl_path}: {e}", file=sys.stderr)

    if not placement_counts:
        return None

    avg_placements = sum(placement_counts) / len(placement_counts)
    return {"avg_placements_per_dataset": round(avg_placements, 1)}


def extract_grid_preset_params(grid_name: str) -> Optional[Dict[str, Any]]:
    """Extract generation parameters from grid preset."""
    if grid_name not in GRID_PRESETS:
        return None

    preset = GRID_PRESETS[grid_name]

    # Convert queue distributions to readable format
    queue_dists = []
    for qd in preset.get("queue_distributions", []):
        if isinstance(qd, tuple) and len(qd) == 7:
            queue_dists.append({
                "name": qd[0],
                "distribution": qd[1],
                "param1": qd[2],
                "param2": qd[3],
                "min_val": qd[4],
                "max_val": qd[5],
                "multiplier": qd[6]
            })

    params = {
        "connection_probabilities": preset.get("connection_probabilities", []),
        "replica_configs": preset.get("replica_configs", []),
        "queue_distributions": queue_dists,
        "seeds": preset.get("seeds", [])
    }

    # Add optional topology params
    if "topology_type" in preset:
        params["topology_type"] = preset["topology_type"]
    if "k_core_values" in preset:
        params["k_core_values"] = preset["k_core_values"]
    if "hub_seeker_fractions" in preset:
        params["hub_seeker_fractions"] = preset["hub_seeker_fractions"]

    return params


def extract_collection_metadata(collection_name: str, collection_path: Path) -> Dict[str, Any]:
    """Extract complete metadata for a collection."""
    print(f"Extracting metadata for {collection_name}...")

    # Get manual metadata
    info = COLLECTION_INFO.get(collection_name, {})

    # Get grid preset
    grid_preset = COLLECTION_TO_GRID.get(collection_name)
    generation_params = extract_grid_preset_params(grid_preset) if grid_preset else None

    # Count datasets
    total_datasets, completed_datasets = count_datasets(collection_path)

    # Sample physics config
    physics = sample_infrastructure_physics(collection_path)
    if not physics:
        physics = {"warmth_model": "node_disk_v2", "queue_feature_contract": "legacy_v0"}

    # Compute statistics
    queue_stats = compute_queue_statistics(collection_path)
    placement_stats = compute_placement_statistics(collection_path)

    # Build metadata
    metadata = {
        "collection_name": collection_name,
        "version": info.get("version", "unknown"),
        "status": info.get("status", "active"),
        "created_date": info.get("created_date", "unknown"),
        "grid_preset": grid_preset,
        "problem_category": info.get("problem_category", "unknown"),
        "generation_params": generation_params,
        "physics": physics,
        "purpose": info.get("purpose", ""),
        "hypothesis": info.get("hypothesis", ""),
        "results": {
            "total_datasets": total_datasets,
            "completed_datasets": completed_datasets,
        }
    }

    # Add queue stats if available
    if queue_stats:
        metadata["results"].update(queue_stats)

    # Add placement stats if available
    if placement_stats:
        metadata["results"].update(placement_stats)

    # Add archive reason if deprecated/archived
    if info.get("archive_reason"):
        metadata["archive_reason"] = info["archive_reason"]

    return metadata


def write_collection_metadata(collection_path: Path, metadata: Dict[str, Any]):
    """Write metadata to collection directory."""
    metadata_path = collection_path / "METADATA.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Wrote {metadata_path}")


def generate_registry(simulation_data_dir: Path, all_metadata: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate global registry from all collection metadata."""
    # Group by status
    by_status = {
        "active": [m for m in all_metadata if m["status"] == "active"],
        "deprecated": [m for m in all_metadata if m["status"] == "deprecated"],
        "archived": [m for m in all_metadata if m["status"] == "archived"],
    }

    # Group by problem category
    by_category = {}
    for m in all_metadata:
        cat = m["problem_category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(m["collection_name"])

    # Compute totals
    total_datasets = sum(m["results"]["total_datasets"] for m in all_metadata)
    completed_datasets = sum(m["results"]["completed_datasets"] for m in all_metadata)

    registry = {
        "schema_version": "1.0.0",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "summary": {
            "total_collections": len(all_metadata),
            "active_collections": len(by_status["active"]),
            "deprecated_collections": len(by_status["deprecated"]),
            "archived_collections": len(by_status["archived"]),
            "total_datasets": total_datasets,
            "completed_datasets": completed_datasets
        },
        "collections_by_status": {
            "active": [m["collection_name"] for m in by_status["active"]],
            "deprecated": [m["collection_name"] for m in by_status["deprecated"]],
            "archived": [m["collection_name"] for m in by_status["archived"]],
        },
        "collections_by_category": by_category,
        "collections": {m["collection_name"]: m for m in all_metadata}
    }

    return registry


def main():
    parser = argparse.ArgumentParser(description="Extract metadata from co-simulation dataset collections")
    parser.add_argument("--all", action="store_true", help="Process all collections")
    parser.add_argument("--collection", type=str, help="Process specific collection")
    parser.add_argument("--data-dir", type=Path, default=Path("simulation_data"),
                       help="Path to simulation_data directory")

    args = parser.parse_args()

    if not args.all and not args.collection:
        parser.error("Must specify --all or --collection")

    simulation_data_dir = args.data_dir
    if not simulation_data_dir.exists():
        print(f"Error: {simulation_data_dir} does not exist", file=sys.stderr)
        return 1

    # Find collections to process
    if args.all:
        collections = [
            d for d in simulation_data_dir.iterdir()
            if d.is_dir() and not d.name.startswith("_") and not d.name.startswith(".")
            and d.name != "METADATA_SCHEMA.json"
        ]
    else:
        collection_path = simulation_data_dir / args.collection
        if not collection_path.exists():
            print(f"Error: {collection_path} does not exist", file=sys.stderr)
            return 1
        collections = [collection_path]

    # Extract metadata for each collection
    all_metadata = []
    for collection_path in sorted(collections):
        collection_name = collection_path.name

        # Skip if not a known collection
        if collection_name not in COLLECTION_INFO:
            print(f"Skipping {collection_name} (not in COLLECTION_INFO)", file=sys.stderr)
            continue

        metadata = extract_collection_metadata(collection_name, collection_path)
        write_collection_metadata(collection_path, metadata)
        all_metadata.append(metadata)

    # Generate global registry if processing all
    if args.all:
        print("\nGenerating global registry...")
        registry = generate_registry(simulation_data_dir, all_metadata)
        registry_path = simulation_data_dir / "REGISTRY.json"
        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=2)
        print(f"✓ Wrote {registry_path}")

        print(f"\n=== Summary ===")
        print(f"Total collections: {registry['summary']['total_collections']}")
        print(f"  Active: {registry['summary']['active_collections']}")
        print(f"  Deprecated: {registry['summary']['deprecated_collections']}")
        print(f"  Archived: {registry['summary']['archived_collections']}")
        print(f"Total datasets: {registry['summary']['total_datasets']}")
        print(f"Completed datasets: {registry['summary']['completed_datasets']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
