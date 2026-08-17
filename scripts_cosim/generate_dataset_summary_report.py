#!/usr/bin/env python3
"""
Step 5: Consolidate dataset metadata/validation/compatibility into one health
report and flag collections worth enriching for GNN training.

Combines REGISTRY.json + COMPATIBILITY_MATRIX.json + per-collection
VALIDATION_REPORT.json (all produced by extract_dataset_metadata.py,
validate_dataset_collection.py, compute_compatibility_matrix.py) and scores
each active collection on:
- structural completeness (missing placements.jsonl etc.)
- coupling rate (low/zero coupling means little multi-task placement signal
  for the GNN to learn from -- see CLAUDE.md "Key Requirement for GNN Advantage")
- queue depth drift from the declared distribution
- dataset count (single-dataset probes aren't trainable)

Usage:
    python scripts_cosim/generate_dataset_summary_report.py
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

MIN_TRAINABLE_DATASETS = 10
LOW_COUPLING_THRESHOLD = 0.10
LOW_COMPLETENESS_THRESHOLD = 0.95


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def score_collection(name: str, meta: Dict[str, Any], validation: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    results = meta.get("results", {})
    total = results.get("total_datasets", 0)
    completed = results.get("completed_datasets", 0)

    struct = (validation or {}).get("structural_completeness", {})
    coupling = (validation or {}).get("coupling_rate_estimation", {})
    queue_depth = (validation or {}).get("queue_depth_validation", {})

    completeness_rate = struct.get("completeness_rate")
    coupling_rate = coupling.get("coupling_rate")
    qd_status = queue_depth.get("status")

    reasons = []
    if total < MIN_TRAINABLE_DATASETS:
        reasons.append(f"only {total} datasets (probe-sized, not trainable)")
    if completeness_rate is not None and completeness_rate < LOW_COMPLETENESS_THRESHOLD:
        reasons.append(f"structural completeness {completeness_rate*100:.1f}% (missing required files)")
    if coupling_rate is not None and coupling_rate < LOW_COUPLING_THRESHOLD:
        reasons.append(f"coupling rate {coupling_rate*100:.1f}% (weak multi-task placement signal for GNN)")
    if qd_status == "FAIL":
        reasons.append("queue depths drift from declared distribution")

    return {
        "collection_name": name,
        "problem_category": meta.get("problem_category"),
        "total_datasets": total,
        "completed_datasets": completed,
        "completeness_rate": completeness_rate,
        "coupling_rate": coupling_rate,
        "queue_depth_status": qd_status,
        "enrichment_reasons": reasons,
        "needs_enrichment": bool(reasons),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate consolidated dataset health report")
    parser.add_argument("--data-dir", type=Path, default=Path("simulation_data"),
                         help="Path to simulation_data directory")
    args = parser.parse_args()

    data_dir = args.data_dir
    registry = load_json(data_dir / "REGISTRY.json")
    if registry is None:
        print(f"Error: {data_dir / 'REGISTRY.json'} does not exist. Run extract_dataset_metadata.py first.", file=sys.stderr)
        return 1

    compatibility = load_json(data_dir / "COMPATIBILITY_MATRIX.json")
    if compatibility is None:
        print(f"Error: {data_dir / 'COMPATIBILITY_MATRIX.json'} does not exist. Run compute_compatibility_matrix.py first.", file=sys.stderr)
        return 1

    scored: List[Dict[str, Any]] = []
    for name, meta in registry["collections"].items():
        if meta["status"] != "active":
            continue
        validation = load_json(data_dir / name / "VALIDATION_REPORT.json")
        scored.append(score_collection(name, meta, validation))

    scored.sort(key=lambda r: (not r["needs_enrichment"], r["problem_category"] or "", r["collection_name"]))

    report = {
        "schema_version": "1.0.0",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "registry_generated_at": registry.get("generated_at"),
        "compatibility_generated_at": compatibility.get("generated_at"),
        "training_groups": compatibility.get("training_groups", {}),
        "collections": scored,
        "enrichment_candidates": [r["collection_name"] for r in scored if r["needs_enrichment"]],
    }

    output_path = data_dir / "DATASET_HEALTH_REPORT.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {output_path}")

    print(f"\n=== Enrichment candidates ({len(report['enrichment_candidates'])}/{len(scored)} active collections) ===")
    for r in scored:
        if not r["needs_enrichment"]:
            continue
        print(f"\n{r['collection_name']} ({r['problem_category']}, {r['completed_datasets']}/{r['total_datasets']} datasets)")
        for reason in r["enrichment_reasons"]:
            print(f"  - {reason}")

    healthy = [r["collection_name"] for r in scored if not r["needs_enrichment"]]
    print(f"\n=== Healthy, training-ready ({len(healthy)}) ===")
    for name in healthy:
        print(f"  - {name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
