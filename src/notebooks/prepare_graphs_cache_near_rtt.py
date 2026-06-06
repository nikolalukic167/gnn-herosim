#!/usr/bin/env python3
from __future__ import annotations

"""
Prepare a graph cache for near-optimal RTT ranking.

This wraps the standard non-unique graph cache builder, then adds
valid_combos_map.pkl so training can sample exact co-sim RTT bands without
re-streaming placements every epoch.
"""

import argparse
import pickle
import sys
from pathlib import Path

from non_unique_lib.cache_io import (
    build_valid_combos_map_from_chunked_cache,
    save_valid_combos_map,
)

import prepare_graphs_cache


def _safe_extended_state_from_infrastructure(dataset_dir: Path) -> Dict[str, Dict[str, Any]]:
    import json

    result: Dict[str, Dict[str, Any]] = {
        "queue_snapshot": {},
        "temporal_state": {},
    }
    infra_path = dataset_dir / "infrastructure.json"
    if not infra_path.exists():
        return result

    try:
        with open(infra_path, "r") as fh:
            infra_data = json.load(fh)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return result

    merged_queues: Dict[str, int] = {}
    for queues in infra_data.get("queue_distributions", {}).values():
        if not isinstance(queues, dict):
            continue
        for key, queue_length in queues.items():
            q = prepare_graphs_cache._queue_length_int(queue_length)
            merged_queues[key] = max(q, merged_queues.get(key, 0))

    result["queue_snapshot"] = merged_queues
    return result


_ORIGINAL_LOAD_EXTENDED_STATE = prepare_graphs_cache.load_extended_state_data


def _load_extended_state_data_safe(dataset_dir: Path) -> Dict[str, Dict[str, Any]]:
    try:
        return _ORIGINAL_LOAD_EXTENDED_STATE(dataset_dir)
    except AttributeError as exc:
        if "'NoneType' object has no attribute 'items'" not in str(exc):
            raise
        return _safe_extended_state_from_infrastructure(dataset_dir)


def _arg_value(args: list[str], name: str) -> str | None:
    if name not in args:
        return None
    idx = args.index(name)
    if idx + 1 >= len(args):
        return None
    return args[idx + 1]


def _arg_values(args: list[str], name: str) -> list[str]:
    if name not in args:
        return []
    idx = args.index(name) + 1
    values: list[str] = []
    while idx < len(args) and not args[idx].startswith("--"):
        values.append(args[idx])
        idx += 1
    return values


def _resolve_cache_dir(project_root: Path, args: list[str]) -> Path:
    explicit_cache = _arg_value(args, "--cache-dir")
    if explicit_cache:
        return Path(explicit_cache).expanduser().resolve()

    base_dirs = _arg_values(args, "--base-dirs")
    if base_dirs:
        first_base = Path(base_dirs[0]).expanduser()
        if not first_base.is_absolute():
            first_base = project_root / first_base
        if "--merge-datasets" in args:
            return first_base.parent / "graphs_cache_merged_2_3_4_tasks"
        return first_base.parent / f"graphs_cache_{first_base.name}"

    # Mirrors prepare_graphs_cache._default_base_dirs for the non-merge case.
    base = (
        project_root
        / "simulation_data"
        / "artifacts"
        / "run_queue_big"
        / "gnn_datasets_4tasks_overnight_260422"
    )
    return base.parent / f"graphs_cache_{base.name}"


def _load_parent_ids(cache_dir: Path) -> set[str]:
    dataset_ids_path = cache_dir / "dataset_ids.pkl"
    if not dataset_ids_path.exists():
        raise FileNotFoundError(f"Missing dataset IDs cache: {dataset_ids_path}")

    with open(dataset_ids_path, "rb") as fh:
        dataset_ids = pickle.load(fh)

    parent_ids: set[str] = set()
    for dataset_id in dataset_ids:
        parent_ids.add(str(dataset_id).split("@seq", 1)[0])
    return parent_ids


def _write_near_rtt_metadata(cache_dir: Path) -> None:
    metadata_path = cache_dir / "metadata.json"
    if not metadata_path.exists():
        return

    import json

    with open(metadata_path, "r") as fh:
        metadata = json.load(fh)

    metadata["near_rtt_training"] = True
    metadata["exact_combo_sidecar"] = "valid_combos_map.pkl"
    metadata["near_rtt_note"] = (
        "valid_combos_map.pkl stores exact co-sim RTT combos sorted by RTT "
        "for near-optimal pairwise ranking."
    )

    with open(metadata_path, "w") as fh:
        json.dump(metadata, fh, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    known, _ = parser.parse_known_args()
    project_root = known.project_root.expanduser().resolve()

    # Reuse the standard cache builder with the original CLI. It writes graphs,
    # dataset IDs, optimal RTTs, and chunked RTT lookup files.
    prepare_graphs_cache.load_extended_state_data = _load_extended_state_data_safe
    prepare_graphs_cache.main()

    cache_dir = _resolve_cache_dir(project_root, sys.argv[1:])
    parent_ids = _load_parent_ids(cache_dir)
    valid_combos_map = build_valid_combos_map_from_chunked_cache(cache_dir, parent_ids)
    save_valid_combos_map(cache_dir, valid_combos_map)
    _write_near_rtt_metadata(cache_dir)

    total_combos = sum(len(v) for v in valid_combos_map.values())
    print(
        f"[near-rtt cache] Wrote exact combo sidecar for "
        f"{len(valid_combos_map)} datasets ({total_combos:,} combos)"
    )


if __name__ == "__main__":
    main()
