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
from typing import Any, Dict

from non_unique_lib.cache_io import (
    build_capped_valid_combos_map_from_chunked_cache,
    build_valid_combos_map_from_chunked_cache,
    save_capped_valid_combos_map,
    save_valid_combos_map,
)

import prepare_graphs_cache


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

    # Default to the repaired 1060-dataset archive (post SSC warmup fix).
    base = (
        project_root
        / "simulation_data"
        / "artifacts"
        / "run_queue_big"
        / "gnn_datasets_4tasks_1060"
    )
    return base.parent / f"graphs_cache_{base.name}_scheduler_adaptive"


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


def _write_near_rtt_metadata(cache_dir: Path, sidecar_name: str) -> None:
    metadata_path = cache_dir / "metadata.json"
    if not metadata_path.exists():
        return

    import json

    with open(metadata_path, "r") as fh:
        metadata = json.load(fh)

    metadata["near_rtt_training"] = True
    metadata["exact_combo_sidecar"] = sidecar_name
    metadata["near_rtt_note"] = (
        f"{sidecar_name} stores a capped near-RTT sidecar "
        "(optimum + reservoir-sampled near/close/mid/far bands) for ranking loss."
    )

    with open(metadata_path, "w") as fh:
        json.dump(metadata, fh, indent=2)


_NEAR_RTT_ONLY_FLAGS = {
    "--full-combos",
    "--skip-cache-build",
    "--near-cap",
    "--close-cap",
    "--mid-cap",
    "--far-cap",
    "--trash-cap",
    "--near-delta",
    "--close-delta",
    "--mid-delta",
    "--trash-delta",
    "--sidecar-name",
}


def _argv_for_prepare_graphs_cache(argv: list[str]) -> list[str]:
    """Strip near-RTT-only flags so prepare_graphs_cache argparse stays valid."""
    filtered: list[str] = []
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg in _NEAR_RTT_ONLY_FLAGS:
            if arg not in {"--full-combos", "--skip-cache-build"}:
                skip_next = True
            continue
        filtered.append(arg)
    return filtered


def _parse_near_rtt_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare graph cache plus capped near-RTT training sidecar.",
        add_help=True,
    )
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--full-combos",
        action="store_true",
        help="Also materialize valid_combos_map.pkl (full RTT table; very large).",
    )
    parser.add_argument("--near-cap", type=int, default=256)
    parser.add_argument("--close-cap", type=int, default=384)
    parser.add_argument("--mid-cap", type=int, default=256, help="0.3s-1.0s band reservoir cap (raised for eval coverage).")
    parser.add_argument("--far-cap", type=int, default=192, help=">1.0s band reservoir cap (raised for bad-layout coverage).")
    parser.add_argument("--trash-cap", type=int, default=0, help=">trash-delta band reservoir cap.")
    parser.add_argument("--near-delta", type=float, default=0.05)
    parser.add_argument("--close-delta", type=float, default=0.30)
    parser.add_argument("--mid-delta", type=float, default=1.00)
    parser.add_argument("--trash-delta", type=float, default=5.00)
    parser.add_argument("--sidecar-name", type=str, default="valid_combos_near_rtt_capped.pkl")
    parser.add_argument(
        "--skip-cache-build",
        action="store_true",
        help="Only (re)build the capped sidecar; assume graphs/rtt chunks already exist.",
    )
    known, _ = parser.parse_known_args(argv)
    return known


def main() -> None:
    near_args = _parse_near_rtt_args(sys.argv[1:])
    project_root = near_args.project_root.expanduser().resolve()

    if not near_args.skip_cache_build:
        # Reuse the standard cache builder with the original CLI. It writes graphs,
        # dataset IDs, optimal RTTs, and chunked RTT lookup files.
        original_argv = sys.argv
        try:
            sys.argv = [original_argv[0], *_argv_for_prepare_graphs_cache(original_argv[1:])]
            prepare_graphs_cache.main()
        finally:
            sys.argv = original_argv

    cache_dir = _resolve_cache_dir(project_root, sys.argv[1:])
    parent_ids = _load_parent_ids(cache_dir)
    capped_map = build_capped_valid_combos_map_from_chunked_cache(
        cache_dir,
        parent_ids,
        near_cap=max(0, near_args.near_cap),
        close_cap=max(0, near_args.close_cap),
        mid_cap=max(0, near_args.mid_cap),
        far_cap=max(0, near_args.far_cap),
        trash_cap=max(0, near_args.trash_cap),
        near_delta=near_args.near_delta,
        close_delta=near_args.close_delta,
        mid_delta=near_args.mid_delta,
        trash_delta=near_args.trash_delta,
        sidecar_name=near_args.sidecar_name,
    )
    save_capped_valid_combos_map(cache_dir, capped_map, sidecar_name=near_args.sidecar_name)
    _write_near_rtt_metadata(cache_dir, near_args.sidecar_name)

    total_combos = sum(len(v) for v in capped_map.values())
    print(
        f"[near-rtt cache] Wrote capped sidecar for "
        f"{len(capped_map)} datasets ({total_combos:,} combos)"
    )

    if near_args.full_combos:
        valid_combos_map = build_valid_combos_map_from_chunked_cache(cache_dir, parent_ids)
        save_valid_combos_map(cache_dir, valid_combos_map)
        full_total = sum(len(v) for v in valid_combos_map.values())
        print(
            f"[near-rtt cache] Wrote full valid_combos_map.pkl for "
            f"{len(valid_combos_map)} datasets ({full_total:,} combos)"
        )


if __name__ == "__main__":
    main()
