#!/usr/bin/env python3
from __future__ import annotations

"""Prepare a HeteroData cache plus capped near-RTT sidecar."""

import argparse
import json
import pickle
import sys
from pathlib import Path

from non_unique_lib.cache_io import (
    build_capped_valid_combos_map_from_chunked_cache,
    build_valid_combos_map_from_chunked_cache,
    save_capped_valid_combos_map,
    save_valid_combos_map,
)

import prepare_graphs_cache_hetero


def _arg_value(args: list[str], name: str) -> str | None:
    if name not in args:
        return None
    idx = args.index(name)
    if idx + 1 >= len(args):
        return None
    return args[idx + 1]


def _load_parent_ids(cache_dir: Path) -> set[str]:
    dataset_ids_path = cache_dir / "dataset_ids.pkl"
    if not dataset_ids_path.exists():
        raise FileNotFoundError(f"Missing dataset IDs cache: {dataset_ids_path}")
    with open(dataset_ids_path, "rb") as fh:
        dataset_ids = pickle.load(fh)
    return {str(dataset_id).split("@seq", 1)[0] for dataset_id in dataset_ids}


def _write_near_rtt_metadata(cache_dir: Path, sidecar_name: str) -> None:
    metadata_path = cache_dir / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r") as fh:
            metadata = json.load(fh)
    metadata["near_rtt_training"] = True
    metadata["exact_combo_sidecar"] = sidecar_name
    metadata["near_rtt_note"] = (
        f"{sidecar_name} stores a capped near-RTT sidecar "
        "(optimum + reservoir-sampled near/close/mid/far/trash bands) for ranking loss."
    )
    with open(metadata_path, "w") as fh:
        json.dump(metadata, fh, indent=2)


_NEAR_RTT_ONLY_FLAGS = {
    "--full-combos",
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


def _argv_for_hetero_cache(argv: list[str]) -> list[str]:
    filtered: list[str] = []
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg in _NEAR_RTT_ONLY_FLAGS:
            if arg != "--full-combos":
                skip_next = True
            continue
        filtered.append(arg)
    return filtered


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare hetero graph cache plus capped near-RTT sidecar.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--full-combos", action="store_true")
    parser.add_argument("--near-cap", type=int, default=256)
    parser.add_argument("--close-cap", type=int, default=384)
    parser.add_argument("--mid-cap", type=int, default=256)
    parser.add_argument("--far-cap", type=int, default=192)
    parser.add_argument("--trash-cap", type=int, default=0)
    parser.add_argument("--near-delta", type=float, default=0.05)
    parser.add_argument("--close-delta", type=float, default=0.30)
    parser.add_argument("--mid-delta", type=float, default=1.00)
    parser.add_argument("--trash-delta", type=float, default=5.00)
    parser.add_argument("--sidecar-name", type=str, default="valid_combos_near_rtt_capped.pkl")
    known, _ = parser.parse_known_args(argv)
    return known


def main() -> None:
    near_args = _parse_args(sys.argv[1:])
    project_root = near_args.project_root.expanduser().resolve()
    cache_dir = prepare_graphs_cache_hetero.resolve_cache_dir(project_root, sys.argv[1:])

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *_argv_for_hetero_cache(original_argv[1:])]
        prepare_graphs_cache_hetero.main()
    finally:
        sys.argv = original_argv

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
    print(f"[near-rtt hetero cache] Wrote capped sidecar for {len(capped_map)} datasets ({total_combos:,} combos)")

    if near_args.full_combos:
        valid_combos_map = build_valid_combos_map_from_chunked_cache(cache_dir, parent_ids)
        save_valid_combos_map(cache_dir, valid_combos_map)
        full_total = sum(len(v) for v in valid_combos_map.values())
        print(f"[near-rtt hetero cache] Wrote full valid_combos_map.pkl for {len(valid_combos_map)} datasets ({full_total:,} combos)")


if __name__ == "__main__":
    main()
