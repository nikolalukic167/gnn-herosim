#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from non_unique_lib.cache_io import (
    build_capped_valid_combos_map_from_chunked_cache,
    create_cache_context,
    load_graphs_from_cache,
    save_capped_valid_combos_map,
)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    default_cache_dir = (
        project_root
        / "simulation_data"
        / "artifacts"
        / "run_queue_big"
        / "graphs_cache_gnn_datasets_4tasks_scheduler_adaptive"
    )

    parser = argparse.ArgumentParser(description="Build capped near-RTT sidecar from chunked RTT cache.")
    parser.add_argument("--cache-dir", type=Path, default=default_cache_dir)
    parser.add_argument("--near-cap", type=int, default=256)
    parser.add_argument("--close-cap", type=int, default=384)
    parser.add_argument("--mid-cap", type=int, default=128)
    parser.add_argument("--far-cap", type=int, default=64)
    parser.add_argument("--near-delta", type=float, default=0.05)
    parser.add_argument("--close-delta", type=float, default=0.30)
    parser.add_argument("--mid-delta", type=float, default=1.00)
    args = parser.parse_args()

    cache_dir = args.cache_dir.expanduser().resolve()
    ctx = create_cache_context(cache_dir)
    _, dataset_ids = load_graphs_from_cache(ctx)
    parent_ids = {str(dataset_id).split("@seq", 1)[0] for dataset_id in dataset_ids}

    capped_map = build_capped_valid_combos_map_from_chunked_cache(
        cache_dir,
        parent_ids,
        near_cap=max(0, args.near_cap),
        close_cap=max(0, args.close_cap),
        mid_cap=max(0, args.mid_cap),
        far_cap=max(0, args.far_cap),
        near_delta=args.near_delta,
        close_delta=args.close_delta,
        mid_delta=args.mid_delta,
    )
    out_path = save_capped_valid_combos_map(cache_dir, capped_map)
    total = sum(len(rows) for rows in capped_map.values())
    print(f"Wrote {total:,} capped near-RTT rows for {len(capped_map)} datasets to {out_path}")


if __name__ == "__main__":
    main()
