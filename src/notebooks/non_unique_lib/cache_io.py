from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from torch_geometric.data import Data


PlacementCombo = Tuple[Tuple[int, int], ...]


@dataclass(frozen=True)
class CacheContext:
    cache_dir: Path
    metadata: Dict
    is_merged_cache: bool
    task_count_dist: Dict[str, int]
    base_dirs: List[Path]
    graphs_cache_path: Path
    dataset_ids_cache_path: Path
    optimal_rtt_cache_path: Path
    rtt_combos_lmdb_path: Path


def create_cache_context(cache_dir: Path) -> CacheContext:
    metadata_path = cache_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        is_merged_cache = metadata.get("merged_datasets", False)
        task_count_dist = metadata.get("statistics", {}).get("task_count_distribution", {})
        base_dirs_raw = metadata.get("base_dirs", [])
        base_dirs = [Path(p) for p in base_dirs_raw if isinstance(p, str)]
    else:
        metadata = {}
        is_merged_cache = False
        task_count_dist = {}
        base_dirs = []

    return CacheContext(
        cache_dir=cache_dir,
        metadata=metadata,
        is_merged_cache=is_merged_cache,
        task_count_dist=task_count_dist,
        base_dirs=base_dirs,
        graphs_cache_path=cache_dir / "graphs.pkl",
        dataset_ids_cache_path=cache_dir / "dataset_ids.pkl",
        optimal_rtt_cache_path=cache_dir / "optimal_rtt.pkl",
        rtt_combos_lmdb_path=cache_dir / "rtt_combos.lmdb",
    )


def load_graphs_from_cache(ctx: CacheContext) -> Tuple[List[Data], List[str]]:
    if not ctx.graphs_cache_path.exists():
        raise FileNotFoundError(
            f"Graphs cache not found at {ctx.graphs_cache_path}. Run prepare_graphs_cache.py first."
        )
    if not ctx.dataset_ids_cache_path.exists():
        raise FileNotFoundError(
            f"Dataset IDs cache not found at {ctx.dataset_ids_cache_path}. Run prepare_graphs_cache.py first."
        )

    print(f"Loading graphs from cache: {ctx.graphs_cache_path}")
    with open(ctx.graphs_cache_path, "rb") as f:
        graphs = pickle.load(f)

    print(f"Loading dataset IDs from cache: {ctx.dataset_ids_cache_path}")
    with open(ctx.dataset_ids_cache_path, "rb") as f:
        dataset_ids = pickle.load(f)

    print(f"Loaded {len(graphs)} graphs with {len(dataset_ids)} dataset IDs")
    return graphs, dataset_ids


def load_optimal_rtt_from_cache(ctx: CacheContext) -> Dict[str, float]:
    if not ctx.optimal_rtt_cache_path.exists():
        raise FileNotFoundError(
            f"Optimal RTT cache not found at {ctx.optimal_rtt_cache_path}. Run prepare_graphs_cache.py first."
        )

    print(f"Loading optimal RTT mapping from cache: {ctx.optimal_rtt_cache_path}")
    with open(ctx.optimal_rtt_cache_path, "rb") as f:
        optimal_rtt_map = pickle.load(f)

    print(f"Loaded optimal RTT for {len(optimal_rtt_map)} datasets")
    return optimal_rtt_map
