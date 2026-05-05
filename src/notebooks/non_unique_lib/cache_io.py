from __future__ import annotations

import json
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from torch_geometric.data import Data
from tqdm import tqdm


PlacementCombo = Tuple[Tuple[int, int], ...]
RttHashTable = Dict[Tuple[str, PlacementCombo], float]


@dataclass(frozen=True)
class CacheContext:
    cache_dir: Path
    metadata: Dict
    is_merged_cache: bool
    task_count_dist: Dict[str, int]
    graphs_cache_path: Path
    dataset_ids_cache_path: Path
    rtt_hash_cache_path: Path
    optimal_rtt_cache_path: Path


def create_cache_context(cache_dir: Path) -> CacheContext:
    metadata_path = cache_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        is_merged_cache = metadata.get("merged_datasets", False)
        task_count_dist = metadata.get("statistics", {}).get("task_count_distribution", {})
    else:
        metadata = {}
        is_merged_cache = False
        task_count_dist = {}

    return CacheContext(
        cache_dir=cache_dir,
        metadata=metadata,
        is_merged_cache=is_merged_cache,
        task_count_dist=task_count_dist,
        graphs_cache_path=cache_dir / "graphs.pkl",
        dataset_ids_cache_path=cache_dir / "dataset_ids.pkl",
        rtt_hash_cache_path=cache_dir / "placement_rtt_hash_table.pkl",
        optimal_rtt_cache_path=cache_dir / "optimal_rtt.pkl",
    )


def load_graphs_from_cache(ctx: CacheContext) -> Tuple[List[Data], List[str]]:
    if not ctx.graphs_cache_path.exists():
        raise FileNotFoundError(f"Graphs cache not found at {ctx.graphs_cache_path}. Run prepare_graphs_cache.py first.")
    if not ctx.dataset_ids_cache_path.exists():
        raise FileNotFoundError(f"Dataset IDs cache not found at {ctx.dataset_ids_cache_path}. Run prepare_graphs_cache.py first.")

    print(f"Loading graphs from cache: {ctx.graphs_cache_path}")
    with open(ctx.graphs_cache_path, "rb") as f:
        graphs = pickle.load(f)

    print(f"Loading dataset IDs from cache: {ctx.dataset_ids_cache_path}")
    with open(ctx.dataset_ids_cache_path, "rb") as f:
        dataset_ids = pickle.load(f)

    print(f"Loaded {len(graphs)} graphs with {len(dataset_ids)} dataset IDs")
    return graphs, dataset_ids


def load_rtt_hash_table_from_cache(ctx: CacheContext) -> RttHashTable:
    meta_path = ctx.cache_dir / "rtt_chunks_meta.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)

        num_chunks = meta["num_chunks"]
        total_entries = meta["total_entries"]
        print(f"Loading RTT hash table from {num_chunks} chunks ({total_entries:,} entries)...")

        placement_rtt_hash_table: RttHashTable = {}
        for i in tqdm(range(num_chunks), desc="Loading RTT chunks"):
            chunk_path = ctx.cache_dir / f"rtt_chunk_{i}.pkl"
            with open(chunk_path, "rb") as f:
                chunk = pickle.load(f)
            placement_rtt_hash_table.update(chunk)

        print(f"Loaded {len(placement_rtt_hash_table):,} placement RTT entries")
        return placement_rtt_hash_table

    if not ctx.rtt_hash_cache_path.exists():
        raise FileNotFoundError(f"RTT hash table cache not found at {ctx.rtt_hash_cache_path}. Run prepare_graphs_cache.py first.")

    print(f"Loading RTT hash table from cache: {ctx.rtt_hash_cache_path}")
    with open(ctx.rtt_hash_cache_path, "rb") as f:
        placement_rtt_hash_table = pickle.load(f)

    print(f"Loaded {len(placement_rtt_hash_table):,} placement RTT entries")
    return placement_rtt_hash_table


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


def build_valid_combos_map(placement_rtt_hash_table: RttHashTable) -> Dict[str, List[Tuple[PlacementCombo, float]]]:
    valid_map: Dict[str, List[Tuple[PlacementCombo, float]]] = defaultdict(list)
    for (ds_id, combo), rtt in placement_rtt_hash_table.items():
        valid_map[ds_id].append((combo, rtt))
    print(f"[valid_combos] Built valid placement combos for {len(valid_map)} datasets")
    return valid_map
