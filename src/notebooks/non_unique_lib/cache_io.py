from __future__ import annotations

import json
import pickle
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from torch_geometric.data import Data
from tqdm import tqdm


PlacementCombo = Tuple[Tuple[int, int], ...]
RttHashKey = Tuple[str, PlacementCombo]
RttHashTable = Dict[RttHashKey, float]
ValidCombosMap = Dict[str, List[Tuple[PlacementCombo, float]]]
PlacementToLogitMap = Dict[str, List[Dict[Tuple[int, int], int]]]
HardNegativeMap = Dict[str, List[Tuple[PlacementCombo, float]]]


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
    rtt_combos_backend: str


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

    backend = str(metadata.get("rtt_combos_backend", "hash_table_chunked"))

    return CacheContext(
        cache_dir=cache_dir,
        metadata=metadata,
        is_merged_cache=is_merged_cache,
        task_count_dist=task_count_dist,
        base_dirs=base_dirs,
        graphs_cache_path=cache_dir / "graphs.pkl",
        dataset_ids_cache_path=cache_dir / "dataset_ids.pkl",
        optimal_rtt_cache_path=cache_dir / "optimal_rtt.pkl",
        rtt_combos_backend=backend,
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


def load_rtt_hash_table_from_cache(cache_dir: Path) -> RttHashTable:
    """Load prebuilt (dataset_id, combo) -> rtt hash table from cache chunks."""
    meta_path = cache_dir / "rtt_chunks_meta.json"
    single_path = cache_dir / "placement_rtt_hash_table.pkl"

    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)
        num_chunks = int(meta.get("num_chunks", 0))
        total_entries = int(meta.get("total_entries", 0))
        print(f"Loading RTT hash table from {num_chunks} chunks ({total_entries:,} entries)...")
        placement_rtt_hash_table: RttHashTable = {}
        for i in tqdm(range(num_chunks), desc="Loading RTT chunks"):
            chunk_path = cache_dir / f"rtt_chunk_{i}.pkl"
            with open(chunk_path, "rb") as f:
                chunk = pickle.load(f)
            placement_rtt_hash_table.update(chunk)
        print(f"Loaded {len(placement_rtt_hash_table):,} placement RTT entries")
        return placement_rtt_hash_table

    if single_path.exists():
        print(f"Loading RTT hash table from cache: {single_path}")
        with open(single_path, "rb") as f:
            placement_rtt_hash_table = pickle.load(f)
        print(f"Loaded {len(placement_rtt_hash_table):,} placement RTT entries")
        return placement_rtt_hash_table

    raise FileNotFoundError(
        f"RTT hash table cache not found in {cache_dir}. Run prepare_graphs_cache.py first."
    )


def build_valid_combos_map(
    placement_rtt_hash_table: RttHashTable,
    dataset_ids_filter: Optional[set] = None,
) -> ValidCombosMap:
    """Group RTT entries by dataset_id.

    dataset_ids_filter: if provided, only include entries whose dataset_id is in
    this set.  Use this when the graph cache was built from a filtered subset so
    the full 36M-entry RTT table can be pruned before the second-level grouping.
    """
    valid_map: ValidCombosMap = defaultdict(list)
    for (ds_id, combo), rtt in placement_rtt_hash_table.items():
        if dataset_ids_filter is not None and ds_id not in dataset_ids_filter:
            continue
        valid_map[ds_id].append((combo, float(rtt)))
    n = len(dataset_ids_filter) if dataset_ids_filter is not None else "all"
    print(f"[valid_combos] Built valid placement combos for {len(valid_map)} datasets (filter={n})")
    return dict(valid_map)


def _task_logit_map_from_graph(graph: Data) -> Dict[int, List[Tuple[int, int]]]:
    task_map = getattr(graph, "task_logit_to_placement", None)
    if task_map is None:
        task_map = getattr(graph, "_task_logit_to_placement", None)
    return task_map or {}


def _optimal_combo_from_graph(
    graph: Data,
    task_map: Dict[int, List[Tuple[int, int]]],
) -> PlacementCombo | None:
    n_tasks = int(graph.n_tasks)
    combo: List[Tuple[int, int]] = []
    for t_idx in range(n_tasks):
        if t_idx not in task_map:
            return None
        opt_idx = int(graph.y[t_idx].item())
        placements = task_map[t_idx]
        if opt_idx < 0 or opt_idx >= len(placements):
            return None
        combo.append(placements[opt_idx])
    return tuple(combo)


def build_regret_training_lookups(
    graphs: List[Data],
    dataset_ids: List[str],
    valid_combos_map: ValidCombosMap,
    hard_negative_fraction: float = 0.5,
    stratified: bool = False,
) -> Tuple[PlacementToLogitMap, HardNegativeMap]:
    """
    Precompute regret sampling pools once so epochs only do O(1) lookups.

    stratified=True: pool is built with stratified bucket sampling so each
    training draw sees near-optimal (40%), moderate (40%), and catastrophic (20%)
    negatives rather than almost exclusively catastrophic ones.
      near-optimal : ΔRTT ≤ 0.05s  (Knative-like mistakes, small RTT gap)
      moderate     : 0.05s < ΔRTT ≤ 1.0s
      catastrophic : ΔRTT > 1.0s   (obviously bad combos, easy to rank)
    When a bucket is empty the remaining fraction is redistributed to the others.
    """
    hard_negative_fraction = min(1.0, max(0.0, hard_negative_fraction))
    placement_to_logit: PlacementToLogitMap = {}
    hard_negative_map: HardNegativeMap = {}
    total_hard_negatives = 0
    strat_bucket_counts: Dict[str, int] = {"near": 0, "mod": 0, "cat": 0}

    for graph, dataset_id in zip(graphs, dataset_ids):
        task_map = _task_logit_map_from_graph(graph)
        if task_map:
            placement_to_logit[dataset_id] = [
                {placement: idx for idx, placement in enumerate(task_map.get(t_idx, []))}
                for t_idx in range(int(graph.n_tasks))
            ]

        if dataset_id in hard_negative_map:
            continue

        valid_combos = valid_combos_map.get(dataset_id, [])
        if not valid_combos:
            hard_negative_map[dataset_id] = []
            continue

        opt_combo = _optimal_combo_from_graph(graph, task_map)
        hard_candidates = [
            (combo, rtt) for combo, rtt in valid_combos
            if opt_combo is None or combo != opt_combo
        ]
        if not hard_candidates:
            hard_negative_map[dataset_id] = []
            continue

        if stratified:
            opt_rtt = min(r for _, r in valid_combos)
            near = [(c, r) for c, r in hard_candidates if r - opt_rtt <= 0.05]
            mod  = [(c, r) for c, r in hard_candidates if 0.05 < r - opt_rtt <= 1.0]
            cat  = [(c, r) for c, r in hard_candidates if r - opt_rtt > 1.0]
            total_pool = max(1, int(len(hard_candidates) * hard_negative_fraction))
            # Target counts per stratum; redistribute from empty buckets
            targets = {"near": 0.40, "mod": 0.40, "cat": 0.20}
            buckets = {"near": near, "mod": mod, "cat": cat}
            # Redistribute fractions from empty buckets proportionally
            avail = {k: len(v) > 0 for k, v in buckets.items()}
            if not any(avail.values()):
                pool = hard_candidates[:total_pool]
            else:
                empty_frac = sum(targets[k] for k, ok in avail.items() if not ok)
                avail_keys = [k for k, ok in avail.items() if ok]
                adj = {k: targets[k] + empty_frac / len(avail_keys) for k in avail_keys}
                pool: List[Tuple] = []
                for k in avail_keys:
                    n = max(1, round(total_pool * adj[k]))
                    bucket = buckets[k]
                    before = len(pool)
                    if len(bucket) >= n:
                        pool.extend(random.sample(bucket, n))
                    else:
                        pool.extend(bucket)
                    strat_bucket_counts[k] += len(pool) - before
                if not pool:
                    pool = hard_candidates[:1]
            hard_negative_map[dataset_id] = pool
            total_hard_negatives += len(pool)
        else:
            hard_candidates.sort(key=lambda x: x[1], reverse=True)
            keep_n = max(1, int(len(hard_candidates) * hard_negative_fraction))
            hard_negative_map[dataset_id] = hard_candidates[:keep_n]
            total_hard_negatives += keep_n

    if stratified:
        print(
            f"[regret lookups] Stratified pools for {len(hard_negative_map)} datasets "
            f"({total_hard_negatives:,} total) — "
            f"near={strat_bucket_counts['near']:,} mod={strat_bucket_counts['mod']:,} "
            f"cat={strat_bucket_counts['cat']:,}"
        )
    else:
        print(
            f"[regret lookups] Precomputed hard-negative pools for {len(hard_negative_map)} datasets "
            f"({total_hard_negatives:,} total samples)"
        )
    return placement_to_logit, hard_negative_map


def build_task_logit_to_queue_key(
    task_logit_to_placement: Dict[int, List[Tuple[int, int]]],
    node_id_to_name: Dict[int, str],
) -> Dict[int, List[str]]:
    """Map each task's logit index to a queue snapshot key (node_name:platform_id)."""
    queue_keys: Dict[int, List[str]] = {}
    for t_idx, placements in task_logit_to_placement.items():
        queue_keys[t_idx] = [
            f"{node_id_to_name.get(node_id, str(node_id))}:{plat_id}"
            for node_id, plat_id in placements
        ]
    return queue_keys


def _queue_length_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _load_queue_snapshot(dataset_dir: Path) -> Dict[str, int]:
    ssc_path = dataset_dir / "system_state_captured_unique.json"
    if not ssc_path.exists():
        return {}
    try:
        with open(ssc_path, "r") as f:
            data = json.load(f)
        task_placements = data.get("task_placements", [])
        if not task_placements:
            return {}
        full_queue_snapshot = task_placements[0].get("full_queue_snapshot", {})
        return {k: _queue_length_int(v) for k, v in full_queue_snapshot.items()}
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}


def _load_node_id_to_name(dataset_dir: Path) -> Dict[int, str]:
    opt_path = dataset_dir / "optimal_result.json"
    if opt_path.exists():
        try:
            with open(opt_path, "r") as f:
                opt_data = json.load(f)
            infra_nodes = opt_data.get("config", {}).get("infrastructure", {}).get("nodes", [])
            if infra_nodes:
                mapping: Dict[int, str] = {}
                for idx, node in enumerate(infra_nodes):
                    node_id = node.get("id", idx)
                    node_name = node.get("node_name", f"node_{node_id}")
                    mapping[int(node_id)] = str(node_name)
                return mapping
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass

    infra_path = dataset_dir / "infrastructure.json"
    if infra_path.exists():
        try:
            with open(infra_path, "r") as f:
                infra_data = json.load(f)
            nodes = infra_data.get("nodes", [])
            mapping = {}
            for idx, node in enumerate(nodes):
                if not isinstance(node, dict):
                    continue
                node_id = node.get("id", idx)
                node_name = node.get("node_name", f"node_{node_id}")
                mapping[int(node_id)] = str(node_name)
            if mapping:
                return mapping
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass

    if opt_path.exists():
        try:
            with open(opt_path, "r") as f:
                opt_data = json.load(f)
            mapping = {}
            for node_result in opt_data.get("stats", {}).get("nodeResults", []):
                node_id = node_result.get("nodeId")
                node_name = node_result.get("nodeName")
                if node_id is not None and node_name:
                    mapping[int(node_id)] = str(node_name)
            return mapping
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass

    return {}


def resolve_dataset_dir(
    project_root: Path,
    dataset_id: str,
    base_dirs: List[Path],
) -> Optional[Path]:
    parts = dataset_id.split("/", 1)
    if len(parts) != 2:
        return None
    collection_name, ds_name = parts
    for base_dir in base_dirs:
        resolved = base_dir if base_dir.is_absolute() else project_root / base_dir
        if resolved.name == collection_name:
            candidate = resolved / ds_name
            if candidate.exists():
                return candidate
    fallback = project_root / collection_name / ds_name
    return fallback if fallback.exists() else None


def load_decode_metadata_for_dataset(
    project_root: Path,
    dataset_id: str,
    base_dirs: List[Path],
    task_logit_to_placement: Dict[int, List[Tuple[int, int]]],
) -> Tuple[Dict[str, int], Dict[int, List[str]]]:
    dataset_dir = resolve_dataset_dir(project_root, dataset_id, base_dirs)
    if dataset_dir is None:
        return {}, {}
    queue_snapshot = _load_queue_snapshot(dataset_dir)
    node_id_to_name = _load_node_id_to_name(dataset_dir)
    queue_keys = build_task_logit_to_queue_key(task_logit_to_placement, node_id_to_name)
    return queue_snapshot, queue_keys


def enrich_graphs_with_decode_metadata(
    graphs: List[Data],
    dataset_ids: List[str],
    project_root: Path,
    ctx: CacheContext,
) -> int:
    """
    Attach queue_snapshot and task_logit_to_queue_key to graphs for sequential eval decode.
    Uses a sidecar pickle in the cache dir when present; otherwise loads from dataset dirs once.
    """
    sidecar_path = ctx.cache_dir / "decode_metadata.pkl"
    sidecar: Dict[str, Tuple[Dict[str, int], Dict[int, List[str]]]] = {}
    if sidecar_path.exists():
        with open(sidecar_path, "rb") as f:
            sidecar = pickle.load(f)
        print(f"Loaded decode metadata sidecar for {len(sidecar)} datasets")

    base_dirs = ctx.base_dirs or []
    if not base_dirs and ctx.metadata.get("base_dirs"):
        base_dirs = [Path(p) for p in ctx.metadata["base_dirs"]]

    enriched = 0
    missing_sidecar: Dict[str, Tuple[Dict[str, int], Dict[int, List[str]]]] = {}

    for graph, dataset_id in zip(graphs, dataset_ids):
        if getattr(graph, "queue_snapshot", None) and getattr(graph, "task_logit_to_queue_key", None):
            enriched += 1
            continue

        if dataset_id in sidecar:
            queue_snapshot, queue_keys = sidecar[dataset_id]
        else:
            task_map = _task_logit_map_from_graph(graph)
            queue_snapshot, queue_keys = load_decode_metadata_for_dataset(
                project_root, dataset_id, base_dirs, task_map
            )
            if queue_snapshot and queue_keys:
                missing_sidecar[dataset_id] = (queue_snapshot, queue_keys)

        if not queue_snapshot or not queue_keys:
            continue

        graph.queue_snapshot = queue_snapshot
        graph.task_logit_to_queue_key = queue_keys
        enriched += 1

    if missing_sidecar:
        sidecar.update(missing_sidecar)
        with open(sidecar_path, "wb") as f:
            pickle.dump(sidecar, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved decode metadata sidecar ({len(sidecar)} datasets) to {sidecar_path}")

    print(f"[decode metadata] Enriched {enriched}/{len(graphs)} graphs for sequential eval")
    return enriched


def build_regret_training_lookups_from_hash_table(
    graphs: List[Data],
    dataset_ids: List[str],
    placement_rtt_hash_table: RttHashTable,
    hard_negative_fraction: float = 0.5,
    stratified: bool = False,
) -> Tuple[PlacementToLogitMap, HardNegativeMap]:
    """Build regret lookups from the hash table and drop the intermediate valid-combos map.

    Automatically limits the valid_combos_map to only the dataset IDs present in
    dataset_ids (the graph cache) so filtered caches don't wastefully iterate over
    the full RTT table for excluded datasets.
    """
    ids_in_graphs = {ds_id.split("@seq")[0] for ds_id in dataset_ids}
    valid_combos_map = build_valid_combos_map(placement_rtt_hash_table, dataset_ids_filter=ids_in_graphs)
    placement_to_logit, hard_negative_map = build_regret_training_lookups(
        graphs,
        dataset_ids,
        valid_combos_map,
        hard_negative_fraction=hard_negative_fraction,
        stratified=stratified,
    )
    del valid_combos_map
    return placement_to_logit, hard_negative_map
