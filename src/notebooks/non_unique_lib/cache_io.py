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
# Precomputed logit indices per task + exact co-sim RTT, sorted ascending by RTT.
ExactRttEntry = Tuple[List[int], float]
ExactRttLookupMap = Dict[str, List[ExactRttEntry]]


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


def load_rtt_hash_table_from_cache(
    cache_dir: Path,
    dataset_ids_filter: Optional[set] = None,
) -> RttHashTable:
    """Load prebuilt (dataset_id, combo) -> rtt hash table from cache chunks.

    dataset_ids_filter: if provided, only entries whose dataset_id is in this set
    are kept, drastically reducing memory when the cache contains more datasets
    than the graph cache (e.g. overnight cache built from all 3990 datasets but
    seq graphs were built for only 1710 high-queue ones).
    """
    meta_path = cache_dir / "rtt_chunks_meta.json"
    single_path = cache_dir / "placement_rtt_hash_table.pkl"

    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)
        num_chunks = int(meta.get("num_chunks", 0))
        total_entries = int(meta.get("total_entries", 0))
        filter_note = f", filtering to {len(dataset_ids_filter)} datasets" if dataset_ids_filter is not None else ""
        print(f"Loading RTT hash table from {num_chunks} chunks ({total_entries:,} entries{filter_note})...")
        placement_rtt_hash_table: RttHashTable = {}
        for i in tqdm(range(num_chunks), desc="Loading RTT chunks"):
            chunk_path = cache_dir / f"rtt_chunk_{i}.pkl"
            with open(chunk_path, "rb") as f:
                chunk = pickle.load(f)
            if dataset_ids_filter is not None:
                chunk = {k: v for k, v in chunk.items() if k[0] in dataset_ids_filter}
            placement_rtt_hash_table.update(chunk)
        print(f"Loaded {len(placement_rtt_hash_table):,} placement RTT entries")
        return placement_rtt_hash_table

    if single_path.exists():
        print(f"Loading RTT hash table from cache: {single_path}")
        with open(single_path, "rb") as f:
            placement_rtt_hash_table = pickle.load(f)
        if dataset_ids_filter is not None:
            placement_rtt_hash_table = {
                k: v for k, v in placement_rtt_hash_table.items() if k[0] in dataset_ids_filter
            }
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


def _rtt_chunks_meta(cache_dir: Path) -> Tuple[int, int]:
    meta_path = cache_dir / "rtt_chunks_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"RTT chunk metadata not found in {cache_dir}")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    return int(meta.get("num_chunks", 0)), int(meta.get("total_entries", 0))


def _finalize_streaming_hard_negatives(
    near: List[Tuple[PlacementCombo, float]],
    mod: List[Tuple[PlacementCombo, float]],
    cat: List[Tuple[PlacementCombo, float]],
    hard_negative_fraction: float,
    stratified: bool,
) -> List[Tuple[PlacementCombo, float]]:
    hard_candidates = near + mod + cat
    if not hard_candidates:
        return []
    if not stratified:
        hard_candidates.sort(key=lambda x: x[1], reverse=True)
        keep_n = max(1, int(len(hard_candidates) * hard_negative_fraction))
        return hard_candidates[:keep_n]

    opt_rtt = min(r for _, r in hard_candidates)
    near_f = [(c, r) for c, r in near if r - opt_rtt <= 0.05]
    mod_f = [(c, r) for c, r in mod if 0.05 < r - opt_rtt <= 1.0]
    cat_f = [(c, r) for c, r in cat if r - opt_rtt > 1.0]
    total_pool = max(1, int(len(hard_candidates) * hard_negative_fraction))
    targets = {"near": 0.40, "mod": 0.40, "cat": 0.20}
    buckets = {"near": near_f, "mod": mod_f, "cat": cat_f}
    avail = {k: len(v) > 0 for k, v in buckets.items()}
    if not any(avail.values()):
        return hard_candidates[:total_pool]
    empty_frac = sum(targets[k] for k, ok in avail.items() if not ok)
    avail_keys = [k for k, ok in avail.items() if ok]
    adj = {k: targets[k] + empty_frac / len(avail_keys) for k in avail_keys}
    pool: List[Tuple[PlacementCombo, float]] = []
    for k in avail_keys:
        n = max(1, round(total_pool * adj[k]))
        bucket = buckets[k]
        if len(bucket) >= n:
            pool.extend(random.sample(bucket, n))
        else:
            pool.extend(bucket)
    return pool if pool else hard_candidates[:1]


def build_regret_training_lookups_from_chunked_cache(
    cache_dir: Path,
    graphs: List[Data],
    dataset_ids: List[str],
    hard_negative_fraction: float = 0.5,
    stratified: bool = False,
    per_dataset_pool_cap: int = 25_000,
) -> Tuple[PlacementToLogitMap, HardNegativeMap]:
    """Stream RTT chunks once; build regret lookups without loading the full hash table."""
    target_parents = set(dataset_ids)
    num_chunks, total_entries = _rtt_chunks_meta(cache_dir)
    print(
        f"[regret chunked] Streaming {num_chunks} RTT chunks ({total_entries:,} entries) "
        f"for {len(target_parents)} parent datasets..."
    )

    placement_to_logit: PlacementToLogitMap = {}
    opt_combo_by_parent: Dict[str, PlacementCombo | None] = {}
    opt_rtt_by_parent: Dict[str, float] = {}
    near: Dict[str, List[Tuple[PlacementCombo, float]]] = defaultdict(list)
    mod: Dict[str, List[Tuple[PlacementCombo, float]]] = defaultdict(list)
    cat: Dict[str, List[Tuple[PlacementCombo, float]]] = defaultdict(list)

    for graph, dataset_id in zip(graphs, dataset_ids):
        task_map = _task_logit_map_from_graph(graph)
        if task_map:
            placement_to_logit[dataset_id] = [
                {placement: idx for idx, placement in enumerate(task_map.get(t_idx, []))}
                for t_idx in range(int(graph.n_tasks))
            ]
        opt_combo_by_parent[dataset_id] = _optimal_combo_from_graph(graph, task_map)
        opt_rtt_val = getattr(graph, "opt_rtt", None)
        if opt_rtt_val is None:
            opt_rtt_val = float("inf")
        else:
            try:
                opt_rtt_val = float(opt_rtt_val)
            except (TypeError, ValueError):
                opt_rtt_val = float("inf")
        opt_rtt_by_parent[dataset_id] = opt_rtt_val

    cap = max(1000, per_dataset_pool_cap)

    for i in tqdm(range(num_chunks), desc="Streaming RTT chunks for regret"):
        chunk_path = cache_dir / f"rtt_chunk_{i}.pkl"
        with open(chunk_path, "rb") as f:
            chunk = pickle.load(f)
        for (ds_id, combo), rtt in chunk.items():
            if ds_id not in target_parents:
                continue
            opt_combo = opt_combo_by_parent.get(ds_id)
            if opt_combo is not None and combo == opt_combo:
                continue
            opt_rtt = opt_rtt_by_parent.get(ds_id, float("inf"))
            if stratified and opt_rtt != float("inf"):
                delta = float(rtt) - opt_rtt
                if delta <= 0.05:
                    bucket = near[ds_id]
                elif delta <= 1.0:
                    bucket = mod[ds_id]
                else:
                    bucket = cat[ds_id]
            else:
                bucket = cat[ds_id]
            if len(bucket) < cap:
                bucket.append((combo, float(rtt)))
            elif random.random() < cap / (cap + 1):
                bucket[random.randrange(cap)] = (combo, float(rtt))
        chunk.clear()

    hard_negative_map: HardNegativeMap = {}
    total_hard = 0
    for ds_id in target_parents:
        pool = _finalize_streaming_hard_negatives(
            near.get(ds_id, []),
            mod.get(ds_id, []),
            cat.get(ds_id, []),
            hard_negative_fraction=hard_negative_fraction,
            stratified=stratified,
        )
        hard_negative_map[ds_id] = pool
        total_hard += len(pool)
        near.pop(ds_id, None)
        mod.pop(ds_id, None)
        cat.pop(ds_id, None)

    print(
        f"[regret chunked] Built hard-negative pools for {len(hard_negative_map)} datasets "
        f"({total_hard:,} total samples)"
    )
    return placement_to_logit, hard_negative_map


class LazyChunkedRttLookup:
    """On-demand RTT lookup from chunked cache (one parent dataset in memory at a time)."""

    def __init__(self, cache_dir: Path, parent_dataset_ids: Optional[set] = None):
        self.cache_dir = cache_dir
        self.parent_dataset_ids = parent_dataset_ids
        self._num_chunks, _ = _rtt_chunks_meta(cache_dir)
        self._by_parent: Dict[str, Dict[PlacementCombo, float]] = {}

    def get(self, key: RttHashKey) -> Optional[float]:
        ds_id, combo = key
        if self.parent_dataset_ids is not None and ds_id not in self.parent_dataset_ids:
            return None
        if ds_id not in self._by_parent:
            self._by_parent.clear()
            self._by_parent[ds_id] = self._load_parent(ds_id)
        return self._by_parent[ds_id].get(combo)

    def _load_parent(self, parent_dataset_id: str) -> Dict[PlacementCombo, float]:
        combos: Dict[PlacementCombo, float] = {}
        for i in range(self._num_chunks):
            chunk_path = self.cache_dir / f"rtt_chunk_{i}.pkl"
            with open(chunk_path, "rb") as f:
                chunk = pickle.load(f)
            for (ds_id, combo), rtt in chunk.items():
                if ds_id == parent_dataset_id:
                    combos.setdefault(combo, float(rtt))
            chunk.clear()
        return combos


class CombinedRttLookup:
    """Primary in-memory hash table plus optional chunked fallback for eval."""

    def __init__(self, primary: RttHashTable, chunked: Optional[LazyChunkedRttLookup] = None):
        self.primary = primary
        self.chunked = chunked

    def get(self, key: RttHashKey) -> Optional[float]:
        val = self.primary.get(key)
        if val is not None:
            return val
        if self.chunked is not None:
            return self.chunked.get(key)
        return None

    def __len__(self) -> int:
        return len(self.primary)


def _combo_to_logit_indices(
    combo: PlacementCombo,
    task_logit_to_placement: Dict[int, List[Tuple[int, int]]],
    placement_to_logit_by_task: Optional[List[Dict[Tuple[int, int], int]]],
) -> Optional[List[int]]:
    n_tasks = len(combo)
    indices: List[int] = []
    for t_idx in range(n_tasks):
        target_node_id, target_plat_id = combo[t_idx]
        found_idx = None
        if placement_to_logit_by_task and t_idx < len(placement_to_logit_by_task):
            found_idx = placement_to_logit_by_task[t_idx].get((target_node_id, target_plat_id))
        if found_idx is None:
            for logit_idx, (node_id, plat_id) in enumerate(task_logit_to_placement.get(t_idx, [])):
                if node_id == target_node_id and plat_id == target_plat_id:
                    found_idx = logit_idx
                    break
        if found_idx is None:
            return None
        indices.append(int(found_idx))
    return indices


def build_valid_combos_map_from_chunked_cache(
    cache_dir: Path,
    parent_dataset_ids: set,
) -> ValidCombosMap:
    """Stream RTT chunks once; keep every exact (combo, rtt) for target parent datasets."""
    num_chunks, total_entries = _rtt_chunks_meta(cache_dir)
    print(
        f"[exact RTT] Streaming {num_chunks} RTT chunks ({total_entries:,} entries) "
        f"for {len(parent_dataset_ids)} parent datasets..."
    )
    valid_map: ValidCombosMap = defaultdict(list)
    for i in tqdm(range(num_chunks), desc="Building exact valid combos map"):
        chunk_path = cache_dir / f"rtt_chunk_{i}.pkl"
        with open(chunk_path, "rb") as f:
            chunk = pickle.load(f)
        for (ds_id, combo), rtt in chunk.items():
            if ds_id in parent_dataset_ids:
                valid_map[ds_id].append((combo, float(rtt)))
        chunk.clear()

    sorted_map: ValidCombosMap = {}
    total = 0
    for ds_id in parent_dataset_ids:
        combos = valid_map.get(ds_id, [])
        if not combos:
            sorted_map[ds_id] = []
            continue
        combos.sort(key=lambda x: x[1])
        sorted_map[ds_id] = combos
        total += len(combos)
    print(
        f"[exact RTT] Built sorted valid combos for {len(sorted_map)} parent datasets "
        f"({total:,} total entries)"
    )
    return sorted_map


def _capped_sidecar_path(cache_dir: Path, sidecar_name: str = "valid_combos_near_rtt_capped.pkl") -> Path:
    return cache_dir / sidecar_name


def _capped_sidecar_meta_path(cache_dir: Path, sidecar_name: str = "valid_combos_near_rtt_capped.pkl") -> Path:
    return cache_dir / f"{Path(sidecar_name).stem}_meta.json"


def _reservoir_add(
    bucket: List[Tuple[PlacementCombo, float]],
    seen_count: int,
    cap: int,
    combo: PlacementCombo,
    rtt: float,
) -> None:
    if cap <= 0:
        return
    if len(bucket) < cap:
        bucket.append((combo, float(rtt)))
        return
    replace_idx = random.randrange(seen_count)
    if replace_idx < cap:
        bucket[replace_idx] = (combo, float(rtt))


def build_capped_valid_combos_map_from_chunked_cache(
    cache_dir: Path,
    parent_dataset_ids: set,
    near_cap: int = 256,
    close_cap: int = 384,
    mid_cap: int = 256,
    far_cap: int = 192,
    trash_cap: int = 0,
    near_delta: float = 0.05,
    close_delta: float = 0.30,
    mid_delta: float = 1.00,
    trash_delta: float = 5.00,
    sidecar_name: str = "valid_combos_near_rtt_capped.pkl",
) -> ValidCombosMap:
    """Stream RTT chunks and keep a bounded near-RTT training sidecar per dataset.

    Each dataset keeps the exact optimum, then reservoir samples from:
      near  : 0 < delta <= near_delta
      close : near_delta < delta <= close_delta
      mid   : close_delta < delta <= mid_delta
      far   : mid_delta < delta <= trash_delta
      trash : delta > trash_delta

    This gives the near-RTT ranking loss fine-grained post-plateau pairs without
    materializing the full 36M-row RTT table as Python objects.
    """
    parent_dataset_ids = {str(ds_id).split("@seq", 1)[0] for ds_id in parent_dataset_ids}
    num_chunks, total_entries = _rtt_chunks_meta(cache_dir)
    print(
        f"[near RTT capped] First pass over {num_chunks} chunks ({total_entries:,} entries) "
        f"for {len(parent_dataset_ids)} parent datasets..."
    )

    opt_by_dataset: Dict[str, Tuple[PlacementCombo, float]] = {}
    for i in tqdm(range(num_chunks), desc="Finding optimal RTT per dataset"):
        chunk_path = cache_dir / f"rtt_chunk_{i}.pkl"
        with open(chunk_path, "rb") as f:
            chunk = pickle.load(f)
        for (ds_id, combo), rtt_raw in chunk.items():
            if ds_id not in parent_dataset_ids:
                continue
            rtt = float(rtt_raw)
            current = opt_by_dataset.get(ds_id)
            if current is None or rtt < current[1]:
                opt_by_dataset[ds_id] = (combo, rtt)
        chunk.clear()

    near: Dict[str, List[Tuple[PlacementCombo, float]]] = defaultdict(list)
    close: Dict[str, List[Tuple[PlacementCombo, float]]] = defaultdict(list)
    mid: Dict[str, List[Tuple[PlacementCombo, float]]] = defaultdict(list)
    far: Dict[str, List[Tuple[PlacementCombo, float]]] = defaultdict(list)
    trash: Dict[str, List[Tuple[PlacementCombo, float]]] = defaultdict(list)
    seen: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"near": 0, "close": 0, "mid": 0, "far": 0, "trash": 0}
    )

    print("[near RTT capped] Second pass sampling near/close/mid/far/trash bands...")
    for i in tqdm(range(num_chunks), desc="Sampling capped near-RTT sidecar"):
        chunk_path = cache_dir / f"rtt_chunk_{i}.pkl"
        with open(chunk_path, "rb") as f:
            chunk = pickle.load(f)
        for (ds_id, combo), rtt_raw in chunk.items():
            opt = opt_by_dataset.get(ds_id)
            if opt is None:
                continue
            opt_combo, opt_rtt = opt
            if combo == opt_combo:
                continue

            rtt = float(rtt_raw)
            delta = rtt - opt_rtt
            if delta <= near_delta:
                key = "near"
                bucket = near[ds_id]
                cap = near_cap
            elif delta <= close_delta:
                key = "close"
                bucket = close[ds_id]
                cap = close_cap
            elif delta <= mid_delta:
                key = "mid"
                bucket = mid[ds_id]
                cap = mid_cap
            elif delta <= trash_delta or trash_cap <= 0:
                key = "far"
                bucket = far[ds_id]
                cap = far_cap
            else:
                key = "trash"
                bucket = trash[ds_id]
                cap = trash_cap
            seen[ds_id][key] += 1
            _reservoir_add(bucket, seen[ds_id][key], cap, combo, rtt)
        chunk.clear()

    capped_map: ValidCombosMap = {}
    total = 0
    bucket_totals = {"opt": 0, "near": 0, "close": 0, "mid": 0, "far": 0, "trash": 0}
    for ds_id in sorted(parent_dataset_ids):
        rows: List[Tuple[PlacementCombo, float]] = []
        opt = opt_by_dataset.get(ds_id)
        if opt is not None:
            rows.append(opt)
            bucket_totals["opt"] += 1
        rows.extend(near.get(ds_id, []))
        rows.extend(close.get(ds_id, []))
        rows.extend(mid.get(ds_id, []))
        rows.extend(far.get(ds_id, []))
        rows.extend(trash.get(ds_id, []))
        rows.sort(key=lambda x: x[1])
        capped_map[ds_id] = rows
        total += len(rows)
        bucket_totals["near"] += len(near.get(ds_id, []))
        bucket_totals["close"] += len(close.get(ds_id, []))
        bucket_totals["mid"] += len(mid.get(ds_id, []))
        bucket_totals["far"] += len(far.get(ds_id, []))
        bucket_totals["trash"] += len(trash.get(ds_id, []))

    meta = {
        "sidecar": _capped_sidecar_path(cache_dir, sidecar_name).name,
        "num_datasets": len(capped_map),
        "total_entries": total,
        "caps": {
            "near": int(near_cap),
            "close": int(close_cap),
            "mid": int(mid_cap),
            "far": int(far_cap),
            "trash": int(trash_cap),
        },
        "deltas": {
            "near": float(near_delta),
            "close": float(close_delta),
            "mid": float(mid_delta),
            "trash": float(trash_delta),
        },
        "bucket_totals": bucket_totals,
    }
    with open(_capped_sidecar_meta_path(cache_dir, sidecar_name), "w") as f:
        json.dump(meta, f, indent=2)

    print(
        f"[near RTT capped] Built capped sidecar for {len(capped_map)} datasets "
        f"({total:,} entries; near={bucket_totals['near']:,}, "
        f"close={bucket_totals['close']:,}, mid={bucket_totals['mid']:,}, "
        f"far={bucket_totals['far']:,}, trash={bucket_totals['trash']:,})"
    )
    return capped_map


def save_capped_valid_combos_map(
    cache_dir: Path,
    valid_combos_map: ValidCombosMap,
    sidecar_name: str = "valid_combos_near_rtt_capped.pkl",
) -> Path:
    out_path = _capped_sidecar_path(cache_dir, sidecar_name)
    with open(out_path, "wb") as f:
        pickle.dump(valid_combos_map, f, protocol=pickle.HIGHEST_PROTOCOL)
    total = sum(len(v) for v in valid_combos_map.values())
    print(f"[near RTT capped] Saved {out_path.name} ({total:,} entries) to {out_path}")
    return out_path


def load_capped_valid_combos_map(
    cache_dir: Path,
    sidecar_name: str = "valid_combos_near_rtt_capped.pkl",
) -> Optional[ValidCombosMap]:
    path = _capped_sidecar_path(cache_dir, sidecar_name)
    if not path.exists():
        return None
    with open(path, "rb") as f:
        valid_combos_map = pickle.load(f)
    total = sum(len(v) for v in valid_combos_map.values())
    print(f"[near RTT capped] Loaded {path.name} ({total:,} entries)")
    return valid_combos_map


def save_valid_combos_map(cache_dir: Path, valid_combos_map: ValidCombosMap) -> Path:
    out_path = cache_dir / "valid_combos_map.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(valid_combos_map, f, protocol=pickle.HIGHEST_PROTOCOL)
    total = sum(len(v) for v in valid_combos_map.values())
    print(f"[exact RTT] Saved valid_combos_map.pkl ({total:,} entries) to {out_path}")
    return out_path


def load_valid_combos_map(cache_dir: Path) -> Optional[ValidCombosMap]:
    path = cache_dir / "valid_combos_map.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        valid_combos_map = pickle.load(f)
    total = sum(len(v) for v in valid_combos_map.values())
    print(f"[exact RTT] Loaded valid_combos_map.pkl ({total:,} entries)")
    return valid_combos_map


def build_exact_rtt_index_lookups(
    graphs: List[Data],
    dataset_ids: List[str],
    valid_combos_map: ValidCombosMap,
) -> Tuple[PlacementToLogitMap, ExactRttLookupMap]:
    """
    Map every co-sim combo to logit indices + exact RTT (sorted ascending by RTT).
    Uses one final-step graph per parent dataset for placement decoding.
    """
    placement_to_logit: PlacementToLogitMap = {}
    exact_lookup: ExactRttLookupMap = {}
    skipped_unmapped = 0

    by_parent: Dict[str, Tuple[Data, str]] = {}
    for graph, graph_id in zip(graphs, dataset_ids):
        parent_id = getattr(graph, "parent_dataset_id", None)
        if parent_id is None:
            by_parent[graph_id] = (graph, graph_id)
            continue
        step = int(getattr(graph, "seq_step", -1))
        n_tasks = int(getattr(graph, "seq_n_tasks", 0))
        if step == n_tasks - 1:
            by_parent[parent_id] = (graph, parent_id)

    for parent_id, (graph, _) in by_parent.items():
        task_map = _task_logit_map_from_graph(graph)
        if task_map:
            placement_to_logit[parent_id] = [
                {placement: idx for idx, placement in enumerate(task_map.get(t_idx, []))}
                for t_idx in range(int(graph.n_tasks))
            ]

        entries: List[ExactRttEntry] = []
        placement_by_task = placement_to_logit.get(parent_id)
        for combo, rtt in valid_combos_map.get(parent_id, []):
            indices = _combo_to_logit_indices(combo, task_map, placement_by_task)
            if indices is None:
                skipped_unmapped += 1
                continue
            entries.append((indices, float(rtt)))

        entries.sort(key=lambda x: x[1])
        exact_lookup[parent_id] = entries

    total = sum(len(v) for v in exact_lookup.values())
    print(
        f"[exact RTT] Built index lookups for {len(exact_lookup)} parent datasets "
        f"({total:,} mapped combos, {skipped_unmapped:,} skipped unmapped)"
    )
    return placement_to_logit, exact_lookup

