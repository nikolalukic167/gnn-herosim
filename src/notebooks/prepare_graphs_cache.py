#!/usr/bin/env python3
from __future__ import annotations

"""
Pre-generate and cache graphs for GNN training (NON-UNIQUE VERSION).

REQUIRES placements/placements.jsonl per dataset for rtt_chunk_*.pkl (placement–RTT hash).
repair + recache does NOT replace JSONL. memory/placements_jsonl_required.md

This script builds all graphs and saves them to pickle files for faster training iterations.

NON-UNIQUE PLACEMENTS:
- Supports datasets where multiple tasks can be placed on the same replica
- Creates edges between tasks and all compatible platforms (no uniqueness constraint)
- Compatible with gnn_datasets_2tasks, gnn_datasets_3tasks, and gnn_datasets_4tasks
- Includes system state, temporal features, queue info, and consolidation metrics
"""

import argparse
import json
import logging
import math
import os
import pickle
import random
import shutil
import sys
import time
import concurrent.futures
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from tqdm import tqdm

_NOTEBOOKS_DIR = Path(__file__).resolve().parent
if str(_NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOKS_DIR))

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from non_unique_lib.training_contract import load_sweep_minimum
from src.placement.queue_features import (
    DEFAULT_QUEUE_FEATURE_CONTRACT,
    VALID_QUEUE_FEATURE_CONTRACTS,
    queue_depth_norm,
    usage_ratio_feature,
    validate_queue_feature_contract,
)
from src.placement.warmth import (
    estimated_pull_remaining_sec,
    normalize_estimated_pull_remaining_sec,
    unit_pull_sec_from_task_priors,
)
from src.placement.network_graph import (
    NETWORK_GRAPH_CONTRACT_OFF,
    attach_network_graph_block,
    build_network_graph_block,
    resolve_network_graph_contract,
)
from src.placement.temporal_features import temporal_remainders
from src.placement.topology_features import build_source_feature_context

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Strictly positive scale for divisions in feature construction (avoid Inf/NaN tensors).
FEATURE_DIV_EPS = 1e-12

TaskPriors = Dict[str, Any]
PlacementCombo = Tuple[Tuple[int, int], ...]

def _require_finite_feature_array(name: str, arr: np.ndarray) -> None:
    """Fail fast at cache build time if features are still non-finite."""
    if np.isfinite(arr).all():
        return
    bad = int(np.size(arr) - np.sum(np.isfinite(arr)))
    raise ValueError(f"{name} has {bad} non-finite value(s); fix normalization or input sanitization")


def _safe_float(v: Any, default: float = 0.0) -> float:
    """Finite float for JSON/simulator values; never returns NaN or Inf."""
    if v is None:
        return default
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(x):
        return default
    return x


def _safe_positive(d: float, eps: float = FEATURE_DIV_EPS) -> float:
    """Lower-bound a divisor so normalization cannot divide by zero."""
    if not math.isfinite(d):
        return eps
    return float(d) if d > eps else eps


def _queue_length_int(v: Any) -> int:
    """Non-negative queue length from snapshot JSON (handles null / non-finite floats)."""
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return max(0, v)
    fv = _safe_float(v, 0.0)
    return max(0, int(fv))


def _finite_positive_exec_values(exec_map: Mapping[str, Any]) -> List[float]:
    """Execution times from priors suitable for min/mean (exclude NaN, Inf, <= 0)."""
    out: List[float] = []
    for v in exec_map.values():
        x = _safe_float(v, 0.0)
        if x > 0.0:
            out.append(x)
    return out


# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ============================================================================
# Configuration
# ============================================================================
@dataclass
class Config:
    base_dirs: List[Path]
    cache_dir: Path
    priors_path: Path
    merge_datasets: bool = False
    queue_norm_factor: float = 50.0
    queue_norm_mode: str = "scheduler_adaptive"
    queue_feature_contract: str = DEFAULT_QUEUE_FEATURE_CONTRACT
    platform_feature_dim: int = 16
    require_queue_data: bool = True
    oversample_manifest: Optional[Path] = None


def load_oversample_weights(manifest_path: Path) -> Dict[str, int]:
    """dataset_id (corp/ds_*) -> repeat count (>=1). Weight 0 excludes."""
    raw = json.loads(manifest_path.read_text())
    if isinstance(raw, dict) and "weights" in raw:
        weights = raw["weights"]
    else:
        weights = raw
    out: Dict[str, int] = {}
    for k, v in weights.items():
        repeat = int(v)
        if repeat > 0:
            out[str(k)] = repeat
    if not out:
        raise ValueError(f"oversample manifest empty: {manifest_path}")
    return out


def _default_base_dirs(project_root: Path, merge_datasets: bool) -> List[Path]:
    artifacts_dir = project_root / "simulation_data" / "artifacts" / "run_queue_big"
    if merge_datasets:
        return [
            artifacts_dir / "gnn_datasets_2tasks",
            artifacts_dir / "gnn_datasets_3tasks",
            artifacts_dir / "gnn_datasets_4tasks",
        ]
    return [artifacts_dir / "gnn_datasets_4tasks_overnight_260422"]


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Pre-generate and cache GNN graphs.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--merge-datasets", action="store_true")
    parser.add_argument("--base-dirs", nargs="+", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--priors-path", type=Path)
    parser.add_argument("--queue-norm-factor", type=float, default=50.0)
    parser.add_argument(
        "--queue-norm-mode",
        choices=["scheduler_adaptive", "fixed"],
        default="scheduler_adaptive",
        help=(
            "Queue normalization mode: "
            "'scheduler_adaptive' matches GNNScheduler p90/cap logic, "
            "'fixed' uses --queue-norm-factor."
        ),
    )
    parser.add_argument(
        "--queue-feature-contract",
        choices=sorted(VALID_QUEUE_FEATURE_CONTRACTS),
        default=DEFAULT_QUEUE_FEATURE_CONTRACT,
        help=(
            "Scaling contract for platform dim7/dim13. 'legacy_v0' reproduces every "
            "pre-2026-08-13 cache bit-exactly; 'scale_invariant_v1' uncaps the dim7 divisor "
            "and log1p-compresses dim13 so live queue depths ~400x deeper than training stay "
            "on the training manifold. Checkpoints must be served under the contract they "
            "were trained on."
        ),
    )
    parser.add_argument(
        "--platform-feature-dim",
        type=int,
        choices=[14, 16],
        default=16,
        help=(
            "16 (dim24 layout) keeps the pull observables added in CACHE 5.6; 14 (dim22) drops "
            "them, matching pre-5.6 caches such as the 873/v5.5 deploy cache. Use 14 when a "
            "retrain must be comparable to a dim22 checkpoint."
        ),
    )
    parser.add_argument("--allow-missing-queue-data", action="store_true")
    parser.add_argument(
        "--oversample-manifest",
        type=Path,
        default=None,
        help="JSON with weights{dataset_id: repeat_count} for strategic merge oversampling.",
    )
    args = parser.parse_args()

    if args.queue_norm_mode == "fixed" and args.queue_norm_factor <= 0:
        parser.error("--queue-norm-factor must be positive (division by zero in queue length normalization).")

    base_dirs = args.base_dirs or _default_base_dirs(args.project_root, args.merge_datasets)
    if args.cache_dir:
        cache_dir = args.cache_dir
    elif args.merge_datasets:
        cache_dir = base_dirs[0].parent / "graphs_cache_merged_2_3_4_tasks"
    else:
        cache_dir = base_dirs[0].parent / f"graphs_cache_{base_dirs[0].name}"

    priors_path = args.priors_path or (args.project_root / "data" / "nofs-ids" / "task-types.json")
    cache_dir.mkdir(parents=True, exist_ok=True)

    return Config(
        base_dirs=base_dirs,
        cache_dir=cache_dir,
        priors_path=priors_path,
        merge_datasets=args.merge_datasets,
        queue_norm_factor=args.queue_norm_factor,
        queue_norm_mode=args.queue_norm_mode,
        queue_feature_contract=validate_queue_feature_contract(args.queue_feature_contract),
        platform_feature_dim=int(args.platform_feature_dim),
        require_queue_data=not args.allow_missing_queue_data,
        oversample_manifest=args.oversample_manifest,
    )


@contextmanager
def time_block(description: str):
    start = time.perf_counter()
    yield
    logger.info(f"{description} completed in {time.perf_counter() - start:.2f}s")

# Version for cache invalidation (increment when graph construction logic changes)
CACHE_VERSION = "5.7"  # + queue_feature_contract in metadata (dim7 divisor / dim13 scaling)
# - Labels y / opt_rtt from placements.jsonl sweep minima (not optimal_result.sample.placement_plan)
# - previous_task_type_name preserved for is_warm; replicas from SSC scheduling-time state
# - graph.parent_dataset_id attached for parent-safe splits / @os RTT identity
# - RTT combos loaded from chunked hash table at train time (rtt_chunk_*.pkl)
# - Sanitized queue/temporal JSON, safe divisors, finite exec-time priors; asserts finite task/platform features
# - Removed QoS features (qos_deviation, deadline) since co-simulation doesn't capture QoS violations as ground truth
# - Supports datasets where 2+ tasks can be placed on the same (node_id, platform_id)
# - Platform dims 14–15: absolute node_cold_count + estimated_pull_remaining_sec/100 (FilterStore depth)
# - metadata.queue_feature_contract records the dim7/dim13 scaling a cache was built under;
#   'legacy_v0' is bit-identical to 5.6, 'scale_invariant_v1' is not servable by 5.6 checkpoints
STRICT_TASK_RESULTS = True
REQUIRED_TASK_FIELDS = (
    "taskId",
    "elapsedTime",
    "queueTime",
    "waitTime",
    "coldStartTime",
    "executionTime",
    "communicationsTime",
    "networkLatency",
    "sourceNode",
    "executionNode",
)

# ============================================================================
# DATA LOADING (same as main script)
# ============================================================================

def extract_dataset_to_dataframes(
    optimal_result_path: Path,
    *,
    placement_plan: Dict[str, Any],
    opt_rtt: float,
    replicas_by_task: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:
    """Extract DataFrames. Labels/opt_rtt/replicas MUST be caller-supplied (sweep + SSC)."""
    with open(optimal_result_path, "r") as f:
        result = json.load(f)
    
    dataset_id = optimal_result_path.parent.name
    infra_nodes = result.get("config", {}).get("infrastructure", {}).get("nodes", [])
    stats = result.get("stats", {})
    task_results = stats.get("taskResults", [])

    if not placement_plan:
        raise ValueError(f"{dataset_id}: empty sweep-min placement_plan")
    if not math.isfinite(float(opt_rtt)):
        raise ValueError(f"{dataset_id}: non-finite sweep-min opt_rtt={opt_rtt!r}")
    if not replicas_by_task:
        raise ValueError(
            f"{dataset_id}: empty scheduling-time replicas; refuse terminal-state fallback"
        )

    if STRICT_TASK_RESULTS and not task_results:
        raise ValueError(
            f"{optimal_result_path.parent.name}: stats.taskResults is empty "
            f"(taskResultsIncluded={stats.get('taskResultsIncluded')}, "
            f"schema={stats.get('statsSchemaVersion')})"
        )

    if STRICT_TASK_RESULTS:
        missing_fields = set()
        for tr in task_results:
            for field in REQUIRED_TASK_FIELDS:
                if field not in tr:
                    missing_fields.add(field)
        if missing_fields:
            raise ValueError(
                f"{optimal_result_path.parent.name}: taskResults missing fields: "
                f"{sorted(missing_fields)}"
            )
    
    # NODES
    nodes_data = []
    for i, node in enumerate(infra_nodes):
        node_name = node.get("node_name", f"node_{i}")
        platforms = node.get("platforms", [])
        network_map = node.get("network_map", {})
        
        nodes_data.append({
            'node_id': i,
            'node_name': node_name,
            'node_type': node.get("type", "unknown"),
            'is_client': node_name.startswith('client_node'),
            'network_map': network_map
        })
    
    df_nodes = pd.DataFrame(nodes_data)
    
    # TASKS
    placement_plan_task_ids = set()
    for k in placement_plan.keys():
        task_id = int(k)
        if task_id >= 0:
            placement_plan_task_ids.add(task_id)
    
    tasks_data = []
    task_ids_seen = []
    
    for task_result in task_results:
        task_id = task_result.get("taskId")
        
        if task_id is None or task_id < 0 or task_id not in placement_plan_task_ids:
            continue
        
        task_ids_seen.append(task_id)
        
        placement = placement_plan.get(str(task_id), [None, None])
        
        if isinstance(placement, list) and len(placement) >= 2:
            opt_node_id, opt_platform_id = placement[0], placement[1]
        else:
            raise ValueError(
                f"{dataset_id}: sweep-min plan missing valid placement for task {task_id}"
            )
        
        tasks_data.append({
            'task_id': task_id,
            'task_type': task_result.get("taskType", {}).get("name", "unknown"),
            'source_node': task_result.get("sourceNode", ""),
            'execution_node': task_result.get("executionNode", ""),
            'execution_platform': task_result.get("executionPlatform", -1),
            'optimal_node_id': opt_node_id,
            'optimal_platform_id': opt_platform_id,
            'elapsed_time': task_result.get("elapsedTime", 0),
            'queue_time': task_result.get("queueTime", 0),
            'wait_time': task_result.get("waitTime", 0),
            'cold_start_time': task_result.get("coldStartTime", 0),
            'execution_time': task_result.get("executionTime", 0),
            'communications_time': task_result.get("communicationsTime", 0),
            'network_latency': task_result.get("networkLatency", 0),
        })
    
    tasks_data.sort(key=lambda x: x['task_id'])
    
    if len(task_ids_seen) != len(placement_plan_task_ids):
        raise RuntimeError(
            f"{dataset_id}: task filtering mismatch "
            f"seen={len(task_ids_seen)} plan={len(placement_plan_task_ids)}"
        )
    if not tasks_data:
        raise RuntimeError(f"{dataset_id}: no tasks after applying sweep-min plan")
    
    df_tasks = pd.DataFrame(tasks_data)
    
    # PLATFORMS — replica flags from scheduling-time SSC only (caller-supplied)
    platforms_data = []
    node_results = stats.get("nodeResults", [])
    
    for node_result in node_results:
        node_id = node_result.get("nodeId")
        node_name = infra_nodes[node_id].get("node_name") if node_id < len(infra_nodes) else f"node_{node_id}"
        
        for plat_result in node_result.get("platformResults", []):
            plat_id = plat_result.get("platformId")
            plat_type = plat_result.get("platformType", {}).get("shortName", "unknown")
            
            has_dnn1_replica = False
            has_dnn2_replica = False
            
            for task_type, replica_list in replicas_by_task.items():
                if isinstance(replica_list, list):
                    for replica in replica_list:
                        if isinstance(replica, list) and len(replica) >= 2:
                            if replica[0] == node_name and replica[1] == plat_id:
                                if task_type == "dnn1":
                                    has_dnn1_replica = True
                                elif task_type == "dnn2":
                                    has_dnn2_replica = True
            
            platforms_data.append({
                'platform_id': plat_id,
                'node_id': node_id,
                'node_name': node_name,
                'platform_type': plat_type,
                'has_dnn1_replica': has_dnn1_replica,
                'has_dnn2_replica': has_dnn2_replica
            })
    
    df_platforms = pd.DataFrame(platforms_data)
    df_metrics = pd.DataFrame([{'dataset_id': dataset_id, 'total_rtt': float(opt_rtt)}])
    
    return {
        'nodes': df_nodes,
        'tasks': df_tasks,
        'platforms': df_platforms,
        'metrics': df_metrics,
        # link_contention_v1 routes + per-link capacities. None for every corpus generated
        # without a network.backbone block, which is most of them; build_graph then emits
        # no network entities and the cached graph is unchanged.
        'link_topology': result.get("config", {}).get("infrastructure", {}).get("link_topology"),
    }


def load_extended_state_data(dataset_dir: Path) -> Dict[str, Any]:
    """
    Load extended state data from system_state_captured_unique.json.

    Returns dict with:
    - queue_snapshot: Dict mapping "node_name:platform_id" -> queue_length
    - temporal_state: Dict mapping "node_name:platform_id" -> {current_task_remaining, cold_start_remaining, comm_remaining}

    Raises FileNotFoundError or ValueError when the SSC file is missing or incomplete.
    No infrastructure.json fallback — repair datasets with refresh_optimal_full_stats.py first.
    """
    ssc_path = dataset_dir / "system_state_captured_unique.json"
    if not ssc_path.exists():
        msg = (
            f"Missing system_state_captured_unique.json for {dataset_dir.name}; "
            "run scripts_cosim/refresh_optimal_full_stats.py --repair"
        )
        logger.warning(msg)
        raise FileNotFoundError(msg)

    try:
        with open(ssc_path, 'r') as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        msg = f"Failed to read {ssc_path}: {e}"
        logger.warning(msg)
        raise ValueError(msg) from e

    task_placements = data.get('task_placements', [])
    if not task_placements:
        msg = f"Empty task_placements in {ssc_path} for {dataset_dir.name}"
        logger.warning(msg)
        raise ValueError(msg)

    full_queue_snapshot = task_placements[0].get('full_queue_snapshot') or {}
    queue_snapshot = {k: _queue_length_int(v) for k, v in full_queue_snapshot.items()}
    if not queue_snapshot:
        msg = (
            f"Empty full_queue_snapshot in {ssc_path} for {dataset_dir.name}; "
            "run scripts_cosim/refresh_optimal_full_stats.py --repair --force"
        )
        logger.warning(msg)
        raise ValueError(msg)

    full_temporal = task_placements[0].get('full_temporal_state_at_scheduling') or {}
    merged_temporal_state: Dict[str, Dict[str, Any]] = {}
    temporal_sources = [full_temporal] if full_temporal else []
    if not temporal_sources:
        temporal_sources = [
            tp.get('temporal_state_at_scheduling', {})
            for tp in task_placements
            if isinstance(tp.get('temporal_state_at_scheduling'), dict)
        ]
    for temp_state in temporal_sources:
        for platform_key, state_dict in temp_state.items():
            if isinstance(state_dict, dict):
                entry: Dict[str, Any] = {
                    'current_task_remaining': _safe_float(
                        state_dict.get('current_task_remaining', 0.0), 0.0
                    ),
                    'cold_start_remaining': _safe_float(
                        state_dict.get('cold_start_remaining', 0.0), 0.0
                    ),
                    'comm_remaining': _safe_float(
                        state_dict.get('comm_remaining', 0.0), 0.0
                    ),
                }
                prev_type = state_dict.get('previous_task_type_name')
                if prev_type is not None:
                    entry['previous_task_type_name'] = str(prev_type)
                merged_temporal_state[platform_key] = entry
    if not merged_temporal_state:
        msg = (
            f"Empty temporal_state in {ssc_path} for {dataset_dir.name}; "
            "run scripts_cosim/refresh_optimal_full_stats.py --repair --force"
        )
        logger.warning(msg)
        raise ValueError(msg)

    # Extract initialized_snapshot — stored at SSC top level by backfill_initialized_snapshot.py
    # (or any run after the shared-fate feature was added).  Falls back to the first
    # task_placement entry for SSC files written by the GNN scheduler, then to {} for
    # legacy datasets so that build_graph() silently zeros the shared-fate dim.
    raw_initialized = (
        data.get('initialized_snapshot')
        or task_placements[0].get('initialized_snapshot')
        or {}
    )
    initialized_snapshot: Dict[str, bool] = {
        k: bool(v) for k, v in raw_initialized.items()
    }

    replicas_by_task = data.get('replicas')
    if not isinstance(replicas_by_task, dict) or not replicas_by_task:
        msg = (
            f"Missing scheduling-time replicas in {ssc_path} for {dataset_dir.name}; "
            "refuse terminal optimal_result systemStateResults fallback"
        )
        logger.warning(msg)
        raise ValueError(msg)

    return {
        'queue_snapshot': queue_snapshot,
        'temporal_state': merged_temporal_state,
        'initialized_snapshot': initialized_snapshot,
        'replicas': replicas_by_task,
    }


def load_all_datasets(
    base_dirs: List[Path], require_queue_data: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Load all datasets from multiple gnn_datasets directories (supports merging).
    
    Args:
        base_dirs: List of Paths to gnn_datasets directories (can be single or multiple)
        require_queue_data: If True, fail on missing/incomplete system_state_captured_unique.json
    """
    all_datasets = {}
    failed_queue_data: List[str] = []
    
    for base_dir in base_dirs:
        if not base_dir.exists():
            logger.warning("Directory %s does not exist, skipping", base_dir)
            continue
        
        dataset_dirs = sorted(base_dir.glob("ds_*"))
        logger.info("Loading %s datasets from %s...", len(dataset_dirs), base_dir.name)
        start_time = time.perf_counter()
        
        for dataset_dir in tqdm(dataset_dirs, desc=f"Loading {base_dir.name}", unit="dataset"):
            optimal_result_path = dataset_dir / "optimal_result.json"
            jsonl_path = dataset_dir / "placements" / "placements.jsonl"
            if not optimal_result_path.exists():
                continue
            if not jsonl_path.is_file():
                # Excluded / incomplete sweeps are archived away from placements.jsonl.
                continue
            
            try:
                extended_state = load_extended_state_data(dataset_dir)
            except (FileNotFoundError, ValueError) as e:
                failed_queue_data.append(f"{base_dir.name}/{dataset_dir.name}: {e}")
                if require_queue_data:
                    raise
                continue
            
            try:
                sweep_plan, sweep_rtt, _sweep_combo = load_sweep_minimum(jsonl_path)
                dataframes = extract_dataset_to_dataframes(
                    optimal_result_path,
                    placement_plan=sweep_plan,
                    opt_rtt=sweep_rtt,
                    replicas_by_task=extended_state["replicas"],
                )
                # Use unique key: base_dir_name/dataset_name to avoid collisions
                unique_key = f"{base_dir.name}/{dataset_dir.name}"
                all_datasets[unique_key] = {
                    **dataframes,
                    'dataset_dir': dataset_dir,
                    'source_dir': base_dir.name,  # Track which directory this came from
                    'num_tasks': len(dataframes['tasks']),  # Track task count
                    'queue_snapshot': extended_state.get('queue_snapshot', {}),
                    'temporal_state': extended_state.get('temporal_state', {}),
                    'initialized_snapshot': extended_state.get('initialized_snapshot', {}),
                    'replicas': extended_state.get('replicas', {}),
                    'sweep_opt_rtt': float(sweep_rtt),
                }
            except Exception as e:
                tqdm.write(f"  Error loading {dataset_dir.name}: {e}")
        
        elapsed = time.perf_counter() - start_time
        logger.info(
            "  Loaded %s datasets from %s in %.2fs",
            len([k for k in all_datasets if k.startswith(base_dir.name)]),
            base_dir.name,
            elapsed,
        )
    
    logger.info("\nTotal datasets loaded: %s", len(all_datasets))
    if failed_queue_data:
        logger.warning(
            "  %s datasets missing valid system_state_captured_unique.json",
            len(failed_queue_data),
        )
        for entry in failed_queue_data[:10]:
            logger.warning("    %s", entry)
        if len(failed_queue_data) > 10:
            logger.warning("    ... and %s more", len(failed_queue_data) - 10)
        if require_queue_data:
            raise RuntimeError(
                f"{len(failed_queue_data)} datasets lack valid SSC; "
                "repair with scripts_cosim/refresh_optimal_full_stats.py --repair"
            )
    
    # Print task count distribution
    task_counts = {}
    for ds_dict in all_datasets.values():
        n_tasks = ds_dict['num_tasks']
        task_counts[n_tasks] = task_counts.get(n_tasks, 0) + 1
    logger.info("\nTask count distribution:")
    for n_tasks in sorted(task_counts.keys()):
        logger.info("  %s tasks: %s datasets", n_tasks, task_counts[n_tasks])
    
    return all_datasets


def export_task_metrics_for_analysis(
    all_datasets: Dict[str, Dict[str, Any]],
    output_csv: Path,
) -> None:
    """Export normalized per-task rows used by graph generation for gap analysis."""
    rows = []
    for dataset_id, dataset_dict in all_datasets.items():
        tasks_df = dataset_dict.get("tasks")
        if tasks_df is None or tasks_df.empty:
            continue
        source_dir = dataset_dict.get("source_dir", "unknown")
        for rec in tasks_df.to_dict(orient="records"):
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "source_dir": source_dir,
                    "num_tasks": int(dataset_dict.get("num_tasks", 0)),
                    **rec,
                }
            )
    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info("Exported %s task metric rows to %s", len(df), output_csv)


# ============================================================================
# RTT HASH TABLE BUILD (chunked pickle backend)
# ============================================================================
def _placement_combos_from_jsonl(jsonl_path: Path) -> Tuple[str, List[Tuple[PlacementCombo, float]]]:
    """Parse one placements.jsonl; return (dataset_id, list of (placement combo, rtt))."""
    combos: List[Tuple[PlacementCombo, float]] = []
    ds_name = jsonl_path.parent.parent.name
    source_dir = jsonl_path.parent.parent.parent.name
    dataset_id = f"{source_dir}/{ds_name}"
    try:
        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    placement_plan = data.get("placement_plan", {})
                    rtt_val = data.get("rtt")
                    if not placement_plan or rtt_val is None:
                        continue
                    sorted_tasks = sorted(placement_plan.keys(), key=lambda x: int(x))
                    combo: PlacementCombo = tuple(
                        (int(placement_plan[task][0]), int(placement_plan[task][1]))
                        for task in sorted_tasks
                        if isinstance(placement_plan[task], list) and len(placement_plan[task]) >= 2
                    )
                    if len(combo) == 0:
                        continue
                    combos.append((combo, float(rtt_val)))
                except (json.JSONDecodeError, ValueError, KeyError, IndexError):
                    continue
    except OSError as e:
        logger.warning("Failed to read JSONL file %s: %s", jsonl_path, e)
    return dataset_id, combos


def _parse_jsonl_file_to_dict(
    jsonl_path: Path,
) -> Dict[Tuple[str, PlacementCombo], float]:
    """
    Parse one placements.jsonl into {(dataset_id, combo): rtt}.
    """
    results: Dict[Tuple[str, PlacementCombo], float] = {}
    dataset_id, combos = _placement_combos_from_jsonl(jsonl_path)
    for combo, rtt in combos:
        key = (dataset_id, combo)
        if key not in results:
            results[key] = float(rtt)
    return results


def _remove_legacy_rtt_artifacts(cache_dir: Path) -> None:
    for stale_chunk in cache_dir.glob("rtt_chunk_*.pkl"):
        stale_chunk.unlink(missing_ok=True)
    (cache_dir / "rtt_chunks_meta.json").unlink(missing_ok=True)
    (cache_dir / "placement_rtt_hash_table.pkl").unlink(missing_ok=True)
    legacy_rtt_store = cache_dir / "rtt_combos.lmdb"
    if legacy_rtt_store.is_dir():
        shutil.rmtree(legacy_rtt_store)
    else:
        legacy_rtt_store.unlink(missing_ok=True)


def build_and_save_rtt_hash_table_chunked(
    base_dirs: List[Path],
    cache_dir: Path,
    n_jobs: int = 12,
    chunk_size: int = 200_000,
    parse_batch_size: int = 200,
) -> Tuple[int, int]:
    """
    Build (dataset_id, combo)->rtt hash table in chunks for O(1) lookup at train time.

    Returns:
        (num_datasets_with_combos, total_entries)
    """
    _remove_legacy_rtt_artifacts(cache_dir)

    all_jsonl_files: List[Path] = []
    for base_dir in base_dirs:
        if not base_dir.exists():
            logger.warning("Base directory does not exist, skipping: %s", base_dir)
            continue
        files = sorted(base_dir.glob("ds_*/placements/placements.jsonl"))
        all_jsonl_files.extend(files)
        logger.info("Found %s JSONL files in %s", len(files), base_dir.name)

    n_jobs = max(1, min(n_jobs, os.cpu_count() or 1))
    parse_batch_size = max(1, parse_batch_size)

    chunk_idx = 0
    current_chunk: Dict[Tuple[str, PlacementCombo], float] = {}
    total_entries = 0
    num_duplicates = 0
    datasets_with_combos: set[str] = set()

    batch_ranges = range(0, len(all_jsonl_files), parse_batch_size)
    for batch_start in tqdm(batch_ranges, desc="Parsing JSONL batches", unit="batch"):
        batch_files = all_jsonl_files[batch_start:batch_start + parse_batch_size]
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
            for parsed_dict in executor.map(_parse_jsonl_file_to_dict, batch_files):
                for key, rtt in parsed_dict.items():
                    ds_id = key[0]
                    datasets_with_combos.add(ds_id)
                    if key not in current_chunk:
                        current_chunk[key] = rtt
                        total_entries += 1
                    else:
                        num_duplicates += 1
                    if len(current_chunk) >= chunk_size:
                        chunk_path = cache_dir / f"rtt_chunk_{chunk_idx}.pkl"
                        with open(chunk_path, "wb") as f:
                            pickle.dump(current_chunk, f, protocol=pickle.HIGHEST_PROTOCOL)
                        logger.info(
                            "  Saved chunk %s (%s entries) to %s",
                            chunk_idx,
                            f"{len(current_chunk):,}",
                            chunk_path,
                        )
                        chunk_idx += 1
                        current_chunk = {}
                parsed_dict.clear()

    if current_chunk:
        chunk_path = cache_dir / f"rtt_chunk_{chunk_idx}.pkl"
        with open(chunk_path, "wb") as f:
            pickle.dump(current_chunk, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(
            "  Saved chunk %s (%s entries) to %s",
            chunk_idx,
            f"{len(current_chunk):,}",
            chunk_path,
        )
        chunk_idx += 1

    chunk_meta = {
        "num_chunks": chunk_idx,
        "total_entries": total_entries,
        "chunk_size": chunk_size,
    }
    with open(cache_dir / "rtt_chunks_meta.json", "w") as f:
        json.dump(chunk_meta, f)

    if num_duplicates > 0:
        logger.info("Found %s duplicate hash keys (kept first occurrence)", f"{num_duplicates:,}")

    logger.info(
        "Built RTT hash table: %s datasets, %s entries, %s chunks",
        len(datasets_with_combos),
        f"{total_entries:,}",
        chunk_idx,
    )
    return len(datasets_with_combos), total_entries


# ============================================================================
# GRAPH CONSTRUCTION (same as main script)
# ============================================================================

TASK_PLATFORM_COMPATIBILITY = {
    'dnn1': ['rpiCpu', 'xavierGpu', 'xavierCpu', 'pynqFpga'],
    'dnn2': ['rpiCpu', 'xavierGpu', 'xavierCpu']
}


def _scheduler_adaptive_queue_norm(
    queue_values: np.ndarray, contract: str = DEFAULT_QUEUE_FEATURE_CONTRACT
) -> float:
    """p90 of ALL platforms (idle zeros included), min 1.0; capped only under legacy_v0.

    NOTE: p90-of-all collapses when >=90% of platforms are idle. legacy_v0 keeps the
    historical collapse-to-1.0 behaviour; scale_invariant_v1 falls back to the busy-platform
    p90 (see src/placement/queue_features.py).
    """
    return queue_depth_norm(queue_values.tolist(), "scheduler_adaptive", contract)


def _scheduler_adaptive_queue_norm_nonzero(
    queue_values: np.ndarray, contract: str = DEFAULT_QUEUE_FEATURE_CONTRACT
) -> float:
    """
    Robust adaptive queue normalization: p90 of non-zero queues only.
    Fixes the collapse-to-1.0 failure when most platforms are idle.
    Training and inference must use the same mode to preserve the feature contract.
    queue_norm_mode='adaptive_nonzero' selects this path.
    """
    return queue_depth_norm(queue_values.tolist(), "adaptive_nonzero", contract)


def build_graph(
    df_nodes: pd.DataFrame,
    df_tasks: pd.DataFrame,
    df_platforms: pd.DataFrame,
    task_priors: TaskPriors,
    queue_norm_factor: float,
    queue_norm_mode: str,
    queue_snapshot: Optional[Mapping[str, int]] = None,
    temporal_state: Optional[Mapping[str, Mapping[str, float]]] = None,
    initialized_snapshot: Optional[Mapping[str, bool]] = None,
    queue_feature_contract: str = DEFAULT_QUEUE_FEATURE_CONTRACT,
    link_topology: Optional[Mapping[str, Any]] = None,
    network_graph_contract: Optional[str] = None,
) -> Data:
    """
    Build a bipartite graph with tasks and platforms as nodes.
    
    Args:
        df_nodes: DataFrame with node information
        df_tasks: DataFrame with task information
        df_platforms: DataFrame with platform information
        queue_snapshot: Dict mapping "node_name:platform_id" -> queue_length (from full_queue_snapshot)
        temporal_state: Dict mapping "node_name:platform_id" -> {current_task_remaining, ...}
    """
    
    # Basic sizes / offsets
    n_tasks = len(df_tasks)
    n_platforms = len(df_platforms)
    task_offset = 0
    platform_offset = n_tasks
    
    # Precompute lookups
    first_idx_per_name = (
        df_nodes.reset_index()[['index', 'node_name']]
        .groupby('node_name', as_index=True)['index']
        .first()
        .to_dict()
    )
    
    plat_pos_by_id = {row.platform_id: i for i, row in enumerate(df_platforms.itertuples(index=False))}
    
    plats_by_node = {}
    node_names_arr = df_platforms['node_name'].to_numpy()
    for pos, name in enumerate(node_names_arr):
        plats_by_node.setdefault(name, []).append(pos)
    
    network_map_by_node = {row.node_name: row.network_map for row in df_nodes.itertuples(index=False)}
    
    plat_types_by_pos = df_platforms['platform_type'].to_numpy()
    plat_node_by_pos = df_platforms['node_name'].to_numpy()
    plat_ids_arr = df_platforms['platform_id'].to_numpy()
    
    # TASK FEATURES (3 dims: 2 type + 1 source)
    # Note: QoS features removed since co-simulation doesn't capture QoS violations as ground truth
    task_types_vocab = np.array(['dnn1', 'dnn2'])
    task_type_arr = df_tasks['task_type'].to_numpy()
    task_onehot = (task_type_arr[:, None] == task_types_vocab[None, :]).astype(float)
    
    src_names = df_tasks['source_node'].to_numpy()
    source_ctx = build_source_feature_context(
        df_nodes['node_name'].tolist(),
        network_map_by_node,
        first_idx_by_name=first_idx_per_name,
    )
    src_feat = np.fromiter((source_ctx.feature(n) for n in src_names),
                           dtype=np.float64, count=n_tasks).reshape(-1, 1)

    task_features = np.concatenate([task_onehot, src_feat], axis=1)
    _require_finite_feature_array("task_features", task_features)
    task_features_tensor = torch.from_numpy(task_features).to(torch.float32)
    
    # PLATFORM FEATURES (16 dims: 5 type + 2 replica + 1 queue + 1 shared-fate + 3 temporal
    # + 2 consolidation + node_cold_count + estimated_pull_remaining_sec/100)
    platform_types_vocab = np.array(['rpiCpu','xavierCpu','xavierGpu','xavierDla','pynqFpga'])
    plat_type_arr = df_platforms['platform_type'].to_numpy()
    plat_onehot = (plat_type_arr[:, None] == platform_types_vocab[None, :]).astype(float)
    
    has_dnn1_arr = df_platforms['has_dnn1_replica'].to_numpy(dtype=bool)
    has_dnn2_arr = df_platforms['has_dnn2_replica'].to_numpy(dtype=bool)
    
    has_dnn1 = has_dnn1_arr.astype(float).reshape(-1, 1)
    has_dnn2 = has_dnn2_arr.astype(float).reshape(-1, 1)
    
    # QUEUE LENGTH FEATURE (normalized by QUEUE_NORM_FACTOR)
    queue_lengths = np.zeros(n_platforms, dtype=np.float64)
    if queue_snapshot:
        for pos in range(n_platforms):
            node_name = str(plat_node_by_pos[pos])
            plat_id = int(plat_ids_arr[pos])
            key = f"{node_name}:{plat_id}"
            queue_lengths[pos] = float(_queue_length_int(queue_snapshot.get(key, 0)))
    
    # Normalize queue lengths with scheduler-aligned adaptive mode or fixed mode.
    queue_feature_contract = validate_queue_feature_contract(queue_feature_contract)
    if queue_norm_mode == "scheduler_adaptive":
        active_queue_norm = _scheduler_adaptive_queue_norm(queue_lengths, queue_feature_contract)
    elif queue_norm_mode == "adaptive_nonzero":
        active_queue_norm = _scheduler_adaptive_queue_norm_nonzero(
            queue_lengths, queue_feature_contract
        )
    else:
        active_queue_norm = _safe_positive(float(queue_norm_factor))
    queue_lengths_norm = (queue_lengths / active_queue_norm).reshape(-1, 1)

    # SHARED-FATE SIGNAL (1 dim) — fraction of co-located platforms on the same
    # physical node that were cold (not yet initialized) at scheduling time.
    # Captures density risk but saturates at 1.0 when all co-located plats are cold
    # (scarce N=12 and remote N=1 both → 1.0). Absolute cold count + pull-remaining
    # (dims 14–15) break that tie. Falls back to 0 if initialized_snapshot absent.
    node_cold_replicas_arr = np.zeros(n_platforms, dtype=np.float64)
    node_cold_count_arr = np.zeros(n_platforms, dtype=np.float64)
    estimated_pull_remaining_arr = np.zeros(n_platforms, dtype=np.float64)
    if initialized_snapshot:
        # Group platform positions by physical node name
        node_platform_positions: Dict[str, List[int]] = {}
        for pos in range(n_platforms):
            name = str(plat_node_by_pos[pos])
            node_platform_positions.setdefault(name, []).append(pos)

        for pos in range(n_platforms):
            node_name = str(plat_node_by_pos[pos])
            plat_id = int(plat_ids_arr[pos])
            co_located_positions = node_platform_positions.get(node_name, [pos])
            cold_count = sum(
                1 for p_pos in co_located_positions
                if not initialized_snapshot.get(
                    f"{node_name}:{int(plat_ids_arr[p_pos])}", True
                )
            )
            node_cold_replicas_arr[pos] = cold_count / max(len(co_located_positions), 1)
            node_cold_count_arr[pos] = float(cold_count)
            unit_pull = unit_pull_sec_from_task_priors(
                task_priors, str(plat_types_by_pos[pos])
            )
            estimated_pull_remaining_arr[pos] = estimated_pull_remaining_sec(
                float(cold_count), unit_pull
            )
    node_cold_replicas_norm = node_cold_replicas_arr.reshape(-1, 1)
    node_cold_count_feat = node_cold_count_arr.reshape(-1, 1)
    estimated_pull_remaining_norm = np.asarray(
        [
            normalize_estimated_pull_remaining_sec(float(v))
            for v in estimated_pull_remaining_arr
        ],
        dtype=np.float64,
    ).reshape(-1, 1)

    # TEMPORAL STATE FEATURES (current task remaining times)
    # Since we don't have exact temporal state, we approximate:
    # - If queue > 0: platform is busy, estimate remaining time
    # - Otherwise: platform is idle
    current_task_remaining = np.zeros(n_platforms, dtype=np.float64)
    cold_start_remaining = np.zeros(n_platforms, dtype=np.float64)
    comm_remaining = np.zeros(n_platforms, dtype=np.float64)
    
    # Shared with live inference and the other two cache builders — see
    # src/placement/temporal_features.py for the two bugs this replaced (a snapshot-level
    # estimate gate, and an average over task types no corpus dispatches).
    for pos in range(n_platforms):
        node_name = str(plat_node_by_pos[pos])
        plat_id = int(plat_ids_arr[pos])
        key = f"{node_name}:{plat_id}"
        (
            current_task_remaining[pos],
            cold_start_remaining[pos],
            comm_remaining[pos],
        ) = temporal_remainders(
            queue_depth=queue_lengths[pos],
            recorded=(temporal_state or {}).get(key),
            platform_type=str(plat_types_by_pos[pos]),
            task_types_data=task_priors,
            task_types_vocab=task_types_vocab,
        )

    # Normalize temporal features (assume max ~10s)
    current_task_remaining_norm = (current_task_remaining / 10.0).reshape(-1, 1)
    cold_start_remaining_norm = (cold_start_remaining / 10.0).reshape(-1, 1)
    comm_remaining_norm = (comm_remaining / 10.0).reshape(-1, 1)
    
    # CONSOLIDATION METRICS (target concurrency and usage ratio)
    # Calculate target concurrency per platform (similar to HRC logic)
    # Baseline: fastest platform for each task type
    target_concurrencies = np.zeros(n_platforms, dtype=np.float64)
    usage_ratios = np.zeros(n_platforms, dtype=np.float64)
    
    # For each platform, calculate target concurrency based on task types it supports
    for pos in range(n_platforms):
        plat_type = str(plat_types_by_pos[pos])
        # Find which task types can run on this platform
        supported_task_types = []
        for task_type in task_types_vocab:
            task_type_priors = task_priors.get(str(task_type), {})
            platforms = task_type_priors.get("platforms", [])
            if plat_type in platforms:
                supported_task_types.append(str(task_type))
        
        # Calculate target concurrency: average of baseline concurrency for supported task types
        # HRC uses baseline platform (fastest) as reference
        baseline_concurrency = 5.0  # Default target (can be tuned)
        if supported_task_types:
            # Find fastest platform for each supported task type
            min_exec_times = []
            for task_type in supported_task_types:
                task_type_priors = task_priors.get(task_type, {})
                exec_map = task_type_priors.get("executionTime", {})
                if isinstance(exec_map, dict) and exec_map:
                    pos_exec = _finite_positive_exec_values(exec_map)
                    if pos_exec:
                        min_exec_times.append(min(pos_exec))
            
            if min_exec_times:
                # Target concurrency inversely related to execution time
                avg_min_exec = float(np.mean(min_exec_times))
                if not math.isfinite(avg_min_exec) or avg_min_exec <= 0:
                    avg_min_exec = 1.0
                exec_map_this = task_priors.get(supported_task_types[0], {}).get("executionTime", {})
                this_exec = (
                    _safe_float(exec_map_this.get(plat_type, avg_min_exec), avg_min_exec)
                    if isinstance(exec_map_this, dict)
                    else avg_min_exec
                )
                if this_exec > 0:
                    target_concurrencies[pos] = baseline_concurrency * (avg_min_exec / this_exec)
                else:
                    target_concurrencies[pos] = baseline_concurrency
            else:
                target_concurrencies[pos] = baseline_concurrency
        else:
            target_concurrencies[pos] = baseline_concurrency
        
        # Usage ratio: queue_length vs target_concurrency, scaled per the active contract.
        usage_ratios[pos] = usage_ratio_feature(
            queue_lengths[pos], target_concurrencies[pos], queue_feature_contract
        )
    
    # Normalize consolidation metrics
    target_concurrency_norm = (target_concurrencies / _safe_positive(20.0)).reshape(-1, 1)
    usage_ratio_norm = usage_ratios.reshape(-1, 1)
    
    # Concatenate all platform features
    platform_features = np.concatenate([
        plat_onehot,                    # dims 0-4  (5)
        has_dnn1, has_dnn2,             # dims 5-6  (2)
        queue_lengths_norm,             # dim  7    (1)
        node_cold_replicas_norm,        # dim  8    (1) shared-fate signal
        current_task_remaining_norm, cold_start_remaining_norm, comm_remaining_norm,  # dims 9-11 (3)
        target_concurrency_norm, usage_ratio_norm,  # dims 12-13 (2)
        node_cold_count_feat,           # dim 14    (1) absolute cold platforms on node
        estimated_pull_remaining_norm,  # dim 15    (1) cold_count × T_pull / 100
    ], axis=1)
    if platform_features.shape[1] != 16:
        raise ValueError(
            f"Expected 16 platform feature dims, got {platform_features.shape[1]}"
        )
    _require_finite_feature_array("platform_features", platform_features)
    platform_features_tensor = torch.from_numpy(platform_features).to(torch.float32)
    queue_key_to_platform_meta: Dict[str, Dict[str, Any]] = {}
    for pos, row in enumerate(df_platforms.itertuples(index=False)):
        node_name = str(row.node_name)
        platform_id = int(plat_ids_arr[pos])
        queue_key = f"{node_name}:{platform_id}"
        queue_key_to_platform_meta[queue_key] = {
            "platform_type": str(plat_type_arr[pos]),
            "target_concurrency": float(target_concurrencies[pos]),
            "node_name": node_name,
            "platform_id": platform_id,
            "node_id": int(row.node_id),
            "platform_pos": int(pos),
        }
    
    # Cache feasible platforms per source node
    feasible_plats_cache = {}
    def feasible_platform_positions(src_node_name: str) -> np.ndarray:
        """Get network-feasible platform positions."""
        hit = feasible_plats_cache.get(src_node_name)
        if hit is not None:
            return hit
        nm = network_map_by_node.get(src_node_name, {})
        feasible_nodes = [src_node_name, *nm.keys()] if isinstance(nm, dict) else [src_node_name]
        out = []
        for node in feasible_nodes:
            out.extend(plats_by_node.get(node, ()))
        arr = np.fromiter(out, dtype=np.int64, count=len(out)) if out else np.empty(0, dtype=np.int64)
        feasible_plats_cache[src_node_name] = arr
        return arr
    
    # Compatibility filtering
    allowed_types_dnn1 = np.array(TASK_PLATFORM_COMPATIBILITY.get('dnn1', []))
    allowed_types_dnn2 = np.array(TASK_PLATFORM_COMPATIBILITY.get('dnn2', []))
    
    plat_type_compat_dnn1 = np.isin(plat_type_arr, allowed_types_dnn1)
    plat_type_compat_dnn2 = np.isin(plat_type_arr, allowed_types_dnn2)
    
    def filter_compatible_platforms(
        network_feasible_plats: np.ndarray,
        task_type: str
    ) -> np.ndarray:
        """Filter platforms by compatibility rules."""
        if network_feasible_plats.size == 0:
            return network_feasible_plats
        
        if task_type == 'dnn1':
            type_mask = plat_type_compat_dnn1
        elif task_type == 'dnn2':
            type_mask = plat_type_compat_dnn2
        else:
            return np.empty(0, dtype=np.int64)
        
        compatible_mask = type_mask[network_feasible_plats]
        return network_feasible_plats[compatible_mask]
    
    # EDGES + LABELS
    edge_src, edge_dst = [], []
    edge_attrs = []
    y_list = []
    
    # NEW: Build per-task mapping from logit index -> (node_id, platform_id)
    # This is needed for StructuredRegretLoss to look up RTT in hash table
    # task_logit_to_placement[task_idx][logit_idx] = (node_id, platform_id)
    task_logit_to_placement: Dict[int, List[Tuple[int, int]]] = {}
    task_logit_to_queue_key: Dict[int, List[str]] = {}
    
    # Build node_name -> node_id mapping
    node_name_to_id = {row.node_name: row.node_id for row in df_nodes.itertuples(index=False)}
    
    optimal_platform_ids = df_tasks['optimal_platform_id'].to_numpy()
    task_types_arr = df_tasks['task_type'].to_numpy()
    
    for t_pos, (src_name, opt_pid, task_type) in enumerate(zip(src_names, optimal_platform_ids, task_types_arr)):
        network_feas_plats = feasible_platform_positions(src_name)
        compat_plats = filter_compatible_platforms(network_feas_plats, task_type)
        # Replica limiting (match simulator / RTT hash space):
        # Only allow platforms that currently have a replica for this task type.
        if compat_plats.size:
            if task_type == 'dnn1':
                compat_plats = compat_plats[has_dnn1_arr[compat_plats]]
            elif task_type == 'dnn2':
                compat_plats = compat_plats[has_dnn2_arr[compat_plats]]
            else:
                compat_plats = np.empty(0, dtype=np.int64)
        
        if compat_plats.size:
            # Sort compatible platforms so their order matches the per-task
            # edge ordering produced by to_undirected (lexicographic by column).
            compat_plats = np.sort(compat_plats)
            
            task_node_idx = task_offset + t_pos
            edge_src.extend([task_node_idx] * compat_plats.size)
            dst_list = (platform_offset + compat_plats).tolist()
            edge_dst.extend(dst_list)
            
            # Build logit_idx -> (node_id, platform_id) mapping for this task
            task_logit_to_placement[t_pos] = []
            task_logit_to_queue_key[t_pos] = []
            
            task_type = str(task_type)
            task_type_priors = task_priors.get(task_type, {})
            exec_map = task_type_priors.get("executionTime", {})
            src_nm = network_map_by_node.get(src_name, {})
            for logit_idx, plat_pos in enumerate(compat_plats.tolist()):
                plat_type = str(plat_types_by_pos[plat_pos])
                plat_node_name = str(plat_node_by_pos[plat_pos])
                plat_id = int(plat_ids_arr[plat_pos])
                node_id = node_name_to_id.get(plat_node_name, -1)
                
                # Store mapping: logit_idx -> (node_id, platform_id)
                task_logit_to_placement[t_pos].append((node_id, plat_id))
                task_logit_to_queue_key[t_pos].append(f"{plat_node_name}:{plat_id}")
                
                exec_time = (
                    _safe_float(exec_map.get(plat_type, 0.0), 0.0) if isinstance(exec_map, dict) else 0.0
                )
                
                # Network latency
                lat_entry = src_nm.get(plat_node_name, {}) if isinstance(src_nm, dict) else {}
                if isinstance(lat_entry, dict):
                    latency = _safe_float(lat_entry.get('latency', 0.0), 0.0)
                else:
                    latency = _safe_float(lat_entry, 0.0)
                
                # is_warm: matches the simulator's actual cold-start predicate.
                # platform_process() fires cold start when
                #   previous_task.type["name"] != task.type["name"].
                # Using the autoscaler's replica flag (has_dnn1/2) diverges from
                # this when the platform last served a different task type.
                plat_key_for_warm = f"{plat_node_name}:{plat_id}"
                prev_type = (
                    (temporal_state or {})
                    .get(plat_key_for_warm, {})
                    .get("previous_task_type_name")
                )
                is_warm = 1.0 if (prev_type is not None and prev_type == task_type) else 0.0
                
                # Energy consumption (from task-types.json)
                energy = 0.0
                energy_map = task_type_priors.get("energy", {})
                if isinstance(energy_map, dict):
                    energy = _safe_float(energy_map.get(plat_type, 0.0), 0.0)
                
                # Communication time (storage read + write)
                # Estimate from state sizes and typical storage throughput
                comm_time = 0.0
                state_size_map = task_type_priors.get("stateSize", {})
                if isinstance(state_size_map, dict):
                    # Use first application type's state size (approximation)
                    app_state = list(state_size_map.values())[0] if state_size_map else {}
                    if isinstance(app_state, dict):
                        input_size = app_state.get("input", 0)  # bytes
                        output_size = app_state.get("output", 0)  # bytes
                        # Typical storage: 100 MB/s throughput, 1ms latency
                        storage_throughput = 100.0 * 1024 * 1024  # bytes/s
                        storage_latency = 0.001  # seconds
                        read_time = (float(input_size) / _safe_positive(storage_throughput)) + storage_latency
                        write_time = (float(output_size) / _safe_positive(storage_throughput)) + storage_latency
                        comm_time = _safe_float(read_time + write_time, 0.0)
                
                # Edge attributes: [exec_time, latency, is_warm, energy, comm_time] (5 dims)
                # Note: penalty_score removed since co-simulation doesn't capture QoS violations as ground truth
                edge_attrs.append([exec_time, latency, is_warm, energy, comm_time])
            
            opt_pos = plat_pos_by_id.get(opt_pid, None)
            if opt_pos is None:
                y_list.append(-1)
            else:
                matches = np.nonzero(compat_plats == opt_pos)[0]
                if matches.size:
                    y_list.append(int(matches[0]))
                else:
                    y_list.append(-1)
        else:
            y_list.append(-1)
    
    # Stack edges
    if edge_src:
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_attr_tensor = torch.tensor(edge_attrs, dtype=torch.float32) if edge_attrs else torch.empty((0, 5), dtype=torch.float32)
        if edge_attr_tensor.numel() > 0 and not torch.isfinite(edge_attr_tensor).all():
            raise ValueError("edge_attr contains non-finite values; check priors / latency JSON")
        num_nodes = n_tasks + n_platforms
        if edge_attr_tensor.numel() > 0:
            # Use PyG to duplicate and align edge attributes with undirected edges
            edge_index, edge_attr_tensor = to_undirected(edge_index, edge_attr_tensor, num_nodes=num_nodes)
        else:
            edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr_tensor = torch.empty((0, 5), dtype=torch.float32)
    
    y = torch.tensor(y_list, dtype=torch.long)
    
    # Create PyG Data
    data = Data(
        edge_index=edge_index,
        y=y,
        n_tasks=n_tasks,
        n_platforms=n_platforms,
        task_features=task_features_tensor,
        platform_features=platform_features_tensor,
    )
    data.edge_attr = edge_attr_tensor
    # Same-node platform<->platform edges for GIN message passing (node aggregation).
    # Lets the GNN propagate contention/co-location signal between platforms that share a
    # physical node (shared FilterStore pulls / node bandwidth) — relational structure a
    # pointwise MLP cannot express. Name contains 'index' so PyG batches it like edge_index.
    node_edge_src: List[int] = []
    node_edge_dst: List[int] = []
    for _positions in plats_by_node.values():
        _pos = sorted(set(int(p) for p in _positions))
        if len(_pos) < 2:
            continue
        for _a in range(len(_pos)):
            for _b in range(_a + 1, len(_pos)):
                _ga = n_tasks + _pos[_a]
                _gb = n_tasks + _pos[_b]
                node_edge_src.extend([_ga, _gb])
                node_edge_dst.extend([_gb, _ga])
    if node_edge_src:
        data.node_edge_index = torch.tensor([node_edge_src, node_edge_dst], dtype=torch.long)
    else:
        data.node_edge_index = torch.empty((2, 0), dtype=torch.long)
    # Network entities (physical nodes + core links + route edges), built by the SAME
    # shared code path live inference uses — see src/placement/network_graph.py. Default
    # OFF, so this is a no-op for every existing cache.
    net_contract = resolve_network_graph_contract(network_graph_contract)
    if net_contract != NETWORK_GRAPH_CONTRACT_OFF:
        attach_network_graph_block(
            data,
            build_network_graph_block(
                node_names=df_nodes['node_name'].tolist(),
                platform_node_names=[str(name) for name in plat_node_by_pos],
                task_source_names=[str(name) for name in src_names],
                task_candidate_node_names=[
                    [
                        str(queue_key_to_platform_meta[key]["node_name"])
                        for key in task_logit_to_queue_key.get(t_pos, [])
                    ]
                    for t_pos in range(n_tasks)
                ],
                link_topology=link_topology,
                n_tasks=n_tasks,
                n_platforms=n_platforms,
                contract=net_contract,
            ),
        )
    # Per-task mapping from logit index -> (node_id, platform_id) for regret loss and decoding.
    # Use non-underscore attr so DataLoader worker IPC preserves it.
    data.task_logit_to_placement = task_logit_to_placement
    # Backward compatibility for older training scripts.
    data._task_logit_to_placement = task_logit_to_placement
    data.queue_snapshot = dict(queue_snapshot) if queue_snapshot else {}
    data.task_logit_to_queue_key = task_logit_to_queue_key
    data.queue_key_to_platform_meta = queue_key_to_platform_meta
    
    return data


# ============================================================================
# MAIN SCRIPT
# ============================================================================

def main():
    config = parse_args()
    script_start_time = time.perf_counter()

    logger.info("=" * 80)
    logger.info("PRE-GENERATING GRAPH CACHE%s", " (MERGED DATASETS)" if config.merge_datasets else "")
    logger.info("=" * 80)
    if config.merge_datasets:
        logger.info("Merging datasets from %s directories:", len(config.base_dirs))
        for bd in config.base_dirs:
            logger.info("  - %s", bd)
    else:
        logger.info("Loading from: %s", config.base_dirs[0])

    graphs_cache_path = config.cache_dir / "graphs.pkl"
    dataset_ids_cache_path = config.cache_dir / "dataset_ids.pkl"
    optimal_rtt_cache_path = config.cache_dir / "optimal_rtt.pkl"
    metadata_cache_path = config.cache_dir / "metadata.json"

    with open(config.priors_path, "r") as f:
        task_priors = json.load(f)
    logger.info("Loaded task priors from %s", config.priors_path)

    step1_start = time.perf_counter()
    with time_block("Step 1: Building RTT hash table chunks"):
        num_combo_datasets, num_rtt_combos_written = build_and_save_rtt_hash_table_chunked(
            config.base_dirs,
            config.cache_dir,
        )
    step1_time = time.perf_counter() - step1_start

    step2_start = time.perf_counter()
    with time_block("Step 2: Loading datasets"):
        all_datasets = load_all_datasets(config.base_dirs, require_queue_data=config.require_queue_data)
    step2_time = time.perf_counter() - step2_start

    if len(all_datasets) == 0:
        logger.error("No datasets loaded")
        sys.exit(1)

    analysis_export_path = config.cache_dir / "task_metrics_analysis.csv"
    export_task_metrics_for_analysis(all_datasets, analysis_export_path)

    step3_start = time.perf_counter()
    with time_block("Step 3: Building optimal RTT map"):
        optimal_rtt_map = {
            ds_id: float(ds_dict['metrics']['total_rtt'].iloc[0]) if 'metrics' in ds_dict and not ds_dict['metrics'].empty else 0.0
            for ds_id, ds_dict in all_datasets.items()
        }
    step3_time = time.perf_counter() - step3_start
    logger.info("Built optimal RTT map for %s datasets", len(optimal_rtt_map))

    step4_start = time.perf_counter()
    oversample_weights: Optional[Dict[str, int]] = None
    if config.oversample_manifest is not None:
        oversample_weights = load_oversample_weights(config.oversample_manifest)
        logger.info(
            "Oversample manifest: %s (%s datasets, %s graph slots)",
            config.oversample_manifest,
            len(oversample_weights),
            sum(oversample_weights.values()),
        )

    graphs = []
    dataset_ids = []
    parent_dataset_ids = []
    graph_build_failures: List[str] = []
    with time_block("Step 4: Building graphs"):
        for dataset_id, dataset_dict in tqdm(all_datasets.items(), desc="Building graphs", unit="dataset"):
            if oversample_weights is not None and dataset_id not in oversample_weights:
                continue
            repeat = oversample_weights.get(dataset_id, 1) if oversample_weights else 1
            try:
                for rep in range(repeat):
                    graph = build_graph(
                        dataset_dict['nodes'],
                        dataset_dict['tasks'],
                        dataset_dict['platforms'],
                        task_priors=task_priors,
                        queue_norm_factor=config.queue_norm_factor,
                        queue_norm_mode=config.queue_norm_mode,
                        queue_snapshot=dataset_dict.get('queue_snapshot', {}),
                        temporal_state=dataset_dict.get('temporal_state', {}),
                        initialized_snapshot=dataset_dict.get('initialized_snapshot', {}),
                        queue_feature_contract=config.queue_feature_contract,
                        link_topology=dataset_dict.get('link_topology'),
                    )
                    if config.platform_feature_dim != 16:
                        graph.platform_features = graph.platform_features[
                            :, : config.platform_feature_dim
                        ]
                    invalid = int((graph.y < 0).sum().item())
                    if invalid:
                        raise RuntimeError(
                            f"{dataset_id}: {invalid}/{int(graph.n_tasks)} sweep-min labels "
                            "absent from scheduling-time candidate edges"
                        )
                    graph_id = dataset_id if repeat == 1 else f"{dataset_id}@os{rep}"
                    graph.dataset_id = graph_id
                    graph.parent_dataset_id = dataset_id
                    graphs.append(graph)
                    dataset_ids.append(graph_id)
                    parent_dataset_ids.append(dataset_id)
            except Exception as e:
                msg = f"{dataset_id}: {e}"
                tqdm.write(f"  Error building graph for {msg}")
                graph_build_failures.append(msg)
    if graph_build_failures:
        raise RuntimeError(
            f"{len(graph_build_failures)} datasets failed graph build under training-contract "
            f"5.5 (sweep-min labels must lie on SSC+network candidate edges). "
            f"Exclude them from the integrity manifest before recache. "
            f"Examples: {graph_build_failures[:5]}"
        )
    step4_time = time.perf_counter() - step4_start

    stats_start = time.perf_counter()
    ys = np.concatenate([g.y.numpy() for g in graphs])
    task_count_dist = {}
    for g in graphs:
        n = int(g.n_tasks)
        task_count_dist[n] = task_count_dist.get(n, 0) + 1
    stats_time = time.perf_counter() - stats_start

    logger.info("\nStatistics:")
    logger.info("  Total graphs: %s", len(graphs))
    logger.info("  Task count distribution:")
    for n_tasks in sorted(task_count_dist.keys()):
        logger.info(
            "    %s tasks: %s graphs (%.1f%%)",
            n_tasks,
            task_count_dist[n_tasks],
            task_count_dist[n_tasks] / len(graphs) * 100,
        )
    logger.info("  Valid labels: %s / %s (%.1f%%)", np.sum(ys >= 0), len(ys), np.sum(ys >= 0) / len(ys) * 100)
    logger.info("  Graphs with no edges: %s / %s", sum([g.edge_index.numel() == 0 for g in graphs]), len(graphs))
    logger.info("  Avg edges per graph: %.1f", np.mean([g.edge_index.size(1) for g in graphs]))
    logger.info("  Avg valid tasks per graph: %.2f", np.mean([(g.y >= 0).sum().item() for g in graphs]))
    logger.info("  Statistics computed in %.2fs", stats_time)

    logger.info("\nStep 5: Saving to cache...")
    step5_start = time.perf_counter()

    save_start = time.perf_counter()
    with open(graphs_cache_path, 'wb') as f:
        pickle.dump(graphs, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("  Saved %s graphs to %s (%.2fs)", len(graphs), graphs_cache_path, time.perf_counter() - save_start)

    save_start = time.perf_counter()
    with open(dataset_ids_cache_path, 'wb') as f:
        pickle.dump(dataset_ids, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("  Saved dataset IDs to %s (%.2fs)", dataset_ids_cache_path, time.perf_counter() - save_start)

    logger.info(
        "  RTT hash table: %s datasets, %s entries",
        num_combo_datasets,
        f"{num_rtt_combos_written:,}",
    )

    save_start = time.perf_counter()
    with open(optimal_rtt_cache_path, 'wb') as f:
        pickle.dump(optimal_rtt_map, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(
        "  Saved optimal RTT mapping (%s datasets) to %s (%.2fs)",
        len(optimal_rtt_map),
        optimal_rtt_cache_path,
        time.perf_counter() - save_start,
    )

    metadata = {
        'version': CACHE_VERSION,
        'merged_datasets': config.merge_datasets,
        'base_dirs': [str(bd) for bd in config.base_dirs],
        'oversample_manifest': str(config.oversample_manifest) if config.oversample_manifest else None,
        'num_graphs': len(graphs),
        'rtt_combos_backend': 'hash_table_chunked',
        'rtt_hash_chunks_meta_rel_path': 'rtt_chunks_meta.json',
        'num_combo_datasets': num_combo_datasets,
        'num_rtt_combo_rows': num_rtt_combos_written,
        'num_datasets': len(all_datasets),
        'dataset_ids': dataset_ids,
        'parent_dataset_ids': parent_dataset_ids,
        'platform_feature_dim': config.platform_feature_dim,
        'inference_feature_layout': 'dim24' if config.platform_feature_dim == 16 else 'dim22',
        'queue_norm_mode': config.queue_norm_mode,
        'queue_feature_contract': config.queue_feature_contract,
        # build_graph resolves this from the process env; record what was actually used so
        # the trainer can read it from the cache instead of trusting its own shell — the
        # same bug class the inference_feature_layout confound (40.8% of total_rtt) had.
        'topology_feature_contract': resolve_topology_feature_contract(),
        'training_contract': {
            'label_source': 'placements.jsonl_sweep_minimum',
            'replica_source': 'ssc_scheduling_time_replicas',
            'warmth_source': 'ssc_previous_task_type_name',
            'canonical_parent_attr': 'parent_dataset_id',
            'ab_report': 'simulation_data/training_contract_ab_20260804.json',
            'pull_observables': [
                'node_cold_count',
                'estimated_pull_remaining_sec',
            ],
        },
        'statistics': {
            'valid_labels': int(np.sum(ys >= 0)),
            'total_labels': len(ys),
            'graphs_with_no_edges': int(sum([g.edge_index.numel() == 0 for g in graphs])),
            'avg_edges': float(np.mean([g.edge_index.size(1) for g in graphs])),
            'avg_valid_tasks': float(np.mean([(g.y >= 0).sum().item() for g in graphs])),
            'task_count_distribution': {str(k): v for k, v in task_count_dist.items()},
        },
        'timing': {
            'step1_build_rtt_hash_chunks': step1_time,
            'step2_load_datasets': step2_time,
            'step3_build_optimal_rtt_map': step3_time,
            'step4_build_graphs': step4_time,
            'step5_save_cache': time.perf_counter() - step5_start,
            'total_time': time.perf_counter() - script_start_time,
        }
    }
    with open(metadata_cache_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info("  Saved metadata to %s", metadata_cache_path)

    step5_time = time.perf_counter() - step5_start
    total_time = time.perf_counter() - script_start_time

    graphs_size = graphs_cache_path.stat().st_size / (1024 * 1024)
    optimal_rtt_size = optimal_rtt_cache_path.stat().st_size / (1024 * 1024)

    logger.info("\n" + "=" * 80)
    logger.info("CACHE GENERATION COMPLETE!")
    logger.info("=" * 80)
    logger.info("Cache directory: %s", config.cache_dir)
    logger.info("Graphs cache: %s (%.2f MB)", graphs_cache_path, graphs_size)
    logger.info("RTT combos backend: preloaded hash table chunks")
    logger.info("Optimal RTT cache: %s (%.2f MB)", optimal_rtt_cache_path, optimal_rtt_size)
    logger.info("Total cache size: %.2f MB", graphs_size + optimal_rtt_size)
    logger.info("Cache version: %s", CACHE_VERSION)
    logger.info("Timing Summary:")
    logger.info("  Step 1 - Build RTT hash chunks: %7.2fs", step1_time)
    logger.info("  Step 2 - Load datasets:        %7.2fs", step2_time)
    logger.info("  Step 3 - Build helper maps:    %7.2fs", step3_time)
    logger.info("  Step 4 - Build graphs:         %7.2fs", step4_time)
    logger.info("  Step 5 - Save cache:           %7.2fs", step5_time)
    logger.info("  Total time:                    %7.2fs", total_time)


if __name__ == "__main__":
    main()