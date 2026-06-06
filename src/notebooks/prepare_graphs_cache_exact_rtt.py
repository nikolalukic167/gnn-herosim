#!/usr/bin/env python3
from __future__ import annotations

"""
Pre-generate and cache SEQUENTIAL counterfactual graphs for EXACT RTT training.

Builds the same sequential graphs as prepare_graphs_cache_seq.py, plus
valid_combos_map.pkl: every co-sim (placement combo, exact RTT) sorted by RTT.
Used by train_exact_rtt.py for pairwise ranking loss (no random negatives).

For each co-sim dataset with N tasks, emits N graphs:
  - Step s uses queue snapshot after placing optimal replicas for tasks 0..s-1
  - Cross-entropy trains only on task s via ce_label_mask (full optimal y kept on all tasks)
  - Final step (s=N-1) sets ce_label_mask on all tasks for joint regret on parent dataset

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

from non_unique_lib.cache_io import (
    build_valid_combos_map_from_chunked_cache,
    save_valid_combos_map,
)
from non_unique_lib.seq_training_utils import (
    apply_prefix_optimal_labels,
    logit_index_for_placement,
    optimal_combo_from_tasks,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Strictly positive scale for divisions in feature construction (avoid Inf/NaN tensors).
FEATURE_DIV_EPS = 1e-12

TaskPriors = Dict[str, Any]
PlacementCombo = Tuple[Tuple[int, int], ...]
ComboRTT = Tuple[PlacementCombo, float]

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
    require_queue_data: bool = True
    filter_ids: Optional[set] = None  # if set, only build graphs for these dataset IDs
    rtt_only: bool = False  # rebuild RTT chunks only (skip graph generation)
    prune_graphs_no_rtt: bool = False  # drop graphs whose parent dataset has no RTT combos
    enrich_only: bool = False  # write valid_combos_map.pkl from existing RTT chunks


def _default_base_dirs(project_root: Path, merge_datasets: bool) -> List[Path]:
    artifacts_dir = project_root / "simulation_data" / "artifacts" / "run_queue_big"
    if merge_datasets:
        return [
            artifacts_dir / "gnn_datasets_2tasks",
            artifacts_dir / "gnn_datasets_3tasks",
            artifacts_dir / "gnn_datasets_4tasks",
        ]
    return [artifacts_dir / "gnn_datasets_4tasks"]


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
    parser.add_argument("--allow-missing-queue-data", action="store_true")
    parser.add_argument(
        "--filter-ids-file",
        type=Path,
        default=None,
        help=(
            "Path to a text file with one dataset ID per line (e.g. ds_00003). "
            "Only datasets whose directory name appears in the file are included. "
            "Used to oversample high-queue-load subsets."
        ),
    )
    parser.add_argument(
        "--rtt-only",
        action="store_true",
        help="Rebuild RTT hash chunks only (requires existing graph cache in --cache-dir).",
    )
    parser.add_argument(
        "--prune-graphs-no-rtt",
        action="store_true",
        help=(
            "After RTT build, remove graphs whose parent dataset produced zero RTT combos. "
            "Use with --rtt-only to refresh an existing cache."
        ),
    )
    parser.add_argument(
        "--enrich-only",
        action="store_true",
        help=(
            "Skip graph/RTT rebuild; stream existing RTT chunks and write valid_combos_map.pkl "
            "for parent datasets already in the cache."
        ),
    )
    args = parser.parse_args()

    if args.queue_norm_mode == "fixed" and args.queue_norm_factor <= 0:
        parser.error("--queue-norm-factor must be positive (division by zero in queue length normalization).")

    base_dirs = args.base_dirs or _default_base_dirs(args.project_root, args.merge_datasets)
    if args.cache_dir:
        cache_dir = args.cache_dir
    elif args.merge_datasets:
        cache_dir = base_dirs[0].parent / "graphs_cache_merged_2_3_4_tasks_exact_rtt"
    else:
        cache_dir = base_dirs[0].parent / f"graphs_cache_{base_dirs[0].name}_exact_rtt"

    priors_path = args.priors_path or (args.project_root / "data" / "nofs-ids" / "task-types.json")
    cache_dir.mkdir(parents=True, exist_ok=True)

    filter_ids: Optional[set] = None
    if args.filter_ids_file:
        filter_ids = set()
        with open(args.filter_ids_file) as fh:
            for line in fh:
                ds_id = line.strip()
                if ds_id:
                    filter_ids.add(ds_id)
        logger.info("Filter file loaded: %d allowed dataset IDs", len(filter_ids))

    return Config(
        base_dirs=base_dirs,
        cache_dir=cache_dir,
        priors_path=priors_path,
        merge_datasets=args.merge_datasets,
        queue_norm_factor=args.queue_norm_factor,
        queue_norm_mode=args.queue_norm_mode,
        require_queue_data=not args.allow_missing_queue_data,
        filter_ids=filter_ids,
        rtt_only=args.rtt_only,
        prune_graphs_no_rtt=args.prune_graphs_no_rtt,
        enrich_only=args.enrich_only,
    )


@contextmanager
def time_block(description: str):
    start = time.perf_counter()
    yield
    logger.info(f"{description} completed in {time.perf_counter() - start:.2f}s")

# Version for cache invalidation (increment when graph construction logic changes)
CACHE_VERSION = "6.3-exact-rtt"  # sequential graphs + valid_combos_map.pkl for exact RTT ranking training
# - RTT combos consumed lazily from placements/placements.jsonl during training (no LMDB build)
# - Sanitized queue/temporal JSON, safe divisors, finite exec-time priors; asserts finite task/platform features
# - Removed QoS features (qos_deviation, deadline) since co-simulation doesn't capture QoS violations as ground truth
# - Supports datasets where 2+ tasks can be placed on the same (node_id, platform_id)
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

def extract_dataset_to_dataframes(optimal_result_path: Path) -> Dict[str, pd.DataFrame]:
    """Extract a single optimal_result.json into DataFrames."""
    with open(optimal_result_path, "r") as f:
        result = json.load(f)
    
    dataset_id = optimal_result_path.parent.name
    infra_nodes = result.get("config", {}).get("infrastructure", {}).get("nodes", [])
    stats = result.get("stats", {})
    task_results = stats.get("taskResults", [])
    placement_plan = result.get("sample", {}).get("placement_plan", {})

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
            opt_node_id, opt_platform_id = None, None
        
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
        logger.error("Task filtering mismatch: %s != %s", len(task_ids_seen), len(placement_plan_task_ids))
    
    df_tasks = pd.DataFrame(tasks_data)
    
    # PLATFORMS
    platforms_data = []
    node_results = stats.get("nodeResults", [])
    system_state = stats.get("systemStateResults", [{}])[-1] if stats.get("systemStateResults") else {}
    replicas_by_task = system_state.get("replicas", {})
    
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
    
    # METRICS
    best_json_path = optimal_result_path.parent / "best.json"
    best_rtt = None
    if best_json_path.exists():
        with open(best_json_path, "r") as f:
            best_rtt = json.load(f).get("rtt")
    if best_rtt is None:
        best_rtt = sum(tr.get("elapsedTime", 0) for tr in task_results)
    
    df_metrics = pd.DataFrame([{'dataset_id': dataset_id, 'total_rtt': best_rtt}])
    
    return {
        'nodes': df_nodes,
        'tasks': df_tasks,
        'platforms': df_platforms,
        'metrics': df_metrics
    }


def load_extended_state_data(dataset_dir: Path) -> Dict[str, Any]:
    """
    Load extended state data from system_state_captured_unique.json and infrastructure.json.
    For run_non_unique datasets, also tries to load from infrastructure.json queue_distributions.
    Returns dict with:
    - queue_snapshot: Dict mapping "node_name:platform_id" -> queue_length
    - temporal_state: Dict mapping "node_name:platform_id" -> {current_task_remaining, cold_start_remaining, comm_remaining}
    Note: QoS data removed since co-simulation doesn't capture QoS violations as ground truth.
    """
    result = {
        'queue_snapshot': {},
        'temporal_state': {}
    }
    
    # Load queue snapshot from system_state_captured_unique.json (if available)
    ssc_path = dataset_dir / "system_state_captured_unique.json"
    if ssc_path.exists():
        try:
            with open(ssc_path, 'r') as f:
                data = json.load(f)
            
            task_placements = data.get('task_placements', [])
            if task_placements:
                # Use first task's full_queue_snapshot (same for all tasks in batch)
                full_queue_snapshot = task_placements[0].get('full_queue_snapshot', {})
                result['queue_snapshot'] = {k: _queue_length_int(v) for k, v in full_queue_snapshot.items()}
                
                # Extract temporal state from task placements
                # Each task has temporal_state_at_scheduling for its valid replica platforms
                # Merge across all tasks to get complete platform coverage
                merged_temporal_state = {}
                for tp in task_placements:
                    temp_state = tp.get('temporal_state_at_scheduling', {})
                    if isinstance(temp_state, dict):
                        # Merge temporal state (later tasks may overwrite earlier ones for same platform)
                        # This is fine since all tasks in batch see same snapshot, just filtered differently
                        for platform_key, state_dict in temp_state.items():
                            if isinstance(state_dict, dict):
                                # Convert values to float (they should already be floats in JSON)
                                merged_temporal_state[platform_key] = {
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
                
                if merged_temporal_state:
                    result['temporal_state'] = merged_temporal_state
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as e:
            logger.warning("Failed to load extended state from %s: %s", ssc_path, e)

    # Prefer batch full-queue snapshot from optimal_result (phase-1 capture often has 1 task only).
    opt_path = dataset_dir / "optimal_result.json"
    if opt_path.exists():
        try:
            with open(opt_path, "r") as f:
                opt_data = json.load(f)
            task_results = opt_data.get("stats", {}).get("taskResults", [])
            if task_results:
                first_tr = task_results[0]
                full_queue = first_tr.get("fullQueueSnapshot") or first_tr.get("full_queue_snapshot") or {}
                if isinstance(full_queue, dict) and full_queue:
                    batch_queues = {k: _queue_length_int(v) for k, v in full_queue.items()}
                    if len(batch_queues) > len(result["queue_snapshot"]):
                        result["queue_snapshot"] = batch_queues
                if not result["temporal_state"]:
                    merged_temporal_state: Dict[str, Dict[str, float]] = {}
                    for tr in task_results:
                        temp_state = tr.get("temporalStateAtScheduling") or tr.get(
                            "temporal_state_at_scheduling", {}
                        )
                        if not isinstance(temp_state, dict):
                            continue
                        for platform_key, state_dict in temp_state.items():
                            if isinstance(state_dict, dict):
                                merged_temporal_state[platform_key] = {
                                    "current_task_remaining": _safe_float(
                                        state_dict.get("current_task_remaining", 0.0), 0.0
                                    ),
                                    "cold_start_remaining": _safe_float(
                                        state_dict.get("cold_start_remaining", 0.0), 0.0
                                    ),
                                    "comm_remaining": _safe_float(
                                        state_dict.get("comm_remaining", 0.0), 0.0
                                    ),
                                }
                    if merged_temporal_state:
                        result["temporal_state"] = merged_temporal_state
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as e:
            logger.warning("Failed to load extended state from %s: %s", opt_path, e)
    
    # Fallback: Load queue data from infrastructure.json (run_non_unique format)
    if not result['queue_snapshot']:
        infra_path = dataset_dir / "infrastructure.json"
        if infra_path.exists():
            try:
                with open(infra_path, 'r') as f:
                    infra_data = json.load(f)
                
                # Load queue_distributions: task_type -> { "node_name:platform_id": queue_length }
                queue_distributions = infra_data.get('queue_distributions', {})
                
                # Merge queue distributions across all task types into single queue_snapshot
                # If same platform appears in multiple task types, take the maximum queue length
                merged_queues = {}
                for task_type, queues in queue_distributions.items():
                    for key, queue_length in queues.items():
                        # key is "node_name:platform_id"
                        q = _queue_length_int(queue_length)
                        if key not in merged_queues:
                            merged_queues[key] = q
                        else:
                            merged_queues[key] = max(merged_queues[key], q)
                
                result['queue_snapshot'] = merged_queues
            except (OSError, json.JSONDecodeError, TypeError, ValueError) as e:
                logger.warning("Failed to load queue data from %s: %s", infra_path, e)
    
    # Note: QoS data loading removed since co-simulation doesn't capture QoS violations as ground truth
    
    return result


def extract_per_task_scheduling_snapshots(
    optimal_result_path: Path,
) -> Tuple[List[Dict[str, int]], List[Dict[str, Dict[str, float]]]]:
    """
    Per-task queue/temporal at scheduling from optimal_result taskResults (B.6).
    Returns lists ordered by task_id ascending.
    """
    per_task_queues: List[Dict[str, int]] = []
    per_task_temporal: List[Dict[str, Dict[str, float]]] = []
    try:
        with open(optimal_result_path, "r") as f:
            result = json.load(f)
        task_results = result.get("stats", {}).get("taskResults", [])
        ordered = sorted(
            (tr for tr in task_results if tr.get("taskId") is not None),
            key=lambda tr: int(tr["taskId"]),
        )
        for tr in ordered:
            qsnap = tr.get("queueSnapshotAtScheduling") or tr.get(
                "queue_snapshot_at_scheduling", {}
            )
            if isinstance(qsnap, dict) and qsnap:
                per_task_queues.append(
                    {str(k): _queue_length_int(v) for k, v in qsnap.items()}
                )
            else:
                per_task_queues.append({})

            temp = tr.get("temporalStateAtScheduling") or tr.get(
                "temporal_state_at_scheduling", {}
            )
            if isinstance(temp, dict) and temp:
                per_task_temporal.append(
                    {
                        str(pk): {
                            "current_task_remaining": _safe_float(
                                sd.get("current_task_remaining", 0.0), 0.0
                            ),
                            "cold_start_remaining": _safe_float(
                                sd.get("cold_start_remaining", 0.0), 0.0
                            ),
                            "comm_remaining": _safe_float(
                                sd.get("comm_remaining", 0.0), 0.0
                            ),
                        }
                        for pk, sd in temp.items()
                        if isinstance(sd, dict)
                    }
                )
            else:
                per_task_temporal.append({})
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as e:
        logger.warning("Per-task scheduling snapshots unavailable for %s: %s", optimal_result_path, e)
    return per_task_queues, per_task_temporal


def load_all_datasets(
    base_dirs: List[Path],
    require_queue_data: bool = True,
    filter_ids: Optional[set] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Load all datasets from multiple gnn_datasets directories (supports merging).
    
    Args:
        base_dirs: List of Paths to gnn_datasets directories (can be single or multiple)
        require_queue_data: If True, skip datasets without system_state_captured_unique.json
    """
    all_datasets = {}
    skipped_no_queue = 0
    
    for base_dir in base_dirs:
        if not base_dir.exists():
            logger.warning("Directory %s does not exist, skipping", base_dir)
            continue
        
        dataset_dirs = sorted(base_dir.glob("ds_*"))
        logger.info("Loading %s datasets from %s...", len(dataset_dirs), base_dir.name)
        start_time = time.perf_counter()
        
        for dataset_dir in tqdm(dataset_dirs, desc=f"Loading {base_dir.name}", unit="dataset"):
            if filter_ids is not None and dataset_dir.name not in filter_ids:
                continue
            optimal_result_path = dataset_dir / "optimal_result.json"
            if not optimal_result_path.exists():
                continue
            
            # Load extended state data (queue, temporal)
            extended_state = load_extended_state_data(dataset_dir)
            
            # Skip if queue data is required but not available
            if require_queue_data and not extended_state.get('queue_snapshot'):
                skipped_no_queue += 1
                continue
            
            try:
                dataframes = extract_dataset_to_dataframes(optimal_result_path)
                per_task_queues, per_task_temporal = extract_per_task_scheduling_snapshots(
                    optimal_result_path
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
                    'per_task_queue_snapshots': per_task_queues,
                    'per_task_temporal_snapshots': per_task_temporal,
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
    if skipped_no_queue > 0:
        logger.info("  Skipped %s datasets without queue data", skipped_no_queue)
    
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
    (cache_dir / "rtt_combos.lmdb").unlink(missing_ok=True)


def build_and_save_rtt_hash_table_chunked(
    base_dirs: List[Path],
    cache_dir: Path,
    n_jobs: int = 12,
    chunk_size: int = 200_000,
    parse_batch_size: int = 200,
    filter_ids: Optional[set] = None,
) -> Tuple[int, int, set]:
    """
    Build (dataset_id, combo)->rtt hash table in chunks for O(1) lookup at train time.

    Returns:
        (num_datasets_with_combos, total_entries, datasets_with_combos)
    """
    _remove_legacy_rtt_artifacts(cache_dir)

    all_jsonl_files: List[Path] = []
    for base_dir in base_dirs:
        if not base_dir.exists():
            logger.warning("Base directory does not exist, skipping: %s", base_dir)
            continue
        files = sorted(base_dir.glob("ds_*/placements/placements.jsonl"))
        if filter_ids is not None:
            files = [f for f in files if f.parent.parent.name in filter_ids]
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
    return len(datasets_with_combos), total_entries, datasets_with_combos


def prune_graph_cache_to_rtt_datasets(
    cache_dir: Path,
    datasets_with_rtt: set,
) -> Tuple[int, int, int]:
    """
    Drop graphs whose parent dataset has no RTT combos.

    Returns:
        (graphs_before, graphs_after, parent_datasets_removed)
    """
    graphs_path = cache_dir / "graphs.pkl"
    dataset_ids_path = cache_dir / "dataset_ids.pkl"
    optimal_rtt_path = cache_dir / "optimal_rtt.pkl"
    metadata_path = cache_dir / "metadata.json"

    with open(graphs_path, "rb") as f:
        graphs = pickle.load(f)
    with open(dataset_ids_path, "rb") as f:
        dataset_ids = pickle.load(f)
    with open(optimal_rtt_path, "rb") as f:
        optimal_rtt_map = pickle.load(f)

    before = len(graphs)
    parents_before = {ds_id.split(SEQ_DATASET_ID_SEP)[0] for ds_id in dataset_ids}

    kept_graphs = []
    kept_ids: List[str] = []
    for graph, ds_id in zip(graphs, dataset_ids):
        parent_id = ds_id.split(SEQ_DATASET_ID_SEP)[0]
        if parent_id in datasets_with_rtt:
            kept_graphs.append(graph)
            kept_ids.append(ds_id)

    after = len(kept_graphs)
    parents_after = {ds_id.split(SEQ_DATASET_ID_SEP)[0] for ds_id in kept_ids}
    removed_parents = parents_before - parents_after

    kept_id_set = set(kept_ids)
    kept_parent_set = parents_after
    optimal_rtt_map = {
        k: v
        for k, v in optimal_rtt_map.items()
        if k in kept_id_set or k.split(SEQ_DATASET_ID_SEP)[0] in kept_parent_set
    }

    with open(graphs_path, "wb") as f:
        pickle.dump(kept_graphs, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(dataset_ids_path, "wb") as f:
        pickle.dump(kept_ids, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(optimal_rtt_path, "wb") as f:
        pickle.dump(optimal_rtt_map, f, protocol=pickle.HIGHEST_PROTOCOL)

    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        metadata["num_graphs"] = after
        metadata["num_parent_datasets"] = len(parents_after)
        metadata["num_datasets"] = len(parents_after)
        metadata["dataset_ids"] = kept_ids
        metadata["pruned_no_rtt_parent_datasets"] = sorted(removed_parents)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    logger.info(
        "Pruned graph cache: %s -> %s graphs (%s parent datasets removed, no RTT)",
        before,
        after,
        len(removed_parents),
    )
    return before, after, len(removed_parents)


def _write_rtt_parent_ids(cache_dir: Path, datasets_with_rtt: set) -> None:
    out_path = cache_dir / "rtt_parent_dataset_ids.txt"
    parent_names = sorted({ds_id.rsplit("/", 1)[-1] for ds_id in datasets_with_rtt})
    with open(out_path, "w") as f:
        for name in parent_names:
            f.write(f"{name}\n")
    logger.info("Wrote %s parent dataset IDs with RTT to %s", len(parent_names), out_path)


# ============================================================================
# GRAPH CONSTRUCTION (same as main script)
# ============================================================================

TASK_PLATFORM_COMPATIBILITY = {
    'dnn1': ['rpiCpu', 'xavierGpu', 'xavierCpu', 'pynqFpga'],
    'dnn2': ['rpiCpu', 'xavierGpu', 'xavierCpu']
}


def _scheduler_adaptive_queue_norm(queue_values: np.ndarray) -> float:
    """
    Match GNNScheduler adaptive queue normalization:
    - 90th percentile
    - min 1.0
    - cap 100.0
    """
    if queue_values.size == 0:
        return 50.0
    q = np.sort(queue_values.astype(np.float64))
    idx = int(len(q) * 0.9)
    percentile_90 = q[idx] if idx < len(q) else q[-1]
    return float(min(max(1.0, percentile_90), 100.0))

def build_graph(
    df_nodes: pd.DataFrame,
    df_tasks: pd.DataFrame,
    df_platforms: pd.DataFrame,
    task_priors: TaskPriors,
    queue_norm_factor: float,
    queue_norm_mode: str,
    queue_snapshot: Optional[Mapping[str, int]] = None,
    temporal_state: Optional[Mapping[str, Mapping[str, float]]] = None,
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
    src_idx = np.fromiter((first_idx_per_name.get(n, 0) for n in src_names),
                          dtype=np.float64, count=n_tasks)
    src_norm = (src_idx / max(len(df_nodes), 1)).reshape(-1, 1)
    
    task_features = np.concatenate([task_onehot, src_norm], axis=1)
    _require_finite_feature_array("task_features", task_features)
    task_features_tensor = torch.from_numpy(task_features).to(torch.float32)
    
    # PLATFORM FEATURES (13 dims: 5 type + 2 replica + 1 queue + 3 temporal + 2 consolidation)
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
    if queue_norm_mode == "scheduler_adaptive":
        active_queue_norm = _scheduler_adaptive_queue_norm(queue_lengths)
    else:
        active_queue_norm = _safe_positive(float(queue_norm_factor))
    queue_lengths_norm = (queue_lengths / active_queue_norm).reshape(-1, 1)
    
    # TEMPORAL STATE FEATURES (current task remaining times)
    # Since we don't have exact temporal state, we approximate:
    # - If queue > 0: platform is busy, estimate remaining time
    # - Otherwise: platform is idle
    current_task_remaining = np.zeros(n_platforms, dtype=np.float64)
    cold_start_remaining = np.zeros(n_platforms, dtype=np.float64)
    comm_remaining = np.zeros(n_platforms, dtype=np.float64)
    
    if temporal_state:
        for pos in range(n_platforms):
            node_name = str(plat_node_by_pos[pos])
            plat_id = int(plat_ids_arr[pos])
            key = f"{node_name}:{plat_id}"
            temp_state = temporal_state.get(key, {})
            current_task_remaining[pos] = _safe_float(temp_state.get('current_task_remaining', 0.0), 0.0)
            cold_start_remaining[pos] = _safe_float(temp_state.get('cold_start_remaining', 0.0), 0.0)
            comm_remaining[pos] = _safe_float(temp_state.get('comm_remaining', 0.0), 0.0)
    else:
        # Approximate: if queue > 0, estimate some remaining time
        for pos in range(n_platforms):
            if queue_lengths[pos] > 0:
                # Estimate: average execution time for platform type
                plat_type = str(plat_types_by_pos[pos])
                # Get average exec time across task types for this platform
                avg_exec = 0.0
                count = 0
                for task_type in task_types_vocab:
                    task_type_priors = task_priors.get(str(task_type), {})
                    exec_map = task_type_priors.get("executionTime", {})
                    if isinstance(exec_map, dict):
                        exec_time = _safe_float(exec_map.get(plat_type, 0.0), 0.0)
                        if exec_time > 0:
                            avg_exec += exec_time
                            count += 1
                if count > 0:
                    current_task_remaining[pos] = avg_exec / count
                    # Cold start typically much shorter than execution for warm platforms
                    cold_start_remaining[pos] = current_task_remaining[pos] * 0.1
                    comm_remaining[pos] = current_task_remaining[pos] * 0.05
    
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
        
        # Usage ratio: queue_length / target_concurrency
        tc = float(target_concurrencies[pos])
        if math.isfinite(tc) and tc > 0:
            usage_ratios[pos] = queue_lengths[pos] / tc
        else:
            usage_ratios[pos] = 0.0
    
    # Normalize consolidation metrics
    target_concurrency_norm = (target_concurrencies / _safe_positive(20.0)).reshape(-1, 1)
    usage_ratio_norm = (usage_ratios / _safe_positive(5.0)).reshape(-1, 1)
    
    # Concatenate all platform features
    platform_features = np.concatenate([
        plat_onehot,  # 5 dims
        has_dnn1, has_dnn2,  # 2 dims
        queue_lengths_norm,  # 1 dim
        current_task_remaining_norm, cold_start_remaining_norm, comm_remaining_norm,  # 3 dims
        target_concurrency_norm, usage_ratio_norm  # 2 dims
    ], axis=1)
    _require_finite_feature_array("platform_features", platform_features)
    platform_features_tensor = torch.from_numpy(platform_features).to(torch.float32)
    
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
                
                # Warm replica flag
                if task_type == 'dnn1':
                    is_warm = float(has_dnn1_arr[plat_pos])
                elif task_type == 'dnn2':
                    is_warm = float(has_dnn2_arr[plat_pos])
                else:
                    is_warm = 0.0
                
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
    # Per-task mapping from logit index -> (node_id, platform_id) for regret loss and decoding.
    # Use non-underscore attr so DataLoader worker IPC preserves it.
    data.task_logit_to_placement = task_logit_to_placement
    # Backward compatibility for older training scripts.
    data._task_logit_to_placement = task_logit_to_placement
    data.queue_snapshot = dict(queue_snapshot) if queue_snapshot else {}
    data.task_logit_to_queue_key = task_logit_to_queue_key
    
    return data


SEQ_DATASET_ID_SEP = "@seq"


def _queue_key_for_placement(
    node_id: int,
    platform_id: int,
    node_id_to_name: Dict[int, str],
) -> str:
    node_name = node_id_to_name.get(int(node_id), str(node_id))
    return f"{node_name}:{int(platform_id)}"


def _increment_queue_snapshot(
    queue_snapshot: Dict[str, int],
    node_id: int,
    platform_id: int,
    node_id_to_name: Dict[int, str],
) -> None:
    key = _queue_key_for_placement(node_id, platform_id, node_id_to_name)
    queue_snapshot[key] = _queue_length_int(queue_snapshot.get(key, 0)) + 1


def _ce_label_mask_for_seq_step(n_tasks: int, step: int) -> torch.Tensor:
    """True on task indices that contribute to CE at this sequential step."""
    mask = torch.zeros(n_tasks, dtype=torch.bool)
    if step == n_tasks - 1:
        mask[:] = True
    elif 0 <= step < n_tasks:
        mask[step] = True
    return mask


def _attach_ce_label_mask(graph: Data, step: int, n_tasks: int) -> None:
    graph.ce_label_mask = _ce_label_mask_for_seq_step(n_tasks, step)


def _sequential_label_statistics_dict(
    graphs: List[Data],
    task_count_dist: Dict[int, int],
) -> Dict[str, Any]:
    ys = np.concatenate([g.y.numpy() for g in graphs])
    placement_valid = int(np.sum(ys >= 0))
    ce_active = int(sum(int(g.ce_label_mask.sum()) for g in graphs))
    ce_missing = 0
    for g in graphs:
        mask = g.ce_label_mask.numpy()
        y_np = g.y.numpy()
        for t_idx in range(len(y_np)):
            if mask[t_idx] and y_np[t_idx] < 0:
                ce_missing += 1
    return {
        'placement_labels_valid': placement_valid,
        'total_label_slots': len(ys),
        'ce_training_targets': ce_active,
        'ce_targets_missing_placement': ce_missing,
        'graphs_with_no_edges': int(sum(g.edge_index.numel() == 0 for g in graphs)),
        'avg_edges': float(np.mean([g.edge_index.size(1) for g in graphs])),
        'avg_ce_targets_per_graph': float(np.mean([g.ce_label_mask.sum().item() for g in graphs])),
        'task_count_distribution': {str(k): v for k, v in task_count_dist.items()},
    }


def _log_sequential_label_statistics(graphs: List[Data]) -> None:
    ys = np.concatenate([g.y.numpy() for g in graphs])
    placement_valid = int(np.sum(ys >= 0))
    total_slots = len(ys)

    ce_active = 0
    ce_missing_placement = 0
    for g in graphs:
        mask = g.ce_label_mask.numpy()
        ce_active += int(mask.sum())
        y_np = g.y.numpy()
        for t_idx in range(len(y_np)):
            if mask[t_idx] and y_np[t_idx] < 0:
                ce_missing_placement += 1

    logger.info(
        "  Placement labels (y>=0, all tasks): %s / %s (%.1f%%)",
        placement_valid,
        total_slots,
        placement_valid / total_slots * 100 if total_slots else 0.0,
    )
    logger.info(
        "  CE training targets (sequential mask): %s / %s (%.1f%%) — "
        "one task per step 0..N-2, all N on final step",
        ce_active,
        total_slots,
        ce_active / total_slots * 100 if total_slots else 0.0,
    )
    if ce_missing_placement:
        logger.warning(
            "  CE targets missing placement in graph: %s / %s",
            ce_missing_placement,
            ce_active,
        )
    else:
        logger.info("  CE targets missing placement in graph: 0")


def build_sequential_graphs_for_dataset(
    dataset_id: str,
    dataset_dict: Dict[str, Any],
    task_priors: TaskPriors,
    queue_norm_factor: float,
    queue_norm_mode: str,
) -> Tuple[List[Data], List[str], Dict[str, int]]:
    """
    Build N counterfactual graphs for one dataset (N = number of tasks).

    Queue state at step s reflects optimal placements for tasks 0..s-1.

    Returns queue_snapshot_source counts: from_cosim vs roll_forward.
    """
    df_nodes = dataset_dict["nodes"]
    df_tasks = dataset_dict["tasks"].sort_values("task_id").reset_index(drop=True)
    df_platforms = dataset_dict["platforms"]
    n_tasks = len(df_tasks)
    if n_tasks == 0:
        return [], [], {"from_cosim": 0, "roll_forward": 0}

    node_id_to_name = {
        int(row.node_id): str(row.node_name)
        for row in df_nodes.itertuples(index=False)
    }
    base_queue = dataset_dict.get("queue_snapshot") or {}
    initial_queue: Dict[str, int] = {
        str(k): _queue_length_int(v) for k, v in base_queue.items()
    }
    live_queue = dict(initial_queue)
    temporal_state = dataset_dict.get("temporal_state") or {}
    per_task_queues: List[Dict[str, int]] = dataset_dict.get("per_task_queue_snapshots") or []
    per_task_temporal: List[Dict[str, Dict[str, float]]] = (
        dataset_dict.get("per_task_temporal_snapshots") or []
    )

    optimal_combo = optimal_combo_from_tasks(
        [
            (
                int(row.task_id),
                row.optimal_node_id,
                row.optimal_platform_id,
            )
            for row in df_tasks.itertuples(index=False)
        ]
    )

    placement_combos: List[ComboRTT] = []
    dataset_dir = dataset_dict.get("dataset_dir")
    if dataset_dir is not None:
        jsonl_path = Path(dataset_dir) / "placements" / "placements.jsonl"
        if jsonl_path.exists():
            _, placement_combos = _placement_combos_from_jsonl(jsonl_path)

    graphs: List[Data] = []
    seq_dataset_ids: List[str] = []
    queue_source_counts = {"from_cosim": 0, "roll_forward": 0}

    for step in range(n_tasks):
        if (
            per_task_queues
            and step < len(per_task_queues)
            and per_task_queues[step]
        ):
            graph_queue = dict(per_task_queues[step])
            queue_source_counts["from_cosim"] += 1
        else:
            graph_queue = dict(live_queue)
            queue_source_counts["roll_forward"] += 1

        step_temporal = temporal_state
        if per_task_temporal and step < len(per_task_temporal) and per_task_temporal[step]:
            step_temporal = per_task_temporal[step]

        graph = build_graph(
            df_nodes,
            df_tasks,
            df_platforms,
            task_priors=task_priors,
            queue_norm_factor=queue_norm_factor,
            queue_norm_mode=queue_norm_mode,
            queue_snapshot=graph_queue,
            temporal_state=step_temporal,
        )
        _attach_ce_label_mask(graph, step, n_tasks)

        if placement_combos and optimal_combo:
            task_map = getattr(graph, "task_logit_to_placement", None) or getattr(
                graph, "_task_logit_to_placement", {}
            )
            apply_prefix_optimal_labels(
                graph.y,
                graph.ce_label_mask,
                task_map,
                optimal_combo,
                placement_combos,
            )

        seq_id = f"{dataset_id}{SEQ_DATASET_ID_SEP}{step}"
        graph.dataset_id = seq_id
        graph.parent_dataset_id = dataset_id
        graph.seq_step = int(step)
        graph.seq_n_tasks = int(n_tasks)
        graph.initial_queue_snapshot = dict(initial_queue)
        graph.prefix_augment = False

        graphs.append(graph)
        seq_dataset_ids.append(seq_id)

        row = df_tasks.iloc[step]
        opt_node = row.get("optimal_node_id")
        opt_plat = row.get("optimal_platform_id")
        if opt_node is None or opt_plat is None:
            continue
        _increment_queue_snapshot(
            live_queue,
            int(opt_node),
            int(opt_plat),
            node_id_to_name,
        )

    if placement_combos and optimal_combo and len(placement_combos) >= 2:
        sorted_combos = sorted(placement_combos, key=lambda x: x[1])
        second_combo = sorted_combos[1][0]
        if second_combo != optimal_combo:
            aug_queue = dict(initial_queue)
            graph = build_graph(
                df_nodes,
                df_tasks,
                df_platforms,
                task_priors=task_priors,
                queue_norm_factor=queue_norm_factor,
                queue_norm_mode=queue_norm_mode,
                queue_snapshot=aug_queue,
                temporal_state=temporal_state,
            )
            _attach_ce_label_mask(graph, 0, n_tasks)
            task_map = getattr(graph, "task_logit_to_placement", None) or getattr(
                graph, "_task_logit_to_placement", {}
            )
            if len(second_combo) > 0:
                neg_idx = logit_index_for_placement(
                    task_map, 0, second_combo[0][0], second_combo[0][1]
                )
                if neg_idx is not None:
                    graph.y[0] = int(neg_idx)
            aug_id = f"{dataset_id}{SEQ_DATASET_ID_SEP}neg1"
            graph.dataset_id = aug_id
            graph.parent_dataset_id = dataset_id
            graph.seq_step = 0
            graph.seq_n_tasks = int(n_tasks)
            graph.initial_queue_snapshot = dict(initial_queue)
            graph.prefix_augment = True
            graphs.append(graph)
            seq_dataset_ids.append(aug_id)

    return graphs, seq_dataset_ids, queue_source_counts


# ============================================================================
# MAIN SCRIPT
# ============================================================================

def _parent_ids_from_cache(cache_dir: Path) -> set:
    dataset_ids_path = cache_dir / "dataset_ids.pkl"
    if not dataset_ids_path.exists():
        raise FileNotFoundError(f"dataset_ids.pkl not found in {cache_dir}")
    with open(dataset_ids_path, "rb") as f:
        dataset_ids = pickle.load(f)
    return {ds_id.split(SEQ_DATASET_ID_SEP)[0] for ds_id in dataset_ids}


def _enrich_exact_rtt_sidecar(cache_dir: Path) -> None:
    parent_ids = _parent_ids_from_cache(cache_dir)
    valid_combos_map = build_valid_combos_map_from_chunked_cache(cache_dir, parent_ids)
    save_valid_combos_map(cache_dir, valid_combos_map)
    metadata_path = cache_dir / "metadata.json"
    metadata: Dict = {}
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    metadata.update(
        {
            "exact_rtt_training": True,
            "valid_combos_map_rel_path": "valid_combos_map.pkl",
            "num_exact_rtt_parent_datasets": len(valid_combos_map),
            "num_exact_rtt_combo_rows": sum(len(v) for v in valid_combos_map.values()),
        }
    )
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def main():
    config = parse_args()
    script_start_time = time.perf_counter()

    logger.info("=" * 80)
    logger.info("PRE-GENERATING SEQUENTIAL GRAPH CACHE%s", " (MERGED DATASETS)" if config.merge_datasets else "")
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

    if config.enrich_only:
        logger.info("Enrich-only mode: building valid_combos_map.pkl in %s", config.cache_dir)
        _enrich_exact_rtt_sidecar(config.cache_dir)
        logger.info("Enrich-only complete.")
        return

    if config.rtt_only:
        logger.info("RTT-only mode: rebuilding hash chunks in %s", config.cache_dir)
        if config.filter_ids is None:
            logger.error("--rtt-only requires --filter-ids-file (or datasets with no filter)")
            sys.exit(1)
        step1_start = time.perf_counter()
        with time_block("Step 1: Building RTT hash table chunks (filtered)"):
            num_combo_datasets, num_rtt_combos_written, datasets_with_rtt = (
                build_and_save_rtt_hash_table_chunked(
                    config.base_dirs,
                    config.cache_dir,
                    filter_ids=config.filter_ids,
                )
            )
        step1_time = time.perf_counter() - step1_start
        _write_rtt_parent_ids(config.cache_dir, datasets_with_rtt)

        graphs_before = graphs_after = None
        if config.prune_graphs_no_rtt:
            if not graphs_cache_path.exists():
                logger.error("Cannot prune: %s not found", graphs_cache_path)
                sys.exit(1)
            graphs_before, graphs_after, _ = prune_graph_cache_to_rtt_datasets(
                config.cache_dir,
                datasets_with_rtt,
            )

        metadata: Dict = {}
        if metadata_cache_path.exists():
            with open(metadata_cache_path, "r") as f:
                metadata = json.load(f)
        metadata.update(
            {
                "rtt_combos_backend": "hash_table_chunked",
                "rtt_hash_chunks_meta_rel_path": "rtt_chunks_meta.json",
                "num_combo_datasets": num_combo_datasets,
                "num_rtt_combo_rows": num_rtt_combos_written,
                "rtt_built_from_filter_ids": True,
                "rtt_filter_ids_count": len(config.filter_ids),
                "rtt_datasets_with_combos": num_combo_datasets,
                "timing": {
                    **metadata.get("timing", {}),
                    "step1_build_rtt_hash_chunks": step1_time,
                    "total_time": time.perf_counter() - script_start_time,
                },
            }
        )
        if graphs_after is not None:
            metadata["num_graphs"] = graphs_after
        with open(metadata_cache_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(
            "RTT-only complete: %s datasets, %s entries (graphs %s -> %s)",
            num_combo_datasets,
            f"{num_rtt_combos_written:,}",
            graphs_before if graphs_before is not None else "?",
            graphs_after if graphs_after is not None else "?",
        )
        return

    with open(config.priors_path, "r") as f:
        task_priors = json.load(f)
    logger.info("Loaded task priors from %s", config.priors_path)

    step1_start = time.perf_counter()
    with time_block("Step 1: Building RTT hash table chunks"):
        num_combo_datasets, num_rtt_combos_written, datasets_with_rtt = (
            build_and_save_rtt_hash_table_chunked(
                config.base_dirs,
                config.cache_dir,
                filter_ids=config.filter_ids,
            )
        )
    step1_time = time.perf_counter() - step1_start
    _write_rtt_parent_ids(config.cache_dir, datasets_with_rtt)

    step2_start = time.perf_counter()
    with time_block("Step 2: Loading datasets"):
        all_datasets = load_all_datasets(
            config.base_dirs,
            require_queue_data=config.require_queue_data,
            filter_ids=config.filter_ids,
        )
    step2_time = time.perf_counter() - step2_start

    if len(all_datasets) == 0:
        logger.error("No datasets loaded")
        sys.exit(1)

    analysis_export_path = config.cache_dir / "task_metrics_analysis.csv"
    export_task_metrics_for_analysis(all_datasets, analysis_export_path)

    step3_start = time.perf_counter()
    with time_block("Step 3: Building optimal RTT map"):
        parent_optimal_rtt_map = {
            ds_id: float(ds_dict['metrics']['total_rtt'].iloc[0]) if 'metrics' in ds_dict and not ds_dict['metrics'].empty else 0.0
            for ds_id, ds_dict in all_datasets.items()
        }
    step3_time = time.perf_counter() - step3_start
    logger.info("Built optimal RTT map for %s parent datasets", len(parent_optimal_rtt_map))

    step4_start = time.perf_counter()
    graphs = []
    dataset_ids = []
    parent_dataset_ids: List[str] = []
    queue_source_totals = {"from_cosim": 0, "roll_forward": 0}
    with time_block("Step 4: Building sequential counterfactual graphs"):
        for dataset_id, dataset_dict in tqdm(
            all_datasets.items(),
            desc="Building seq graphs",
            unit="dataset",
        ):
            try:
                seq_graphs, seq_ids, qsrc = build_sequential_graphs_for_dataset(
                    dataset_id,
                    dataset_dict,
                    task_priors=task_priors,
                    queue_norm_factor=config.queue_norm_factor,
                    queue_norm_mode=config.queue_norm_mode,
                )
                graphs.extend(seq_graphs)
                dataset_ids.extend(seq_ids)
                parent_dataset_ids.extend([dataset_id] * len(seq_ids))
                for k in queue_source_totals:
                    queue_source_totals[k] += int(qsrc.get(k, 0))
            except Exception as e:
                tqdm.write(f"  Error building seq graphs for {dataset_id}: {e}")
    step4_time = time.perf_counter() - step4_start

    optimal_rtt_map: Dict[str, float] = dict(parent_optimal_rtt_map)
    for seq_id, parent_id in zip(dataset_ids, parent_dataset_ids):
        optimal_rtt_map[seq_id] = float(parent_optimal_rtt_map.get(parent_id, 0.0))

    stats_start = time.perf_counter()
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
    _log_sequential_label_statistics(graphs)
    q_total = queue_source_totals["from_cosim"] + queue_source_totals["roll_forward"]
    if q_total:
        logger.info(
            "  Queue snapshot source: co-sim per-task %s (%.1f%%), roll-forward %s (%.1f%%)",
            queue_source_totals["from_cosim"],
            100.0 * queue_source_totals["from_cosim"] / q_total,
            queue_source_totals["roll_forward"],
            100.0 * queue_source_totals["roll_forward"] / q_total,
        )
    logger.info("  Graphs with no edges: %s / %s", sum([g.edge_index.numel() == 0 for g in graphs]), len(graphs))
    logger.info("  Avg edges per graph: %.1f", np.mean([g.edge_index.size(1) for g in graphs]))
    logger.info("  Avg valid tasks per graph: %.2f", np.mean([(g.y >= 0).sum().item() for g in graphs]))
    if len(all_datasets) > 0:
        logger.info(
            "  Avg seq graphs per parent dataset: %.2f",
            len(graphs) / len(all_datasets),
        )
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
        'sequential_counterfactual': True,
        'exact_rtt_training': True,
        'valid_combos_map_rel_path': 'valid_combos_map.pkl',
        'seq_dataset_id_sep': SEQ_DATASET_ID_SEP,
        'num_parent_datasets': len(all_datasets),
        'merged_datasets': config.merge_datasets,
        'base_dirs': [str(bd) for bd in config.base_dirs],
        'num_graphs': len(graphs),
        'rtt_combos_backend': 'hash_table_chunked',
        'rtt_hash_chunks_meta_rel_path': 'rtt_chunks_meta.json',
        'num_combo_datasets': num_combo_datasets,
        'num_rtt_combo_rows': num_rtt_combos_written,
        'num_datasets': len(all_datasets),
        'dataset_ids': dataset_ids,
        'statistics': _sequential_label_statistics_dict(graphs, task_count_dist),
        'queue_snapshot_sources': queue_source_totals,
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

    parent_ids = {ds_id.split(SEQ_DATASET_ID_SEP)[0] for ds_id in dataset_ids}
    valid_combos_map = build_valid_combos_map_from_chunked_cache(config.cache_dir, parent_ids)
    save_valid_combos_map(config.cache_dir, valid_combos_map)
    metadata["num_exact_rtt_parent_datasets"] = len(valid_combos_map)
    metadata["num_exact_rtt_combo_rows"] = sum(len(v) for v in valid_combos_map.values())
    with open(metadata_cache_path, 'w') as f:
        json.dump(metadata, f, indent=2)

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