#!/usr/bin/env python3
from __future__ import annotations

"""
Pre-generate and cache graphs for GNN training (NON-UNIQUE VERSION) — **RAM-oriented**.

Stores every dataset's ``[(combo, rtt), ...]`` **on each graph
object** inside ``graphs.pkl`` so ``train.py`` loads everything in one ``pickle.load`` and never
touches LMDB (fastest for training at the cost of a larger single pickle and higher peak RAM).

Optional **staging** (``--staging-dir``, ``--ram-staging``, env): write pickles to **node-local RAM
or fast scratch** first, then copy the bundle to ``--cache-dir`` (e.g. ``/share/...``). The heavy
``graphs.pkl`` (with embedded RTT lists) is built off slow shared disk.

**dataLAB-style job:** create ``/scratch/${USER}_${SLURM_JOB_ID}``, then e.g.
``export HEROSIM_PREPARE_SCRATCH=/scratch/...`` and run with ``--ram-staging``, or pass
``--staging-dir /scratch/.../prep``.

NON-UNIQUE PLACEMENTS:
- Supports datasets where multiple tasks can be placed on the same replica
- Creates edges between tasks and all compatible platforms (no uniqueness constraint)
- Compatible with gnn_datasets_2tasks, gnn_datasets_3tasks, and gnn_datasets_4tasks
- Includes system state, temporal features, queue info, and consolidation metrics
"""

import argparse
import concurrent.futures
import shutil
import json
import logging
import math
import os
import pickle
import random
import sys
import time
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
    work_dir: Path
    priors_path: Path
    merge_datasets: bool = False
    queue_norm_factor: float = 50.0
    require_queue_data: bool = True
    rtt_workers: int = 1
    rtt_batch_size: int = 0


def _default_base_dirs(project_root: Path, merge_datasets: bool) -> List[Path]:
    artifacts_dir = project_root / "simulation_data" / "artifacts" / "run_queue_big"
    if merge_datasets:
        return [
            artifacts_dir / "gnn_datasets_2tasks",
            artifacts_dir / "gnn_datasets_3tasks",
            artifacts_dir / "gnn_datasets_4tasks",
        ]
    return [artifacts_dir / "gnn_datasets_4tasks_overnight_260422"]


def _resolve_ram_staging_path() -> Path:
    """
    Fast local path for intermediate pickles (RAM-ish tmpfs or node scratch).
    Not used for final cache; publish step copies to --cache-dir.
    """
    slurm_tmp = os.environ.get("SLURM_TMPDIR")
    if slurm_tmp:
        job = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
        return Path(slurm_tmp) / f"herosim_prepare_{job}"
    scratch = os.environ.get("HEROSIM_PREPARE_SCRATCH")
    if scratch:
        base = Path(scratch)
        base.mkdir(parents=True, exist_ok=True)
        return base / f"herosim_build_{os.getpid()}"
    shm = Path(f"/dev/shm/herosim_prepare_{os.getuid()}_{os.getpid()}")
    shm.mkdir(parents=True, exist_ok=True)
    return shm


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Pre-generate and cache GNN graphs.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--merge-datasets", action="store_true")
    parser.add_argument("--base-dirs", nargs="+", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=None,
        help=(
            "Build pickles here first (tmpfs / $SLURM_TMPDIR / /scratch), then copy to --cache-dir. "
            "Avoids slow shared-disk writes during large pickle.dumps. "
            "Alternative: set HEROSIM_PREPARE_STAGING_DIR."
        ),
    )
    parser.add_argument(
        "--ram-staging",
        action="store_true",
        help=(
            "Pick a fast staging directory automatically unless --staging-dir / "
            "HEROSIM_PREPARE_STAGING_DIR is set. Order: HEROSIM_PREPARE_STAGING_DIR, "
            "SLURM_TMPDIR/herosim_prepare_${SLURM_JOB_ID|pid}, "
            "HEROSIM_PREPARE_SCRATCH/herosim_build_${pid}, "
            "/dev/shm/herosim_prepare_${uid}_${pid}."
        ),
    )
    parser.add_argument("--priors-path", type=Path)
    parser.add_argument("--queue-norm-factor", type=float, default=50.0)
    parser.add_argument("--allow-missing-queue-data", action="store_true")
    parser.add_argument(
        "--rtt-workers",
        type=int,
        default=int(os.environ.get("HEROSIM_RTT_WORKERS", os.environ.get("SLURM_CPUS_PER_TASK", "1"))),
        help=(
            "Worker processes for parsing placements JSONL into the in-RAM RTT map. "
            "Defaults to HEROSIM_RTT_WORKERS, then SLURM_CPUS_PER_TASK, then 1."
        ),
    )
    parser.add_argument(
        "--rtt-batch-size",
        type=int,
        default=int(os.environ.get("HEROSIM_RTT_BATCH_SIZE", "0")),
        help=(
            "Number of JSONL files submitted to RTT workers at once. "
            "0 means auto: max(2 * --rtt-workers, 1). Higher values can use more RAM."
        ),
    )
    args = parser.parse_args()

    if args.queue_norm_factor <= 0:
        parser.error("--queue-norm-factor must be positive (division by zero in queue length normalization).")
    if args.rtt_workers <= 0:
        parser.error("--rtt-workers must be positive")
    if args.rtt_batch_size < 0:
        parser.error("--rtt-batch-size must be >= 0")

    base_dirs = args.base_dirs or _default_base_dirs(args.project_root, args.merge_datasets)
    if args.cache_dir:
        cache_dir = args.cache_dir
    elif args.merge_datasets:
        cache_dir = base_dirs[0].parent / "graphs_cache_merged_2_3_4_tasks_embedded"
    else:
        cache_dir = base_dirs[0].parent / f"graphs_cache_{base_dirs[0].name}_embedded"

    priors_path = args.priors_path or (args.project_root / "data" / "nofs-ids" / "task-types.json")

    staging: Optional[Path] = args.staging_dir
    if staging is None and os.environ.get("HEROSIM_PREPARE_STAGING_DIR"):
        staging = Path(os.environ["HEROSIM_PREPARE_STAGING_DIR"])
    elif staging is None and args.ram_staging:
        staging = _resolve_ram_staging_path()
    cache_dir_resolved = cache_dir.expanduser().resolve()
    if staging is not None:
        staging_resolved = staging.expanduser().resolve()
        work_dir = staging_resolved if staging_resolved != cache_dir_resolved else cache_dir_resolved
    else:
        work_dir = cache_dir_resolved

    cache_dir_resolved.mkdir(parents=True, exist_ok=True)
    if work_dir != cache_dir_resolved:
        work_dir.mkdir(parents=True, exist_ok=True)

    return Config(
        base_dirs=base_dirs,
        cache_dir=cache_dir_resolved,
        work_dir=work_dir,
        priors_path=priors_path,
        merge_datasets=args.merge_datasets,
        queue_norm_factor=args.queue_norm_factor,
        require_queue_data=not args.allow_missing_queue_data,
        rtt_workers=args.rtt_workers,
        rtt_batch_size=args.rtt_batch_size,
    )


@contextmanager
def time_block(description: str):
    start = time.perf_counter()
    yield
    logger.info(f"{description} completed in {time.perf_counter() - start:.2f}s")

# Version for cache invalidation (increment when graph construction logic changes)
CACHE_VERSION = "5.1"  # Same graph construction as cache 5.0; RTT combos embedded in graphs.pkl (RAM train path)
# - RTT valid combos stored on each Data object (valid_combos); no rtt_combos.lmdb in this script
# - Optional --staging-dir / HEROSIM_PREPARE_STAGING_DIR: build on fast/local RAM, copy to --cache-dir
# - Sanitized queue/temporal JSON, safe divisors, finite exec-time priors; asserts finite task/platform features
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


def load_all_datasets(
    base_dirs: List[Path], require_queue_data: bool = True
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
                # Use unique key: base_dir_name/dataset_name to avoid collisions
                unique_key = f"{base_dir.name}/{dataset_dir.name}"
                all_datasets[unique_key] = {
                    **dataframes,
                    'dataset_dir': dataset_dir,
                    'source_dir': base_dir.name,  # Track which directory this came from
                    'num_tasks': len(dataframes['tasks']),  # Track task count
                    'queue_snapshot': extended_state.get('queue_snapshot', {}),
                    'temporal_state': extended_state.get('temporal_state', {})
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
# RTT COMBOS (in-memory map → attached to each graph; no LMDB in this script)
# ============================================================================


def _remove_legacy_rtt_artifacts(cache_dir: Path) -> None:
    """Drop SQLite / chunked pickle RTT stores and any prior LMDB env directory."""
    for stale_chunk in cache_dir.glob("rtt_chunk_*.pkl"):
        stale_chunk.unlink(missing_ok=True)
    (cache_dir / "rtt_chunks_meta.json").unlink(missing_ok=True)
    (cache_dir / "placement_rtt_hash_table.sqlite3").unlink(missing_ok=True)
    (cache_dir / "placement_rtt_hash_table.pkl").unlink(missing_ok=True)
    lmdb_dir = cache_dir / "rtt_combos.lmdb"
    if lmdb_dir.exists():
        shutil.rmtree(lmdb_dir, ignore_errors=True)


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


def _add_rtt_result(
    rtt_map: Dict[str, List[Tuple[PlacementCombo, float]]],
    dataset_id: str,
    cleaned: List[Tuple[PlacementCombo, float]],
) -> Tuple[int, int]:
    """Attach one parsed dataset to the RTT map; return dataset/row increments."""
    if not cleaned:
        return 0, 0
    rtt_map[dataset_id] = cleaned
    return 1, len(cleaned)


def _auto_rtt_batch_size(num_files: int, workers: int, requested: int) -> int:
    """Bound worker result buffering while still giving each worker enough files."""
    if num_files <= 0:
        return 1
    if requested > 0:
        return min(requested, num_files)
    return min(max(workers * 2, 1), num_files)


def build_rtt_combos_map(
    base_dirs: List[Path], work_dir: Path, workers: int = 1, batch_size: int = 0
) -> Tuple[Dict[str, List[Tuple[PlacementCombo, float]]], int, int]:
    """
    Parse all placements.jsonl under base_dirs; return a map dataset_id -> [(combo, rtt), ...].

    Returns:
        (rtt_map, num_datasets_with_combos, total_combo_rows)
    """
    _remove_legacy_rtt_artifacts(work_dir)

    jsonl_files: List[Path] = []
    for base_dir in base_dirs:
        if not base_dir.exists():
            logger.warning("Base directory does not exist, skipping: %s", base_dir)
            continue
        files = sorted(base_dir.glob("ds_*/placements/placements.jsonl"))
        jsonl_files.extend(files)
        logger.info("Found %s JSONL files in %s", len(files), base_dir.name)

    rtt_map: Dict[str, List[Tuple[PlacementCombo, float]]] = {}
    total_datasets = 0
    total_combos = 0
    if not jsonl_files:
        logger.warning("No placements.jsonl files found under configured base directories")
    elif workers <= 1:
        logger.info("Parsing RTT combos sequentially")
        for jsonl_path in tqdm(jsonl_files, desc="Loading RTT combos (RAM)"):
            dataset_id, cleaned = _placement_combos_from_jsonl(jsonl_path)
            ds_inc, combo_inc = _add_rtt_result(rtt_map, dataset_id, cleaned)
            total_datasets += ds_inc
            total_combos += combo_inc
    else:
        workers = min(workers, len(jsonl_files))
        effective_batch_size = _auto_rtt_batch_size(len(jsonl_files), workers, batch_size)
        logger.info(
            "Parsing RTT combos with %s worker processes (batch size %s)",
            workers,
            effective_batch_size,
        )
        batches = range(0, len(jsonl_files), effective_batch_size)
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            for start in tqdm(batches, desc="Loading RTT combos (RAM batches)", unit="batch"):
                batch = jsonl_files[start:start + effective_batch_size]
                futures = [executor.submit(_placement_combos_from_jsonl, p) for p in batch]
                for future in concurrent.futures.as_completed(futures):
                    dataset_id, cleaned = future.result()
                    ds_inc, combo_inc = _add_rtt_result(rtt_map, dataset_id, cleaned)
                    total_datasets += ds_inc
                    total_combos += combo_inc

    logger.info(
        "Loaded %s datasets (%s total (combo, rtt) rows) into RAM for graph attachment",
        total_datasets,
        f"{total_combos:,}",
    )
    if total_datasets == 0:
        logger.warning(
            "RTT combo map is empty (no placements.jsonl rows parsed). "
            "Regret negatives will have no combos until JSONL paths are fixed."
        )
    return rtt_map, total_datasets, total_combos


def _format_bytes(n_bytes: int) -> str:
    value = float(n_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TiB"


class _ProgressWriter:
    """File-like wrapper that logs pickle write progress without changing pickle format."""

    def __init__(self, f: Any, label: str, path: Path, log_seconds: float = 15.0, log_bytes: int = 1 << 30) -> None:
        self._f = f
        self._label = label
        self._path = path
        self._log_seconds = log_seconds
        self._log_bytes = log_bytes
        self._written = 0
        self._last_log_time = time.perf_counter()
        self._last_log_bytes = 0
        logger.info("  Writing %s to %s ...", label, path)

    def write(self, data: bytes) -> int:
        n = self._f.write(data)
        self._written += n
        now = time.perf_counter()
        if (
            now - self._last_log_time >= self._log_seconds
            or self._written - self._last_log_bytes >= self._log_bytes
        ):
            logger.info("  Writing %s: %s written", self._label, _format_bytes(self._written))
            self._last_log_time = now
            self._last_log_bytes = self._written
        return n

    def flush(self) -> None:
        self._f.flush()


def _pickle_dump_with_progress(obj: Any, path: Path, label: str) -> None:
    start = time.perf_counter()
    with open(path, "wb") as f:
        writer = _ProgressWriter(f, label, path)
        pickle.dump(obj, writer, protocol=pickle.HIGHEST_PROTOCOL)
        writer.flush()
    size = path.stat().st_size if path.exists() else 0
    logger.info(
        "  Finished %s: %s written to %s (%.2fs)",
        label,
        _format_bytes(size),
        path,
        time.perf_counter() - start,
    )


def _copy_file_with_progress(src: Path, dst: Path, label: str, chunk_size: int = 64 * 1024 * 1024) -> None:
    total = src.stat().st_size
    copied = 0
    start = time.perf_counter()
    last_log_time = start
    last_log_bytes = 0
    logger.info("  Publishing %s: %s -> %s (%s)", label, src, dst, _format_bytes(total))
    with open(src, "rb") as rf, open(dst, "wb") as wf:
        while True:
            chunk = rf.read(chunk_size)
            if not chunk:
                break
            wf.write(chunk)
            copied += len(chunk)
            now = time.perf_counter()
            if now - last_log_time >= 15.0 or copied - last_log_bytes >= (1 << 30):
                pct = (copied / total * 100.0) if total else 100.0
                logger.info(
                    "  Publishing %s: %s / %s (%.1f%%)",
                    label,
                    _format_bytes(copied),
                    _format_bytes(total),
                    pct,
                )
                last_log_time = now
                last_log_bytes = copied
    shutil.copystat(src, dst)
    logger.info(
        "  Finished publishing %s: %s copied (%.2fs)",
        label,
        _format_bytes(copied),
        time.perf_counter() - start,
    )


def _publish_cache_to_final(work_dir: Path, final_dir: Path) -> None:
    """Copy prepared artifacts to the persistent cache directory; drop stale LMDB if any."""
    final_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "graphs.pkl",
        "dataset_ids.pkl",
        "optimal_rtt.pkl",
        "metadata.json",
        "task_metrics_analysis.csv",
    ):
        src = work_dir / name
        if src.exists():
            _copy_file_with_progress(src, final_dir / name, name)
    stale_lmdb = final_dir / "rtt_combos.lmdb"
    if stale_lmdb.exists():
        shutil.rmtree(stale_lmdb, ignore_errors=True)
        logger.info("Removed stale %s (embedded RTT cache)", stale_lmdb)


# ============================================================================
# GRAPH CONSTRUCTION (same as main script)
# ============================================================================

TASK_PLATFORM_COMPATIBILITY = {
    'dnn1': ['rpiCpu', 'xavierGpu', 'xavierCpu', 'pynqFpga'],
    'dnn2': ['rpiCpu', 'xavierGpu', 'xavierCpu']
}

def build_graph(
    df_nodes: pd.DataFrame,
    df_tasks: pd.DataFrame,
    df_platforms: pd.DataFrame,
    task_priors: TaskPriors,
    queue_norm_factor: float,
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
    
    # Normalize queue lengths (queue_norm_factor validated > 0 in CLI; still guard here)
    queue_lengths_norm = (queue_lengths / _safe_positive(float(queue_norm_factor))).reshape(-1, 1)
    
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
    # Per-task mapping from logit index -> (node_id, platform_id) for regret loss and decoding
    data._task_logit_to_placement = task_logit_to_placement
    
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
    if config.work_dir.resolve() != config.cache_dir.resolve():
        logger.info(
            "Work directory (staging): %s  →  publish to cache-dir: %s",
            config.work_dir,
            config.cache_dir,
        )
    else:
        logger.info("Work directory: %s", config.work_dir)

    graphs_cache_path = config.work_dir / "graphs.pkl"
    dataset_ids_cache_path = config.work_dir / "dataset_ids.pkl"
    optimal_rtt_cache_path = config.work_dir / "optimal_rtt.pkl"
    metadata_cache_path = config.work_dir / "metadata.json"

    with open(config.priors_path, "r") as f:
        task_priors = json.load(f)
    logger.info("Loaded task priors from %s", config.priors_path)

    step1_start = time.perf_counter()
    with time_block("Step 1: Building RTT combos map (RAM)"):
        rtt_map, num_rtt_datasets, num_rtt_combos_rows = build_rtt_combos_map(
            config.base_dirs,
            config.work_dir,
            workers=config.rtt_workers,
            batch_size=config.rtt_batch_size,
        )
    step1_time = time.perf_counter() - step1_start

    step2_start = time.perf_counter()
    with time_block("Step 2: Loading datasets"):
        all_datasets = load_all_datasets(config.base_dirs, require_queue_data=config.require_queue_data)
    step2_time = time.perf_counter() - step2_start

    if len(all_datasets) == 0:
        logger.error("No datasets loaded")
        sys.exit(1)

    analysis_export_path = config.work_dir / "task_metrics_analysis.csv"
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
    graphs = []
    dataset_ids = []
    with time_block("Step 4: Building graphs"):
        for dataset_id, dataset_dict in tqdm(all_datasets.items(), desc="Building graphs", unit="dataset"):
            try:
                graph = build_graph(
                    dataset_dict['nodes'],
                    dataset_dict['tasks'],
                    dataset_dict['platforms'],
                    task_priors=task_priors,
                    queue_norm_factor=config.queue_norm_factor,
                    queue_snapshot=dataset_dict.get('queue_snapshot', {}),
                    temporal_state=dataset_dict.get('temporal_state', {})
                )
                graph.dataset_id = dataset_id
                graph.valid_combos = rtt_map.get(dataset_id, [])
                graphs.append(graph)
                dataset_ids.append(dataset_id)
            except Exception as e:
                tqdm.write(f"  Error building graph for {dataset_id}: {e}")
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

    _pickle_dump_with_progress(graphs, graphs_cache_path, f"graphs.pkl ({len(graphs)} graphs)")

    _pickle_dump_with_progress(dataset_ids, dataset_ids_cache_path, "dataset_ids.pkl")

    logger.info(
        "  Embedded RTT combos on graphs: %s datasets, %s rows",
        num_rtt_datasets,
        f"{num_rtt_combos_rows:,}",
    )

    _pickle_dump_with_progress(
        optimal_rtt_map,
        optimal_rtt_cache_path,
        f"optimal_rtt.pkl ({len(optimal_rtt_map)} datasets)",
    )

    metadata = {
        'version': CACHE_VERSION,
        'merged_datasets': config.merge_datasets,
        'base_dirs': [str(bd) for bd in config.base_dirs],
        'num_graphs': len(graphs),
        'rtt_combos_backend': 'embedded_in_graphs',
        'num_rtt_datasets': num_rtt_datasets,
        'num_rtt_combo_rows': num_rtt_combos_rows,
        'num_datasets': len(all_datasets),
        'dataset_ids': dataset_ids,
        'statistics': {
            'valid_labels': int(np.sum(ys >= 0)),
            'total_labels': len(ys),
            'graphs_with_no_edges': int(sum([g.edge_index.numel() == 0 for g in graphs])),
            'avg_edges': float(np.mean([g.edge_index.size(1) for g in graphs])),
            'avg_valid_tasks': float(np.mean([(g.y >= 0).sum().item() for g in graphs])),
            'task_count_distribution': {str(k): v for k, v in task_count_dist.items()},
        },
        'timing': {
            'step1_build_rtt_map': step1_time,
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

    if config.work_dir.resolve() != config.cache_dir.resolve():
        logger.info("Publishing cache to %s", config.cache_dir)
        _publish_cache_to_final(config.work_dir, config.cache_dir)

    step5_time = time.perf_counter() - step5_start
    total_time = time.perf_counter() - script_start_time

    out_graphs = config.cache_dir / "graphs.pkl"
    optimal_out = config.cache_dir / "optimal_rtt.pkl"
    graphs_size = out_graphs.stat().st_size / (1024 * 1024)
    optimal_rtt_size = optimal_out.stat().st_size / (1024 * 1024)

    logger.info("\n" + "=" * 80)
    logger.info("CACHE GENERATION COMPLETE!")
    logger.info("=" * 80)
    logger.info("Cache directory (train with --cache-dir): %s", config.cache_dir)
    logger.info("Graphs cache (includes embedded RTT): %s (%.2f MB)", out_graphs, graphs_size)
    logger.info("Optimal RTT cache: %s (%.2f MB)", optimal_out, optimal_rtt_size)
    logger.info("Total primary cache size: %.2f MB", graphs_size + optimal_rtt_size)
    logger.info("Cache version: %s", CACHE_VERSION)
    logger.info("Timing Summary:")
    logger.info("  Step 1 - Build RTT map (RAM):   %7.2fs", step1_time)
    logger.info("  Step 2 - Load datasets:         %7.2fs", step2_time)
    logger.info("  Step 3 - Build helper maps:     %7.2fs", step3_time)
    logger.info("  Step 4 - Build graphs:          %7.2fs", step4_time)
    logger.info("  Step 5 - Save + publish:        %7.2fs", step5_time)
    logger.info("  Total time:                     %7.2fs", total_time)


if __name__ == "__main__":
    main()