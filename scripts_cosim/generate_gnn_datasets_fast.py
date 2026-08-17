#!/usr/bin/env python3
"""
Optimized GNN Dataset Generation Script

MANDATORY OUTPUT: placements/placements.jsonl
  Every successful dataset MUST persist the full brute-force sweep:
  one JSONL line per (placement_plan, rtt). prepare_graphs_cache* builds
  rtt_chunk_*.pkl from this file for counterfactual RTT / near-RTT training.

  refresh_optimal_full_stats.py --repair does NOT create JSONL.
  Repair + recache alone leaves ~40% of CE rows without counterfactual RTT lookup.

  Never delete .bf_scratch until placements/placements.jsonl exists and is non-empty.
  --resume must not skip datasets that have best.json but lack placements.jsonl.
  See memory/placements_jsonl_required.md

This Python script replaces generate_gnn_datasets.sh with significant performance improvements:
1. Eliminates jq overhead (native Python JSON handling)
2. Eliminates subprocess spawning for infrastructure generation
3. Single Python process for all operations
4. Supports --quiet mode for faster execution
5. Uses orjson for faster JSON serialization when available

Usage:
    python scripts_cosim/generate_gnn_datasets_fast.py [--quiet] [--max-datasets N] [--workers N]

Example:
    # Generate up to 100 datasets with quiet mode
    python scripts_cosim/generate_gnn_datasets_fast.py --quiet --max-datasets 100
"""

import argparse
import json
import logging
import os
import random
import shutil
import signal
import subprocess
import sys
import time
import re
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Try to use orjson for faster JSON
try:
    import orjson
    
    def _convert_keys_to_str(obj):
        """Recursively convert dict keys to strings for orjson compatibility."""
        if isinstance(obj, dict):
            return {str(k): _convert_keys_to_str(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_convert_keys_to_str(item) for item in obj]
        elif isinstance(obj, tuple):
            return [_convert_keys_to_str(item) for item in obj]
        return obj
    
    def json_dumps(obj):
        return orjson.dumps(_convert_keys_to_str(obj)).decode('utf-8')
    def json_dumps_pretty(obj):
        return orjson.dumps(_convert_keys_to_str(obj), option=orjson.OPT_INDENT_2).decode('utf-8')
    HAS_ORJSON = True
except ImportError:
    def json_dumps(obj):
        return json.dumps(obj, separators=(',', ':'))
    def json_dumps_pretty(obj):
        return json.dumps(obj, indent=2)
    HAS_ORJSON = False

from src.generate_infrastructure import generate_deterministic_infrastructure
from src.executecosimulation import execute_brute_force_optimized, load_simulation_inputs
from src.sample_loader import load_primary_sample_and_mapping

# Timeout for brute-force simulation (1 hour per dataset)
SIMULATION_TIMEOUT = 900  # seconds
# Skip datasets with excessive placement combinations (OOM guard)
MAX_PLACEMENT_COMBINATIONS_SKIP_DEFAULT = 250000


# =============================================================================
# CONFIGURATION GRIDS
# =============================================================================

GridPreset = Dict[str, Any]

# warmth_v2: fixed conn=0.50, moderate-to-heavy replicas/queues (500 ds).
WARMTH_V2_GRID: GridPreset = {
    "connection_probabilities": [0.50],
    "replica_configs": [
        (1, 2, 0.0, 0.0),
        (2, 2, 0.0, 0.0),
        (1, 3, 0.3, 0.5),
        (2, 3, 0.3, 0.5),
        (2, 2, 0.5, 0.7),
    ],
    "queue_distributions": [
        ("pois16", "poisson", 16, 0, 0, 42, 1),
        ("norm22", "normal", 22, 7, 0, 56, 1),
        ("pois28", "poisson", 28, 0, 0, 72, 1),
        ("norm35", "normal", 35, 11, 0, 96, 1),
        ("uniform20_80", "uniform", 20, 80, 0, 120, 1),
    ],
    "seeds": list(range(101, 121)),
    "default_output_subdir": "gnn_datasets_4tasks_1060_warmth_v2",
}

# sparse_warmth_v2 Option A: sparse topology + light replicas + low/mid queues (351 ds).
SPARSE_WARMTH_V2_GRID: GridPreset = {
    "connection_probabilities": [0.25, 0.30, 0.35],
    "replica_configs": [
        (1, 1, 0.0, 0.0),
        (1, 2, 0.0, 0.0),
        (2, 2, 0.0, 0.0),
    ],
    "queue_distributions": [
        ("pois12", "poisson", 12, 0, 0, 24, 1),
        ("norm16", "normal", 16, 5, 0, 32, 1),
        ("pois16", "poisson", 16, 0, 0, 42, 1),
    ],
    "seeds": list(range(121, 134)),
    "default_output_subdir": "gnn_datasets_4tasks_sparse_warmth_v2",
}

# skew_warmth_v2: degree_skewed_core hub topology + 5/30ms asymmetric latency (288 ds).
SKEW_WARMTH_V2_GRID: GridPreset = {
    "topology_type": "degree_skewed_core",
    "k_core_values": [4, 6, 8],
    "hub_seeker_fractions": [0.35, 0.50, 0.65],
    "latency_core_ms": 5,
    "latency_periphery_ms": 30,
    "connection_probabilities": [],
    "replica_configs": [
        (1, 2, 0.0, 0.0),
        (2, 2, 0.0, 0.0),
    ],
    "queue_distributions": [
        ("pois16", "poisson", 16, 0, 0, 42, 1),
        ("pois28", "poisson", 28, 0, 0, 72, 1),
    ],
    "seeds": list(range(141, 149)),
    "default_output_subdir": "gnn_datasets_4tasks_skew_warmth_v2",
}

# contention_v1: deliberately non-separable regime for GNN-necessity study.
# Levers (verified to drive joint coupling, see separability_diagnostic):
#   - sparse topology (few reachable platforms) => tasks must compete for the same options
#   - all-cold replicas (client_pct=server_pct=0) => simultaneous pulls serialize on node
#     FilterStore (N x pullTime), so co-locating cold tasks on one node is a real cliff
#   - heavy queues => stacking two tasks on one platform serializes execution
# MUST be generated with --allow-non-unique-replicas so the oracle can express collisions.
CONTENTION_V1_GRID: GridPreset = {
    "connection_probabilities": [0.25, 0.35],
    "replica_configs": [
        (1, 1, 0.0, 0.0),
        (1, 2, 0.0, 0.0),
        (2, 2, 0.0, 0.0),
    ],
    "queue_distributions": [
        ("pois28", "poisson", 28, 0, 0, 72, 1),
        ("norm35", "normal", 35, 11, 0, 96, 1),
        ("uniform20_80", "uniform", 20, 80, 0, 120, 1),
    ],
    "seeds": list(range(201, 215)),
    "default_output_subdir": "gnn_datasets_4tasks_contention_v1",
}

# contention_v2: SCARCE-WARM-RESOURCE regime. contention_v1 (all-cold) showed that
# cranking collision frequency does NOT break per-task greedy (regret stayed 0.00%):
# when colliding is optimal, both tasks independently prefer the same platform anyway.
# Greedy only breaks when two tasks share a #1 platform AND co-locating there is
# EXPENSIVE, forcing the optimum to split them. That needs a scarce *attractive* resource:
#   - warm replicas (high preinit) => a few platforms are clearly best => tasks compete
#   - heavy/pre-loaded queues => stacking two tasks on the warm platform serializes,
#     destroying its advantage so the optimum must split => anti-correlated preferences
#   - sparse topology => few fallback platforms => the split is non-trivial
# MUST be generated with --allow-non-unique-replicas.
CONTENTION_V2_GRID: GridPreset = {
    "connection_probabilities": [0.25, 0.35],
    "replica_configs": [
        (1, 1, 0.7, 0.9),
        (1, 2, 0.7, 0.9),
        (2, 2, 0.5, 0.7),
    ],
    "queue_distributions": [
        ("norm35", "normal", 35, 11, 0, 96, 1),
        ("uniform20_80", "uniform", 20, 80, 0, 120, 1),
        ("pois28", "poisson", 28, 0, 0, 72, 1),
    ],
    "seeds": list(range(301, 351)),
    "default_output_subdir": "gnn_datasets_4tasks_contention_v2",
}

# contention_v4_deepq: MATCH THE LIVE QUEUE REGIME.
# Measured 2026-08-13: the live failure of 873/v5.5 GNN/MLP is pure queueing, not pull
# serialization — live GNN sparse_p35 s42 has averageQueueTime 503.4s out of 503.4s
# averageElapsedTime, with averagePullTime 0.017s and averageColdStartTime 0.0003s.
# Meanwhile every contention_v2 label is warm-path: 0/900 datasets have any cold start or
# pull, and optimum averageQueueTime spans only 0.27-5.47s. The trained models therefore
# never see a queue deep enough for co-locating a batch to be catastrophic, which is exactly
# the mistake they make live (penaltyProportion ~80%).
# Lever: keep contention_v2's scarce-warm attractor (a few clearly-best warm platforms) but
# push pre-seeded queue depth deep enough that stacking two tasks on the attractive platform
# costs tens of seconds, so the optimum MUST split them.
# Depth is calibrated to the queue band the WINNING live policy occupies, not the failing one:
# live seed-42 averageQueueTime is Knative 7.7/30.6/44.9s vs MLP 25.2/79.5/216.1s vs GNN
# 87.9/210.0/503.4s on skew/p25/p35. Target label band ~5-50s => ~10x contention_v2's mean 35
# warmup tasks (which yielded only 0.27-5.47s). Overshooting to mean 1200 was measured at
# >150s/dataset locally because warmup tasks are materialized per placement, so keep the mean
# in the hundreds.
# MUST be generated with --allow-non-unique-replicas.
CONTENTION_V4_DEEPQ_GRID: GridPreset = {
    "connection_probabilities": [0.25, 0.35],
    "replica_configs": [
        (1, 1, 0.7, 0.9),
        (1, 2, 0.7, 0.9),
        (2, 2, 0.5, 0.7),
    ],
    "queue_distributions": [
        ("norm350", "normal", 350, 120, 0, 900, 1),
        ("uniform150_700", "uniform", 150, 700, 0, 900, 1),
        ("pois400", "poisson", 400, 0, 0, 900, 1),
    ],
    "seeds": list(range(601, 651)),
    "default_output_subdir": "gnn_datasets_4tasks_contention_v4_deepq",
}

# contention_v5_quick_test: QUICK TEST for deep queues + coupling optimization.
# Hypothesis: norm450/pois500 (12.9x deeper than v2's norm35) + scarce warm resources
# should push coupling rate from v2's 7.1% to 15-25%, giving GNN measurable advantage.
# Target: 216 datasets, 7-10 min walltime with 15 parallel workers on datalab CPU.
# MUST be generated with --allow-non-unique-replicas.
CONTENTION_V5_QUICK_TEST_GRID: GridPreset = {
    "connection_probabilities": [0.25, 0.30, 0.35],
    "replica_configs": [
        (1, 1, 0.7, 0.9),  # Minimal replicas, high warmth (strongest scarcity)
        (1, 2, 0.7, 0.9),  # Slightly more replicas, high warmth
    ],
    "queue_distributions": [
        ("norm450", "normal", 450, 150, 0, 1200, 1),
        ("uniform200_800", "uniform", 200, 800, 0, 1200, 1),
        ("pois500", "poisson", 500, 0, 0, 1200, 1),
    ],
    "seeds": list(range(701, 713)),  # 12 seeds
    "default_output_subdir": "gnn_datasets_4tasks_contention_v5_quick_test",
}

# contention_v3: push coupling further — sparser topology + heavier queues.
# Target: coupled (>1% greedy regret) fraction >25% (v2 was 7.2%).
# MUST be generated with --allow-non-unique-replicas.
CONTENTION_V3_GRID: GridPreset = {
    "connection_probabilities": [0.15, 0.20],
    "replica_configs": [
        (1, 1, 0.7, 0.9),
        (1, 2, 0.7, 0.9),
        (2, 2, 0.5, 0.7),
    ],
    "queue_distributions": [
        ("norm40", "normal", 40, 13, 0, 110, 1),
        ("uniform25_90", "uniform", 25, 90, 0, 130, 1),
        ("pois32", "poisson", 32, 0, 0, 84, 1),
    ],
    "seeds": list(range(401, 451)),
    "default_output_subdir": "gnn_datasets_4tasks_contention_v3",
}

# regime_b_cold_burst_v1: training labels for the frozen Regime B problem.
# Live gate is N=12 under platform_reuse_v1 (see archive/regime_b/scripts_cosim/regime_b_problem_spec.py).
# Co-sim stays at 4-task BF (placement space); physics + scarce-warm lever match live.
# MUST use --warmth-physics platform_reuse_v1 (node_disk_v2 kills FilterStore headroom).
# MUST --allow-non-unique-replicas. Cartesian: 2×3×3×25 = 450.
REGIME_B_COLD_BURST_V1_GRID: GridPreset = {
    "connection_probabilities": [0.25, 0.35],
    "replica_configs": [
        (1, 1, 0.7, 0.9),
        (1, 2, 0.7, 0.9),
        (2, 2, 0.5, 0.7),
    ],
    "queue_distributions": [
        ("norm35", "normal", 35, 11, 0, 96, 1),
        ("uniform20_80", "uniform", 20, 80, 0, 120, 1),
        ("pois28", "poisson", 28, 0, 0, 72, 1),
    ],
    "seeds": list(range(501, 526)),
    "default_output_subdir": "gnn_datasets_4tasks_regime_b_cold_burst_v1",
    "required_warmth_physics": "platform_reuse_v1",
}

GRID_PRESETS: Dict[str, GridPreset] = {
    "warmth_v2": WARMTH_V2_GRID,
    "sparse_warmth_v2": SPARSE_WARMTH_V2_GRID,
    "skew_warmth_v2": SKEW_WARMTH_V2_GRID,
    "contention_v1": CONTENTION_V1_GRID,
    "contention_v2": CONTENTION_V2_GRID,
    "contention_v3": CONTENTION_V3_GRID,
    "contention_v4_deepq": CONTENTION_V4_DEEPQ_GRID,
    "contention_v5_quick_test": CONTENTION_V5_QUICK_TEST_GRID,
    "regime_b_cold_burst_v1": REGIME_B_COLD_BURST_V1_GRID,
}


def resolve_grid_preset(grid_name: str) -> GridPreset:
    if grid_name not in GRID_PRESETS:
        known = ", ".join(sorted(GRID_PRESETS))
        raise ValueError(f"Unknown grid {grid_name!r}; expected one of: {known}")
    return GRID_PRESETS[grid_name]


def grid_topology_axis_count(preset: GridPreset) -> int:
    if preset.get("topology_type") == "degree_skewed_core":
        return len(preset["k_core_values"]) * len(preset["hub_seeker_fractions"])
    return len(preset["connection_probabilities"])


def grid_total_datasets(preset: GridPreset) -> int:
    return (
        grid_topology_axis_count(preset)
        * len(preset["replica_configs"])
        * len(preset["seeds"])
        * len(preset["queue_distributions"])
    )


def grid_topology_variants(preset: GridPreset) -> List[Tuple[str, Dict[str, Any]]]:
    """Ordered (label, kwargs for create_config_for_iteration topology fields)."""
    if preset.get("topology_type") == "degree_skewed_core":
        variants: List[Tuple[str, Dict[str, Any]]] = []
        for k_core in preset["k_core_values"]:
            for hub_frac in preset["hub_seeker_fractions"]:
                seek_pct = int(round(hub_frac * 100))
                variants.append(
                    (
                        f"k{k_core}_seek{seek_pct:02d}",
                        {
                            "topology_type": "degree_skewed_core",
                            "k_core": k_core,
                            "hub_seeker_fraction": hub_frac,
                            "latency_core_ms": preset.get("latency_core_ms", 5),
                            "latency_periphery_ms": preset.get("latency_periphery_ms", 30),
                        },
                    )
                )
        return variants
    return [
        (
            f"conn={conn_prob}",
            {"topology_type": "erdos_renyi", "connection_prob": conn_prob},
        )
        for conn_prob in preset["connection_probabilities"]
    ]

# Task type ratios: (dnn1%, dnn2%)
TASK_TYPE_RATIOS = [
    (0, 100), (50, 50), (100, 0)
]

# Workload parameters (can be overridden via --num-tasks)
NUM_TASKS = 4
NUM_CLIENT_NODES = 20  # matches space_with_network.json client_nodes.count
NUM_WORKLOAD_TEMPLATES = 10


def log(msg: str, quiet: bool = False, force: bool = False):
    """Print message unless in quiet mode."""
    if not quiet or force:
        print(msg)


def _run_shard_tag() -> str:
    """Stable per-process tag so concurrent SLURM array shards / local procs don't collide."""
    job_id = (
        os.environ.get("SLURM_ARRAY_JOB_ID")
        or os.environ.get("SLURM_JOB_ID")
        or str(os.getpid())
    )
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", str(os.getpid()))
    return f"{job_id}_{task_id}"


def workload_templates_dir_for_run(sim_input_path: Path) -> Path:
    """Per-SLURM-task template dir — parallel array shards must not share gnn_templates/."""
    return sim_input_path / "traces" / f"gnn_templates_{_run_shard_tag()}"


def workload_base_file_for_run(sim_input_path: Path) -> Path:
    """Per-shard workload-10 copy. The BF reads this path; concurrent array shards each
    overwrite their OWN copy so they cannot clobber each other's workload mid-run."""
    return sim_input_path / "traces" / f"workload-10_{_run_shard_tag()}.json"


def generate_workload_templates(
    base_workload_path: Path,
    output_dir: Path,
    num_templates: int = NUM_WORKLOAD_TEMPLATES,
    quiet: bool = False
) -> List[Path]:
    """
    Generate workload templates with varied task type ratios.
    
    Returns list of paths to generated template files.
    """
    with open(base_workload_path, 'r') as f:
        base_workload = json.load(f)
    
    templates = []
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for template_idx in range(num_templates):
        # Cycle through task type ratios
        dnn1_pct, dnn2_pct = TASK_TYPE_RATIOS[template_idx % len(TASK_TYPE_RATIOS)]
        
        num_dnn1 = NUM_TASKS * dnn1_pct // 100
        num_dnn2 = NUM_TASKS - num_dnn1
        
        # Create task types list
        task_types = ['dnn1'] * num_dnn1 + ['dnn2'] * num_dnn2
        
        # Random client node assignments
        client_nodes = [random.randint(0, NUM_CLIENT_NODES - 1) for _ in range(NUM_TASKS)]
        
        # Create workload with improved duration for queue accumulation
        workload = {
            'rps': base_workload.get('rps', 1),
            'duration': 1,
            'events': []
        }
        
        base_events = base_workload.get('events', [])
        for idx in range(NUM_TASKS):
            base_event = deepcopy(base_events[idx % len(base_events)])
            task_type = task_types[idx]
            client_node = client_nodes[idx]
            
            base_event['application']['name'] = f"nofs-{task_type}"
            base_event['application']['dag'] = {task_type: []}
            base_event['node_name'] = f"client_node{client_node}"
            
            workload['events'].append(base_event)
        
        # Save template
        template_path = output_dir / f"workload_template_{template_idx}.json"
        with open(template_path, 'w') as f:
            f.write(json_dumps_pretty(workload))
        
        templates.append(template_path)
        
        if not quiet:
            log(f"  Template {template_idx}: {num_dnn1} dnn1 + {num_dnn2} dnn2")
    
    return templates


def create_config_for_iteration(
    base_config: Dict[str, Any],
    replica_cfg: Tuple[int, int, float, float],
    seed: int,
    queue_dist: Tuple[str, str, int, int, int, int, int],
    batch_size: int = 4,
    topology_type: str = "erdos_renyi",
    connection_prob: Optional[float] = None,
    k_core: Optional[int] = None,
    hub_seeker_fraction: Optional[float] = None,
    latency_core_ms: float = 5.0,
    latency_periphery_ms: float = 30.0,
) -> Dict[str, Any]:
    """
    Create a modified config for a specific iteration.
    
    Args:
        batch_size: Batch size for determined scheduler (should match num_tasks)
    """
    config = deepcopy(base_config)
    
    per_client, per_server, client_pct, server_pct = replica_cfg
    qname, qtype, qp1, qp2, qmin, qmax, qstep = queue_dist
    
    # Network topology
    if 'network' not in config:
        config['network'] = {}
    if 'topology' not in config['network']:
        config['network']['topology'] = {}
    topo = config['network']['topology']
    if topology_type == "degree_skewed_core":
        if k_core is None or hub_seeker_fraction is None:
            raise ValueError(
                "degree_skewed_core requires k_core and hub_seeker_fraction"
            )
        topo.update(
            {
                "type": "degree_skewed_core",
                "k_core": k_core,
                "hub_seeker_fraction": hub_seeker_fraction,
                "p_core": 0.95,
                "p_periphery": 0.15,
                "latency_core_ms": latency_core_ms,
                "latency_periphery_ms": latency_periphery_ms,
                "seed": seed,
            }
        )
    else:
        if connection_prob is None:
            raise ValueError("erdos_renyi topology requires connection_prob")
        topo["connection_probability"] = connection_prob
        topo["seed"] = seed
    
    # Preinit configuration
    config['preinit'] = {
        'client_percentage': client_pct,
        'server_percentage': server_pct
    }
    
    # Replica configuration
    config['replicas'] = {
        'dnn1': {'per_client': per_client, 'per_server': per_server},
        'dnn2': {'per_client': per_client, 'per_server': per_server}
    }
    
    # Queue distribution parameters
    if qtype == "constant":
        q_params = {'type': 'constant', 'value': qp1, 'min': qmin, 'max': qmax, 'step': qstep}
    elif qtype == "poisson":
        q_params = {'type': 'poisson', 'lambda': qp1, 'min': qmin, 'max': qmax, 'step': qstep}
    elif qtype == "normal":
        stddev = qp2 if qp2 != 0 else 1
        q_params = {'type': 'normal', 'mean': qp1, 'stddev': stddev, 'min': qmin, 'max': qmax, 'step': qstep}
    elif qtype == "uniform":
        q_params = {'type': 'uniform', 'low': qp1, 'high': qp2, 'min': qmin, 'max': qmax, 'step': qstep}
    else:
        q_params = {'type': 'poisson', 'lambda': 4, 'min': qmin, 'max': qmax, 'step': qstep}
    
    config['prewarm'] = {
        'dnn1': {
            'distribution': 'none',
            'queue_distribution': 'statistical',
            'queue_distribution_params': q_params
        },
        'dnn2': {
            'distribution': 'none',
            'queue_distribution': 'statistical',
            'queue_distribution_params': q_params
        }
    }
    
    # Set scheduler batch_size to match num_tasks (for determined scheduler)
    # This ensures scheduler processes tasks in batches matching the workload
    if 'scheduler' not in config:
        config['scheduler'] = {}
    config['scheduler']['batch_size'] = batch_size
    config['scheduler']['batch_timeout'] = 0.1
    
    return config


def generate_single_dataset(
    dataset_id: str,
    output_dir: Path,
    config: Dict[str, Any],
    workload_template: Path,
    sim_input_path: Path,
    sample_json_file: Path,
    samples_file: Path,
    mapping_file: Path,
    seed: int,
    max_workers: int,
    quiet: bool = False,
    fast_forward_warmup: bool = True,
    fast_forward_threshold: int = 1,
    allow_non_unique_replicas: bool = True,
    warmth_physics: str = "node_disk_v2",
) -> Tuple[bool, float, float]:
    """
    Generate a single GNN dataset.
    
    Returns (success, rtt, duration_seconds)
    """
    start_time = time.time()
    
    try:
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        if os.environ.get("GNN_CAPTURE_DATASET_STATE", "0") != "1":
            ssc_stale = output_dir / "system_state_captured_unique.json"
            if ssc_stale.exists():
                ssc_stale.unlink()
        
        # Save config for this dataset
        config_path = output_dir / "space_with_network.json"
        with open(config_path, 'w') as f:
            f.write(json_dumps_pretty(config))
        
        # Copy workload template
        workload_path = output_dir / "workload.json"
        with open(workload_template, 'r') as f:
            workload = json.load(f)
        with open(workload_path, 'w') as f:
            f.write(json_dumps_pretty(workload))
        
        # Copy workload to a per-shard location for simulation (parallel-array safe).
        traces_dir = sim_input_path / "traces"
        traces_dir.mkdir(parents=True, exist_ok=True)
        shard_workload_file = workload_base_file_for_run(sim_input_path)
        with open(shard_workload_file, 'w') as f:
            f.write(json_dumps_pretty(workload))
        
        # Generate infrastructure
        infra_file = output_dir / "infrastructure.json"
        log(f"  Generating infrastructure...", quiet)
        generate_deterministic_infrastructure(
            str(config_path),
            sim_input_path,
            str(infra_file),
            seed
        )
        
        # Load one scenario sample (JSON preferred, .npy/.pkl fallback)
        sample, mapping, sample_source = load_primary_sample_and_mapping(
            sample_json_path=sample_json_file,
            samples_npy_path=samples_file,
            mapping_pkl_path=mapping_file,
        )
        log(f"  Sample source: {sample_source}", quiet)
        
        # Load apps from config
        apps = list(config['wsc'].keys())
        
        # Per-dataset scratch dir — parallel-safe (shared initial_results_simple races across shards).
        # placements.jsonl lives here during BF; MUST be copied to placements/ before rmtree.
        results_dir = output_dir / ".bf_scratch"
        if results_dir.exists():
            shutil.rmtree(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Run optimized brute-force simulation
        # NOTE: No timeout wrapper here because execute_brute_force_optimized uses ProcessPoolExecutor internally.
        # If it hangs, the process will need to be killed externally. The simulation itself should complete
        # within reasonable time for most datasets. If it hangs, check for deadlocks in the simulation code.
        log(f"  Running brute-force optimization (max {SIMULATION_TIMEOUT}s expected)...", quiet)
        
        sim_start = time.time()
        suppress_sim_prints = (
            os.environ.get("COSIM_SUPPRESS_SIM_PRINTS", "0") == "1"
            or os.environ.get("GNN_CAPTURE_DATASET_STATE", "0") != "1"
        )
        bf_kwargs = dict(
            apps=apps,
            config_file=str(config_path),
            mapping_file=str(mapping_file),
            output_dir=results_dir,
            sample=sample,
            sim_input_path=sim_input_path,
            workload_base_file=str(shard_workload_file),
            max_workers=max_workers,
            infrastructure_file=infra_file,
            quiet=quiet,
            final_dataset_dir=output_dir,  # Write progress files to final dataset directory
            fast_forward_warmup=fast_forward_warmup,
            fast_forward_threshold=fast_forward_threshold,
            allow_non_unique_replicas=allow_non_unique_replicas,
            mapping_override=mapping,
            warmth_physics=warmth_physics,
        )
        if suppress_sim_prints:
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                result_paths = execute_brute_force_optimized(**bf_kwargs)
        else:
            result_paths = execute_brute_force_optimized(**bf_kwargs)
        sim_duration = time.time() - sim_start
        
        # Warn if simulation took too long (but don't fail - it might be legitimate)
        if sim_duration > SIMULATION_TIMEOUT:
            log(f"  WARNING: Simulation took {sim_duration:.1f}s (exceeded {SIMULATION_TIMEOUT}s threshold)", quiet, force=True)
        
        # Check for results and copy to dataset directory
        best_json = results_dir / "best.json"
        placements_file = results_dir / "placements.jsonl"
        
        if best_json.exists():
            # Copy results to dataset directory
            with open(best_json, 'r') as f:
                best_info = json.load(f)
            
            optimal_rtt = best_info.get('rtt', float('inf'))
            optimal_file = best_info.get('file', '')
            
            # Copy best.json
            with open(output_dir / "best.json", 'w') as f:
                f.write(json_dumps(best_info))
            
            # Copy optimal result (use stdlib json to handle numpy types)
            optimal_src = results_dir / optimal_file
            if optimal_src.exists():
                shutil.copy2(optimal_src, output_dir / "optimal_result.json")
                capture_state = os.environ.get("GNN_CAPTURE_DATASET_STATE", "0") == "1"
                ssc_path = output_dir / "system_state_captured_unique.json"
                if capture_state:
                    try:
                        from src.executecosimulation import build_system_state_captured
                        from src.placement.model import DataclassJSONEncoder

                        with open(output_dir / "optimal_result.json", "r") as opt_f:
                            optimal_data = json.load(opt_f)
                        opt_stats = optimal_data.get("stats")
                        if opt_stats and opt_stats.get("taskResults"):
                            captured_state = build_system_state_captured(opt_stats)
                            with open(ssc_path, "w") as ssc_f:
                                json.dump(captured_state, ssc_f, indent=2, cls=DataclassJSONEncoder)
                    except Exception as ssc_exc:
                        log(f"  WARNING: Failed to write system_state_captured_unique.json: {ssc_exc}", quiet, force=True)
                        raise RuntimeError(
                            f"SSC export failed for {output_dir.name}: {ssc_exc}"
                        ) from ssc_exc
                elif ssc_path.exists():
                    ssc_path.unlink()
            
            # REQUIRED: persist full placement sweep for RTT-hash / near-RTT training.
            pub_placements = output_dir / "placements" / "placements.jsonl"
            if not placements_file.exists():
                raise RuntimeError(
                    f"{dataset_id}: best.json exists but .bf_scratch/placements.jsonl missing — "
                    "cannot train counterfactual RTT; keep scratch and re-run BF"
                )
            (output_dir / "placements").mkdir(exist_ok=True)
            shutil.copy2(placements_file, pub_placements)
            if pub_placements.stat().st_size == 0:
                raise RuntimeError(
                    f"{dataset_id}: placements/placements.jsonl is empty after copy"
                )
            
            # Copy placement metadata if it exists (will be written by execute_brute_force_optimized)
            metadata_src = results_dir / "placement_metadata.json"
            if metadata_src.exists():
                shutil.copy2(metadata_src, output_dir / "placement_metadata.json")
            
            # Copy placement progress if it exists
            progress_src = results_dir / "placement_progress.txt"
            if progress_src.exists():
                shutil.copy2(progress_src, output_dir / "placement_progress.txt")
            
            # Only remove scratch after public JSONL is verified (see placements_jsonl_required.md)
            shutil.rmtree(results_dir, ignore_errors=True)

            duration = time.time() - start_time
            return 'success', optimal_rtt, duration
        else:
            # No results - check if this was an infeasible scenario (placements.jsonl empty or missing)
            duration = time.time() - start_time
            if placements_file.exists() and placements_file.stat().st_size == 0:
                # Empty placements file = infeasible scenario, skip gracefully
                shutil.rmtree(results_dir, ignore_errors=True)
                return 'skipped', float('inf'), duration
            else:
                # Some other issue
                shutil.rmtree(results_dir, ignore_errors=True)
                return 'failed', float('inf'), duration
            
    except Exception as e:
        duration = time.time() - start_time
        log(f"  ERROR: {e}", quiet, force=True)
        return 'failed', float('inf'), duration


def main():
    parser = argparse.ArgumentParser(
        description="Optimized GNN Dataset Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--quiet', '-q', default=True, action='store_true',
                        help='Suppress per-placement logging (default: True)')
    parser.add_argument('--no-quiet', action='store_false', dest='quiet',
                        help='Disable quiet mode (show per-placement logging)')
    parser.add_argument('--max-datasets', '-n', type=int, default=300000,
                        help='Maximum number of datasets to generate')
    parser.add_argument('--workers', '-w', type=int, default=None,
                        help='Number of parallel workers (default: CPU count - 1)')
    parser.add_argument(
        '--resume',
        action='store_true',
        help=(
            'Skip datasets that already have best.json AND non-empty '
            'placements/placements.jsonl (JSONL required for RTT-hash training)'
        ),
    )
    parser.add_argument(
        '--only-missing-jsonl',
        action='store_true',
        help=(
            'Only re-run BF for datasets that have best.json but lack non-empty '
            'placements/placements.jsonl (skips tail datasets with no best.json)'
        ),
    )
    parser.add_argument('--start-from', type=int, default=0,
                        help='Start from dataset index (e.g., 118 to start from ds_00118)')
    parser.add_argument('--fast-forward-warmup', default=True, action='store_true',
                        help='Enable fast-forward warmup for queues > 1 task (default: True)')
    parser.add_argument('--no-fast-forward-warmup', action='store_false', dest='fast_forward_warmup',
                        help='Disable fast-forward warmup')
    parser.add_argument('--fast-forward-threshold', type=int, default=1,
                        help='Threshold for fast-forward warmup (default: 1)')
    parser.add_argument(
        '--warmth-physics',
        type=str,
        default='node_disk_v2',
        choices=['platform_reuse_v1', 'node_disk_v2'],
        help='Warmth model for co-sim labels (default: node_disk_v2)',
    )
    parser.add_argument('--allow-non-unique-replicas', action='store_true',
                        help='Allow multiple tasks to share the same replica')
    parser.add_argument('--num-tasks', type=int, choices=[1, 2, 3, 4, 5], default=4,
                        help='Number of tasks per workload (1-5). Sets batch_size accordingly.')
    parser.add_argument(
        '--grid',
        type=str,
        default='warmth_v2',
        choices=sorted(GRID_PRESETS),
        help='Dataset grid preset (default: warmth_v2)',
    )
    parser.add_argument(
        '--output-subdir',
        type=str,
        default=None,
        help='Output subdirectory under simulation_data (default from --grid preset)',
    )
    parser.add_argument(
        '--progress-log-name',
        type=str,
        default=None,
        help='Progress log filename under logs/ (default: progress_{num_tasks}tasks.txt or derived from --output-subdir)',
    )
    args = parser.parse_args()
    if args.only_missing_jsonl and not args.resume:
        args.resume = True

    if os.environ.get("COSIM_SUPPRESS_SIM_PRINTS", "0") == "1":
        logging.getLogger().setLevel(logging.ERROR)
        logging.getLogger("simulation").setLevel(logging.ERROR)
    
    # OOM guard for brute-force placement generation.
    # Keep user-provided value if already exported in environment.
    os.environ.setdefault(
        "MAX_PLACEMENT_COMBINATIONS_SKIP",
        str(MAX_PLACEMENT_COMBINATIONS_SKIP_DEFAULT),
    )
    
    quiet = args.quiet
    grid_preset = resolve_grid_preset(args.grid)
    required_physics = grid_preset.get("required_warmth_physics")
    if required_physics and args.warmth_physics != required_physics:
        raise SystemExit(
            f"FAIL LOUD: grid {args.grid!r} requires --warmth-physics {required_physics} "
            f"(got {args.warmth_physics!r}). Regime B / FilterStore headroom collapses under "
            f"node_disk_v2 same-image — see archive/regime_b/scripts_cosim/regime_b_problem_spec.py."
        )
    topology_variants = grid_topology_variants(grid_preset)
    replica_configs = grid_preset["replica_configs"]
    queue_distributions = grid_preset["queue_distributions"]
    seeds = grid_preset["seeds"]
    grid_total = grid_total_datasets(grid_preset)

    # max_datasets is relative to start_from (e.g., --start-from xyz --max-datasets 1 means generate ds_xyz only)
    max_datasets = args.start_from + args.max_datasets
    cpu_count = os.cpu_count()
    max_workers = args.workers or (cpu_count - 1 if cpu_count and cpu_count > 1 else 1)
    
    # Set NUM_TASKS based on argument
    global NUM_TASKS
    NUM_TASKS = args.num_tasks
    batch_size = NUM_TASKS  # Match batch_size to num_tasks
    
    # Paths
    base_dir = PROJECT_ROOT / "simulation_data"
    config_path = base_dir / "space_with_network.json"
    sample_json_file = base_dir / "sample_simple.json"
    default_output_subdir = grid_preset.get("default_output_subdir") or f"gnn_datasets_{NUM_TASKS}tasks"
    output_subdir = args.output_subdir or default_output_subdir
    output_base = base_dir / output_subdir
    sim_input_path = PROJECT_ROOT / "data" / "nofs-ids"
    samples_file = base_dir / "lhs_samples_simple.npy"
    mapping_file = base_dir / "lhs_samples_simple_mapping.pkl"
    workload_base_file = sim_input_path / "traces" / "workload-10.json"
    workload_templates_dir = workload_templates_dir_for_run(sim_input_path)
    if args.progress_log_name:
        progress_log_name = args.progress_log_name
    elif args.output_subdir:
        safe_subdir = re.sub(r'[^A-Za-z0-9_.-]+', '_', output_subdir)
        progress_log_name = f"progress_{safe_subdir}.txt"
    else:
        progress_log_name = f"progress_{NUM_TASKS}tasks.txt"
    progress_log = PROJECT_ROOT / "logs" / progress_log_name
    
    # Create directories
    output_base.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "logs").mkdir(parents=True, exist_ok=True)
    
    log(f"=== Optimized GNN Dataset Generation ===", quiet)
    log(f"Grid: {args.grid} ({grid_total} combos)", quiet)
    log(f"Num tasks: {NUM_TASKS} (batch_size={batch_size})", quiet)
    log(f"Max datasets: {args.max_datasets} (up to ds_{max_datasets-1:05d})", quiet)
    log(f"Workers: {max_workers}", quiet)
    log(f"Using orjson: {HAS_ORJSON}", quiet)
    log(f"Quiet mode: {quiet}", quiet)
    log(
        f"MAX_PLACEMENT_COMBINATIONS_SKIP: {os.environ.get('MAX_PLACEMENT_COMBINATIONS_SKIP')}",
        quiet,
    )
    
    # Load base config
    with open(config_path, 'r') as f:
        base_config = json.load(f)
    
    # Generate workload templates
    log(f"\nGenerating workload templates...", quiet)
    templates = generate_workload_templates(
        workload_base_file,
        workload_templates_dir,
        NUM_WORKLOAD_TEMPLATES,
        quiet
    )
    log(f"Generated {len(templates)} workload templates", quiet)
    
    # Generate datasets
    log(f"\n=== Starting Dataset Generation ===", quiet)
    
    dataset_idx = args.start_from
    if dataset_idx > 0:
        log(f"Starting from dataset index: {dataset_idx} (ds_{dataset_idx:05d})", quiet)
    template_idx = 0
    total_time = 0
    successful = 0
    skipped = 0
    failed = 0
    
    total_combinations = grid_total
    log(f"Total possible combinations: {total_combinations}", quiet)
    
    start_time = time.time()
    
    # Calculate starting positions for nested loops
    start_from = args.start_from
    current_idx = 0
    
    for topo_label, topo_kwargs in topology_variants:
        for replica_cfg in replica_configs:
            for seed in seeds:
                for queue_dist in queue_distributions:
                    # Skip until we reach the starting index
                    if current_idx < start_from:
                        current_idx += 1
                        template_idx = (template_idx + 1) % NUM_WORKLOAD_TEMPLATES
                        continue
                    
                    if dataset_idx >= max_datasets:
                        break
                    
                    dataset_id = f"ds_{dataset_idx:05d}"
                    output_dir = output_base / dataset_id
                    
                    # Resume only when full BF artifacts exist (JSONL = placement–RTT universe)
                    pub_jsonl = output_dir / "placements" / "placements.jsonl"
                    has_jsonl = pub_jsonl.exists() and pub_jsonl.stat().st_size > 0
                    has_best = (output_dir / "best.json").exists()

                    if args.only_missing_jsonl:
                        if has_jsonl:
                            log(
                                f"[{dataset_id}] Skipping (--only-missing-jsonl: JSONL ok)",
                                quiet,
                            )
                            dataset_idx += 1
                            current_idx += 1
                            template_idx = (template_idx + 1) % NUM_WORKLOAD_TEMPLATES
                            continue
                        if not has_best:
                            log(
                                f"[{dataset_id}] Skipping (--only-missing-jsonl: no best.json)",
                                quiet,
                            )
                            dataset_idx += 1
                            current_idx += 1
                            template_idx = (template_idx + 1) % NUM_WORKLOAD_TEMPLATES
                            continue
                        log(
                            f"[{dataset_id}] BF repair: best.json without placements.jsonl",
                            quiet,
                            force=True,
                        )
                        if pub_jsonl.exists() and pub_jsonl.stat().st_size == 0:
                            pub_jsonl.unlink()
                    elif args.resume and has_best and has_jsonl:
                        log(
                            f"[{dataset_id}] Skipping (best.json + placements.jsonl)",
                            quiet,
                        )
                        dataset_idx += 1
                        current_idx += 1
                        template_idx = (template_idx + 1) % NUM_WORKLOAD_TEMPLATES
                        continue
                    elif args.resume and has_best and not has_jsonl:
                        log(
                            f"[{dataset_id}] Re-running: best.json exists but "
                            "placements/placements.jsonl missing or empty",
                            quiet,
                            force=True,
                        )
                        if pub_jsonl.exists() and pub_jsonl.stat().st_size == 0:
                            pub_jsonl.unlink()
                    elif not has_best and has_jsonl:
                        log(
                            f"[{dataset_id}] Re-running: stale placements.jsonl without best.json",
                            quiet,
                            force=True,
                        )
                        pub_jsonl.unlink()
                    
                    # Get workload template
                    template = templates[template_idx]
                    
                    # Create config for this iteration (with batch_size matching num_tasks)
                    config = create_config_for_iteration(
                        base_config,
                        replica_cfg,
                        seed,
                        queue_dist,
                        batch_size=batch_size,
                        **topo_kwargs,
                    )
                    
                    qname = queue_dist[0]
                    per_client, per_server, client_pct, server_pct = replica_cfg
                    
                    log(
                        f"\n[{dataset_id}] topo={topo_label} rpc={per_client} "
                        f"rps={per_server} cpct={client_pct} spct={server_pct} "
                        f"seed={seed} q={qname}",
                        quiet,
                    )
                    
                    # Generate dataset
                    status, rtt, duration = generate_single_dataset(
                        dataset_id=dataset_id,
                        output_dir=output_dir,
                        config=config,
                        workload_template=template,
                        sim_input_path=sim_input_path,
                        sample_json_file=sample_json_file,
                        samples_file=samples_file,
                        mapping_file=mapping_file,
                        seed=seed,
                        max_workers=max_workers,
                        quiet=quiet,
                        fast_forward_warmup=args.fast_forward_warmup,
                        fast_forward_threshold=args.fast_forward_threshold,
                        allow_non_unique_replicas=args.allow_non_unique_replicas,
                        warmth_physics=args.warmth_physics,
                    )
                    
                    total_time += duration
                    
                    # Match logs/non_unique_progress_* line shape: existing= new= best_rtt=
                    # (brute-force run: no prior placements file → existing=0, new=completed sims)
                    num_existing = 0
                    num_new = 0
                    metadata_file = output_dir / "placement_metadata.json"
                    if metadata_file.exists():
                        try:
                            with open(metadata_file, 'r') as mf:
                                metadata = json.load(mf)
                            num_new = int(metadata.get('completed', metadata.get('num_placements', 0)))
                        except Exception:
                            pass
                    
                    if status == 'success':
                        successful += 1
                        log(f"  SUCCESS: RTT={rtt:.3f}s ({duration:.1f}s)", quiet)
                        with open(progress_log, 'a') as f:
                            f.write(
                                f"{dataset_id} SUCCESS {datetime.now().isoformat()} "
                                f"{duration:.1f}s existing={num_existing} new={num_new} "
                                f"best_rtt={rtt:.3f}s\n"
                            )
                    elif status == 'skipped':
                        skipped += 1
                        log(f"  SKIPPED: infeasible configuration ({duration:.1f}s)", quiet)
                        with open(progress_log, 'a') as f:
                            f.write(
                                f"{dataset_id} SKIPPED {datetime.now().isoformat()} "
                                f"infeasible\n"
                            )
                    else:
                        failed += 1
                        log(f"  FAILED ({duration:.1f}s)", quiet)
                        with open(progress_log, 'a') as f:
                            f.write(
                                f"{dataset_id} FAILED {datetime.now().isoformat()} "
                                f"{duration:.1f}s\n"
                            )
                    
                    dataset_idx += 1
                    current_idx += 1
                    template_idx = (template_idx + 1) % NUM_WORKLOAD_TEMPLATES
                    
                    # Progress update
                    if dataset_idx % 10 == 0:
                        elapsed = time.time() - start_time
                        rate = dataset_idx / elapsed if elapsed > 0 else 0
                        log(f"\n--- Progress: {dataset_idx}/{max_datasets} "
                            f"({100*dataset_idx/max_datasets:.1f}%) - "
                            f"{rate:.2f} datasets/min ---", quiet)
                
                if dataset_idx >= max_datasets:
                    break
            if dataset_idx >= max_datasets:
                break
        if dataset_idx >= max_datasets:
            break
    
    # Summary
    total_elapsed = time.time() - start_time
    
    log(f"\n=== Generation Complete ===", quiet, force=True)
    log(f"Total attempted: {dataset_idx}", quiet, force=True)
    log(f"Successful: {successful}", quiet, force=True)
    log(f"Skipped (infeasible): {skipped}", quiet, force=True)
    log(f"Failed: {failed}", quiet, force=True)
    if successful > 0:
        log(f"Success rate: {100*successful/(successful+failed+skipped):.1f}%", quiet, force=True)
    log(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)", quiet, force=True)
    log(f"Average time per dataset: {total_elapsed/max(1, dataset_idx):.1f}s", quiet, force=True)
    log(f"Output directory: {output_base}", quiet, force=True)
    log(f"Progress log: {progress_log}", quiet, force=True)


if __name__ == "__main__":
    main()
