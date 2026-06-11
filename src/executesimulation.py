"""
Unified Simulation Executor with Policy Selection

This script runs simulations with different policies (vanilla knative or vanilla gnn)
for real simulation (full workload, no warmup tasks, autoscaling from zero).

Workflow:
1. Load space_with_network.json config file
2. Generate infrastructure (nodes + network topology) deterministically
3. Load workload from file
4. Run simulation with chosen policy (kn_network_kn_network or gnn_gnn)
5. Save simulation results

Usage:
    python -m src.executesimulation --config <space_config.json> --workload <workload.json> --policy knative [--seed <seed>]
    python -m src.executesimulation --config <space_config.json> --workload <workload.json> --policy gnn [--seed <seed>]
    python -m src.executesimulation --config <space_config.json> --workload <workload.json> --policy roundrobin [--seed <seed>]
    python -m src.executesimulation --config <space_config.json> --workload <workload.json> --policy knative_network [--seed <seed>]
    python -m src.executesimulation --config <space_config.json> --workload <workload.json> --policy herocache_network [--seed <seed>]
"""

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

from src.generate_infrastructure import (
    apply_degree_skew_core_server_device_types,
    generate_network_topology_deterministic,
)
from src.motivational.constants import KEEP_ALIVE, QUEUE_LENGTH, RECONCILE_INTERVAL
from src.placement.executor import execute_sim
from src.placement.model import SimulationData, DataclassJSONEncoder
from src.policy.tabular.constants import PLATFORM_FEATURE_DIM, TASK_FEATURE_DIM

REQUIRED_SIM_FILES = [
    'application-types.json',
    'platform-types.json',
    'qos-types.json',
    'storage-types.json',
    'task-types.json'
]


def setup_logging(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger('simulation')
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    logger.propagate = False
    return logger


def load_simulation_inputs(sim_input_path: Path) -> Dict[str, Any]:
    """Load all required simulation input files."""
    sim_inputs = {}

    missing_files = []
    for filename in REQUIRED_SIM_FILES:
        if not (sim_input_path / filename).exists():
            missing_files.append(filename)

    if missing_files:
        raise FileNotFoundError(
            f"Missing required simulation input files: {', '.join(missing_files)}"
        )

    for filename in REQUIRED_SIM_FILES:
        file_path = sim_input_path / filename
        with open(file_path, 'r') as f:
            key = filename.replace('.json', '').replace('-', '_')
            sim_inputs[key] = json.load(f)

    return sim_inputs


def prepare_infrastructure_for_real_simulation(
        space_config: Dict[str, Any],
        seed: Optional[int] = None,
        sim_input_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Prepare infrastructure configuration for real simulation (no warmup, autoscaling from zero).
    
    This generates:
    - Nodes (client and server nodes with platforms)
    - Network topology (deterministic with seed)
    
    Does NOT generate:
    - Replica placements (autoscaling handles this)
    - Queue distributions (no warmup tasks)
    
    Args:
        space_config: Configuration from space_with_network.json
        seed: Random seed for deterministic network topology (default: from config or 42)
    
    Returns:
        Infrastructure configuration dictionary
    """
    # Get seed from config or use default
    if seed is None:
        topology_config = space_config.get('network', {}).get('topology', {})
        seed = topology_config.get('seed', 42)
    
    # Create seeded RNG
    rng = random.Random(seed)
    
    # Get node counts
    client_nodes_count = space_config['nodes']['client_nodes']['count']
    server_nodes_count = space_config['nodes']['server_nodes']['count']
    device_types = list(space_config['pci'].keys())
    
    # Generate nodes
    nodes = []
    
    # Generate client nodes
    for i in range(client_nodes_count):
        device_type = device_types[i % len(device_types)]
        device_specs = space_config['pci'][device_type]['specs']
        node_config = device_specs.copy()
        node_config['node_name'] = f"client_node{i}"
        node_config['type'] = device_type
        # network_map will be assigned after topology generation
        nodes.append(node_config)
    
    # Generate server nodes
    for i in range(server_nodes_count):
        device_type = device_types[i % len(device_types)]
        device_specs = space_config['pci'][device_type]['specs']
        node_config = device_specs.copy()
        node_config['node_name'] = f"node{i}"
        node_config['type'] = device_type
        # network_map will be assigned after topology generation
        nodes.append(node_config)

    apply_degree_skew_core_server_device_types(nodes, space_config)
    
    # Load task-types.json for platform compatibility checks
    task_types_data = None
    if sim_input_path is not None:
        task_types_path = sim_input_path / "task-types.json"
        if task_types_path.exists():
            with open(task_types_path, 'r') as f:
                task_types_data = json.load(f)
    
    topology_config = space_config.get('network', {}).get('topology', {})
    topology_type = topology_config.get('type', 'sparse')
    connection_probability = topology_config.get('connection_probability', 0.85)
    clients = [n for n in nodes if n['node_name'].startswith('client_node')]
    servers = [n for n in nodes if not n['node_name'].startswith('client_node')]
    print(f"\n=== Network Topology Generation ===")
    print(f"Topology type: {topology_type}")
    if topology_type != 'degree_skewed_core':
        print(f"Connection probability: {connection_probability} ({connection_probability*100:.1f}%)")
    print(f"Nodes: {len(clients)} clients, {len(servers)} servers")

    # Generate network topology deterministically
    network_maps = generate_network_topology_deterministic(nodes, space_config, rng, task_types_data=task_types_data)
    
    # Assign network maps to nodes
    for node in nodes:
        node['network_map'] = network_maps.get(node['node_name'], {})
    
    # Get network bandwidth (default to 1000.0 if not specified)
    network_bandwidth = space_config.get('network', {}).get('bandwidth', 1000.0)
    
    # Build infrastructure configuration for real simulation
    # NO preinitialize_platforms, NO replica_plan, NO deterministic placements
    infrastructure_config = {
        "network": {
            "bandwidth": float(network_bandwidth)
        },
        "nodes": nodes,
        # Real simulation: start with zero replicas, rely on autoscaling
        # No preinitialize_platforms flag
        # No replica_plan
        # No deterministic_replica_placements
        # No deterministic_queue_distributions
    }
    
    return infrastructure_config


def execute_simulation(
        config: Dict[str, Any],
        sim_inputs: Dict[str, Any],
        scheduling_strategy: str,
        cache_policy='fifo',
        task_priority='fifo',
        keep_alive=30,
        queue_length=30,
        models=None,
        reconcile_interval=1,
) -> Dict[str, Any]:
    """Execute simulation with full configuration and simulation inputs."""

    simulation_data = SimulationData(
        platform_types=sim_inputs['platform_types'],
        storage_types=sim_inputs['storage_types'],
        qos_types=sim_inputs['qos_types'],
        application_types=sim_inputs['application_types'],
        task_types=sim_inputs['task_types'],
    )

    stats = execute_sim(
        simulation_data,
        config['infrastructure'],
        cache_policy,
        keep_alive,
        task_priority,
        queue_length,
        scheduling_strategy,
        config['workload'],
        'workload-simulation',
        models=models,
        reconcile_interval=reconcile_interval,
    )
    return {
        "status": "success",
        "config": config,
        "sim_inputs": sim_inputs,
        "stats": stats
    }


def load_gnn_model(model_path: Path):
    """Load the trained GNN model."""
    import torch
    from src.policy.gnn.gnn_model import TaskPlacementGNN
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading GNN model from {model_path} on {device}...", flush=True)

        state_dict = torch.load(model_path, map_location='cpu')
        task_feature_dim = int(state_dict["task_encoder.net.0.weight"].shape[1])
        platform_feature_dim = int(state_dict["platform_encoder.net.0.weight"].shape[1])
        embedding_dim = 64
        edge_fc1_in = int(state_dict["edge_scorer.fc1.weight"].shape[1])
        edge_dim = edge_fc1_in - 2 * embedding_dim
        if edge_dim < 0:
            raise ValueError(
                f"Cannot infer edge_dim from edge_scorer.fc1 in_dim={edge_fc1_in}"
            )

        layout = os.environ.get("INFERENCE_FEATURE_LAYOUT", "atomic21").strip().lower()
        if task_feature_dim == 3 and platform_feature_dim == 6:
            os.environ["INFERENCE_FEATURE_LAYOUT"] = "ce_reduced"
            print(
                f"Using ce_reduced inference layout "
                f"(task_dim={task_feature_dim}, platform_dim={platform_feature_dim}, edge_dim={edge_dim})",
                flush=True,
            )
        elif layout in ("dim22", "legacy", "22") or (
            task_feature_dim == 3
            and platform_feature_dim == 14
            and layout not in ("atomic21", "21", "ce_reduced")
        ):
            os.environ["INFERENCE_FEATURE_LAYOUT"] = "dim22"
            print(
                f"Using legacy dim22 inference layout "
                f"(task_dim={task_feature_dim}, platform_dim={platform_feature_dim})",
                flush=True,
            )
        elif task_feature_dim != TASK_FEATURE_DIM or platform_feature_dim != PLATFORM_FEATURE_DIM:
            if not (task_feature_dim == 3 and layout in ("atomic21", "21")):
                raise ValueError(
                    f"Checkpoint dims task={task_feature_dim} platform={platform_feature_dim} "
                    f"do not match current constants "
                    f"task={TASK_FEATURE_DIM} platform={PLATFORM_FEATURE_DIM}"
                )
            print(
                f"Using atomic21 inference layout with task_dim={task_feature_dim} checkpoint "
                f"(platform_dim={platform_feature_dim})",
                flush=True,
            )

        model = TaskPlacementGNN(
            task_feature_dim=task_feature_dim,
            platform_feature_dim=platform_feature_dim,
            embedding_dim=embedding_dim,
            hidden_dim=64,
            num_layers=3,
            edge_dim=edge_dim,
        )
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        
        # Clear CUDA cache to avoid memory issues
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        print(f"GNN model loaded successfully ({sum(p.numel() for p in model.parameters()):,} parameters)", flush=True)
        return model, device
    except Exception as e:
        print(f"ERROR loading GNN model: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise


def load_gnn_hetero_model(model_path: Path):
    """Load a trained HeteroData/HeteroConv GNN model."""
    import torch
    from src.policy.gnn_hetero.gnn_model import TaskPlacementGNN

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading hetero GNN model from {model_path} on {device}...", flush=True)

        model = TaskPlacementGNN(
            task_feature_dim=TASK_FEATURE_DIM,
            platform_feature_dim=PLATFORM_FEATURE_DIM,
            embedding_dim=64,
            hidden_dim=64,
            num_layers=3,
        )

        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()

        if device.type == 'cuda':
            torch.cuda.empty_cache()

        print(
            f"Hetero GNN model loaded successfully ({sum(p.numel() for p in model.parameters()):,} parameters)",
            flush=True,
        )
        return model, device
    except Exception as e:
        print(f"ERROR loading hetero GNN model: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise


def load_task_types_data(sim_input_path: Path) -> Dict[str, Any]:
    """Load task-types.json for feature extraction."""
    task_types_path = sim_input_path / "task-types.json"
    with open(task_types_path, 'r') as f:
        return json.load(f)


def build_rtt_overview(
    stats: Dict[str, Any],
    total_rtt: float,
    decode_stats_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Summarize simulated RTT vs wall-clock ML inference not charged to SimPy time.

    Per-task inference is stored in task.gnn_decision_time (GNN and XGBoost batch).
    """
    total_inference = float(stats.get("total_inference_time") or 0.0)
    combined = float(total_rtt) + total_inference
    num_tasks = int(stats.get("num_tasks") or 0)
    tasks_with_inference = int(stats.get("tasks_with_inference") or 0)
    overview: Dict[str, Any] = {
        "simulated_total_rtt_s": float(total_rtt),
        "total_inference_time_s": total_inference,
        "hypothetical_total_with_inference_s": combined,
        "inference_fraction_of_combined": (
            total_inference / combined if combined > 0 else 0.0
        ),
        "num_tasks": num_tasks,
        "tasks_with_inference": tasks_with_inference,
        "average_inference_time_s": float(stats.get("averageGNNDecisionTime") or 0.0),
        "inference_charged_to_simpy_time": False,
        "note": (
            "Inference is wall-clock only (task.gnn_decision_time); "
            "not included in elapsedTime / total_rtt."
        ),
    }
    if decode_stats_summary:
        dt = decode_stats_summary.get("decode_time_ms") or {}
        decode_total_ms = float(dt.get("total") or 0.0)
        overview["decode_phase_wall_time_s"] = decode_total_ms / 1000.0
        overview["decode_phase_note"] = (
            "decode_phase_wall_time_s covers decode step only (seq_decode); "
            "total_inference_time_s includes graph build + forward + decode."
        )
    return overview


def run_simulation(
        config_file: Path,
        workload_file: Path,
        output_file: Path,
        sim_input_path: Path,
        logger: logging.Logger,
        policy: str,
        seed: Optional[int] = None,
        gnn_model: Any = None,
        gnn_device: Any = None,
        task_types_data: Optional[Dict[str, Any]] = None,
        xgb_model_path: Optional[Path] = None,
        mlp_model_path: Optional[Path] = None,
) -> bool:
    """
    Run simulation with the specified policy.
    
    Args:
        config_file: Path to space_with_network.json config file
        workload_file: Path to workload JSON file
        output_file: Path to save simulation results
        sim_input_path: Path to simulation input files
        logger: Logger instance
        policy: Policy name ('knative', 'gnn', or 'roundrobin')
        seed: Random seed for deterministic network topology (optional)
        gnn_model: GNN model (required for gnn policy)
        task_types_data: Task types data (required for gnn policy)
    
    Returns True if successful, False if failed.
    """
    logger.info(f"Running {policy} simulation")

    # Validate policy
    valid_policies = [
        'knative',
        'gnn',
        'gnn_hetero',
        'roundrobin',
        'knative_network',
        'knative_network_ect',
        'knative_network_batch',
        'herocache_network',
        'herocache_network_batch',
        'random_network',
        'offload_network',
        'xgboost_batch',
        'xgboost_single',
        'mlp_batch',
    ]
    if policy not in valid_policies:
        logger.error(
            f"Invalid policy: {policy}. Must be one of: {', '.join(valid_policies)}"
        )
        return False

    # For GNN policy, check if model is provided
    if policy in ('gnn', 'gnn_hetero') and (gnn_model is None or task_types_data is None):
        logger.error(f"{policy} policy requires gnn_model and task_types_data")
        return False

    if policy in ('xgboost_batch', 'xgboost_single'):
        if xgb_model_path is None or not xgb_model_path.exists():
            logger.error(f"{policy} policy requires a valid --xgb-model path (got {xgb_model_path})")
            return False
        if task_types_data is None:
            logger.error(f"{policy} policy requires task_types_data for feature extraction")

    if policy == 'mlp_batch':
        if mlp_model_path is None or not mlp_model_path.exists():
            logger.error(f"mlp_batch policy requires a valid --mlp-model path (got {mlp_model_path})")
            return False
        if task_types_data is None:
            logger.error("mlp_batch policy requires task_types_data for feature extraction")
            return False

    # Check required files exist
    if not config_file.exists():
        logger.error(f"Config file not found: {config_file}")
        return False

    if not workload_file.exists():
        logger.error(f"Workload file not found: {workload_file}")
        return False

    try:
        # Load simulation inputs
        sim_inputs = load_simulation_inputs(sim_input_path)

        # Load space config
        with open(config_file, 'r') as f:
            space_config = json.load(f)

        placement_seed = seed
        if placement_seed is None:
            placement_seed = space_config.get("network", {}).get("topology", {}).get("seed", 42)
        random.seed(placement_seed)

        # Load workload
        with open(workload_file, 'r') as f:
            workload = json.load(f)

        # Prepare infrastructure for real simulation
        infrastructure_config = prepare_infrastructure_for_real_simulation(
            space_config, seed=seed, sim_input_path=sim_input_path
        )

        # Combine into full config
        full_config = {
            "infrastructure": infrastructure_config,
            "workload": workload,
        }

        # Determine scheduling strategy
        scheduling_strategy = None
        models = None
        
        if policy == 'knative':
            scheduling_strategy = 'kn_network_kn_network'
            models = None
        elif policy == 'gnn':
            scheduling_strategy = 'gnn_gnn'
            models = {
                'gnn_model': gnn_model,
                'device': gnn_device,
                'task_types_data': task_types_data,
            }
        elif policy == 'gnn_hetero':
            scheduling_strategy = 'gnn_hetero_gnn_hetero'
            models = {
                'gnn_model': gnn_model,
                'device': gnn_device,
                'task_types_data': task_types_data,
            }
        elif policy == 'roundrobin':
            scheduling_strategy = 'rr_network_rr_network'
            models = None
        elif policy == 'knative_network':
            scheduling_strategy = 'kn_network_kn_network'
            models = None
        elif policy == 'knative_network_ect':
            scheduling_strategy = 'kn_network_ect_kn_network_ect'
            models = None
        elif policy == 'knative_network_batch':
            scheduling_strategy = 'kn_network_batch_kn_network_batch'
            models = None
        elif policy == 'herocache_network':
            scheduling_strategy = 'hrc_network_hrc_network'
            models = None
        elif policy == 'herocache_network_batch':
            scheduling_strategy = 'hrc_network_batch_hrc_network_batch'
            models = None
        elif policy == 'random_network':
            scheduling_strategy = 'rp_network_rp_network'
            models = None
        elif policy == 'offload_network':
            scheduling_strategy = 'offload_network_offload_network'
            models = None
        elif policy == 'xgboost_batch':
            scheduling_strategy = 'xgb_batch_xgb_batch'
            models = {
                'xgb_model_path': str(xgb_model_path),
                'task_types_data': task_types_data,
            }
        elif policy == 'xgboost_single':
            scheduling_strategy = 'xgb_single_xgb_single'
            models = {
                'xgb_model_path': str(xgb_model_path),
                'task_types_data': task_types_data,
            }
        elif policy == 'mlp_batch':
            scheduling_strategy = 'mlp_batch_mlp_batch'
            models = {
                'mlp_model_path': str(mlp_model_path),
                'task_types_data': task_types_data,
            }

        if scheduling_strategy is None:
            logger.error(f"Unknown policy: {policy}")
            return False

        logger.info(f"Running {policy} simulation with strategy {scheduling_strategy}...")
        print(f"  Running {policy} simulation with strategy {scheduling_strategy}...")

        # Execute simulation
        result = execute_simulation(
            full_config,
            sim_inputs,
            scheduling_strategy=scheduling_strategy,
            cache_policy='fifo',
            task_priority='fifo',
            keep_alive=KEEP_ALIVE,
            queue_length=QUEUE_LENGTH,
            models=models,
            reconcile_interval=RECONCILE_INTERVAL,
        )

        # Extract stats
        stats = result.get('stats', {})
        # Use precomputed total_rtt/num_tasks when present (avoids holding full taskResults in memory)
        task_results = stats.get('taskResults', [])
        if stats.get('total_rtt') is not None and stats.get('num_tasks') is not None:
            total_rtt = stats['total_rtt']
            num_tasks = stats['num_tasks']
        else:
            total_rtt = sum(
                tr.get('elapsedTime', 0)
                for tr in task_results
                if tr.get('taskId') is not None and tr.get('taskId') >= 0
            )
            num_tasks = len([tr for tr in task_results if tr.get('taskId') is not None and tr.get('taskId') >= 0])

        # Build result summary
        decode_stats_summary = None
        result_summary = {
            "status": "success",
            "policy": policy,
            "scheduling_strategy": scheduling_strategy,
            "config_file": str(config_file),
            "workload_file": str(workload_file),
            "seed": seed,
            "total_rtt": total_rtt,
            "num_tasks": num_tasks,
            "stats": stats,
        }

        ml_policies = ("gnn", "gnn_hetero", "xgboost_batch", "xgboost_single", "mlp_batch")
        if policy in ml_policies:
            try:
                if policy == "gnn_hetero":
                    from src.policy.gnn_hetero.seq_decode import get_run_decode_stats, write_run_decode_stats
                else:
                    from src.policy.gnn.seq_decode import get_run_decode_stats, write_run_decode_stats

                decode_stats = get_run_decode_stats()
                if decode_stats is not None and decode_stats.gnn_batches > 0:
                    margin = int(os.environ.get("GNN_SEQBLEND_QUEUE_MARGIN", "1"))
                    summary = decode_stats.summary(p1_margin=margin)
                    decode_stats_summary = summary
                    result_summary["decode_stats"] = summary
                    stats_path = output_file.with_suffix(".decode_stats.json")
                    write_run_decode_stats(stats_path, p1_margin=margin)
                    print("\n=== GNN decode stats ===", flush=True)
                    dt = summary.get("decode_time_ms", {})
                    col = summary.get("intra_batch_platform_collisions", {})
                    qv = summary.get("chosen_queue_vs_min", {})
                    print(
                        f"  mode={summary.get('decode_mode')} top_k={summary.get('top_k')} | "
                        f"batches={summary['gnn_batches']:,} tasks={summary['total_decode_tasks']:,}",
                        flush=True,
                    )
                    print(
                        f"  decode_time_ms: mean={dt.get('mean')} p95={dt.get('p95')} total={dt.get('total')}",
                        flush=True,
                    )
                    print(
                        f"  combo_search_size: mean={summary.get('combo_search_size', {}).get('mean')} "
                        f"max={summary.get('combo_search_size', {}).get('max')}",
                        flush=True,
                    )
                    print(
                        f"  intra_batch_collisions: total={col.get('total')} "
                        f"batches={col.get('batches_with_collision')} "
                        f"rate={float(col.get('collision_batch_rate', 0))*100:.2f}%",
                        flush=True,
                    )
                    print(
                        f"  chosen_queue vs min: mean={qv.get('mean')} median={qv.get('median')} p95={qv.get('p95')}",
                        flush=True,
                    )
                    if summary.get("p1_override_count", 0) > 0:
                        print(
                            f"  seqblend p1 overrides={summary['p1_override_count']:,} "
                            f"({summary['p1_override_rate']*100:.2f}%)",
                            flush=True,
                        )
                    print(f"  wrote {stats_path.name}", flush=True)
            except Exception as exc:
                logger.warning(f"Could not attach decode stats: {exc}")

        result_summary["rtt_overview"] = build_rtt_overview(
            stats, total_rtt, decode_stats_summary=decode_stats_summary
        )

        # Save result
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(result_summary, f, indent=2, cls=DataclassJSONEncoder)

        logger.info(f"✓ Saved {output_file}")
        print(f"  ✓ Saved {output_file.name} (RTT: {total_rtt:.3f}s)", flush=True)
        overview = result_summary.get("rtt_overview") or {}
        inf_s = overview.get("total_inference_time_s")
        if inf_s is not None and float(inf_s) > 0:
            combined = overview.get("hypothetical_total_with_inference_s", total_rtt)
            frac = float(overview.get("inference_fraction_of_combined") or 0) * 100
            print(
                f"  RTT overview: inference={float(inf_s):.3f}s "
                f"(not in RTT) | with inference={float(combined):.3f}s "
                f"| inference={frac:.2f}% of combined",
                flush=True,
            )

        return True

    except Exception as e:
        logger.error(f"Error running simulation: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    Main entry point for unified simulation executor.
    
    Usage:
        python -m src.executesimulation --config <space_config.json> --workload <workload.json> --policy knative [--seed <seed>] [--output <output.json>]
        python -m src.executesimulation --config <space_config.json> --workload <workload.json> --policy gnn [--seed <seed>] [--output <output.json>]
        python -m src.executesimulation --config <space_config.json> --workload <workload.json> --policy roundrobin [--seed <seed>] [--output <output.json>]
        python -m src.executesimulation --config <space_config.json> --workload <workload.json> --policy knative_network [--seed <seed>]
        python -m src.executesimulation --config <space_config.json> --workload <workload.json> --policy herocache_network [--seed <seed>] [--output <output.json>]
        python -m src.executesimulation --config <space_config.json> --workload <workload.json> --policy herocache_network [--seed <seed>] [--output <output.json>]
    """
    # Configuration
    sim_input_path = Path("data/nofs-ids")
    _gnn_model_env = os.environ.get("GNN_MODEL_PATH")
    gnn_model_path = Path(_gnn_model_env) if _gnn_model_env else Path("models/desert-galaxy-26.pt")
    _gnn_hetero_model_env = os.environ.get("GNN_HETERO_MODEL_PATH")
    gnn_hetero_model_path = Path(_gnn_hetero_model_env) if _gnn_hetero_model_env else Path("models/hetero.pt")
    _xgb_model_env = os.environ.get("XGB_MODEL_PATH")
    default_xgb_model_path = Path(_xgb_model_env) if _xgb_model_env else Path("models/tabular/batch_edge_ranker.json")
    _xgb_single_model_env = os.environ.get("XGB_SINGLE_MODEL_PATH")
    default_xgb_single_model_path = (
        Path(_xgb_single_model_env)
        if _xgb_single_model_env
        else Path("models/tabular/single_edge_ranker.json")
    )
    _mlp_model_env = os.environ.get("MLP_MODEL_PATH")
    default_mlp_model_path = Path(_mlp_model_env) if _mlp_model_env else Path("models/tabular/batch_edge_mlp.pt")
    default_output_dir = Path("simulation_data/results")

    # Parse arguments
    config_file = None
    workload_file = None
    policy = None
    seed = None
    output_file = None
    xgb_model_path = None
    mlp_model_path = None

    if '--xgb-model' in sys.argv:
        idx = sys.argv.index('--xgb-model')
        if idx + 1 < len(sys.argv):
            xgb_model_path = Path(sys.argv[idx + 1])

    if '--mlp-model' in sys.argv:
        idx = sys.argv.index('--mlp-model')
        if idx + 1 < len(sys.argv):
            mlp_model_path = Path(sys.argv[idx + 1])

    if '--config' in sys.argv:
        idx = sys.argv.index('--config')
        if idx + 1 < len(sys.argv):
            config_file = Path(sys.argv[idx + 1])

    if '--workload' in sys.argv:
        idx = sys.argv.index('--workload')
        if idx + 1 < len(sys.argv):
            workload_file = Path(sys.argv[idx + 1])

    if '--policy' in sys.argv:
        idx = sys.argv.index('--policy')
        if idx + 1 < len(sys.argv):
            policy = sys.argv[idx + 1].lower()

    if '--seed' in sys.argv:
        idx = sys.argv.index('--seed')
        if idx + 1 < len(sys.argv):
            try:
                seed = int(sys.argv[idx + 1])
            except ValueError:
                print(f"ERROR: Invalid seed value: {sys.argv[idx + 1]}")
                sys.exit(1)

    if '--output' in sys.argv:
        idx = sys.argv.index('--output')
        if idx + 1 < len(sys.argv):
            output_file = Path(sys.argv[idx + 1])

    # Validate arguments
    if not config_file:
        print("ERROR: --config is required")
        print("Usage: python -m src.executesimulation --config <space_config.json> --workload <workload.json> --policy <knative|gnn> [--seed <seed>] [--output <output.json>]")
        sys.exit(1)

    if not workload_file:
        print("ERROR: --workload is required")
        print(
            "Usage: python -m src.executesimulation "
            "--config <space_config.json> --workload <workload.json> "
            "--policy <knative|gnn|gnn_hetero|roundrobin|knative_network|knative_network_ect|knative_network_batch|herocache_network|"
            "herocache_network_batch|random_network|offload_network> "
            "[--seed <seed>] [--output <output.json>]"
        )
        sys.exit(1)
    
    if not policy:
        print("ERROR: --policy is required")
        print(
            "Usage: python -m src.executesimulation "
            "--config <space_config.json> --workload <workload.json> "
            "--policy <knative|gnn|gnn_hetero|roundrobin|knative_network|knative_network_ect|knative_network_batch|herocache_network|"
            "herocache_network_batch|random_network|offload_network> "
            "[--seed <seed>] [--output <output.json>]"
        )
        sys.exit(1)
    
    cli_valid_policies = [
        'knative',
        'gnn',
        'gnn_hetero',
        'roundrobin',
        'knative_network',
        'knative_network_ect',
        'knative_network_batch',
        'herocache_network',
        'herocache_network_batch',
        'random_network',
        'offload_network',
        'xgboost_batch',
        'xgboost_single',
        'mlp_batch',
    ]
    if policy not in cli_valid_policies:
        print(f"ERROR: Invalid policy '{policy}'. Must be one of: {', '.join(cli_valid_policies)}")
        sys.exit(1)

    if not config_file.exists():
        print(f"ERROR: Config file not found: {config_file}")
        sys.exit(1)

    if not workload_file.exists():
        print(f"ERROR: Workload file not found: {workload_file}")
        sys.exit(1)

    # Set default output file if not provided
    if not output_file:
        default_output_dir.mkdir(parents=True, exist_ok=True)
        output_file = default_output_dir / f"simulation_result_{policy}.json"

    # Setup logging
    logger = setup_logging(Path("."))

    # Load GNN model if needed
    gnn_model = None
    gnn_device = None
    task_types_data = None
    if policy == 'gnn':
        if not gnn_model_path.exists():
            print(f"ERROR: GNN model not found at {gnn_model_path}")
            sys.exit(1)
        
        gnn_model, gnn_device = load_gnn_model(gnn_model_path)
        task_types_data = load_task_types_data(sim_input_path)
    elif policy == 'gnn_hetero':
        if not gnn_hetero_model_path.exists():
            print(f"ERROR: hetero GNN model not found at {gnn_hetero_model_path}")
            sys.exit(1)

        gnn_model, gnn_device = load_gnn_hetero_model(gnn_hetero_model_path)
        task_types_data = load_task_types_data(sim_input_path)
    elif policy in ('xgboost_batch', 'xgboost_single'):
        if xgb_model_path is None:
            xgb_model_path = (
                default_xgb_model_path
                if policy == 'xgboost_batch'
                else default_xgb_single_model_path
            )
        if not xgb_model_path.exists():
            print(f"ERROR: XGBoost model not found at {xgb_model_path}")
            sys.exit(1)
        task_types_data = load_task_types_data(sim_input_path)
    elif policy == 'mlp_batch':
        if mlp_model_path is None:
            mlp_model_path = default_mlp_model_path
        if not mlp_model_path.exists():
            print(f"ERROR: MLP model not found at {mlp_model_path}")
            sys.exit(1)
        task_types_data = load_task_types_data(sim_input_path)

    # Run simulation
    success = run_simulation(
        config_file, workload_file, output_file, sim_input_path, logger, policy,
        seed=seed, gnn_model=gnn_model, gnn_device=gnn_device, task_types_data=task_types_data,
        xgb_model_path=xgb_model_path, mlp_model_path=mlp_model_path,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

