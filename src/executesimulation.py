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
    build_core_backbone,
    generate_network_topology_deterministic,
)
from src.placement.constants import KEEP_ALIVE, QUEUE_LENGTH, RECONCILE_INTERVAL
from src.placement.executor import execute_sim
from src.placement.model import SimulationData, DataclassJSONEncoder
from src.placement.network_graph import (
    NETWORK_GRAPH_CONTRACT_ENV,
    NETWORK_GRAPH_CONTRACT_OFF,
    require_matching_network_graph_contract,
    resolve_network_graph_contract,
)
from src.placement.queue_features import (
    QUEUE_FEATURE_CONTRACT_ENV,
    require_matching_queue_feature_contract,
    resolve_queue_feature_contract,
    validate_queue_feature_contract,
)
from src.placement.topology_features import (
    CLIENT_NODE_PREFIX,
    TOPOLOGY_FEATURE_CONTRACT_ENV,
    require_matching_topology_feature_contract,
    resolve_topology_feature_contract,
    validate_topology_feature_contract,
)
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

    # link_contention_v1: overlay the core backbone on the live path too. Without this the
    # live gate would run different physics from the corpus the model trained on — the
    # exact class of train/serve mismatch that cost the mp_parity lineage a headline.
    _apply_link_backbone_env_default(space_config)
    link_topology = build_core_backbone(network_maps, nodes, space_config, rng, seed=seed)
    if link_topology is not None:
        print(
            f"Core backbone: {len(link_topology['links'])} links, "
            f"{sum(len(v) for v in link_topology['routes'].values())} routes"
        )

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
        "link_topology": link_topology,
    }
    infrastructure_config.update(
        _regime_b_infrastructure_overrides(space_config)
    )

    return infrastructure_config


# Pilot defaults, chosen by the P0 overlap pre-check
# (scripts_cosim/link_overlap_precheck.py): a pure ring with single attachment forces
# traffic across multiple shared segments — 30.3% of task pairs contend on a core link and
# 91.3% of that contention is between tasks on DIFFERENT destination nodes, which is the
# part no node-occupancy repair column can express. Chords and a second attachment both
# let paths diverge and collapse the overlap (5.2% at n_core=6, attach_degree=2).
_BACKBONE_ENV_DEFAULTS = {
    "n_core": 12,
    "attach_degree": 1,
    "chord_count": 0,
    "core_link_latency_ms": 4.0,
    "access_link_latency_ms": 20.0,
}


def _apply_link_backbone_env_default(space_config: Dict[str, Any]) -> None:
    """Let HEROSIM_LINK_BANDWIDTH_MBPS switch the backbone on, mirroring the ingress knob.

    An explicit network.backbone block always wins; the env var only synthesizes one when
    the config is silent, so a sweep can A/B the physics without editing every config.
    """
    network = space_config.setdefault("network", {})
    if network.get("backbone"):
        return
    raw = os.environ.get("HEROSIM_LINK_BANDWIDTH_MBPS")
    if not raw:
        return
    bandwidth = float(raw)
    if bandwidth <= 0:
        raise ValueError(
            f"HEROSIM_LINK_BANDWIDTH_MBPS must be > 0 when set, got {raw}"
        )
    network["backbone"] = {**_BACKBONE_ENV_DEFAULTS, "bandwidth_mbps": bandwidth}


def _env_bool(name: str) -> Optional[bool]:
    raw = os.environ.get(name)
    if raw is None:
        return None
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _regime_b_infrastructure_overrides(space_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pass Regime B / warmth flags from space config or HEROSIM_* env vars.

    space_config keys win over environment when both are set.
    """
    overrides: Dict[str, Any] = {}

    defer = space_config.get("defer_cold_replica_init")
    if defer is None:
        defer = _env_bool("HEROSIM_DEFER_COLD_REPLICA_INIT")
    if defer is not None:
        overrides["defer_cold_replica_init"] = bool(defer)

    warmth = space_config.get("warmth_physics")
    if warmth is None:
        warmth = os.environ.get("HEROSIM_WARMTH_PHYSICS")
    if warmth:
        overrides["warmth_physics"] = warmth

    ff = space_config.get("fast_forward_warmup")
    if ff is None:
        ff = _env_bool("HEROSIM_FAST_FORWARD_WARMUP")
    if ff is not None:
        overrides["fast_forward_warmup"] = bool(ff)
        overrides["fast_forward_threshold"] = int(
            space_config.get("fast_forward_threshold", 1)
        )

    scheduler = space_config.get("scheduler")
    if scheduler:
        overrides["scheduler"] = scheduler

    # Contention physics. These were previously unreachable from the live-simulation
    # path: node_contention_v3 shipped without a pass-through here, so a live gate could
    # not exercise it at all. Both are opt-in and stay None unless configured, which is
    # node_disk_v2 physics.
    compute_slots = space_config.get("nodes", {}).get("compute_slots_per_node")
    if compute_slots is None:
        raw = os.environ.get("HEROSIM_COMPUTE_SLOTS_PER_NODE")
        compute_slots = int(raw) if raw else None
    if compute_slots is not None:
        overrides["compute_slots_per_node"] = int(compute_slots)

    ingress_bw = space_config.get("nodes", {}).get("ingress_bandwidth_mbps")
    if ingress_bw is None:
        raw = os.environ.get("HEROSIM_INGRESS_BANDWIDTH_MBPS")
        ingress_bw = float(raw) if raw else None
    if ingress_bw is not None:
        overrides["ingress_bandwidth_mbps"] = float(ingress_bw)

    return overrides


def _workload_has_burst_ids(workload: Dict[str, Any]) -> bool:
    events = workload.get("events") or []
    return any(ev.get("burst_id") for ev in events)


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


def _adopt_queue_feature_contract(trained: str, model_label: str, source: str) -> None:
    declared = os.environ.get(QUEUE_FEATURE_CONTRACT_ENV, "").strip()
    if declared:
        require_matching_queue_feature_contract(trained, declared, model_label=model_label)
    else:
        os.environ[QUEUE_FEATURE_CONTRACT_ENV] = validate_queue_feature_contract(trained)
    print(
        f"[QUEUE FEATURES] {model_label} trained under "
        f"{os.environ[QUEUE_FEATURE_CONTRACT_ENV]} (source={source})",
        flush=True,
    )


def apply_checkpoint_queue_feature_contract(model_path: Path, model_label: str) -> None:
    """Adopt (or verify) the queue feature contract a checkpoint was trained under.

    GNN checkpoints are bare state dicts, so the dim7/dim13 scaling cannot be recovered
    from weight shapes; trainers write a `<model>.contract.json` sidecar instead. Absent a
    sidecar the checkpoint predates the split and is legacy_v0 by construction.
    """
    sidecar = model_path.with_suffix(".contract.json")
    if not sidecar.is_file():
        return
    try:
        trained = json.loads(sidecar.read_text()).get("queue_feature_contract")
    except json.JSONDecodeError as exc:
        raise ValueError(f"{sidecar} is not valid JSON: {exc}") from exc
    if not trained:
        raise ValueError(f"{sidecar} has no queue_feature_contract field")
    _adopt_queue_feature_contract(trained, model_label, sidecar.name)


def _read_checkpoint_sidecar(model_path: Path) -> dict:
    """The `<model>.contract.json` payload, or {} when the checkpoint predates sidecars."""
    sidecar = model_path.with_suffix(".contract.json")
    if not sidecar.is_file():
        return {}
    try:
        return json.loads(sidecar.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{sidecar} is not valid JSON: {exc}") from exc


def apply_checkpoint_topology_feature_contract(model_path: Path, model_label: str) -> None:
    """Adopt (or verify) the topology feature contract a checkpoint was trained under.

    Task feature dim 2 is `index(src)/n_nodes` under `src_index_v0` and reachable-server
    fraction under `size_invariant_v1` — different quantities on different scales. Nothing
    in the weights distinguishes them, so serving the wrong one silently corrupts every
    placement decision rather than failing. Sidecar-less checkpoints predate the split and
    are `src_index_v0` by construction, matching the resolver's default.
    """
    trained = _read_checkpoint_sidecar(model_path).get("topology_feature_contract")
    if not trained:
        return
    trained = validate_topology_feature_contract(trained)
    declared = os.environ.get(TOPOLOGY_FEATURE_CONTRACT_ENV, "").strip()
    if declared:
        require_matching_topology_feature_contract(
            trained, resolve_topology_feature_contract(), model_label=model_label
        )
    else:
        os.environ[TOPOLOGY_FEATURE_CONTRACT_ENV] = trained
    print(
        f"[TOPOLOGY FEATURES] {model_label} trained under "
        f"{os.environ[TOPOLOGY_FEATURE_CONTRACT_ENV]}",
        flush=True,
    )


def apply_checkpoint_inference_feature_layout(model_path: Path, model_label: str) -> None:
    """Adopt (or verify) the platform-feature layout a checkpoint was trained under.

    Weight shapes pin the feature *dimension* but not its *meaning*: a task_dim=3 /
    platform_dim=14 checkpoint is served as `atomic21` by this loader's default, yet every
    live-gate script in `scripts_cosim/important/` exports `INFERENCE_FEATURE_LAYOUT=dim22`
    for exactly these checkpoints. Whichever is right, it must not depend on whether a
    caller remembered to export the variable — so a checkpoint that declares its layout
    wins, and a conflicting declaration is an error rather than a silent override.
    """
    trained = _read_checkpoint_sidecar(model_path).get("inference_feature_layout")
    if not trained:
        return
    trained = str(trained).strip().lower()
    declared = os.environ.get("INFERENCE_FEATURE_LAYOUT", "").strip().lower()
    if declared and declared != trained:
        raise ValueError(
            f"{model_label} was trained with inference feature layout {trained!r} but this "
            f"run declares INFERENCE_FEATURE_LAYOUT={declared!r}. The layouts assign "
            "different meanings to the same platform feature columns; serving the wrong "
            "one corrupts every score without changing any tensor shape."
        )
    os.environ["INFERENCE_FEATURE_LAYOUT"] = trained
    print(f"[FEATURE LAYOUT] {model_label} trained under {trained}", flush=True)


def check_checkpoint_corpus_compatibility(
    model_path: Path, model_label: str, space_config: Optional[Dict[str, Any]]
) -> None:
    """Compare the live infrastructure against the corpus the checkpoint trained on.

    Two different severities, on purpose:

    * **Raises** on `warmth_physics`. It changes the simulated cost model, so a mismatch
      makes the live number incomparable to the corpus in a way no amount of care at
      analysis time can repair.
    * **Warns loudly** on cluster size and topology density. Both the GNN's
      `build_inference_graph` and `PointwiseEdgeMLP` are candidate-pair based, so a model
      genuinely *can* run at another size — that is the `topology_transfer_v1` question,
      not an error. But it must never happen by accident, unnoticed: the existing
      sealed-holdout gate ran 40/40 p50 against a 20/20 p25 corpus and nothing said so.

    Only fields the sidecar actually declares are checked, so older checkpoints keep
    loading unchanged.
    """
    payload = _read_checkpoint_sidecar(model_path)
    corpus = payload.get("corpus")
    if not corpus or not space_config:
        return

    trained_warmth = corpus.get("warmth_physics")
    if trained_warmth:
        serving_warmth = os.environ.get("HEROSIM_WARMTH_PHYSICS", "").strip()
        if serving_warmth and serving_warmth != trained_warmth:
            raise ValueError(
                f"{model_label} was trained under warmth physics {trained_warmth!r} but "
                f"this run declares HEROSIM_WARMTH_PHYSICS={serving_warmth!r}. The cost "
                "model differs; the resulting RTT is not comparable to the training corpus."
            )
        if not serving_warmth:
            os.environ["HEROSIM_WARMTH_PHYSICS"] = trained_warmth

    warnings: List[str] = []
    live_topology = space_config.get("network", {}).get("topology", {})
    live_shape = {
        "client_node_count": space_config.get("nodes", {}).get("client_nodes", {}).get("count"),
        "server_node_count": space_config.get("nodes", {}).get("server_nodes", {}).get("count"),
        "topology_type": live_topology.get("type"),
        "connection_probability": live_topology.get("connection_probability"),
    }
    for key, live_value in live_shape.items():
        if live_value is None:
            continue
        # A corpus may span several values of an axis (the full siv1 corpus mixes six
        # connection probabilities). Then "in distribution" means membership, not equality.
        allowed = corpus.get(f"{key}_values")
        if allowed is not None:
            if live_value not in allowed:
                warnings.append(f"{key}: live={live_value} not in trained set {sorted(allowed)}")
            continue
        trained_value = corpus.get(key)
        if trained_value is None:
            continue
        if trained_value != live_value:
            warnings.append(f"{key}: trained={trained_value} live={live_value}")

    if warnings:
        print(
            f"\n!! INFRA MISMATCH: {model_label} is being served on infrastructure that "
            f"differs from its training corpus:\n"
            + "".join(f"     - {w}\n" for w in warnings)
            + "   The model will still run (both model classes are candidate-pair based), "
            "but this is\n   an out-of-distribution evaluation. Verify with "
            "scripts_cosim/verify_live_infra_parity.py.\n",
            flush=True,
        )
    else:
        print(
            f"[CORPUS] {model_label} infrastructure matches its training corpus "
            f"({live_shape['client_node_count']}c/{live_shape['server_node_count']}s "
            f"{live_shape['topology_type']} p={live_shape['connection_probability']})",
            flush=True,
        )


def checkpoint_mp_config(model_path: Path) -> dict:
    """Message-passing options a GNN checkpoint was trained with, from its sidecar.

    Serving the wrong message-passing graph is not a soft degradation: it cost 12.4x live
    RTT on 2026-08-16. `mp_node_edges` cannot be recovered from weight shapes, so a
    checkpoint trained with same-node edges MUST carry a sidecar declaring them.
    Sidecar-less checkpoints predate the flag and are bipartite-only by construction.
    """
    payload = _read_checkpoint_sidecar(model_path)
    if not payload:
        return {}
    config = {
        key: bool(payload[key])
        for key in ("mp_residual", "mp_node_edges", "mp_node_edges_candidates_only")
        if key in payload
    }
    # Not a bool: which network entities the training graph contained. Recoverable from
    # weights only as "some encoder exists", never as *which* contract built the features,
    # so it has to come from here.
    if "network_graph_contract" in payload:
        config["network_graph_contract"] = str(payload["network_graph_contract"])
    return config


def apply_mlp_checkpoint_queue_feature_contract(model_path: Path, model_label: str) -> None:
    """Same as the GNN sidecar path, but for MLP checkpoints, which are dicts.

    `MLPBatchScheduler.set_models` also adopts the contract, but it runs *after*
    `build_run_provenance`, so provenance would otherwise record the pre-load default
    rather than what actually served. Adopting here keeps the record truthful; the later
    call then sees a matching declaration and is a no-op.
    """
    import torch

    checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        return
    trained = checkpoint.get("queue_feature_contract")
    if not trained:
        return
    _adopt_queue_feature_contract(trained, model_label, model_path.name)


def resolve_serving_device():
    """Resolve the torch device used for GNN serving.

    Controlled by HEROSIM_GNN_DEVICE: 'cpu' (default), 'cuda' (require CUDA, fail
    loud if absent), or 'auto' (cuda-if-available, the pre-2026-08-25 behavior).

    Default is cpu for PARITY, not for speed. Measured 2026-08-25 on a 30k-event episode
    (cell01, workload-150-100-30k): cuda 72 s vs cpu 86 s — cuda is the FASTER device
    here, so this default costs ~19% on a GPU box. What it buys is that a local run and a
    datalab CPU-amd gate run resolve to the same device: cuda is the only axis PARITY.md
    finds that moves GNN logits at all (1.9e-5), and it is visible end-to-end — the same
    cell's total_rtt differs by 4.6e-6 relative between the two devices here.

    The episode speedup the profiling predicted from "serve on CPU" did NOT come from the
    device. It came from not calling `Data.to()` — see move_graph_tensors_ in
    policy/gnn/scheduler.py, worth 93 s -> 72 s on the same episode, on cuda.
    """
    import torch

    requested = os.environ.get("HEROSIM_GNN_DEVICE", "cpu").strip().lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "HEROSIM_GNN_DEVICE=cuda but torch.cuda.is_available() is False"
            )
        return torch.device("cuda")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError(
        f"HEROSIM_GNN_DEVICE={requested!r} — expected 'cpu', 'cuda' or 'auto'"
    )


def load_gnn_model(model_path: Path, space_config: Optional[Dict[str, Any]] = None):
    """Load the trained GNN model.

    `space_config` is optional so existing callers keep working, but passing it enables
    the corpus-compatibility check — without it a size/density mismatch between the live
    infrastructure and the training corpus goes unreported.
    """
    import torch
    from src.policy.gnn.gnn_model import TaskPlacementGNN
    
    try:
        device = resolve_serving_device()
        print(f"Loading GNN model from {model_path} on {device}...", flush=True)

        state_dict = torch.load(model_path, map_location='cpu')
        _label = f"GNN checkpoint {model_path.name}"
        apply_checkpoint_queue_feature_contract(model_path, _label)
        apply_checkpoint_topology_feature_contract(model_path, _label)
        apply_checkpoint_inference_feature_layout(model_path, _label)
        check_checkpoint_corpus_compatibility(model_path, _label, space_config)
        task_feature_dim = int(state_dict["task_encoder.net.0.weight"].shape[1])
        platform_feature_dim = int(state_dict["platform_encoder.net.0.weight"].shape[1])
        embedding_dim = 64
        edge_fc1_in = int(state_dict["edge_scorer.fc1.weight"].shape[1])
        edge_dim = edge_fc1_in - 2 * embedding_dim
        if edge_dim < 0:
            raise ValueError(
                f"Cannot infer edge_dim from edge_scorer.fc1 in_dim={edge_fc1_in}"
            )

        # A task_dim=3 / platform_dim=14 checkpoint is structurally valid under BOTH
        # atomic21 and dim22 — the layouts assign different meanings to the same platform
        # columns (dim22 normalizes the queue features, atomic21 does not), so the weight
        # shapes cannot disambiguate them and neither load nor forward raises. Defaulting
        # silently to atomic21 is how the prefixctl and tempfix gates came to serve a
        # different layout than every deployed-checkpoint gate (which served dim22 from a
        # sidecar), with nothing in the result but an easily-missed banner line. For that
        # ambiguous shape, refuse to guess.
        declared_layout = os.environ.get("INFERENCE_FEATURE_LAYOUT", "").strip()
        if not declared_layout and task_feature_dim == 3 and platform_feature_dim == 14:
            raise ValueError(
                f"{model_path.name}: task_dim=3 / platform_dim=14 is served under either "
                f"'atomic21' or 'dim22', and this run declares neither — the checkpoint has "
                f"no inference_feature_layout in its .contract.json and "
                f"INFERENCE_FEATURE_LAYOUT is unset. The two layouts give the same tensor "
                f"shapes different meanings (dim22 normalizes the platform queue features), "
                f"so guessing silently changes every score. Declare one explicitly, or "
                f"retrain with a trainer that records it in the sidecar."
            )
        layout = (declared_layout or "atomic21").lower()
        if task_feature_dim == 3 and platform_feature_dim == 6:
            os.environ["INFERENCE_FEATURE_LAYOUT"] = "ce_reduced"
            print(
                f"Using ce_reduced inference layout "
                f"(task_dim={task_feature_dim}, platform_dim={platform_feature_dim}, edge_dim={edge_dim})",
                flush=True,
            )
        elif task_feature_dim == 3 and platform_feature_dim == 16:
            os.environ["INFERENCE_FEATURE_LAYOUT"] = "dim24"
            print(
                f"Using dim24 pull-observable inference layout "
                f"(task_dim={task_feature_dim}, platform_dim={platform_feature_dim})",
                flush=True,
            )
        elif layout in ("dim22", "legacy", "22") or (
            task_feature_dim == 3
            and platform_feature_dim == 14
            and layout not in ("atomic21", "21", "ce_reduced", "dim24", "24")
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

        # Reconstruct the architecture the weights were actually fitted with.
        # `mp_gate` is present iff the checkpoint was trained with the GIN residual, so it
        # is authoritative; the sidecar supplies what weights cannot encode (node edges).
        mp_cfg = checkpoint_mp_config(model_path)
        mp_residual = "mp_gate" in state_dict
        if mp_cfg.get("mp_residual", mp_residual) != mp_residual:
            raise ValueError(
                f"{model_path.name}: sidecar says mp_residual={mp_cfg['mp_residual']} but "
                f"the state dict {'has' if mp_residual else 'lacks'} an 'mp_gate' weight"
            )
        mp_node_edges = mp_cfg.get("mp_node_edges", False)
        # The sidecar is authoritative here; refuse to let a stale env var be silently
        # ignored (it used to be the only control, so it WILL still be set in old scripts).
        _env_node_edges = os.environ.get("GNN_MP_NODE_EDGES", "").strip().lower()
        if _env_node_edges not in ("", "0", "false", "no") and not mp_node_edges:
            raise ValueError(
                f"GNN_MP_NODE_EDGES={_env_node_edges!r} but {model_path.name} was not "
                f"trained with same-node edges (per {model_path.stem}.contract.json). "
                "Serving them is the 12.4x-RTT regression; retrain with "
                "NEAR_RTT_MP_NODE_EDGES=1 instead of forcing it at serve time."
            )

        # Network entities. Unlike mp_node_edges these ARE visible in the weights (two
        # extra encoders), so the state dict is authoritative for *whether* and the
        # sidecar for *which contract* built the features. Both must line up with the
        # contract this run resolves, or the served graph is not the trained graph.
        mp_network_entities = any(
            key.startswith("net_node_encoder.") for key in state_dict
        )
        trained_net_contract = mp_cfg.get("network_graph_contract")
        if mp_network_entities and trained_net_contract is None:
            raise ValueError(
                f"{model_path.name} has network-entity encoders but its sidecar declares "
                f"no network_graph_contract, so there is no way to know which graph it was "
                f"fitted on. Retrain with a trainer that records it."
            )
        declared_net_contract = os.environ.get(NETWORK_GRAPH_CONTRACT_ENV, "").strip()
        if declared_net_contract:
            require_matching_network_graph_contract(
                trained_net_contract if mp_network_entities else NETWORK_GRAPH_CONTRACT_OFF,
                resolve_network_graph_contract(),
                model_label=model_path.name,
            )
        elif mp_network_entities:
            os.environ[NETWORK_GRAPH_CONTRACT_ENV] = trained_net_contract

        model = TaskPlacementGNN(
            task_feature_dim=task_feature_dim,
            platform_feature_dim=platform_feature_dim,
            embedding_dim=embedding_dim,
            hidden_dim=64,
            num_layers=3,
            edge_dim=edge_dim,
            normalize_platform_inputs="platform_input_norm.weight" in state_dict,
            mp_residual=mp_residual,
            mp_node_edges=mp_node_edges,
            mp_node_edges_candidates_only=mp_cfg.get("mp_node_edges_candidates_only", True),
            mp_network_entities=mp_network_entities,
        )
        print(
            f"[GNN] message passing: residual={mp_residual} node_edges={mp_node_edges} "
            f"candidates_only={mp_cfg.get('mp_node_edges_candidates_only', True)} "
            f"network_entities={mp_network_entities}"
            + (f" ({trained_net_contract})" if mp_network_entities else ""),
            flush=True,
        )
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        mp = os.environ.get("GNN_DISABLE_MESSAGE_PASSING", "").strip().lower()
        if mp in ("1", "true", "yes"):
            print(
                "[GNN] GNN_DISABLE_MESSAGE_PASSING=1 — GIN aggregation skipped; "
                "encoder embeddings go straight to the edge scorer",
                flush=True,
            )
        
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
        device = resolve_serving_device()
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


def build_run_provenance(space_config: Dict[str, Any], policy: str) -> Dict[str, Any]:
    """
    Capture the run knobs that silently change results across sweeps.

    warmth_physics alone moves live total RTT by ~100x, and batch window / decode
    mode / feature layout are env-driven, so results must carry them.
    """
    from src.placement.env_fingerprint import (
        describe_code_provenance,
        describe_python_env,
        env_fingerprint,
        format_code_banner,
        format_env_banner,
    )
    from src.placement.warmth import describe_warmth_physics, require_explicit_warmth_physics

    descriptor = describe_warmth_physics(space_config.get("warmth_physics"))
    require_explicit_warmth_physics(descriptor)

    provenance: Dict[str, Any] = dict(descriptor)
    provenance["defer_cold_replica_init"] = space_config.get(
        "defer_cold_replica_init", _env_bool("HEROSIM_DEFER_COLD_REPLICA_INIT")
    )
    provenance["env"] = {
        name: os.environ.get(name)
        for name in (
            "GNN_BATCH_SIZE",
            "GNN_BATCH_TIMEOUT",
            "GNN_DECODE_MODE",
            "GNN_DECODE_TOP_K",
            "GNN_QUEUE_NORM_MODE",
            "GNN_DISABLE_MESSAGE_PASSING",
            "GNN_MP_NODE_EDGES",
            "GNN_LQB_LAMBDA",
            "GNN_QUEUE_FILTER_MAX_DELTA",
            "GNN_SEQBLEND_QUEUE_MARGIN",
            "HEROSIM_GNN_DEVICE",
            "INFERENCE_FEATURE_LAYOUT",
            "KNATIVE_BATCH_SIZE",
            "KNATIVE_BATCH_TIMEOUT",
            "GNN_MODEL_PATH",
            "MLP_MODEL_PATH",
            "TOPOLOGY_FEATURE_CONTRACT",
            "NETWORK_GRAPH_CONTRACT",
            QUEUE_FEATURE_CONTRACT_ENV,
        )
    }
    provenance["slurm"] = {
        name: os.environ.get(name)
        for name in (
            "SLURM_JOB_ID",
            "SLURM_ARRAY_JOB_ID",
            "SLURM_ARRAY_TASK_ID",
        )
        if os.environ.get(name) is not None
    }
    provenance["policy"] = policy
    # dim7/dim13 scaling changes queue ranking, so it belongs next to warmth_physics.
    provenance["queue_feature_contract"] = resolve_queue_feature_contract()

    # Which code, and which interpreter, actually produced this number. Both axes have
    # silently invalidated a gate: an uncommitted feature fix (23.3% of total_rtt, job
    # 708549) and an undeclared venv. See src/placement/env_fingerprint.py.
    provenance["code"] = describe_code_provenance()
    python_env = describe_python_env()
    provenance["python_env"] = python_env
    provenance["env_fingerprint"] = env_fingerprint(python_env)
    print(format_code_banner(provenance["code"]), flush=True)
    print(format_env_banner(python_env), flush=True)

    banner = (
        f"[PHYSICS] warmth_physics={provenance['warmth_physics']} "
        f"(source={provenance['warmth_physics_source']})"
    )
    print(banner, flush=True)
    if provenance["warmth_physics_source"] == "default":
        print(
            "[PHYSICS] WARNING: physics not declared by config or env — "
            "this run is NOT comparable to sweeps that declared node_disk_v2.",
            flush=True,
        )
    return provenance


def _resolve_queue_length(explicit: Optional[int] = None) -> int:
    """Target concurrency per platform for Knative-family autoscaling."""
    if explicit is not None:
        return int(explicit)
    env_raw = os.environ.get("HEROSIM_QUEUE_LENGTH")
    if env_raw is not None and env_raw.strip() != "":
        return int(env_raw)
    return int(QUEUE_LENGTH)


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
        queue_length: Optional[int] = None,
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
        'knative_network_ect_pull',
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

        # Adopt the MLP's contract before provenance so the record matches what serves.
        if policy == 'mlp_batch' and mlp_model_path is not None:
            _mlp_label = f"MLP checkpoint {mlp_model_path.name}"
            apply_mlp_checkpoint_queue_feature_contract(mlp_model_path, _mlp_label)
            # Same protections the GNN path gets (no-ops on sidecar-less checkpoints):
            # topology contract and warmth/corpus compatibility must not depend on which
            # model class is being served.
            apply_checkpoint_topology_feature_contract(mlp_model_path, _mlp_label)
            check_checkpoint_corpus_compatibility(mlp_model_path, _mlp_label, space_config)

        # Before any simulation work: declared physics decides comparability.
        run_provenance = build_run_provenance(space_config, policy)

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
        elif policy == 'knative_network_ect_pull':
            scheduling_strategy = 'kn_network_ect_pull_kn_network_ect_pull'
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

        resolved_queue_length = _resolve_queue_length(queue_length)
        logger.info(
            f"Running {policy} simulation with strategy {scheduling_strategy} "
            f"(target_concurrency/queue_length={resolved_queue_length})..."
        )
        print(
            f"  Running {policy} simulation with strategy {scheduling_strategy} "
            f"(queue_length={resolved_queue_length})..."
        )

        # Execute simulation
        result = execute_simulation(
            full_config,
            sim_inputs,
            scheduling_strategy=scheduling_strategy,
            cache_policy='fifo',
            task_priority='fifo',
            keep_alive=KEEP_ALIVE,
            queue_length=resolved_queue_length,
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
            # The seed actually used: the CLI value defaults to the topology seed (or 42)
            # at placement_seed resolution above. "seed": null does NOT mean unseeded.
            "placement_seed": placement_seed,
            "queue_length": resolved_queue_length,
            "total_rtt": total_rtt,
            "num_tasks": num_tasks,
            "run_provenance": run_provenance,
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

        if _workload_has_burst_ids(workload):
            from scripts_cosim.regime_b_metrics import regime_b_metrics_from_simulation
            from src.executecosimulation import extract_task_metrics

            regime_b = regime_b_metrics_from_simulation(
                workload,
                stats,
                extract_task_metrics_fn=extract_task_metrics,
            )
            result_summary["regime_b"] = regime_b
            result_summary["regime_b_primary_score_s"] = regime_b["regime_b_primary_score_s"]
            print(
                f"  Regime B primary score: {regime_b['regime_b_primary_score_s']:.3f}s "
                f"(total_rtt trap ratio: "
                f"{regime_b['total_rtt_trap']['total_rtt_over_primary_ratio']:.2f}x)",
                flush=True,
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
    queue_length: Optional[int] = None

    if '--queue-length' in sys.argv:
        idx = sys.argv.index('--queue-length')
        if idx + 1 < len(sys.argv):
            try:
                queue_length = int(sys.argv[idx + 1])
            except ValueError:
                print(f"ERROR: Invalid --queue-length value: {sys.argv[idx + 1]}")
                sys.exit(1)

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
            "--policy <knative|gnn|gnn_hetero|roundrobin|knative_network|knative_network_ect|knative_network_ect_pull|knative_network_batch|herocache_network|"
            "herocache_network_batch|random_network|offload_network> "
            "[--seed <seed>] [--output <output.json>]"
        )
        sys.exit(1)
    
    if not policy:
        print("ERROR: --policy is required")
        print(
            "Usage: python -m src.executesimulation "
            "--config <space_config.json> --workload <workload.json> "
            "--policy <knative|gnn|gnn_hetero|roundrobin|knative_network|knative_network_ect|knative_network_ect_pull|knative_network_batch|herocache_network|"
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
        'knative_network_ect_pull',
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

        # `run_simulation` loads this again for the sim itself; the checkpoint's
        # corpus-compatibility check needs it before the model is constructed.
        with open(config_file, 'r') as _f:
            space_config_for_checkpoint = json.load(_f)
        gnn_model, gnn_device = load_gnn_model(
            gnn_model_path, space_config=space_config_for_checkpoint
        )
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
        queue_length=queue_length,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

