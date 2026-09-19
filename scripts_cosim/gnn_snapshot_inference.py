"""Build GNN graphs from live audit snapshots with live-scheduler parity."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from src.notebooks.non_unique_lib.seq_training_utils import decode_sequential_argmax_placement
from src.placement.queue_features import (
    LEGACY_QUEUE_NORM_CAP,
    queue_depth_norm,
    resolve_queue_feature_contract,
)

TASK_PLATFORM_COMPATIBILITY = {
    "dnn1": ["rpiCpu", "xavierGpu", "xavierCpu", "pynqFpga"],
    "dnn2": ["rpiCpu", "xavierGpu", "xavierCpu"],
}

PLATFORM_TYPES_VOCAB = ["rpiCpu", "xavierCpu", "xavierGpu", "xavierDla", "pynqFpga"]
TASK_TYPES_VOCAB = ["dnn1", "dnn2"]
QUEUE_NORM_CAP = LEGACY_QUEUE_NORM_CAP


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def enumerate_infra_platforms(
    infrastructure_nodes: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Assign node/platform ids the same way as create_nodes() in simulation."""
    platforms_info: List[Dict[str, Any]] = []
    node_name_to_id: Dict[str, int] = {}
    node_id = 0
    platform_id = 0
    for node in infrastructure_nodes:
        node_name = str(node.get("node_name", ""))
        node_name_to_id[node_name] = node_id
        for plat_type in node.get("platforms", []):
            platforms_info.append(
                {
                    "node_id": node_id,
                    "platform_id": platform_id,
                    "plat_type": str(plat_type),
                    "node_name": node_name,
                    "network_map": dict(node.get("network_map") or {}),
                }
            )
            platform_id += 1
        node_id += 1
    return platforms_info, node_name_to_id


def replicas_from_snapshot(snapshot: Mapping[str, Any]) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
    dnn1: Set[Tuple[int, int]] = set()
    dnn2: Set[Tuple[int, int]] = set()
    for task in snapshot.get("tasks", []):
        task_type = str(task.get("task_type", ""))
        target = dnn1 if task_type == "dnn1" else dnn2 if task_type == "dnn2" else None
        if target is None:
            continue
        for candidate in task.get("candidates", []):
            node_id = int(candidate.get("node_id", -1))
            plat_id = int(candidate.get("platform_id", -1))
            if node_id >= 0 and plat_id >= 0:
                target.add((node_id, plat_id))
    return dnn1, dnn2


def candidate_lookup_from_tasks(
    tasks: Sequence[Mapping[str, Any]],
) -> Dict[Tuple[int, int, int], Dict[str, Any]]:
    """Map (task_idx, node_id, platform_id) -> audit candidate payload."""
    out: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    for task_idx, task in enumerate(tasks):
        for candidate in task.get("candidates", []):
            key = (
                task_idx,
                int(candidate.get("node_id", -1)),
                int(candidate.get("platform_id", -1)),
            )
            out[key] = candidate
    return out


def temporal_state_from_snapshot(snapshot: Mapping[str, Any]) -> Dict[str, Dict[str, float]]:
    temporal: Dict[str, Dict[str, float]] = {}
    for task in snapshot.get("tasks", []):
        for candidate in task.get("candidates", []):
            queue_key = str(candidate.get("queue_key") or placement_key_from_candidate(candidate))
            temporal[queue_key] = {
                "current_task_remaining": _safe_float(candidate.get("current_task_remaining")),
                "cold_start_remaining": _safe_float(candidate.get("cold_start_remaining")),
                "comm_remaining": _safe_float(candidate.get("comm_remaining")),
            }
    return temporal


def placement_key_from_candidate(candidate: Mapping[str, Any]) -> str:
    return f"{candidate.get('node_name')}:{candidate.get('platform_id')}"


def _calculate_adaptive_queue_norm(queue_snapshot: Mapping[str, int]) -> float:
    if not queue_snapshot:
        return 50.0
    return queue_depth_norm(
        [int(v) for v in queue_snapshot.values()],
        "scheduler_adaptive",
        resolve_queue_feature_contract(),
    )


def _network_latency(source_node: str, target_node: Mapping[str, Any]) -> float:
    target_name = str(target_node.get("node_name", ""))
    if source_node == target_name:
        return 0.0
    network_map = target_node.get("network_map") or {}
    entry = network_map.get(source_node)
    if isinstance(entry, dict):
        return _safe_float(entry.get("latency"))
    return _safe_float(entry)


def _approx_comm_time(task_types_data: Mapping[str, Any], task_type: str) -> float:
    task_priors = task_types_data.get(task_type, {})
    state_size_map = task_priors.get("stateSize", {})
    if not isinstance(state_size_map, dict) or not state_size_map:
        return 0.0
    app_state = next(iter(state_size_map.values()))
    if not isinstance(app_state, dict):
        return 0.0
    input_size = _safe_float(app_state.get("input"))
    output_size = _safe_float(app_state.get("output"))
    storage_throughput = 100.0 * 1024.0 * 1024.0
    storage_latency = 0.001
    return (
        (input_size / storage_throughput + storage_latency)
        + (output_size / storage_throughput + storage_latency)
    )


def build_snapshot_gnn_graph(
    *,
    tasks: Sequence[Mapping[str, Any]],
    snapshot: Mapping[str, Any],
    infrastructure_nodes: Sequence[Mapping[str, Any]],
    task_types_data: Mapping[str, Any],
) -> Tuple[Optional[Data], Optional[Dict[int, List[Tuple[int, int]]]], Optional[Dict[int, List[str]]]]:
    """Full-infra graph + replica-limited edges (matches live GNN scheduler)."""
    if not tasks:
        return None, None, None

    platforms_info, node_name_to_id = enumerate_infra_platforms(infrastructure_nodes)
    if not platforms_info:
        return None, None, None

    queue_snapshot = {
        str(k): int(v or 0)
        for k, v in (snapshot.get("full_queue_snapshot") or {}).items()
    }
    dnn1_replicas, dnn2_replicas = replicas_from_snapshot(snapshot)
    temporal_by_key = temporal_state_from_snapshot(snapshot)

    initialized_by_key: Dict[str, bool] = {}
    for task in snapshot.get("tasks", []):
        for candidate in task.get("candidates", []):
            queue_key = str(candidate.get("queue_key") or placement_key_from_candidate(candidate))
            if queue_key not in initialized_by_key:
                initialized_by_key[queue_key] = bool(candidate.get("initialized", True))

    n_tasks = len(tasks)
    n_platforms = len(platforms_info)

    task_features: List[List[float]] = []
    for task in tasks:
        task_type = str(task.get("task_type", ""))
        onehot = [1.0 if task_type == name else 0.0 for name in TASK_TYPES_VOCAB]
        task_features.append(onehot)

    platform_features: List[List[float]] = []
    for plat in platforms_info:
        node_id = int(plat["node_id"])
        plat_id = int(plat["platform_id"])
        plat_type = str(plat["plat_type"])
        node_name = str(plat["node_name"])
        queue_key = f"{node_name}:{plat_id}"
        queue_len_raw = int(queue_snapshot.get(queue_key, 0))
        queue_len = float(queue_len_raw)

        onehot = [1.0 if plat_type == name else 0.0 for name in PLATFORM_TYPES_VOCAB]
        has_dnn1 = 1.0 if (node_id, plat_id) in dnn1_replicas else 0.0
        has_dnn2 = 1.0 if (node_id, plat_id) in dnn2_replicas else 0.0
        is_cold = 0.0 if initialized_by_key.get(queue_key, True) else 1.0

        temporal = temporal_by_key.get(queue_key, {})
        current_task_remaining = _safe_float(temporal.get("current_task_remaining"))
        cold_start_remaining = _safe_float(temporal.get("cold_start_remaining"))
        comm_remaining = _safe_float(temporal.get("comm_remaining"))
        if queue_len_raw > 0 and current_task_remaining == 0.0:
            avg_exec = 0.0
            count = 0
            for task_type_name, task_priors in task_types_data.items():
                exec_map = task_priors.get("executionTime", {})
                if isinstance(exec_map, dict):
                    exec_time = _safe_float(exec_map.get(plat_type))
                    if exec_time > 0:
                        avg_exec += exec_time
                        count += 1
            if count > 0:
                current_task_remaining = avg_exec / count
                cold_start_remaining = current_task_remaining * 0.1
                comm_remaining = current_task_remaining * 0.05

        baseline_concurrency = 5.0
        target_concurrency = baseline_concurrency
        supported = [
            task_type_name
            for task_type_name, task_priors in task_types_data.items()
            if plat_type in (task_priors.get("platforms") or [])
        ]
        min_exec_times: List[float] = []
        for task_type_name in supported:
            exec_map = task_types_data.get(task_type_name, {}).get("executionTime", {})
            if isinstance(exec_map, dict) and exec_map:
                min_exec_times.append(min(_safe_float(v) for v in exec_map.values()))
        if min_exec_times:
            avg_min_exec = sum(min_exec_times) / len(min_exec_times)
            exec_map_this = task_types_data.get(supported[0], {}).get("executionTime", {})
            exec_time_this = _safe_float(exec_map_this.get(plat_type), avg_min_exec)
            if exec_time_this > 0:
                target_concurrency = max(1.0, avg_min_exec / exec_time_this * baseline_concurrency)

        platform_features.append(
            onehot
            + [has_dnn1, has_dnn2, queue_len]
            + [is_cold]
            + [
                current_task_remaining / 10.0,
                cold_start_remaining / 10.0,
                comm_remaining / 10.0,
                float(target_concurrency),
                0.0,
            ]
        )

    plat_pos_by_key = {
        (int(plat["node_id"]), int(plat["platform_id"])): pos
        for pos, plat in enumerate(platforms_info)
    }
    node_by_name = {str(n.get("node_name")): n for n in infrastructure_nodes}

    edge_src: List[int] = []
    edge_dst: List[int] = []
    edge_attrs: List[List[float]] = []
    task_logit_to_placement: Dict[int, List[Tuple[int, int]]] = {}
    task_logit_to_queue_key: Dict[int, List[str]] = {}

    for t_idx, task in enumerate(tasks):
        task_type = str(task.get("task_type", ""))
        source_node = str(task.get("source_node", ""))
        compatible_types = TASK_PLATFORM_COMPATIBILITY.get(task_type, [])
        task_logit_to_placement[t_idx] = []
        task_logit_to_queue_key[t_idx] = []

        for plat in platforms_info:
            node_id = int(plat["node_id"])
            plat_id = int(plat["platform_id"])
            plat_type = str(plat["plat_type"])
            node_name = str(plat["node_name"])

            if plat_type not in compatible_types:
                continue

            is_local = source_node == node_name
            is_server = not node_name.startswith("client_node")
            if not is_local:
                if not is_server:
                    continue
                target_node = node_by_name.get(node_name)
                if target_node is None:
                    continue
                if source_node not in (target_node.get("network_map") or {}):
                    continue

            if task_type == "dnn1" and (node_id, plat_id) not in dnn1_replicas:
                continue
            if task_type == "dnn2" and (node_id, plat_id) not in dnn2_replicas:
                continue

            pos = plat_pos_by_key.get((node_id, plat_id))
            if pos is None:
                continue

            edge_src.append(t_idx)
            edge_dst.append(n_tasks + pos)

            task_priors = task_types_data.get(task_type, {})
            exec_time = _safe_float((task_priors.get("executionTime") or {}).get(plat_type))
            latency = 0.0 if is_local else _network_latency(source_node, node_by_name.get(node_name, {}))
            is_warm = 1.0 if (
                (task_type == "dnn1" and (node_id, plat_id) in dnn1_replicas)
                or (task_type == "dnn2" and (node_id, plat_id) in dnn2_replicas)
            ) else 0.0
            energy = _safe_float((task_priors.get("energy") or {}).get(plat_type))
            comm_time = _approx_comm_time(task_types_data, task_type)

            edge_attrs.append([exec_time, latency, is_warm, energy, comm_time])
            task_logit_to_placement[t_idx].append((node_id, plat_id))
            task_logit_to_queue_key[t_idx].append(f"{node_name}:{plat_id}")

    if not edge_src:
        return None, None, None

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_index = to_undirected(edge_index, num_nodes=n_tasks + n_platforms)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
    edge_attr = torch.cat([edge_attr, edge_attr.clone()], dim=0)

    data = Data(
        edge_index=edge_index,
        edge_attr=edge_attr,
        n_tasks=n_tasks,
        n_platforms=n_platforms,
        task_features=torch.tensor(task_features, dtype=torch.float32),
        platform_features=torch.tensor(platform_features, dtype=torch.float32),
    )
    return data, task_logit_to_placement, task_logit_to_queue_key


def choose_gnn_live_decode(
    tasks: Sequence[Mapping[str, Any]],
    snapshot: Mapping[str, Any],
    model: Any,
    device: torch.device,
    infrastructure_nodes: Sequence[Mapping[str, Any]],
    task_types_data: Mapping[str, Any],
) -> Optional[List[Dict[str, Any]]]:
    """Sequential argmax + queue roll-forward on a full-infra graph."""
    graph, task_map, queue_keys = build_snapshot_gnn_graph(
        tasks=tasks,
        snapshot=snapshot,
        infrastructure_nodes=infrastructure_nodes,
        task_types_data=task_types_data,
    )
    if graph is None or task_map is None:
        return None

    queue_snapshot = {
        str(k): int(v or 0)
        for k, v in (snapshot.get("full_queue_snapshot") or {}).items()
    }
    candidate_lookup = candidate_lookup_from_tasks(tasks)

    graph = graph.to(device)
    with torch.no_grad():
        logits_per_task = model(graph)

    combo = decode_sequential_argmax_placement(
        logits_per_task,
        task_map,
        len(tasks),
        queue_snapshot,
        queue_keys,
    )
    if combo is None:
        return None

    choices: List[Dict[str, Any]] = []
    for task_idx, (node_id, plat_id) in enumerate(combo):
        candidate = candidate_lookup.get((task_idx, node_id, plat_id))
        if candidate is None:
            return None
        choices.append(candidate)
    return choices
