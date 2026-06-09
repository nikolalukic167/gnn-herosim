"""
Shared 22-d edge feature construction for tabular rankers and GNN inference.

Train/serve parity: same scaling as prepare_graphs_cache_seq.py / GNNScheduler.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

if TYPE_CHECKING:
    from src.placement.infrastructure import Node, Platform, Task
    from src.placement.model import SystemState

from src.policy.tabular.constants import FEATURE_DIM

TASK_PLATFORM_COMPATIBILITY = {
    "dnn1": ["rpiCpu", "xavierGpu", "xavierCpu", "pynqFpga"],
    "dnn2": ["rpiCpu", "xavierGpu", "xavierCpu"],
}

TASK_TYPES_VOCAB = ["dnn1", "dnn2"]
PLATFORM_TYPES_VOCAB = ["rpiCpu", "xavierCpu", "xavierGpu", "xavierDla", "pynqFpga"]


@dataclass
class PlatformInfo:
    node: Any
    platform: Any
    node_id: int
    platform_id: int
    platform_type: str
    node_name: str
    position: int


@dataclass
class InferenceFeatureBundle:
    """Features and mappings for one inference step (1..N tasks)."""

    n_tasks: int
    n_platforms: int
    task_features: np.ndarray
    platform_features: np.ndarray
    edge_attr_directed: np.ndarray
    task_logit_to_placement: Dict[int, List[Tuple[int, int]]]
    task_logit_to_queue_key: Dict[int, List[str]]
    queue_key_to_platform_meta: Dict[str, Dict[str, Any]] = field(default_factory=dict)


def calculate_adaptive_queue_norm(
    queue_snapshot: Mapping[str, int],
    *,
    norm_mode: str = "adaptive",
) -> float:
    """Match GNNScheduler._calculate_adaptive_queue_norm."""
    if not queue_snapshot:
        return 50.0

    queue_values = sorted(int(v) for v in queue_snapshot.values())
    if not queue_values:
        return 50.0

    if norm_mode == "adaptive_nonzero":
        non_zero = [v for v in queue_values if v > 0]
        if not non_zero:
            return 1.0
        idx = int(len(non_zero) * 0.9)
        p90 = non_zero[min(idx, len(non_zero) - 1)]
    else:
        idx = int(len(queue_values) * 0.9)
        p90 = queue_values[min(idx, len(queue_values) - 1)]

    return float(min(max(1.0, p90), 100.0))


def _collect_platforms_info(nodes: Sequence[Any]) -> List[PlatformInfo]:
    platforms_info: List[PlatformInfo] = []
    pos = 0
    for node in nodes:
        for platform in node.platforms.items:
            platforms_info.append(
                PlatformInfo(
                    node=node,
                    platform=platform,
                    node_id=int(node.id),
                    platform_id=int(platform.id),
                    platform_type=str(platform.type["shortName"]),
                    node_name=str(node.node_name),
                    position=pos,
                )
            )
            pos += 1
    return platforms_info


def _replica_id_sets(system_state: "SystemState") -> Tuple[set, set]:
    dnn1_replicas = set()
    dnn2_replicas = set()
    for node, plat in system_state.replicas.get("dnn1", set()):
        dnn1_replicas.add((int(node.id), int(plat.id)))
    for node, plat in system_state.replicas.get("dnn2", set()):
        dnn2_replicas.add((int(node.id), int(plat.id)))
    return dnn1_replicas, dnn2_replicas


def _network_maps(nodes: Sequence[Any]) -> Dict[str, Any]:
    network_maps: Dict[str, Any] = {}
    for node in nodes:
        if hasattr(node, "network_map"):
            network_maps[str(node.node_name)] = node.network_map
    return network_maps


def _node_name_to_id(nodes: Sequence[Any]) -> Dict[str, int]:
    return {str(node.node_name): int(node.id) for node in nodes}


def build_inference_feature_bundle(
    batch_tasks: Sequence["Task"],
    system_state: "SystemState",
    queue_snapshot: Mapping[str, int],
    *,
    nodes: Sequence[Any],
    task_types_data: Optional[Mapping[str, Any]] = None,
    queue_norm_mode: str = "adaptive",
    temporal_state: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> Optional[InferenceFeatureBundle]:
    """
    Build tabular/GNN features from live or cached system state.

    Returns None when no feasible edges exist.
    """
    if not batch_tasks:
        return None

    platforms_info = _collect_platforms_info(nodes)
    if not platforms_info:
        return None

    graph_queue_snapshot = {
        f"{info.node_name}:{info.platform_id}": int(
            queue_snapshot.get(f"{info.node_name}:{info.platform_id}", 0)
        )
        for info in platforms_info
    }
    adaptive_queue_norm = calculate_adaptive_queue_norm(
        graph_queue_snapshot, norm_mode=queue_norm_mode
    )
    n_tasks = len(batch_tasks)
    n_platforms = len(platforms_info)
    node_name_to_id = _node_name_to_id(nodes)
    dnn1_replicas, dnn2_replicas = _replica_id_sets(system_state)
    network_maps = _network_maps(nodes)

    node_to_platform_objs: Dict[str, list] = {}
    for info in platforms_info:
        node_to_platform_objs.setdefault(info.node_name, []).append(info.platform)

    task_features = []
    for task in batch_tasks:
        task_type = str(task.type["name"])
        onehot = [1.0 if task_type == t else 0.0 for t in TASK_TYPES_VOCAB]
        src_node_idx = node_name_to_id.get(str(task.node_name), 0)
        src_norm = src_node_idx / max(len(nodes), 1)
        task_features.append(onehot + [src_norm])
    task_features_arr = np.asarray(task_features, dtype=np.float32)

    platform_features: List[List[float]] = []
    queue_key_to_platform_meta: Dict[str, Dict[str, Any]] = {}

    for info in platforms_info:
        onehot = [1.0 if info.platform_type == t else 0.0 for t in PLATFORM_TYPES_VOCAB]
        has_dnn1 = 1.0 if (info.node_id, info.platform_id) in dnn1_replicas else 0.0
        has_dnn2 = 1.0 if (info.node_id, info.platform_id) in dnn2_replicas else 0.0
        queue_key = f"{info.node_name}:{info.platform_id}"
        queue_len_raw = int(queue_snapshot.get(queue_key, 0))
        queue_len = queue_len_raw / adaptive_queue_norm

        co_located = node_to_platform_objs.get(info.node_name, [])
        node_cold = sum(1 for p in co_located if not p.initialized.triggered)
        shared_fate_signal = node_cold / max(len(co_located), 1)

        temporal = (temporal_state or {}).get(queue_key, {})
        current_task_remaining = float(temporal.get("current_task_remaining", 0.0))
        cold_start_remaining = float(temporal.get("cold_start_remaining", 0.0))
        comm_remaining = float(temporal.get("comm_remaining", 0.0))
        if queue_len_raw > 0 and current_task_remaining == 0.0 and task_types_data:
            avg_exec = 0.0
            count = 0
            for _task_type_name, task_priors in task_types_data.items():
                exec_map = task_priors.get("executionTime", {})
                if isinstance(exec_map, dict):
                    exec_time = exec_map.get(info.platform_type, 0.0)
                    if exec_time > 0:
                        avg_exec += float(exec_time)
                        count += 1
            if count > 0:
                current_task_remaining = avg_exec / count
                cold_start_remaining = current_task_remaining * 0.1
                comm_remaining = current_task_remaining * 0.05

        current_task_remaining_norm = current_task_remaining / 10.0
        cold_start_remaining_norm = cold_start_remaining / 10.0
        comm_remaining_norm = comm_remaining / 10.0

        baseline_concurrency = 5.0
        target_concurrency = baseline_concurrency
        if task_types_data:
            supported_task_types = [
                task_type_name
                for task_type_name, task_priors in task_types_data.items()
                if info.platform_type in task_priors.get("platforms", [])
            ]
            min_exec_times = []
            for task_type_name in supported_task_types:
                task_priors = task_types_data.get(task_type_name, {})
                exec_map = task_priors.get("executionTime", {})
                if isinstance(exec_map, dict) and exec_map:
                    min_exec_times.append(min(exec_map.values()))
            if min_exec_times:
                avg_min_exec = sum(min_exec_times) / len(min_exec_times)
                exec_map_this = task_types_data.get(supported_task_types[0], {}).get("executionTime", {})
                exec_time_this = (
                    exec_map_this.get(info.platform_type, avg_min_exec)
                    if isinstance(exec_map_this, dict)
                    else avg_min_exec
                )
                if exec_time_this > 0:
                    target_concurrency = max(1.0, avg_min_exec / exec_time_this * baseline_concurrency)

        usage_ratio = (queue_len_raw / target_concurrency) if target_concurrency > 0 else 0.0
        target_concurrency_norm = target_concurrency / 20.0
        usage_ratio_norm = usage_ratio / 5.0

        queue_key_to_platform_meta[queue_key] = {
            "platform_type": str(info.platform_type),
            "target_concurrency": float(target_concurrency),
            "node_name": str(info.node_name),
            "platform_id": int(info.platform_id),
            "node_id": int(info.node_id),
            "platform_pos": int(info.position),
        }

        platform_features.append(
            onehot
            + [has_dnn1, has_dnn2, queue_len]
            + [shared_fate_signal]
            + [current_task_remaining_norm, cold_start_remaining_norm, comm_remaining_norm]
            + [target_concurrency_norm, usage_ratio_norm]
        )

    platform_features_arr = np.asarray(platform_features, dtype=np.float32)

    task_offset = 0
    platform_offset = n_tasks
    edge_attrs: List[List[float]] = []
    task_logit_to_placement: Dict[int, List[Tuple[int, int]]] = {}
    task_logit_to_queue_key: Dict[int, List[str]] = {}

    for t_idx, task in enumerate(batch_tasks):
        task_type = str(task.type["name"])
        source_node = str(task.node_name)
        compatible_types = TASK_PLATFORM_COMPATIBILITY.get(task_type, [])
        task_logit_to_placement[t_idx] = []
        task_logit_to_queue_key[t_idx] = []

        for info in platforms_info:
            if info.platform_type not in compatible_types:
                continue

            is_local = source_node == info.node_name
            is_server = not info.node_name.startswith("client_node")
            if not is_local:
                if not is_server:
                    continue
                if info.node_name not in network_maps:
                    continue
                if source_node not in network_maps[info.node_name]:
                    continue

            if task_type == "dnn1" and (info.node_id, info.platform_id) not in dnn1_replicas:
                continue
            if task_type == "dnn2" and (info.node_id, info.platform_id) not in dnn2_replicas:
                continue

            exec_time = 0.0
            if task_types_data and task_type in task_types_data:
                exec_time = float(
                    task_types_data[task_type].get("executionTime", {}).get(info.platform_type, 0.0)
                )

            latency = 0.0
            if not is_local and info.node_name in network_maps:
                lat_entry = network_maps[info.node_name].get(source_node, {})
                if isinstance(lat_entry, dict):
                    latency = float(lat_entry.get("latency", 0.0))
                else:
                    try:
                        latency = float(lat_entry)
                    except (TypeError, ValueError):
                        latency = 0.0

            prev_task = info.platform.previous_task if hasattr(info.platform, "previous_task") else None
            prev_type_name = prev_task.type["name"] if prev_task is not None else None
            is_warm = 1.0 if (prev_type_name is not None and prev_type_name == task_type) else 0.0

            energy = 0.0
            if task_types_data and task_type in task_types_data:
                energy_map = task_types_data[task_type].get("energy", {})
                if isinstance(energy_map, dict):
                    energy = float(energy_map.get(info.platform_type, 0.0))

            comm_time = 0.0
            if task_types_data and task_type in task_types_data:
                state_size_map = task_types_data[task_type].get("stateSize", {})
                if isinstance(state_size_map, dict) and state_size_map:
                    app_state = next(iter(state_size_map.values()))
                    if isinstance(app_state, dict):
                        input_size = app_state.get("input", 0)
                        output_size = app_state.get("output", 0)
                        storage_throughput = 100.0 * 1024 * 1024
                        storage_latency = 0.001
                        read_time = (input_size / storage_throughput) + storage_latency
                        write_time = (output_size / storage_throughput) + storage_latency
                        comm_time = read_time + write_time

            edge_attrs.append([exec_time, latency, is_warm, energy, comm_time])
            task_logit_to_placement[t_idx].append((info.node_id, info.platform_id))
            task_logit_to_queue_key[t_idx].append(f"{info.node_name}:{info.platform_id}")

    if not edge_attrs:
        return None

    edge_attr_directed = np.asarray(edge_attrs, dtype=np.float32)
    if edge_attr_directed.shape[1] != 5:
        raise ValueError(f"Expected 5 edge attrs, got {edge_attr_directed.shape[1]}")

    return InferenceFeatureBundle(
        n_tasks=n_tasks,
        n_platforms=n_platforms,
        task_features=task_features_arr,
        platform_features=platform_features_arr,
        edge_attr_directed=edge_attr_directed,
        task_logit_to_placement=task_logit_to_placement,
        task_logit_to_queue_key=task_logit_to_queue_key,
        queue_key_to_platform_meta=queue_key_to_platform_meta,
    )


def edge_row_features(
    bundle: InferenceFeatureBundle,
    task_idx: int,
    logit_idx: int,
) -> np.ndarray:
    """Concatenate 22-d Option B features for one candidate edge."""
    candidates = bundle.task_logit_to_placement.get(task_idx, [])
    if logit_idx >= len(candidates):
        raise IndexError(f"logit_idx={logit_idx} out of range for task {task_idx}")

    offset = sum(len(bundle.task_logit_to_placement[s]) for s in range(task_idx))
    global_edge_idx = offset + logit_idx

    x_task = bundle.task_features[task_idx]
    node_id, plat_id = candidates[logit_idx]
    queue_keys = bundle.task_logit_to_queue_key.get(task_idx, [])
    queue_key = str(queue_keys[logit_idx]) if logit_idx < len(queue_keys) else ""
    meta = bundle.queue_key_to_platform_meta.get(queue_key)
    if meta is None or "platform_pos" not in meta:
        raise ValueError(f"platform_pos missing for queue_key={queue_key!r}")
    plat_pos = int(meta["platform_pos"])
    x_plat = bundle.platform_features[plat_pos]
    x_edge = bundle.edge_attr_directed[global_edge_idx]
    feat = np.concatenate([x_task, x_plat, x_edge]).astype(np.float32)
    if feat.shape[0] != FEATURE_DIM:
        raise ValueError(f"Feature dim mismatch: {feat.shape[0]} != {FEATURE_DIM}")
    if not np.isfinite(feat).all():
        raise ValueError("Non-finite feature vector")
    return feat


def build_pyg_inference_graph(
    batch_tasks: Sequence["Task"],
    system_state: "SystemState",
    queue_snapshot: Mapping[str, int],
    *,
    nodes: Sequence[Any],
    task_types_data: Optional[Mapping[str, Any]] = None,
    queue_norm_mode: str = "adaptive",
    temporal_state: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> Tuple[Optional[Data], Optional[Dict[int, List[Tuple[int, int]]]]]:
    """Build PyG Data for GNN/XGB batch schedulers."""
    bundle = build_inference_feature_bundle(
        batch_tasks,
        system_state,
        queue_snapshot,
        nodes=nodes,
        task_types_data=task_types_data,
        queue_norm_mode=queue_norm_mode,
        temporal_state=temporal_state,
    )
    if bundle is None:
        return None, None

    n_tasks = bundle.n_tasks
    n_platforms = bundle.n_platforms
    task_offset = 0
    platform_offset = n_tasks

    edge_src: List[int] = []
    edge_dst: List[int] = []
    edge_offset = 0
    for t_idx in range(n_tasks):
        candidates = bundle.task_logit_to_placement.get(t_idx, [])
        for logit_idx, _ in enumerate(candidates):
            edge_src.append(task_offset + t_idx)
            plat_pos = int(
                bundle.queue_key_to_platform_meta[
                    bundle.task_logit_to_queue_key[t_idx][logit_idx]
                ]["platform_pos"]
            )
            edge_dst.append(platform_offset + plat_pos)
            edge_offset += 1

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_attr = torch.tensor(bundle.edge_attr_directed, dtype=torch.float32)
    num_nodes = n_tasks + n_platforms
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    if edge_attr.numel() > 0:
        edge_attr = torch.cat([edge_attr, edge_attr.clone()], dim=0)

    data = Data(
        edge_index=edge_index,
        n_tasks=n_tasks,
        n_platforms=n_platforms,
        task_features=torch.tensor(bundle.task_features, dtype=torch.float32),
        platform_features=torch.tensor(bundle.platform_features, dtype=torch.float32),
    )
    data.edge_attr = edge_attr
    data._task_logit_to_queue_key = bundle.task_logit_to_queue_key
    data.task_logit_to_queue_key = bundle.task_logit_to_queue_key
    data.queue_key_to_platform_meta = bundle.queue_key_to_platform_meta
    return data, bundle.task_logit_to_placement
