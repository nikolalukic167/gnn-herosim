"""
Shared 21-d edge feature construction for tabular rankers and GNN inference.

Train/serve parity: same scaling as prepare_graphs_cache_seq.py / GNNScheduler.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, TYPE_CHECKING

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

if TYPE_CHECKING:
    from src.placement.infrastructure import Node, Platform, Task
    from src.placement.model import SystemState

from src.policy.tabular.constants import FEATURE_DIM
from src.placement.warmth import (
    estimated_pull_remaining_sec,
    node_has_cached_image,
    normalize_estimated_pull_remaining_sec,
    unit_pull_sec_from_task_priors,
)
from src.policy.gnn.gnn_model import build_same_node_edge_index

LEGACY_FEATURE_DIM = 22
LEGACY_TASK_FEATURE_DIM = 3
LEGACY_PLATFORM_FEATURE_DIM = 14

# CACHE 5.6 / dim24: 3-task + 16-plat (+ node_cold_count, estimated_pull_remaining) + 5-edge
DIM24_FEATURE_DIM = 24
DIM24_PLATFORM_FEATURE_DIM = 16

# CE-reduced ablation (train_near_rtt_ce_reduced_features.py on legacy 1060 cache).
CE_REDUCED_TASK_FEATURE_DIM = 3
CE_REDUCED_PLATFORM_FEATURE_DIM = 6
CE_REDUCED_EDGE_FEATURE_DIM = 2
CE_REDUCED_PLATFORM_INDICES = [0, 1, 2, 3, 4, 7]
CE_REDUCED_EDGE_INDICES = [0, 1]
CE_REDUCED_PLATFORM_QUEUE_DIM = 5

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
    feature_dim: int = FEATURE_DIM


def _inference_feature_layout(feature_layout: Optional[str] = None) -> str:
    return (feature_layout or os.environ.get("INFERENCE_FEATURE_LAYOUT", "atomic21")).strip().lower()


def _scheduler_adaptive_queue_norm(queue_values: Sequence[int], queue_norm_mode: str) -> float:
    if not queue_values:
        return 50.0
    values = sorted(int(v) for v in queue_values)
    mode = queue_norm_mode.strip().lower()
    if mode in ("adaptive", "scheduler_adaptive"):
        idx = int(len(values) * 0.9)
        p90 = values[min(idx, len(values) - 1)]
        return float(min(max(1.0, p90), 100.0))
    if mode == "adaptive_nonzero":
        nonzero = [v for v in values if v > 0]
        if not nonzero:
            return 1.0
        idx = int(len(nonzero) * 0.9)
        p90 = nonzero[min(idx, len(nonzero) - 1)]
        return float(min(max(1.0, p90), 100.0))
    return 50.0


def _batch_task_type_names(batch_tasks: Sequence["Task"]) -> Set[str]:
    return {str(task.type["name"]) for task in batch_tasks}


def _platform_node_disk_hit(
    node: Any,
    platform_type: str,
    batch_task_types: Set[str],
) -> float:
    hit = 0.0
    for task_type in batch_task_types:
        task_type_obj = {"name": task_type}
        if node_has_cached_image(node, platform_type, task_type_obj):
            return 1.0
    return hit


def _shared_fate_by_position(platforms_info: Sequence[PlatformInfo]) -> List[float]:
    node_positions: Dict[str, List[int]] = {}
    for info in platforms_info:
        node_positions.setdefault(str(info.node_name), []).append(int(info.position))
    shared_fate = [0.0] * len(platforms_info)
    for info in platforms_info:
        co_located = node_positions.get(str(info.node_name), [info.position])
        cold_count = sum(
            1
            for pos in co_located
            if not platforms_info[pos].platform.initialized.triggered
        )
        shared_fate[info.position] = cold_count / max(len(co_located), 1)
    return shared_fate


def _node_cold_counts_by_position(platforms_info: Sequence[PlatformInfo]) -> List[float]:
    """Absolute cold platform count per node (shared_fate numerator, not density)."""
    node_positions: Dict[str, List[int]] = {}
    for info in platforms_info:
        node_positions.setdefault(str(info.node_name), []).append(int(info.position))
    cold_counts = [0.0] * len(platforms_info)
    for info in platforms_info:
        co_located = node_positions.get(str(info.node_name), [info.position])
        cold_count = sum(
            1
            for pos in co_located
            if not platforms_info[pos].platform.initialized.triggered
        )
        cold_counts[info.position] = float(cold_count)
    return cold_counts


def _uses_dim22_layout(layout: str) -> bool:
    return layout in ("dim22", "legacy", "22", "ce_reduced", "reduced_ce", "reduced1060")


def _uses_dim24_layout(layout: str) -> bool:
    return layout in ("dim24", "24", "pull_obs", "pull_observables")


def _expected_feature_dim_for_layout(layout: str) -> int:
    if _uses_dim24_layout(layout):
        return DIM24_FEATURE_DIM
    if _uses_dim22_layout(layout):
        return LEGACY_FEATURE_DIM
    return FEATURE_DIM


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


def build_inference_feature_bundle(
    batch_tasks: Sequence["Task"],
    system_state: "SystemState",
    queue_snapshot: Mapping[str, int],
    *,
    nodes: Sequence[Any],
    task_types_data: Optional[Mapping[str, Any]] = None,
    queue_norm_mode: str = "adaptive",
    temporal_state: Optional[Mapping[str, Mapping[str, float]]] = None,
    feature_layout: Optional[str] = None,
) -> Optional[InferenceFeatureBundle]:
    """
    Build tabular/GNN features from live or cached system state.

    Returns None when no feasible edges exist.
    """
    layout = _inference_feature_layout(feature_layout)
    use_dim22 = _uses_dim22_layout(layout)
    use_dim24 = _uses_dim24_layout(layout)
    use_norm_queue = use_dim22 or use_dim24
    expected_feature_dim = _expected_feature_dim_for_layout(layout)

    if not batch_tasks:
        return None

    platforms_info = _collect_platforms_info(nodes)
    if not platforms_info:
        return None

    n_tasks = len(batch_tasks)
    n_platforms = len(platforms_info)
    dnn1_replicas, dnn2_replicas = _replica_id_sets(system_state)
    network_maps = _network_maps(nodes)

    node_name_to_idx = {str(node.node_name): idx for idx, node in enumerate(nodes)}

    task_features = []
    for task in batch_tasks:
        task_type = str(task.type["name"])
        onehot = [1.0 if task_type == t else 0.0 for t in TASK_TYPES_VOCAB]
        src_idx = node_name_to_idx.get(str(task.node_name), 0)
        src_norm = float(src_idx) / max(len(nodes), 1)
        task_features.append(onehot + [src_norm])
    task_features_arr = np.asarray(task_features, dtype=np.float32)
    batch_task_types = _batch_task_type_names(batch_tasks)

    raw_queue_by_pos: List[int] = []
    for info in platforms_info:
        queue_key = f"{info.node_name}:{info.platform_id}"
        raw_queue_by_pos.append(int(queue_snapshot.get(queue_key, 0)))
    queue_norm = (
        _scheduler_adaptive_queue_norm(raw_queue_by_pos, queue_norm_mode)
        if use_norm_queue
        else 1.0
    )
    shared_fate_by_pos = (
        _shared_fate_by_position(platforms_info) if use_norm_queue else None
    )
    node_cold_count_by_pos = (
        _node_cold_counts_by_position(platforms_info) if use_dim24 else None
    )

    platform_features: List[List[float]] = []
    queue_key_to_platform_meta: Dict[str, Dict[str, Any]] = {}

    for info in platforms_info:
        onehot = [1.0 if info.platform_type == t else 0.0 for t in PLATFORM_TYPES_VOCAB]
        has_dnn1 = 1.0 if (info.node_id, info.platform_id) in dnn1_replicas else 0.0
        has_dnn2 = 1.0 if (info.node_id, info.platform_id) in dnn2_replicas else 0.0
        queue_key = f"{info.node_name}:{info.platform_id}"
        queue_len_raw = int(queue_snapshot.get(queue_key, 0))
        if use_norm_queue:
            queue_len = float(queue_len_raw) / float(queue_norm)
        else:
            queue_len = float(queue_len_raw)

        is_cold = 0.0 if info.platform.initialized.triggered else 1.0
        shared_fate = (
            float(shared_fate_by_pos[info.position]) if shared_fate_by_pos is not None else 0.0
        )

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

        # Match prepare_graphs_cache.build_graph: only TASK_TYPES_VOCAB (dnn1/dnn2),
        # positive finite exec times, no max(1.0, ...) floor. Including rf/cnn from
        # task-types.json inflates target_concurrency and breaks cache↔live parity.
        baseline_concurrency = 5.0
        target_concurrency = baseline_concurrency
        if task_types_data:
            supported_task_types = [
                task_type_name
                for task_type_name in TASK_TYPES_VOCAB
                if info.platform_type
                in (task_types_data.get(task_type_name) or {}).get("platforms", [])
            ]
            min_exec_times = []
            for task_type_name in supported_task_types:
                task_priors = task_types_data.get(task_type_name, {})
                exec_map = task_priors.get("executionTime", {})
                if isinstance(exec_map, dict) and exec_map:
                    pos_exec = [
                        float(v)
                        for v in exec_map.values()
                        if v is not None and float(v) > 0.0
                    ]
                    if pos_exec:
                        min_exec_times.append(min(pos_exec))
            if min_exec_times:
                avg_min_exec = sum(min_exec_times) / len(min_exec_times)
                if avg_min_exec <= 0.0:
                    avg_min_exec = 1.0
                exec_map_this = task_types_data.get(supported_task_types[0], {}).get(
                    "executionTime", {}
                )
                exec_time_this = (
                    float(exec_map_this.get(info.platform_type, avg_min_exec))
                    if isinstance(exec_map_this, dict)
                    else float(avg_min_exec)
                )
                if exec_time_this > 0:
                    target_concurrency = baseline_concurrency * (
                        avg_min_exec / exec_time_this
                    )

        target_concurrency_raw = float(target_concurrency)
        node_disk_hit = _platform_node_disk_hit(
            info.node, str(info.platform_type), batch_task_types
        )
        if use_norm_queue:
            target_concurrency_feat = target_concurrency_raw / 20.0
            dim13_feat = (
                (float(queue_len_raw) / target_concurrency_raw / 5.0)
                if target_concurrency_raw > 0
                else 0.0
            )
            platform_state_dim = shared_fate
        else:
            target_concurrency_feat = target_concurrency_raw
            dim13_feat = node_disk_hit
            platform_state_dim = is_cold

        queue_key_to_platform_meta[queue_key] = {
            "platform_type": str(info.platform_type),
            "target_concurrency": float(target_concurrency),
            "node_name": str(info.node_name),
            "platform_id": int(info.platform_id),
            "node_id": int(info.node_id),
            "platform_pos": int(info.position),
        }

        plat_row = (
            onehot
            + [has_dnn1, has_dnn2, queue_len]
            + [platform_state_dim]
            + [current_task_remaining_norm, cold_start_remaining_norm, comm_remaining_norm]
            + [target_concurrency_feat, dim13_feat]
        )
        if use_dim24:
            if node_cold_count_by_pos is None:
                raise RuntimeError("dim24 layout requires node_cold_count_by_pos")
            cold_count = float(node_cold_count_by_pos[info.position])
            # Cache path uses default bandwidth only (no per-node override).
            unit_pull = unit_pull_sec_from_task_priors(
                task_types_data,
                str(info.platform_type),
            )
            pull_remaining = estimated_pull_remaining_sec(cold_count, unit_pull)
            plat_row = plat_row + [
                cold_count,
                normalize_estimated_pull_remaining_sec(pull_remaining),
            ]
            if len(plat_row) != DIM24_PLATFORM_FEATURE_DIM:
                raise ValueError(
                    f"Expected {DIM24_PLATFORM_FEATURE_DIM} platform dims for dim24, got {len(plat_row)}"
                )
        elif use_dim22 and len(plat_row) != LEGACY_PLATFORM_FEATURE_DIM:
            raise ValueError(
                f"Expected {LEGACY_PLATFORM_FEATURE_DIM} platform dims for dim22, got {len(plat_row)}"
            )
        platform_features.append(plat_row)

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
        feature_dim=expected_feature_dim,
    )


def edge_row_features(
    bundle: InferenceFeatureBundle,
    task_idx: int,
    logit_idx: int,
) -> np.ndarray:
    """Concatenate 21-d Option B features for one candidate edge."""
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
    expected_dim = int(getattr(bundle, "feature_dim", FEATURE_DIM))
    if feat.shape[0] != expected_dim:
        raise ValueError(f"Feature dim mismatch: {feat.shape[0]} != {expected_dim}")
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
    # Must pass edge_attr into to_undirected — naive clone misaligns reverse-edge
    # attrs after PyG lexicographic reordering (breaks cache↔live edge parity).
    if edge_attr.numel() > 0:
        edge_index, edge_attr = to_undirected(
            edge_index, edge_attr, num_nodes=num_nodes
        )
    else:
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)

    data = Data(
        edge_index=edge_index,
        n_tasks=n_tasks,
        n_platforms=n_platforms,
        task_features=torch.tensor(bundle.task_features, dtype=torch.float32),
        platform_features=torch.tensor(bundle.platform_features, dtype=torch.float32),
    )
    if data.task_features.shape[1] != 3:
        raise ValueError(
            f"Expected 3-d task features (type onehot + src_norm), got {data.task_features.shape[1]}"
        )
    layout = _inference_feature_layout()
    if layout in ("ce_reduced", "reduced_ce", "reduced1060"):
        data.task_features = data.task_features[:, :CE_REDUCED_TASK_FEATURE_DIM]
        data.platform_features = data.platform_features[:, CE_REDUCED_PLATFORM_INDICES]
        if edge_attr.numel() > 0:
            n_dir = edge_attr.shape[0] // 2
            directed = edge_attr[:n_dir][:, CE_REDUCED_EDGE_INDICES]
            # Re-undirect reduced attrs so reverse edges stay aligned with edge_index.
            directed_ei = edge_index[:, :n_dir]
            edge_index, edge_attr = to_undirected(
                directed_ei, directed, num_nodes=num_nodes
            )
    data.edge_attr = edge_attr

    # Same-node platform<->platform edges (GIN co-location signal). Cache always
    # sets this; live must too or train/serve topology diverges.
    node_to_positions: Dict[str, List[int]] = {}
    for meta in bundle.queue_key_to_platform_meta.values():
        node_to_positions.setdefault(str(meta["node_name"]), []).append(
            int(meta["platform_pos"])
        )
    data.node_edge_index = build_same_node_edge_index(node_to_positions, n_tasks)

    data._task_logit_to_queue_key = bundle.task_logit_to_queue_key
    data.task_logit_to_queue_key = bundle.task_logit_to_queue_key
    data.queue_key_to_platform_meta = bundle.queue_key_to_platform_meta
    data.task_logit_to_placement = bundle.task_logit_to_placement
    data._task_logit_to_placement = bundle.task_logit_to_placement
    return data, bundle.task_logit_to_placement
