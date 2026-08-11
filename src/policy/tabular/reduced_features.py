"""
CE-reduced feature ablation for tabular MLP (task=3, platform=6, edge=2).

Slices full sequential graph caches in-process — no cache regen.
Matches train_near_rtt_ce_reduced_features.py / feature_builder ce_reduced layout.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from src.policy.tabular.graph_extraction import (
    TabularEdgeRow,
    _as_numpy,
    _directed_edge_attr_slice,
    _directed_edge_offset,
    _task_placement_map,
    _task_queue_key_map,
    generate_row_id,
    resolve_platform_pos,
    should_emit_graph,
)

REDUCED_TASK_FEATURE_DIM = 3
REDUCED_PLATFORM_FEATURE_DIM = 6
REDUCED_EDGE_FEATURE_DIM = 2
REDUCED_FEATURE_DIM = REDUCED_TASK_FEATURE_DIM + REDUCED_PLATFORM_FEATURE_DIM + REDUCED_EDGE_FEATURE_DIM

FULL_PLATFORM_QUEUE_DIM = 7
REDUCED_PLATFORM_QUEUE_DIM = 5
_PLATFORM_FEATURE_INDICES = [0, 1, 2, 3, 4, FULL_PLATFORM_QUEUE_DIM]
_EDGE_FEATURE_INDICES = [0, 1]

REDUCED_FEATURE_COLUMN_NAMES = [f"x_{i}" for i in range(REDUCED_FEATURE_DIM)]


def parent_dataset_id(dataset_id: Any) -> str:
    """Strip @os / @seq suffixes to the canonical co-sim parent id."""
    s = str(dataset_id or "")
    for sep in ("@os", "@seq"):
        idx = s.find(sep)
        if idx >= 0:
            s = s[:idx]
    return s


@lru_cache(maxsize=512)
def _load_parent_task_src_norms(parent_id: str, repo_root: str) -> Tuple[tuple, int]:
    ds_path = Path(repo_root) / "simulation_data" / parent_id
    optimal_path = ds_path / "optimal_result.json"
    if not optimal_path.exists():
        raise FileNotFoundError(f"Missing optimal_result.json for parent {parent_id!r}: {optimal_path}")

    with open(optimal_path, "r", encoding="utf-8") as f:
        optimal = json.load(f)

    task_results = optimal.get("taskResults")
    if not task_results:
        task_results = (optimal.get("stats") or {}).get("taskResults")
    if not task_results:
        raise ValueError(f"No taskResults in {optimal_path}")

    config = optimal.get("config") or {}
    infra = config.get("infrastructure") or {}
    nodes = infra.get("nodes") if isinstance(infra, dict) else None

    if not nodes:
        config_path = ds_path / "space_with_network.json"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                space_cfg = json.load(f)
            if isinstance(space_cfg.get("infrastructure"), dict):
                nodes = space_cfg["infrastructure"].get("nodes")
            if not nodes and isinstance(space_cfg.get("nodes"), list):
                nodes = space_cfg["nodes"]
    if not nodes:
        raise ValueError(f"Cannot resolve node list for parent {parent_id!r}")

    first_idx: Dict[str, int] = {}
    for i, node in enumerate(nodes):
        name = node.get("node_name") or node.get("nodeName") or node.get("name")
        if name is not None:
            first_idx[str(name)] = i
    n_nodes = len(nodes)

    def _task_sort_key(tr: Mapping[str, Any]) -> int:
        for key in ("taskId", "task_id", "id"):
            if key in tr:
                return int(tr[key])
        raise KeyError(f"task result missing id keys in {parent_id}: {list(tr.keys())[:8]}")

    src_norms: List[float] = []
    for tr in sorted(task_results, key=_task_sort_key):
        src = tr.get("sourceNode") or tr.get("source_node") or ""
        idx = first_idx.get(str(src), 0)
        src_norms.append(float(idx) / max(n_nodes, 1))
    return tuple(src_norms), n_nodes


def enrich_task_features_with_src_norm(graph: Any, repo_root: Path) -> None:
    """Promote 2-d seq-cache task onehot to 3-d (onehot + src_norm) when needed."""
    task_features = graph.task_features
    if not isinstance(task_features, torch.Tensor):
        task_features = torch.as_tensor(task_features, dtype=torch.float32)
    if int(task_features.size(-1)) >= REDUCED_TASK_FEATURE_DIM:
        graph.task_features = task_features[:, :REDUCED_TASK_FEATURE_DIM].clone()
        return

    parent_id = str(getattr(graph, "parent_dataset_id", "") or parent_dataset_id(getattr(graph, "dataset_id", "")))
    if not parent_id:
        raise ValueError("Graph missing parent_dataset_id for src_norm enrichment")

    src_norms, _ = _load_parent_task_src_norms(parent_id, str(repo_root.resolve()))
    n_tasks = int(task_features.shape[0])
    if len(src_norms) < n_tasks:
        raise ValueError(
            f"Parent {parent_id!r} has {len(src_norms)} src_norm values but graph has {n_tasks} tasks"
        )
    src_col = torch.tensor(src_norms[:n_tasks], dtype=torch.float32).reshape(-1, 1)
    graph.task_features = torch.cat([task_features[:, :2], src_col], dim=1)


def apply_reduced_features_to_graph(graph: Any, repo_root: Path) -> Any:
    enrich_task_features_with_src_norm(graph, repo_root)
    graph.task_features = graph.task_features[:, :REDUCED_TASK_FEATURE_DIM].clone()
    graph.platform_features = graph.platform_features[:, _PLATFORM_FEATURE_INDICES].clone()
    if hasattr(graph, "edge_attr") and graph.edge_attr is not None and graph.edge_attr.numel() > 0:
        graph.edge_attr = graph.edge_attr[:, _EDGE_FEATURE_INDICES].clone()
    return graph


DIM22_FEATURE_DIM = 22  # 3-task + 14-platform + 5-edge
DIM22_FEATURE_COLUMN_NAMES = [f"x_{i}" for i in range(DIM22_FEATURE_DIM)]


def _extract_dim22_rows_for_task(
    graph: Any,
    graph_id: str,
    parent_id: str,
    task_idx: int,
    seq_n_tasks: int,
    *,
    prefix_augment: bool = False,
) -> Tuple[List[TabularEdgeRow], Optional[str]]:
    """Shared dim22 row builder for one task decision on a graph."""
    task_placement_map = _task_placement_map(graph)
    task_queue_map = _task_queue_key_map(graph)

    if task_idx not in task_placement_map:
        return [], f"task_idx {task_idx} missing from task_logit_to_placement"

    y_raw = getattr(graph, "y", None)
    if y_raw is None:
        return [], "missing y labels"

    target_class_idx = int(y_raw[task_idx].item()) if isinstance(y_raw, torch.Tensor) else int(y_raw[task_idx])
    if target_class_idx < 0:
        return [], f"invalid label y[{task_idx}]={target_class_idx}"

    task_features = _as_numpy(graph.task_features)
    platform_features = _as_numpy(graph.platform_features)
    edge_attr_all = _as_numpy(graph.edge_attr)
    edge_attr_directed = _directed_edge_attr_slice(edge_attr_all)
    edge_offset = _directed_edge_offset(task_placement_map, task_idx)

    if task_features.shape[1] < 3:
        raise ValueError(
            f"graph {graph_id}: task_features has {task_features.shape[1]} cols, expected >= 3"
        )
    if platform_features.shape[1] < 14:
        raise ValueError(
            f"graph {graph_id}: platform_features has {platform_features.shape[1]} cols, expected >= 14"
        )
    if edge_attr_directed.shape[1] < 5:
        raise ValueError(
            f"graph {graph_id}: edge_attr_directed has {edge_attr_directed.shape[1]} cols, expected >= 5"
        )

    candidates = task_placement_map[task_idx]
    queue_keys = task_queue_map[task_idx]
    if len(candidates) != len(queue_keys):
        raise ValueError(
            f"task {task_idx}: placement count {len(candidates)} != queue key count {len(queue_keys)}"
        )

    decision_graph_id = f"{graph_id}@task{task_idx}"
    rows: List[TabularEdgeRow] = []
    for logit_idx, (node_id, plat_id) in enumerate(candidates):
        queue_key = str(queue_keys[logit_idx])
        plat_pos = resolve_platform_pos(graph, int(node_id), int(plat_id), queue_key)
        global_edge_idx = edge_offset + logit_idx
        if global_edge_idx >= edge_attr_directed.shape[0]:
            raise IndexError(
                f"global_edge_idx={global_edge_idx} out of range for directed edge_attr "
                f"(size={edge_attr_directed.shape[0]}, task_idx={task_idx}, logit_idx={logit_idx})"
            )

        x_task = task_features[task_idx, :3]
        x_plat = platform_features[plat_pos, :14]
        x_edge = edge_attr_directed[global_edge_idx, :5]
        features = np.concatenate([x_task, x_plat, x_edge]).astype(np.float64)
        if features.shape[0] != DIM22_FEATURE_DIM:
            raise ValueError(f"Expected {DIM22_FEATURE_DIM} dim22 features, got {features.shape[0]}")
        if not np.isfinite(features).all():
            raise ValueError(
                f"Non-finite features for graph={decision_graph_id} task={task_idx} logit={logit_idx}"
            )

        y_class = 1 if logit_idx == target_class_idx else 0
        rows.append(
            TabularEdgeRow(
                row_id=generate_row_id(parent_id, decision_graph_id, task_idx, logit_idx),
                parent_dataset_id=parent_id,
                graph_id=decision_graph_id,
                seq_step=task_idx,
                seq_n_tasks=seq_n_tasks,
                task_idx=task_idx,
                logit_idx=logit_idx,
                node_id=int(node_id),
                platform_id=int(plat_id),
                queue_key=queue_key,
                prefix_augment=prefix_augment,
                y_class=y_class,
                y_logit=target_class_idx,
                features=features,
            )
        )

    return rows, None


def extract_rows_dim22_from_batch_graph(graph: Any, graph_id: str) -> Tuple[List[TabularEdgeRow], Optional[str]]:
    """Extract dim22 rows for every task in a batch PyG graph (prepare_graphs_cache.py).

    Batch cache platform features already match dim22 inference: normalized queue (dim 7),
    shared_fate (dim 8), usage_ratio (dim 13) — same as GNN wssm training cache.
    """
    parent_id = str(
        getattr(graph, "parent_dataset_id", None) or parent_dataset_id(graph_id)
    )
    n_tasks = int(getattr(graph, "n_tasks"))
    if n_tasks <= 0:
        return [], "n_tasks <= 0"

    all_rows: List[TabularEdgeRow] = []
    for task_idx in range(n_tasks):
        rows, skip_reason = _extract_dim22_rows_for_task(
            graph,
            graph_id,
            parent_id,
            task_idx,
            n_tasks,
        )
        if skip_reason:
            return [], skip_reason
        all_rows.extend(rows)
    return all_rows, None


def extract_rows_dim22_from_graph(graph: Any, graph_id: str) -> Tuple[List[TabularEdgeRow], Optional[str]]:
    """Extract full 22-dim (3-task + 14-plat + 5-edge) rows from a seq cache graph.

    Unlike extract_rows_from_reduced_graph, this keeps all platform and edge features.
    The resulting model is trained with INFERENCE_FEATURE_LAYOUT=dim22, which matches
    build_inference_feature_bundle(use_dim22=True): 3-task + 14-plat + 5-edge = 22-d.

    Key fix vs ce_reduced: retains is_warm (edge dim 2), is_cold/shared_fate (plat dim 8),
    has_dnn1/dnn2 (plat dims 5-6), and temporal dims. Queue is still raw here (plat dim 7)
    vs normalized at inference, but this mismatch is diluted across 21 other features — same
    situation as the working batch_edge_mlp.pt trained on the v6.5 seq cache.
    """
    parent_id = str(getattr(graph, "parent_dataset_id", graph_id))
    seq_step = int(getattr(graph, "seq_step"))
    seq_n_tasks = int(getattr(graph, "seq_n_tasks"))
    prefix_augment = bool(getattr(graph, "prefix_augment", False))
    task_idx = seq_step
    return _extract_dim22_rows_for_task(
        graph,
        graph_id,
        parent_id,
        task_idx,
        seq_n_tasks,
        prefix_augment=prefix_augment,
    )


def dim22_rows_to_dataframe(rows: Sequence[TabularEdgeRow]):
    import pandas as pd

    records: List[Dict[str, Any]] = []
    for row in rows:
        rec: Dict[str, Any] = {
            "row_id": row.row_id,
            "parent_dataset_id": row.parent_dataset_id,
            "graph_id": row.graph_id,
            "seq_step": row.seq_step,
            "seq_n_tasks": row.seq_n_tasks,
            "task_idx": row.task_idx,
            "logit_idx": row.logit_idx,
            "node_id": row.node_id,
            "platform_id": row.platform_id,
            "queue_key": row.queue_key,
            "prefix_augment": int(row.prefix_augment),
            "y_class": row.y_class,
            "y_logit": row.y_logit,
        }
        for col, val in zip(DIM22_FEATURE_COLUMN_NAMES, row.features):
            rec[col] = float(val)
        records.append(rec)
    return pd.DataFrame.from_records(records)


def validate_dim22_frame(df) -> Dict[str, Any]:
    if len(df) == 0:
        raise ValueError("Extracted dim22 dataframe is empty")
    if not (df["task_idx"] == df["seq_step"]).all():
        raise ValueError("Invariant violated: task_idx != seq_step")
    feature_values = df[DIM22_FEATURE_COLUMN_NAMES].to_numpy()
    if not np.isfinite(feature_values).all():
        raise ValueError("Non-finite feature values in dim22 extracted dataframe")
    pos_per_graph = df.groupby("graph_id")["y_class"].sum()
    if not (pos_per_graph == 1).all():
        bad = pos_per_graph[pos_per_graph != 1]
        raise ValueError(f"Expected exactly one positive edge per graph; bad graphs: {bad.to_dict()}")
    return {
        "num_rows": int(len(df)),
        "num_graphs": int(df["graph_id"].nunique()),
        "num_parents": int(df["parent_dataset_id"].nunique()),
        "positives": int(df["y_class"].sum()),
    }


def extract_rows_from_reduced_graph(graph: Any, graph_id: str) -> Tuple[List[TabularEdgeRow], Optional[str]]:
    """Extract reduced-dim tabular rows for the active seq step on this graph."""
    parent_id = str(getattr(graph, "parent_dataset_id", graph_id))
    seq_step = int(getattr(graph, "seq_step"))
    seq_n_tasks = int(getattr(graph, "seq_n_tasks"))
    prefix_augment = bool(getattr(graph, "prefix_augment", False))
    task_idx = seq_step

    task_placement_map = _task_placement_map(graph)
    task_queue_map = _task_queue_key_map(graph)

    if task_idx not in task_placement_map:
        return [], f"task_idx {task_idx} missing from task_logit_to_placement"

    y_raw = getattr(graph, "y", None)
    if y_raw is None:
        return [], "missing y labels"

    target_class_idx = int(y_raw[task_idx].item()) if isinstance(y_raw, torch.Tensor) else int(y_raw[task_idx])
    if target_class_idx < 0:
        return [], f"invalid label y[{task_idx}]={target_class_idx}"

    task_features = _as_numpy(graph.task_features)
    platform_features = _as_numpy(graph.platform_features)
    edge_attr_all = _as_numpy(graph.edge_attr)
    edge_attr_directed = _directed_edge_attr_slice(edge_attr_all)
    edge_offset = _directed_edge_offset(task_placement_map, task_idx)

    candidates = task_placement_map[task_idx]
    queue_keys = task_queue_map[task_idx]
    if len(candidates) != len(queue_keys):
        raise ValueError(
            f"task {task_idx}: placement count {len(candidates)} != queue key count {len(queue_keys)}"
        )

    rows: List[TabularEdgeRow] = []
    for logit_idx, (node_id, plat_id) in enumerate(candidates):
        queue_key = str(queue_keys[logit_idx])
        plat_pos = resolve_platform_pos(graph, int(node_id), int(plat_id), queue_key)
        global_edge_idx = edge_offset + logit_idx
        if global_edge_idx >= edge_attr_directed.shape[0]:
            raise IndexError(
                f"global_edge_idx={global_edge_idx} out of range for directed edge_attr "
                f"(size={edge_attr_directed.shape[0]}, task_idx={task_idx}, logit_idx={logit_idx})"
            )

        x_task = task_features[task_idx]
        x_plat = platform_features[plat_pos]
        x_edge = edge_attr_directed[global_edge_idx]
        features = np.concatenate([x_task, x_plat, x_edge]).astype(np.float64)
        if features.shape[0] != REDUCED_FEATURE_DIM:
            raise ValueError(f"Expected {REDUCED_FEATURE_DIM} features, got {features.shape[0]}")
        if not np.isfinite(features).all():
            raise ValueError(f"Non-finite features for graph={graph_id} task={task_idx} logit={logit_idx}")

        y_class = 1 if logit_idx == target_class_idx else 0
        rows.append(
            TabularEdgeRow(
                row_id=generate_row_id(parent_id, graph_id, task_idx, logit_idx),
                parent_dataset_id=parent_id,
                graph_id=graph_id,
                seq_step=seq_step,
                seq_n_tasks=seq_n_tasks,
                task_idx=task_idx,
                logit_idx=logit_idx,
                node_id=int(node_id),
                platform_id=int(plat_id),
                queue_key=queue_key,
                prefix_augment=prefix_augment,
                y_class=y_class,
                y_logit=target_class_idx,
                features=features,
            )
        )

    return rows, None


def reduced_rows_to_dataframe(rows: Sequence[TabularEdgeRow]):
    import pandas as pd

    records: List[Dict[str, Any]] = []
    for row in rows:
        rec: Dict[str, Any] = {
            "row_id": row.row_id,
            "parent_dataset_id": row.parent_dataset_id,
            "graph_id": row.graph_id,
            "seq_step": row.seq_step,
            "seq_n_tasks": row.seq_n_tasks,
            "task_idx": row.task_idx,
            "logit_idx": row.logit_idx,
            "node_id": row.node_id,
            "platform_id": row.platform_id,
            "queue_key": row.queue_key,
            "prefix_augment": int(row.prefix_augment),
            "y_class": row.y_class,
            "y_logit": row.y_logit,
        }
        for col, val in zip(REDUCED_FEATURE_COLUMN_NAMES, row.features):
            rec[col] = float(val)
        records.append(rec)
    return pd.DataFrame.from_records(records)


def validate_reduced_frame(df) -> Dict[str, Any]:
    if len(df) == 0:
        raise ValueError("Extracted reduced dataframe is empty")

    if not (df["task_idx"] == df["seq_step"]).all():
        raise ValueError("Invariant violated: task_idx != seq_step")

    feature_values = df[REDUCED_FEATURE_COLUMN_NAMES].to_numpy()
    if not np.isfinite(feature_values).all():
        raise ValueError("Non-finite feature values in reduced extracted dataframe")

    pos_per_graph = df.groupby("graph_id")["y_class"].sum()
    if not (pos_per_graph == 1).all():
        bad = pos_per_graph[pos_per_graph != 1]
        raise ValueError(f"Expected exactly one positive edge per graph; bad graphs: {bad.to_dict()}")

    return {
        "num_rows": int(len(df)),
        "num_graphs": int(df["graph_id"].nunique()),
        "num_parents": int(df["parent_dataset_id"].nunique()),
        "positives": int(df["y_class"].sum()),
    }
