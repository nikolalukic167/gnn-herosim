"""
Flatten sequential PyG graphs into Option B tabular rows (22-d edge features).

Invariant: emit rows only when task_idx == seq_step (one decision time per graph).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from src.policy.tabular.constants import FEATURE_DIM

PlacementPair = Tuple[int, int]


@dataclass(frozen=True)
class TabularEdgeRow:
    row_id: str
    parent_dataset_id: str
    graph_id: str
    seq_step: int
    seq_n_tasks: int
    task_idx: int
    logit_idx: int
    node_id: int
    platform_id: int
    queue_key: str
    prefix_augment: bool
    y_class: int
    y_logit: int
    features: np.ndarray  # shape (22,)


def _as_numpy(tensor_or_array: Any) -> np.ndarray:
    if isinstance(tensor_or_array, torch.Tensor):
        return tensor_or_array.detach().cpu().numpy()
    return np.asarray(tensor_or_array)


def _task_placement_map(graph: Any) -> Dict[int, List[PlacementPair]]:
    mapping = getattr(graph, "task_logit_to_placement", None)
    if not mapping:
        mapping = getattr(graph, "_task_logit_to_placement", None)
    if not mapping:
        raise ValueError("Graph missing task_logit_to_placement")
    return mapping


def _task_queue_key_map(graph: Any) -> Dict[int, List[str]]:
    mapping = getattr(graph, "task_logit_to_queue_key", None)
    if not mapping:
        raise ValueError("Graph missing task_logit_to_queue_key")
    return mapping


def resolve_platform_pos(
    graph: Any,
    node_id: int,
    platform_id: int,
    queue_key: str,
) -> int:
    """Return platform_features row index for a candidate platform."""
    meta_map: Mapping[str, Mapping[str, Any]] = getattr(graph, "queue_key_to_platform_meta", None) or {}
    meta = meta_map.get(queue_key)
    if meta is not None and "platform_pos" in meta:
        return int(meta["platform_pos"])

    for entry in meta_map.values():
        if int(entry["node_id"]) == int(node_id) and int(entry["platform_id"]) == int(platform_id):
            if "platform_pos" in entry:
                return int(entry["platform_pos"])

    raise ValueError(
        f"platform_pos missing for queue_key={queue_key!r} (node_id={node_id}, platform_id={platform_id}). "
        "Rebuild the sequential graph cache with prepare_graphs_cache_seq.py (version >= 6.5-seq-tabular)."
    )


def _directed_edge_offset(task_placement_map: Mapping[int, Sequence[PlacementPair]], task_idx: int) -> int:
    offset = 0
    for step in range(task_idx):
        offset += len(task_placement_map[step])
    return offset


def _directed_edge_attr_slice(edge_attr: np.ndarray) -> np.ndarray:
    if edge_attr.size == 0:
        return edge_attr
    n_directed = edge_attr.shape[0] // 2
    if edge_attr.shape[0] != 2 * n_directed:
        raise ValueError(
            f"edge_attr length {edge_attr.shape[0]} is not an even count after to_undirected duplication"
        )
    return edge_attr[:n_directed]


def generate_row_id(parent_id: str, graph_id: str, task_idx: int, logit_idx: int) -> str:
    seed = f"{parent_id}_{graph_id}_{task_idx}_{logit_idx}".encode("utf-8")
    return hashlib.md5(seed).hexdigest()


def should_emit_graph(
    graph: Any,
    *,
    regime: str,
    exclude_prefix_augment: bool = True,
) -> bool:
    if regime not in ("batch", "single"):
        raise ValueError(f"Unknown regime {regime!r}; expected 'batch' or 'single'")

    seq_step = int(getattr(graph, "seq_step"))
    prefix_augment = bool(getattr(graph, "prefix_augment", False))

    if exclude_prefix_augment and prefix_augment:
        return False
    if regime == "single" and seq_step != 0:
        return False
    if regime == "batch" and getattr(graph, "seq_n_tasks", None) == 1:
        # 1-task marginal cache: always one decision graph
        return True
    return True


def extract_rows_from_graph(graph: Any, graph_id: str) -> Tuple[List[TabularEdgeRow], Optional[str]]:
    """
    Extract tabular rows for the active decision step on this graph.

    Returns (rows, skip_reason). skip_reason is set when the graph is skipped entirely.
    """
    parent_id = str(getattr(graph, "parent_dataset_id", graph_id))
    seq_step = int(getattr(graph, "seq_step"))
    seq_n_tasks = int(getattr(graph, "seq_n_tasks"))
    prefix_augment = bool(getattr(graph, "prefix_augment", False))
    task_idx = seq_step

    if task_idx != seq_step:
        raise AssertionError("internal error: task_idx must equal seq_step")

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
        if features.shape[0] != FEATURE_DIM:
            raise ValueError(f"Expected {FEATURE_DIM} features, got {features.shape[0]}")
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


def rows_to_dataframe(rows: Sequence[TabularEdgeRow]):
    """Convert rows to a pandas DataFrame (lazy import)."""
    import pandas as pd

    from src.policy.tabular.constants import FEATURE_COLUMN_NAMES

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
        for col, val in zip(FEATURE_COLUMN_NAMES, row.features):
            rec[col] = float(val)
        records.append(rec)
    return pd.DataFrame.from_records(records)


def validate_extracted_frame(df) -> Dict[str, Any]:
    """Run post-extraction invariants; raises ValueError on failure."""
    from src.policy.tabular.constants import FEATURE_COLUMN_NAMES

    if len(df) == 0:
        raise ValueError("Extracted dataframe is empty")

    if not (df["task_idx"] == df["seq_step"]).all():
        raise ValueError("Invariant violated: task_idx != seq_step")

    feature_values = df[FEATURE_COLUMN_NAMES].to_numpy()
    if not np.isfinite(feature_values).all():
        raise ValueError("Non-finite feature values in extracted dataframe")

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
