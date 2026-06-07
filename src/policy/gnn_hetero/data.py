from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch_geometric.data import Data, HeteroData


FORWARD_EDGE_TYPE = ("task", "can_run_on", "platform")
REVERSE_EDGE_TYPE = ("platform", "rev_can_run_on", "task")


def build_hetero_graph(
    task_features: Tensor,
    platform_features: Tensor,
    task_to_platform_edge_index: Tensor,
    edge_attr: Tensor | None = None,
    **metadata: Any,
) -> HeteroData:
    """Build the typed bipartite graph used by the hetero trainer and scheduler."""
    data = HeteroData()
    data["task"].x = task_features
    data["platform"].x = platform_features

    if task_to_platform_edge_index.numel() == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=task_features.device)
    else:
        edge_index = task_to_platform_edge_index.to(dtype=torch.long)
    data[FORWARD_EDGE_TYPE].edge_index = edge_index
    data[REVERSE_EDGE_TYPE].edge_index = edge_index.flip(0)

    if edge_attr is not None:
        data[FORWARD_EDGE_TYPE].edge_attr = edge_attr
        data[REVERSE_EDGE_TYPE].edge_attr = edge_attr.clone()

    n_tasks = int(task_features.size(0))
    n_platforms = int(platform_features.size(0))
    data.n_tasks = n_tasks
    data.n_platforms = n_platforms

    for key, value in metadata.items():
        setattr(data, key, value)

    return data


def homogeneous_to_hetero(graph: Data) -> HeteroData:
    """Convert the existing flat bipartite Data graph into HeteroData."""
    if isinstance(graph, HeteroData):
        return graph

    n_tasks = int(graph.n_tasks)
    n_platforms = int(graph.n_platforms)
    edge_index = graph.edge_index

    if edge_index.numel() == 0:
        forward_edge_index = torch.empty((2, 0), dtype=torch.long)
        forward_edge_attr = torch.empty((0, 5), dtype=torch.float32)
    else:
        src = edge_index[0]
        dst = edge_index[1]
        valid = (src >= 0) & (src < n_tasks) & (dst >= n_tasks) & (dst < n_tasks + n_platforms)
        forward_edge_index = torch.stack([src[valid], dst[valid] - n_tasks], dim=0)
        edge_attr = getattr(graph, "edge_attr", None)
        if edge_attr is not None and edge_attr.numel() > 0:
            forward_edge_attr = edge_attr[valid]
        else:
            forward_edge_attr = torch.empty((forward_edge_index.size(1), 5), dtype=torch.float32)

    metadata = {
        "y": getattr(graph, "y", torch.empty((n_tasks,), dtype=torch.long)),
        "task_logit_to_placement": getattr(
            graph,
            "task_logit_to_placement",
            getattr(graph, "_task_logit_to_placement", {}),
        ),
        "_task_logit_to_placement": getattr(
            graph,
            "_task_logit_to_placement",
            getattr(graph, "task_logit_to_placement", {}),
        ),
        "task_logit_to_queue_key": getattr(graph, "task_logit_to_queue_key", {}),
        "_task_logit_to_queue_key": getattr(
            graph,
            "_task_logit_to_queue_key",
            getattr(graph, "task_logit_to_queue_key", {}),
        ),
        "queue_snapshot": getattr(graph, "queue_snapshot", {}),
    }
    for optional in ("dataset_id", "opt_rtt", "parent_dataset_id", "seq_step", "seq_n_tasks"):
        if hasattr(graph, optional):
            metadata[optional] = getattr(graph, optional)

    return build_hetero_graph(
        graph.task_features,
        graph.platform_features,
        forward_edge_index,
        forward_edge_attr,
        **metadata,
    )
