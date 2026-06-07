"""Heterogeneous GNN model for task-to-platform placement prediction."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, MessagePassing

from src.policy.gnn_hetero.data import FORWARD_EDGE_TYPE, REVERSE_EDGE_TYPE


class MLPEncoder(nn.Module):
    """Generic 2-layer MLP encoder with LayerNorm (matches train.py / desert-galaxy-26)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class EdgeScorer(nn.Module):
    """2-layer MLP to score task-platform edges with optional edge attributes."""
    def __init__(self, embedding_dim: int, hidden_dim: int, edge_dim: int = 0) -> None:
        super().__init__()
        in_dim = 2 * embedding_dim + (edge_dim if edge_dim else 0)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(hidden_dim, 1)
    
    def forward(
        self,
        e_task: Tensor,
        e_platform: Tensor,
        e_attr: Optional[Tensor] = None,
    ) -> Tensor:
        x = torch.cat([e_task, e_platform] + ([e_attr] if e_attr is not None else []), dim=-1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze(-1)


class BipartiteEdgeConv(MessagePassing):
    """Edge-attribute-aware bipartite message passing for one hetero relation."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        edge_dim: int = 5,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__(aggr="mean")
        self.message_mlp = nn.Sequential(
            nn.Linear(embedding_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(
        self,
        x: tuple[Tensor, Tensor],
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        x_src, x_dst = x
        if edge_attr is None:
            edge_attr = x_src.new_zeros((edge_index.size(1), 5))
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=(x_src.size(0), x_dst.size(0)))
        return self.update_mlp(torch.cat([x_dst, out], dim=-1))

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        return self.message_mlp(torch.cat([x_j, edge_attr], dim=-1))


class TaskPlacementGNN(nn.Module):
    """
    1. Encode task and platform features separately
    2. HeteroConv to produce typed node embeddings
    3. Edge MLP to score task-platform compatibility
    """
    def __init__(
        self,
        task_feature_dim: int,
        platform_feature_dim: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.task_encoder = MLPEncoder(task_feature_dim, hidden_dim, embedding_dim)
        self.platform_encoder = MLPEncoder(platform_feature_dim, hidden_dim, embedding_dim)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                HeteroConv(
                    {
                        FORWARD_EDGE_TYPE: BipartiteEdgeConv(embedding_dim, hidden_dim, edge_dim=5),
                        REVERSE_EDGE_TYPE: BipartiteEdgeConv(embedding_dim, hidden_dim, edge_dim=5),
                    },
                    aggr="sum",
                )
            )
        self.post_gin_dropout = nn.Dropout(p=0.2)
        self.edge_scorer = EdgeScorer(embedding_dim, hidden_dim, edge_dim=5)

    def forward(self, data: HeteroData) -> List[Tensor]:
        n_tasks: int = int(data.n_tasks)

        x_dict: Dict[str, Tensor] = {
            "task": self.task_encoder(data["task"].x),
            "platform": self.platform_encoder(data["platform"].x),
        }
        edge_index_dict = {
            edge_type: data[edge_type].edge_index
            for edge_type in (FORWARD_EDGE_TYPE, REVERSE_EDGE_TYPE)
        }
        edge_attr_dict = {
            edge_type: getattr(data[edge_type], "edge_attr", None)
            for edge_type in (FORWARD_EDGE_TYPE, REVERSE_EDGE_TYPE)
        }

        for conv in self.convs:
            updated = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
            x_dict = {
                node_type: self.post_gin_dropout(F.relu(updated.get(node_type, x_dict[node_type]))) + x_dict[node_type]
                for node_type in x_dict
            }

        task_emb = x_dict["task"]
        platform_emb = x_dict["platform"]

        # Score edges
        ei = data[FORWARD_EDGE_TYPE].edge_index
        if ei.numel() == 0:
            return [torch.empty(0, device=task_emb.device) for _ in range(n_tasks)]

        ti = ei[0]
        pj = ei[1]

        e_task = task_emb[ti]
        e_platform = platform_emb[pj]
        e_attr: Optional[Tensor] = None
        if hasattr(data[FORWARD_EDGE_TYPE], "edge_attr") and data[FORWARD_EDGE_TYPE].edge_attr.numel() > 0:
            e_attr = data[FORWARD_EDGE_TYPE].edge_attr
        edge_scores = self.edge_scorer(e_task, e_platform, e_attr)

        # Split scores per task
        logits_per_task = []
        for t in range(n_tasks):
            mask_t = (ti == t)
            logits_t = edge_scores[mask_t]
            logits_per_task.append(logits_t)

        return logits_per_task

