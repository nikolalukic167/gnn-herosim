"""
GNN Model for Task-to-Platform Placement Prediction

This is a copy of the model architecture from the training script,
used for inference in the co-simulation.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn.models import GIN


def _env_flag(name: str) -> bool:
    """Parse a boolean ablation flag; fail loud on an unrecognized value."""
    raw = os.environ.get(name, "").strip().lower()
    if raw in ("", "0", "false", "no"):
        return False
    if raw in ("1", "true", "yes"):
        return True
    raise ValueError(f"FAIL LOUD: {name}={raw!r} is not a boolean (use 1/0/true/false/yes/no)")


def build_same_node_edge_index(
    node_to_platform_positions: Dict[object, Sequence[int]],
    n_tasks: int,
) -> Tensor:
    """Build undirected platform<->platform edges for platforms on the SAME physical node.

    The bipartite task<->platform graph cannot propagate co-location/contention signal
    between platforms that share a node (shared FilterStore image pulls, shared node
    bandwidth). These extra edges let GIN message passing aggregate that signal, which is
    the structural capability a pointwise MLP fundamentally lacks.

    Platform global index in the graph is ``n_tasks + platform_pos`` (tasks occupy
    indices ``0..n_tasks-1``). The returned tensor uses 'index' in its eventual attr name
    so PyG batches it with the same +num_nodes increment as ``edge_index``.

    Args:
        node_to_platform_positions: physical node -> list of platform row positions.
        n_tasks: number of task nodes (offset for platform indices).

    Returns:
        LongTensor of shape ``[2, E]`` (E may be 0). Undirected (both directions emitted).
    """
    src: List[int] = []
    dst: List[int] = []
    for positions in node_to_platform_positions.values():
        pos = sorted(set(int(p) for p in positions))
        if len(pos) < 2:
            continue
        for a_i in range(len(pos)):
            for b_i in range(a_i + 1, len(pos)):
                ga = n_tasks + pos[a_i]
                gb = n_tasks + pos[b_i]
                src.extend([ga, gb])
                dst.extend([gb, ga])
    if not src:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor([src, dst], dtype=torch.long)


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


class TaskPlacementGNN(nn.Module):
    """
    1. Encode task and platform features separately
    2. GIN to produce node embeddings
    3. Edge MLP to score task-platform compatibility
    4. Masked softmax to predict placement probabilities
    """
    def __init__(
        self,
        task_feature_dim: int,
        platform_feature_dim: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 3,
        edge_dim: int = 5,
    ) -> None:
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.task_encoder = MLPEncoder(task_feature_dim, hidden_dim, embedding_dim)
        self.platform_encoder = MLPEncoder(platform_feature_dim, hidden_dim, embedding_dim)
        
        self.gin = GIN(
            in_channels=embedding_dim,
            hidden_channels=hidden_dim,
            num_layers=num_layers,
            out_channels=embedding_dim
        )
        self.post_gin_dropout = nn.Dropout(p=0.2)
        self.edge_scorer = EdgeScorer(embedding_dim, hidden_dim, edge_dim=edge_dim)

        self._disable_mp = _env_flag("GNN_DISABLE_MESSAGE_PASSING")
        # Same-node platform<->platform edges are OFF by default because
        # train_near_rtt.py fits weights with `self.gin(x, data.edge_index)` — bipartite
        # only. Serving them changed ~87.5% of argmax decisions on the training cache and
        # cost 12.4x live RTT on sparse_p35 (276.0M -> 22.3M once dropped): there are
        # ~1428 same-node edges vs ~57 bipartite, a 25:1 flood that averages co-located
        # platforms together and erases the queue-depth feature that distinguishes them.
        # Only enable this for a checkpoint actually TRAINED with these edges.
        self._mp_node_edges = _env_flag("GNN_MP_NODE_EDGES")

    def forward(self, data: Data) -> List[Tensor]:
        n_tasks: int = int(data.n_tasks)
        n_platforms: int = int(data.n_platforms)

        task_embeddings = self.task_encoder(data.task_features)
        platform_embeddings = self.platform_encoder(data.platform_features)

        # Message passing. The GIN aggregates over the bipartite task<->platform edges
        # PLUS optional platform<->platform edges for platforms on the same physical node
        # (data.node_edge_index). Same-node edges give the GIN the relational signal an MLP
        # cannot see: contention/co-location coupling between platforms sharing a node.
        # Ablation: skip GIN so each platform's encoded features (including dim7/dim13)
        # reach the scorer unsmoothed. Isolates "message passing dilutes queue" from
        # "the scoring head never learned queue".
        if self._disable_mp:
            task_emb = task_embeddings
            platform_emb = platform_embeddings
        else:
            x = torch.cat([task_embeddings, platform_embeddings], dim=0)
            mp_edge_index = data.edge_index
            node_ei = getattr(data, "node_edge_index", None) if self._mp_node_edges else None
            if node_ei is not None and node_ei.numel() > 0:
                mp_edge_index = torch.cat(
                    [data.edge_index, node_ei.to(data.edge_index.device)], dim=1
                )
            x = self.gin(x, mp_edge_index)
            x = self.post_gin_dropout(x)
            task_emb = x[:n_tasks]
            platform_emb = x[n_tasks:]

        device = task_embeddings.device

        # Score edges. Scoring stays on the bipartite task->platform edges only, so
        # edge_attr alignment is preserved and same-node edges never produce logits.
        ei = data.edge_index
        if ei.numel() == 0:
            return [torch.empty(0, device=device) for _ in range(n_tasks)]

        ti = ei[0]
        pj = ei[1] - n_tasks
        # Defensive: only score edges whose source is a task node (filters any
        # platform<->platform edge that may have been merged into edge_index).
        valid = (pj >= 0) & (pj < n_platforms) & (ti < n_tasks)
        ti = ti[valid]
        pj = pj[valid]
        if ti.numel() == 0:
            return [torch.empty(0, device=device) for _ in range(n_tasks)]

        e_task = task_emb[ti]
        e_platform = platform_emb[pj]
        e_attr: Optional[Tensor] = None
        if hasattr(data, 'edge_attr') and data.edge_attr.numel() > 0:
            try:
                e_attr = data.edge_attr[valid]
            except (IndexError, RuntimeError):
                e_attr = None
        edge_scores = self.edge_scorer(e_task, e_platform, e_attr)

        # Split scores per task
        logits_per_task = []
        for t in range(n_tasks):
            mask_t = (ti == t)
            logits_t = edge_scores[mask_t]
            logits_per_task.append(logits_t)

        return logits_per_task

