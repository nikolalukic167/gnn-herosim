"""The two graph-free arms of r1_attribution_v1 (docs/lineages/r1_attribution_v1.md, "Arms").

* ``mlp_same``  -- the same inputs as GNN-eng, scored per (task, candidate) edge, no graph and no node encoders:
  one MLP over the flat vector ``[task features (+ type one-hot) | platform features | edge_attr | partial-state
  block]``. That vector is exactly the per-edge input the GNN's ``EdgeScorer`` sees before its node embeddings
  are substituted for the raw node features, because both read their edge columns through
  ``gnn_model.gather_scorable_edges``.
* ``set_transformer`` -- the same inputs and the same ``EdgeScorer`` head as GNN-eng, with the graph replaced by
  all-pairs self-attention: SAB blocks over the batch's tasks and ISAB blocks over its candidate platforms. No
  task<->platform edge, no peer edge, no same-node edge, no edge attribute enters a context block. Attention rows
  are softmax-normalised, so every pooled set is a mean-type average (``bipartite_aggr_v1``: never a sum over a
  variable-sized set).

Both expose the interface the decoder and the serving loader use (``_encode``, ``_score``, ``forward``,
``partial_state_edge_dim``, ``task_type_onehot_dim``), so ``make_partial_state_score_fn`` and the masked_topo decode
run unchanged. The partial-state columns enter at the scorer only, never a node feature, which is what lets one
encode be reused across every decode step.

The arm travels in the checkpoint sidecar as ``arm_kind`` (weight-VISIBLE for ``mlp_same`` and ``set_transformer``
through their parameter names, weight-INVISIBLE for ``set_heads``), and ``require_arm_matches_weights`` refuses a
sidecar that disagrees with the weights it sits next to.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data

from src.policy.gnn.gnn_model import (
    EdgeScorer,
    MLPEncoder,
    gather_scorable_edges,
    split_scores_per_task,
)

ARM_GNN = "gnn"
ARM_MLP_SAME = "mlp_same"
ARM_SET_TRANSFORMER = "set_transformer"
ARM_KINDS = (ARM_GNN, ARM_MLP_SAME, ARM_SET_TRANSFORMER)
GRAPH_FREE_ARMS = (ARM_MLP_SAME, ARM_SET_TRANSFORMER)

ARM_ENV = "NEAR_RTT_ARM"
SET_HEADS_ENV = "NEAR_RTT_SET_HEADS"
SET_INDUCING_ENV = "NEAR_RTT_SET_INDUCING"
SERVE_ARM_ENV = "GNN_ARM_KIND"
DEFAULT_SET_HEADS = 4
DEFAULT_SET_INDUCING = 8

# sidecar flags that describe graph operations; any of them true on a graph-free arm is a defect, not an option
GRAPH_FLAGS = (
    "mp_residual", "mp_node_edges", "mp_network_entities", "mp_dag_edges", "mp_peer_edges",
    "mp_bipartite_edge_conv", "mp_bipartite_edge_attr_zero", "mp_bipartite_hetero",
    "plan_raw", "plan_raw_sum", "plan_raw_local", "disable_message_passing",
)


def resolve_arm_kind(raw: Optional[str] = None) -> str:
    value = ((os.environ.get(ARM_ENV) if raw is None else raw) or "").strip() or ARM_GNN
    if value not in ARM_KINDS:
        raise ValueError(f"FAIL LOUD: {ARM_ENV}={value!r}; expected one of {ARM_KINDS}")
    return value


def _env_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    value = int(raw)
    if value < 1:
        raise ValueError(f"FAIL LOUD: {name}={raw!r} must be a positive integer")
    return value


def _task_input_features(data: Data, task_type_onehot_dim: int) -> Tensor:
    """task_features, plus the DAG task-type one-hot when the arm uses it (the GNN's own rule)."""
    tf = data.task_features
    if not task_type_onehot_dim:
        return tf
    onehot = getattr(data, "task_type_onehot4", None)
    if onehot is None or int(onehot.size(-1)) != task_type_onehot_dim:
        got = "absent" if onehot is None else f"width {int(onehot.size(-1))}"
        raise ValueError(
            f"FAIL LOUD: task_type_onehot_dim={task_type_onehot_dim} but the graph's task_type_onehot4 is {got}. "
            "Build the cache with --dag-partial-state."
        )
    return torch.cat([tf, onehot.to(tf.device, dtype=tf.dtype)], dim=-1)


class _GraphFreeArm(nn.Module):
    """What the decoder reads off a model, for an arm with no graph operations at all."""

    arm_kind: str = ""

    def __init__(self, task_type_onehot_dim: int, partial_state_edge_dim: int, edge_dim: int,
                 normalize_platform_inputs: bool, platform_feature_dim: int) -> None:
        super().__init__()
        self.task_type_onehot_dim = int(task_type_onehot_dim)
        self.partial_state_edge_dim = int(partial_state_edge_dim)
        self.edge_dim = int(edge_dim)
        # Read by the decoder and the plan-raw dispatch; a graph-free arm has none of them.
        self.plan_raw = False
        self.plan_raw_sum = False
        self.plan_raw_local = False
        self.mp_peer_edges = False
        self.platform_input_norm = nn.LayerNorm(platform_feature_dim) if normalize_platform_inputs else None

    def _platform_input(self, data: Data) -> Tensor:
        feats = data.platform_features
        return feats if self.platform_input_norm is None else self.platform_input_norm(feats)

    def forward(self, data: Data) -> List[Tensor]:
        task_emb, platform_emb = self._encode(data)
        return self._score(task_emb, platform_emb, data)

    def _encode(self, data: Data) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    def _score(self, task_emb: Tensor, platform_emb: Tensor, data: Data) -> List[Tensor]:
        raise NotImplementedError


class TaskPlacementMLPSame(_GraphFreeArm):
    """GNN-eng's inputs, no graph: one MLP over the flat per-edge vector."""

    arm_kind = ARM_MLP_SAME

    def __init__(
        self,
        task_feature_dim: int,
        platform_feature_dim: int,
        hidden_dim: int = 128,
        edge_dim: int = 5,
        dropout: float = 0.1,
        normalize_platform_inputs: bool = False,
        task_type_onehot_dim: int = 0,
        partial_state_edge_dim: int = 0,
        num_hidden_layers: int = 2,
    ) -> None:
        super().__init__(task_type_onehot_dim, partial_state_edge_dim, edge_dim, normalize_platform_inputs,
                         platform_feature_dim)
        if num_hidden_layers < 1:
            raise ValueError("FAIL LOUD: num_hidden_layers must be >= 1")
        self.task_in_dim = task_feature_dim + self.task_type_onehot_dim
        self.platform_in_dim = platform_feature_dim
        in_dim = self.task_in_dim + platform_feature_dim + edge_dim + self.partial_state_edge_dim
        layers: List[nn.Module] = []
        width = in_dim
        for _ in range(num_hidden_layers):
            layers += [nn.Linear(width, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(p=dropout)]
            width = hidden_dim
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def _encode(self, data: Data) -> Tuple[Tensor, Tensor]:
        # No encoding: the raw node features are the "embeddings", gathered per edge in _score.
        return _task_input_features(data, self.task_type_onehot_dim), self._platform_input(data)

    def edge_inputs(self, task_in: Tensor, platform_in: Tensor, data: Data) -> Optional[Tuple[Tensor, Tensor, Tensor]]:
        """(ti, flat per-edge vector [E, in_dim]) or None; the vector the parity test pins."""
        edges = gather_scorable_edges(data, self.partial_state_edge_dim)
        if edges is None:
            return None
        ti, pj, e_attr = edges
        parts = [task_in[ti], platform_in[pj]]
        if e_attr is not None:
            parts.append(e_attr.to(task_in.device, dtype=task_in.dtype))
        return ti, pj, torch.cat(parts, dim=-1)

    def _score(self, task_emb: Tensor, platform_emb: Tensor, data: Data) -> List[Tensor]:
        n_tasks = int(data.n_tasks)
        built = self.edge_inputs(task_emb, platform_emb, data)
        if built is None:
            return [torch.empty(0, device=task_emb.device) for _ in range(n_tasks)]
        ti, _pj, x = built
        return split_scores_per_task(self.net(x).squeeze(-1), ti, n_tasks)


class _MAB(nn.Module):
    """Multihead attention block (Lee et al., Set Transformer): post-norm attention + feed-forward."""

    def __init__(self, dim: int, heads: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(p=dropout), nn.Linear(hidden, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, q: Tensor, kv: Tensor) -> Tensor:
        h = self.norm1(q + self.attn(q, kv, kv, need_weights=False)[0])
        return self.norm2(h + self.ff(h))


class _SAB(nn.Module):
    def __init__(self, dim: int, heads: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.mab = _MAB(dim, heads, hidden, dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.mab(x, x)


class _ISAB(nn.Module):
    """Induced set attention: m learned inducing points attend to the set, the set attends back to them.
    Cost O(n m) instead of O(n^2) -- the candidate-platform set reaches a hundred members at 80 servers."""

    def __init__(self, dim: int, heads: int, hidden: int, dropout: float, inducing: int) -> None:
        super().__init__()
        self.inducing = nn.Parameter(torch.empty(1, inducing, dim))
        nn.init.xavier_uniform_(self.inducing)
        self.mab_in = _MAB(dim, heads, hidden, dropout)
        self.mab_out = _MAB(dim, heads, hidden, dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.mab_out(x, self.mab_in(self.inducing, x))


class TaskPlacementSetTransformer(_GraphFreeArm):
    """GNN-eng's inputs and scorer head, with all-pairs set attention in place of message passing."""

    arm_kind = ARM_SET_TRANSFORMER

    def __init__(
        self,
        task_feature_dim: int,
        platform_feature_dim: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 3,
        edge_dim: int = 5,
        dropout: float = 0.1,
        normalize_platform_inputs: bool = False,
        task_type_onehot_dim: int = 0,
        partial_state_edge_dim: int = 0,
        heads: int = DEFAULT_SET_HEADS,
        inducing: int = DEFAULT_SET_INDUCING,
    ) -> None:
        super().__init__(task_type_onehot_dim, partial_state_edge_dim, edge_dim, normalize_platform_inputs,
                         platform_feature_dim)
        if embedding_dim % heads:
            raise ValueError(f"FAIL LOUD: embedding_dim={embedding_dim} is not divisible by heads={heads}")
        if num_layers < 1:
            raise ValueError("FAIL LOUD: num_layers must be >= 1")
        self.embedding_dim = embedding_dim
        self.set_heads = int(heads)
        self.set_inducing = int(inducing)
        self.task_encoder = MLPEncoder(task_feature_dim + self.task_type_onehot_dim, hidden_dim, embedding_dim, dropout)
        self.platform_encoder = MLPEncoder(platform_feature_dim, hidden_dim, embedding_dim, dropout)
        self.task_blocks = nn.ModuleList(_SAB(embedding_dim, heads, hidden_dim, dropout) for _ in range(num_layers))
        self.platform_blocks = nn.ModuleList(
            _ISAB(embedding_dim, heads, hidden_dim, dropout, inducing) for _ in range(num_layers)
        )
        self.post_set_dropout = nn.Dropout(p=dropout)
        self.edge_scorer = EdgeScorer(embedding_dim, hidden_dim, edge_dim=edge_dim + self.partial_state_edge_dim,
                                      dropout=dropout)

    def _encode(self, data: Data) -> Tuple[Tensor, Tensor]:
        task = self.task_encoder(_task_input_features(data, self.task_type_onehot_dim)).unsqueeze(0)
        platform = self.platform_encoder(self._platform_input(data)).unsqueeze(0)
        for block in self.task_blocks:
            task = block(task)
        for block in self.platform_blocks:
            platform = block(platform)
        return self.post_set_dropout(task.squeeze(0)), self.post_set_dropout(platform.squeeze(0))

    def _score(self, task_emb: Tensor, platform_emb: Tensor, data: Data) -> List[Tensor]:
        n_tasks = int(data.n_tasks)
        edges = gather_scorable_edges(data, self.partial_state_edge_dim)
        if edges is None:
            return [torch.empty(0, device=task_emb.device) for _ in range(n_tasks)]
        ti, pj, e_attr = edges
        e_attr = None if e_attr is None else e_attr.to(task_emb.device, dtype=task_emb.dtype)
        return split_scores_per_task(self.edge_scorer(task_emb[ti], platform_emb[pj], e_attr), ti, n_tasks)


def arm_kind_from_state_dict(state_dict: Mapping[str, Any]) -> str:
    """The arm a set of weights belongs to, from its parameter names."""
    keys = set(state_dict)
    if any(k.startswith("task_blocks.") for k in keys):
        return ARM_SET_TRANSFORMER
    if any(k.startswith("net.") for k in keys) and "task_encoder.net.0.weight" not in keys:
        return ARM_MLP_SAME
    return ARM_GNN


def require_arm_matches_weights(sidecar: Mapping[str, Any], state_dict: Mapping[str, Any], label: str) -> str:
    """Refuse a checkpoint whose sidecar names one arm and whose weights are another's, a graph-free arm that
    declares a graph operation, and (when set) a serving environment that asks for a different arm."""
    declared = str(sidecar.get("arm_kind") or ARM_GNN)
    if declared not in ARM_KINDS:
        raise ValueError(f"{label}: sidecar arm_kind={declared!r} is not one of {ARM_KINDS}")
    actual = arm_kind_from_state_dict(state_dict)
    if declared != actual:
        raise ValueError(f"{label}: sidecar arm_kind={declared!r} but the weights are a {actual!r} arm's -- "
                         "serving would run a different architecture than the one that was trained")
    wanted = (os.environ.get(SERVE_ARM_ENV) or "").strip()
    if wanted and wanted != declared:
        raise ValueError(f"{label}: {SERVE_ARM_ENV}={wanted!r} but the checkpoint is a {declared!r} arm")
    if declared in GRAPH_FREE_ARMS:
        bad = [k for k in GRAPH_FLAGS if sidecar.get(k)]
        if bad:
            raise ValueError(f"{label}: a {declared!r} arm has no graph, but its sidecar declares {bad}")
    return declared



def build_graph_free_arm(
    arm_kind: str,
    *,
    task_feature_dim: int,
    platform_feature_dim: int,
    embedding_dim: int,
    hidden_dim: int,
    num_layers: int,
    edge_dim: int = 5,
    dropout: float = 0.1,
    normalize_platform_inputs: bool = False,
    task_type_onehot_dim: int = 0,
    partial_state_edge_dim: int = 0,
    heads: int = DEFAULT_SET_HEADS,
    inducing: int = DEFAULT_SET_INDUCING,
) -> _GraphFreeArm:
    common = dict(task_feature_dim=task_feature_dim, platform_feature_dim=platform_feature_dim,
                  hidden_dim=hidden_dim, edge_dim=edge_dim, dropout=dropout,
                  normalize_platform_inputs=normalize_platform_inputs,
                  task_type_onehot_dim=task_type_onehot_dim, partial_state_edge_dim=partial_state_edge_dim)
    if arm_kind == ARM_MLP_SAME:
        return TaskPlacementMLPSame(**common, num_hidden_layers=2)
    if arm_kind == ARM_SET_TRANSFORMER:
        return TaskPlacementSetTransformer(**common, embedding_dim=embedding_dim, num_layers=num_layers,
                                           heads=heads, inducing=inducing)
    raise ValueError(f"FAIL LOUD: {arm_kind!r} is not a graph-free arm ({GRAPH_FREE_ARMS})")


def infer_graph_free_dims(arm_kind: str, state_dict: Mapping[str, Tensor], onehot_dim: int,
                          sidecar: Mapping[str, Any]) -> Dict[str, Any]:
    """Constructor arguments recovered from the weights, plus the weight-invisible ones from the sidecar."""
    psd = int(sidecar.get("partial_state_feature_dim") or 0)
    edge_dim = int(sidecar.get("edge_dim") or 5)
    if arm_kind == ARM_MLP_SAME:
        in_dim = int(state_dict["net.0.weight"].shape[1])
        task_dim = int(sidecar["task_feature_dim"])
        platform_dim = int(sidecar["platform_feature_dim"])
        want = task_dim + onehot_dim + platform_dim + edge_dim + psd
        if in_dim != want:
            raise ValueError(f"mlp_same: first layer takes {in_dim} columns but the sidecar's task {task_dim} + "
                             f"one-hot {onehot_dim} + platform {platform_dim} + edge {edge_dim} + partial {psd} "
                             f"= {want}")
        hidden = int(state_dict["net.0.weight"].shape[0])
        return dict(task_feature_dim=task_dim, platform_feature_dim=platform_dim, hidden_dim=hidden,
                    embedding_dim=hidden, num_layers=2, edge_dim=edge_dim)
    task_w = state_dict["task_encoder.net.0.weight"]
    plat_w = state_dict["platform_encoder.net.0.weight"]
    layers = sum(1 for k in state_dict if k.startswith("task_blocks.") and k.endswith(".mab.attn.in_proj_weight"))
    return dict(task_feature_dim=int(task_w.shape[1]) - onehot_dim, platform_feature_dim=int(plat_w.shape[1]),
                hidden_dim=int(task_w.shape[0]), embedding_dim=int(state_dict["task_encoder.net.4.weight"].shape[0]),
                num_layers=layers, edge_dim=edge_dim,
                heads=int(sidecar["set_heads"]), inducing=int(state_dict["platform_blocks.0.inducing"].shape[1]))
