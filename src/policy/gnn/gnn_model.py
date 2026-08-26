"""
GNN Model for Task-to-Platform Placement Prediction

This is a copy of the model architecture from the training script,
used for inference in the co-simulation.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

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


def restrict_node_edges_to_candidates(
    node_edge_index: Tensor,
    edge_index: Tensor,
    n_tasks: int,
    n_platforms: int,
) -> Tensor:
    """Keep only same-node edges whose BOTH endpoints some task can actually reach.

    A platform is a *candidate* when it appears as a destination of a bipartite
    task->platform edge in this graph. Contention only matters among platforms a task
    could be placed on, so edges between two unreachable platforms carry no decision-
    relevant signal — they only add aggregation mass.

    This is the difference between a flood and a signal. Measured on
    ``graphs_cache_contention_v2_873_v5.7_siv1_dim14`` (300 graphs, 208 platforms):

        bipartite edges                    47.8 mean
        same-node edges, unrestricted    1428.0 mean  (29.9x bipartite)
        same-node edges, candidates only   12.9 mean  ( 0.27x bipartite)

    Unrestricted, the GIN averages every co-located platform together and erases the
    queue-depth feature that distinguishes them (the 12.4x live RTT regression of
    2026-08-16). Restricted, same-node edges are a minority term that sharpens rather
    than swamps the bipartite signal.

    Note 36% of graphs in that cache have zero candidate-restricted same-node edges; for
    those, message passing correctly degenerates to bipartite-only.

    Assumes a single (unbatched) graph: platform global index is ``n_tasks + platform_pos``.
    """
    if node_edge_index.numel() == 0:
        return node_edge_index
    src_pos = node_edge_index[0] - n_tasks
    dst_pos = node_edge_index[1] - n_tasks
    in_range = (
        (src_pos >= 0) & (src_pos < n_platforms)
        & (dst_pos >= 0) & (dst_pos < n_platforms)
    )
    if not bool(in_range.any()):
        return node_edge_index[:, :0]

    bip_pos = edge_index[1] - n_tasks
    bip_valid = (bip_pos >= 0) & (bip_pos < n_platforms) & (edge_index[0] < n_tasks)
    is_candidate = torch.zeros(n_platforms, dtype=torch.bool, device=node_edge_index.device)
    is_candidate[bip_pos[bip_valid]] = True

    keep = in_range.clone()
    keep[in_range] = is_candidate[src_pos[in_range]] & is_candidate[dst_pos[in_range]]
    return node_edge_index[:, keep]


def split_task_platform_embeddings(
    x: Tensor, n_tasks: int, n_platforms: int
) -> tuple[Tensor, Tensor]:
    """Split the post-message-passing stack into its task and platform blocks.

    Bounded on purpose, and shared on purpose. `x[n_tasks:]` is correct only while tasks and
    platforms are the *only* entities in the stack; the moment anything is appended it
    silently feeds foreign rows to the scorer, misaligned against `edge_index`. That
    open-ended slice has now been the bug three times in this repo:

      1. `mp_node_edges` (2026-08-16) — same-node edges merged into the message-passing
         index, 12.4x live RTT.
      2. `TaskPlacementGNN.forward` (2026-08-18) — would have leaked node/link rows into
         `platform_emb` once network entities were appended.
      3. `gnn_necessity_ablation.AblationModel` — the same open slice, in the harness whose
         numbers the pre-registered topology-transfer gate depends on.

    Fixing it per-model as it is found is how it reached three. Every model that concatenates
    entity blocks calls this instead, and the bound is asserted rather than assumed.

    Raises:
        ValueError: if the stack is too short to contain both blocks — which means the
            caller's `n_tasks`/`n_platforms` disagree with what it actually concatenated.
    """
    n_tasks = int(n_tasks)
    n_platforms = int(n_platforms)
    if x.shape[0] < n_tasks + n_platforms:
        raise ValueError(
            f"FAIL LOUD: embedding stack has {x.shape[0]} rows but n_tasks={n_tasks} + "
            f"n_platforms={n_platforms} = {n_tasks + n_platforms} were expected. The model "
            f"concatenated a different set of entities than it is slicing."
        )
    return x[:n_tasks], x[n_tasks : n_tasks + n_platforms]


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
    def __init__(
        self, embedding_dim: int, hidden_dim: int, edge_dim: int = 0, dropout: float = 0.1
    ) -> None:
        super().__init__()
        in_dim = 2 * embedding_dim + (edge_dim if edge_dim else 0)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
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

    THIS IS THE ONE DEFINITION. ``src/notebooks/train_near_rtt.py`` imports this class
    rather than declaring its own copy. Two hand-synced copies is exactly what produced
    the 2026-08-16 train/serve message-passing mismatch (12.4x live RTT); do not
    reintroduce a second one.

    Message-passing options must match between training and serving:

    * ``mp_residual`` is self-describing — it adds a learnable ``mp_gate`` parameter, so a
      residual checkpoint will not strict-load into a non-residual model (or vice versa).
      Silent behavioural drift becomes a loud load error.
    * ``mp_node_edges`` cannot be inferred from weights; trainers record it in the
      ``<model>.contract.json`` sidecar next to the queue feature contract.
    * ``mp_dag_edges`` likewise cannot be inferred from weights, and is recorded in the
      sidecar together with ``mp_dag_edges_undirected`` and ``dag_task_type_vocab``.
      ``task_type_onehot_dim`` and ``partial_state_edge_dim`` ARE weight-visible (they
      widen ``task_encoder.net.0`` and ``edge_scorer.fc1`` respectively), so a strict
      load across those boundaries fails loudly.
    """
    def __init__(
        self,
        task_feature_dim: int,
        platform_feature_dim: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 3,
        edge_dim: int = 5,
        dropout: float = 0.1,
        post_gin_dropout: float = 0.2,
        normalize_platform_inputs: bool = False,
        mp_residual: bool = False,
        mp_node_edges: Optional[bool] = None,
        mp_node_edges_candidates_only: bool = True,
        mp_network_entities: Optional[bool] = None,
        net_node_feature_dim: int = 6,
        net_link_feature_dim: int = 5,
        mp_dag_edges: Optional[bool] = None,
        task_type_onehot_dim: int = 0,
        partial_state_edge_dim: int = 0,
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        # The 4-way task-type one-hot (route_b stage 2) is appended to task_features on
        # the TASK side. It is deliberately NOT covered by platform_input_norm: that
        # LayerNorm is platform-side (driven by the atomic21 layout) and the appended
        # columns are already 0/1 alongside the pre-normalised src_feat scalar, so there
        # is no scale mismatch to fix. Do not "fix" this by widening the norm.
        self.task_type_onehot_dim = int(task_type_onehot_dim)
        self.task_encoder = MLPEncoder(
            task_feature_dim + self.task_type_onehot_dim, hidden_dim, embedding_dim, dropout
        )
        self.platform_input_norm = (
            nn.LayerNorm(platform_feature_dim) if normalize_platform_inputs else None
        )
        self.platform_encoder = MLPEncoder(platform_feature_dim, hidden_dim, embedding_dim, dropout)

        self.gin = GIN(
            in_channels=embedding_dim,
            hidden_channels=hidden_dim,
            num_layers=num_layers,
            out_channels=embedding_dim
        )
        self.post_gin_dropout = nn.Dropout(p=post_gin_dropout)
        # Prefix conditioning (route_b stage 2, T2): the 38 partial-state columns ride
        # in their own data.partial_state_edge_attr and are concatenated onto edge_attr
        # at scoring time. edge_attr itself is NOT widened — its 5-column width is
        # load-bearing for the dim22/dim25cr/dim63crk row extractors that build the
        # A2/A3 baselines this arm is compared against.
        self.partial_state_edge_dim = int(partial_state_edge_dim)
        self.edge_scorer = EdgeScorer(
            embedding_dim,
            hidden_dim,
            edge_dim=edge_dim + self.partial_state_edge_dim,
            dropout=dropout,
        )

        # Residual around the GIN: message passing AUGMENTS the per-node encoding instead
        # of replacing it, so the scorer keeps an undiluted view of each platform's own
        # features (dim7 queue depth, dim13 usage) and GNN capacity >= pointwise capacity.
        # The gate is learnable and initialised to 1.0 (a plain residual); its trained
        # value is a direct readout of how much the model actually relies on the graph.
        self.mp_residual = mp_residual
        if mp_residual:
            self.mp_gate = nn.Parameter(torch.ones(1))

        self._disable_mp = _env_flag("GNN_DISABLE_MESSAGE_PASSING")
        # Same-node platform<->platform edges default OFF: a checkpoint trained with
        # `self.gin(x, data.edge_index)` (bipartite only) must not be served them. Doing so
        # changed ~87.5% of argmax decisions on the training cache and cost 12.4x live RTT
        # on sparse_p35 (276.0M -> 22.3M once dropped) — unrestricted, they outnumber
        # bipartite edges ~30:1 and average co-located platforms together, erasing the
        # queue-depth feature that distinguishes them.
        # Enable ONLY for a checkpoint actually trained with them, and prefer
        # candidates-only scoping (see restrict_node_edges_to_candidates).
        self.mp_node_edges = (
            _env_flag("GNN_MP_NODE_EDGES") if mp_node_edges is None else bool(mp_node_edges)
        )
        self.mp_node_edges_candidates_only = mp_node_edges_candidates_only

        # Network entities: physical nodes and core links, joined by the route table
        # (see src/placement/network_graph.py). Without them the graph contains no
        # topology at all — network latency is a static scalar in edge_attr — so a
        # topology-transfer claim would be made by a model that cannot see topology.
        #
        # Default OFF, same rule as mp_node_edges: a checkpoint trained on the bipartite
        # graph must never be served these. Unlike mp_node_edges this one is
        # self-describing — the two extra encoders make a strict load fail loudly across
        # the boundary instead of drifting silently.
        self.mp_network_entities = (
            _env_flag("GNN_MP_NETWORK_ENTITIES")
            if mp_network_entities is None
            else bool(mp_network_entities)
        )
        if self.mp_network_entities:
            self.net_node_encoder = MLPEncoder(
                net_node_feature_dim, hidden_dim, embedding_dim, dropout
            )
            self.net_link_encoder = MLPEncoder(
                net_link_feature_dim, hidden_dim, embedding_dim, dropout
            )

        # Workload-DAG task<->task edges (route_b stage 2). Emitted UNDIRECTED: PyG
        # aggregates source->target only, so a parent->child-only variant leaves the
        # root task with a bit-identical no-DAG embedding — and the §4 prefix-oracle
        # curve puts nearly all decoder myopia in the first two of four steps, exactly
        # where such a variant is blind. The direction is not lost: task_type_onehot4
        # identifies each node's role and on the frozen diamond4 grids the type
        # determines the DAG position, whereas a missing edge loses reachability
        # outright. A future directed variant is a NEW contract, never a silent edit —
        # hence mp_dag_edges_undirected in the sidecar.
        #
        # Default OFF, same rule as mp_node_edges: weight-invisible, so a checkpoint
        # trained without them must never be served them.
        self.mp_dag_edges = (
            _env_flag("GNN_MP_DAG_EDGES") if mp_dag_edges is None else bool(mp_dag_edges)
        )
        self.mp_dag_edges_undirected = True
        if self.mp_dag_edges and self.task_type_onehot_dim <= 0:
            # Not a style preference: undirected DAG edges make a 4-task block fully
            # connected within 2 hops, and a 3-layer GIN then mixes all four task
            # embeddings. task_features encodes only a 2-way one-hot over
            # ('dnn1','dnn2') plus a source scalar, so on a diamond4 grid 'cnn' and 'rf'
            # are already indistinguishable at the node level and mixing makes them
            # more so. The 4-way one-hot is the prerequisite that keeps them apart (and
            # is a fairness repair — the T1 MLP already sees task type via krank).
            raise ValueError(
                "FAIL LOUD: mp_dag_edges=True requires task_type_onehot_dim > 0. "
                "Undirected DAG message passing over a 4-task block makes task types "
                "that share a task_features encoding (e.g. 'cnn'/'rf' on diamond4) "
                "interchangeable. Pass task_type_onehot_dim=4 and a cache built with "
                "--dag-partial-state."
            )

    def _network_entity_embeddings(self, data: Data) -> List[Tensor]:
        """Encoded [nodes, links], or a loud failure when the graph has none.

        Loud on purpose. A model built with network entities being handed a graph without
        them is the 2026-08-16 train/serve mismatch with the arrow reversed, and it would
        otherwise degrade to a silently different (bipartite) model. Contract `core_v1`
        requires a corpus generated with a `network.backbone` block; run one that has no
        fabric and this is where you find out.
        """
        missing = [
            name
            for name in ("net_node_features", "net_link_features", "net_edge_index")
            if getattr(data, name, None) is None
        ]
        if missing:
            raise ValueError(
                f"FAIL LOUD: mp_network_entities is on but the graph is missing {missing}. "
                f"Build the cache and run live inference with NETWORK_GRAPH_CONTRACT="
                f"core_v1 over a corpus that has a link_topology (network.backbone)."
            )
        return [
            self.net_node_encoder(data.net_node_features),
            self.net_link_encoder(data.net_link_features),
        ]

    def _task_input_features(self, data: Data) -> Tensor:
        """task_features, plus the 4-way DAG task-type one-hot when enabled."""
        tf = data.task_features
        if not self.task_type_onehot_dim:
            return tf
        onehot = getattr(data, "task_type_onehot4", None)
        if onehot is None or int(onehot.size(-1)) != self.task_type_onehot_dim:
            got = "absent" if onehot is None else f"width {int(onehot.size(-1))}"
            raise ValueError(
                f"FAIL LOUD: task_type_onehot_dim={self.task_type_onehot_dim} but the "
                f"graph's task_type_onehot4 is {got}. Build the cache with "
                "--dag-partial-state."
            )
        return torch.cat([tf, onehot.to(tf.device, dtype=tf.dtype)], dim=-1)

    def _encode(self, data: Data) -> Tuple[Tensor, Tensor]:
        """Node encoding + message passing → (task_emb, platform_emb).

        Split out of ``forward`` so a prefix-conditioned decode can reuse one GIN pass
        across every step and every tied plan. That reuse is only valid because the
        partial-state columns enter at the SCORER (see ``_score``) and never touch a
        node feature — ``make_partial_state_score_fn`` asserts that precondition.
        """
        n_tasks: int = int(data.n_tasks)
        n_platforms: int = int(data.n_platforms)

        task_embeddings = self.task_encoder(self._task_input_features(data))
        platform_feats = data.platform_features
        if self.platform_input_norm is not None:
            platform_feats = self.platform_input_norm(platform_feats)
        platform_embeddings = self.platform_encoder(platform_feats)

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
            blocks = [task_embeddings, platform_embeddings]
            extra_edges: List[Tensor] = []

            node_ei = getattr(data, "node_edge_index", None) if self.mp_node_edges else None
            if node_ei is not None and node_ei.numel() > 0:
                node_ei = node_ei.to(data.edge_index.device)
                if self.mp_node_edges_candidates_only:
                    node_ei = restrict_node_edges_to_candidates(
                        node_ei, data.edge_index, n_tasks, n_platforms
                    )
                if node_ei.numel() > 0:
                    extra_edges.append(node_ei)

            # Workload-DAG task<->task edges. Unlike mp_network_entities this adds NO
            # node rows — it reuses existing task indices — so the x0 block order and
            # therefore split_task_platform_embeddings are untouched. This option
            # cannot move the platform block.
            if self.mp_dag_edges:
                dag_ei = getattr(data, "dag_edge_index", None)
                if dag_ei is None:
                    raise ValueError(
                        "FAIL LOUD: mp_dag_edges is on but the graph has no "
                        "dag_edge_index. Build the cache with --dag-partial-state."
                    )
                if dag_ei.numel() > 0:
                    dag_ei = dag_ei.to(data.edge_index.device)
                    if int(dag_ei.max()) >= n_tasks or int(dag_ei.min()) < 0:
                        raise ValueError(
                            f"FAIL LOUD: dag_edge_index indexes outside the task block "
                            f"[0, {n_tasks}): min={int(dag_ei.min())} "
                            f"max={int(dag_ei.max())}. DAG edges are task<->task only."
                        )
                    extra_edges.append(torch.cat([dag_ei, dag_ei.flip(0)], dim=1))

            # Network entities append AFTER platforms, so the task/platform slices below
            # are unchanged and every pre-existing checkpoint keeps its index layout.
            if self.mp_network_entities:
                blocks.extend(self._network_entity_embeddings(data))
                net_ei = data.net_edge_index
                if net_ei.numel() > 0:
                    extra_edges.append(net_ei.to(data.edge_index.device))

            x0 = torch.cat(blocks, dim=0)
            mp_edge_index = (
                torch.cat([data.edge_index] + extra_edges, dim=1)
                if extra_edges
                else data.edge_index
            )
            h = self.post_gin_dropout(self.gin(x0, mp_edge_index))
            x = x0 + self.mp_gate * h if self.mp_residual else h
            task_emb, platform_emb = split_task_platform_embeddings(x, n_tasks, n_platforms)

        return task_emb, platform_emb

    def _score(self, task_emb: Tensor, platform_emb: Tensor, data: Data) -> List[Tensor]:
        """Edge scoring from precomputed node embeddings → per-task logits."""
        n_tasks: int = int(data.n_tasks)
        n_platforms: int = int(data.n_platforms)
        device = task_emb.device

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
            if self.partial_state_edge_dim:
                # No silent swallow when prefix conditioning is on: an alignment fault
                # here would drop edge_attr and produce a quietly wrong T2 arm.
                e_attr = data.edge_attr[valid]
            else:
                try:
                    e_attr = data.edge_attr[valid]
                except (IndexError, RuntimeError):
                    e_attr = None

        # The partial-state prefix block rides in its own attr, aligned row-for-row with
        # the FULL edge_index, and is selected by the same `valid` mask as edge_attr —
        # so its alignment with the per-task logit order is inherited, not re-derived.
        if self.partial_state_edge_dim:
            ps = getattr(data, "partial_state_edge_attr", None)
            if ps is None or int(ps.size(-1)) != self.partial_state_edge_dim:
                got = "absent" if ps is None else f"width {int(ps.size(-1))}"
                raise ValueError(
                    f"FAIL LOUD: partial_state_edge_dim={self.partial_state_edge_dim} "
                    f"but the graph's partial_state_edge_attr is {got}. Populate it via "
                    "src.policy.gnn.partial_state_edges.refresh_partial_state_edge_attr."
                )
            ps_valid = ps.to(e_task.device, dtype=e_task.dtype)[valid]
            e_attr = ps_valid if e_attr is None else torch.cat([e_attr, ps_valid], dim=-1)

        edge_scores = self.edge_scorer(e_task, e_platform, e_attr)

        # Split scores per task
        logits_per_task = []
        for t in range(n_tasks):
            mask_t = (ti == t)
            logits_t = edge_scores[mask_t]
            logits_per_task.append(logits_t)

        return logits_per_task

    def forward(self, data: Data) -> List[Tensor]:
        task_emb, platform_emb = self._encode(data)
        return self._score(task_emb, platform_emb, data)

