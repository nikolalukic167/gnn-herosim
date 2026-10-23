"""partial_state_v8: the static link block as a conv edge attribute (`link_static_attr`, [E, 3], live_link_features_v1).

S1-S3 (placement/link_features.static_edge_columns) are prefix-free: they depend on the captured flows and the batch only, never on the committed set. So they are
written ONCE per graph, from the graph's own `partial_state_ctx` (the cache builder and the live batch fill the same tables), and every row of the
bipartite edge_index carries them: the forward row of a (task, candidate) pair and its reverse row alike (as plan_raw's `_edge_platforms` pairs them).
The model appends them to the conv's edge_attr under the weight-visible flag `mp_link_static_attr`; the scorers read the same three numbers as columns 30-32 of
`partial_state_edge_attr` (reduced_features.partial_state_columns), computed by the same function.
"""
from __future__ import annotations

from typing import Any

import torch

from src.placement import link_features
from src.policy.gnn.partial_state_edges import candidate_edge_rows
from src.policy.gnn.plan_raw import _edge_platforms
from src.policy.tabular.reduced_features import _peer_adj, build_partial_state_context_from_graph

ATTR = "link_static_attr"


def static_partners(ctx: Any, task_id: int):
    """(candidate nodes of the peer, payload bytes) for EVERY peer of the task, in `peer_pairs` order: the argument `static_edge_columns` takes."""
    return [(ctx.cand_nodes.get(j) or (), b) for j, b in _peer_adj(ctx).get(task_id, ()) if j != task_id]


def link_static_attr(data: Any) -> torch.Tensor:
    ctx = build_partial_state_context_from_graph(data)
    if ctx.link_counts is None or ctx.link_routes is None:
        raise ValueError("FAIL LOUD: link_static_attr needs link_counts and link_routes in the graph's partial_state_ctx (a v8 cache or a live v8 batch)")
    rows = candidate_edge_rows(data)
    _, reverse = _edge_platforms(data)
    attr = torch.zeros((int(data.edge_index.size(1)), link_features.LINK_STATIC_DIM), dtype=torch.float32)
    for t in range(int(data.n_tasks)):
        partners = static_partners(ctx, t)
        for k, cand in enumerate(data.task_logit_to_placement[t]):
            node = ctx.node_of[tuple(int(v) for v in cand)]
            vals = link_features.static_edge_columns(ctx.link_counts, ctx.link_routes, ctx.node_exchange, partners, node)
            row = rows[t][k]
            attr[row] = torch.tensor(vals, dtype=torch.float64).to(torch.float32)
            attr[reverse[row]] = attr[row]
    return attr.to(data.edge_index.device)


def attach_link_static(data: Any) -> None:
    """Write `data.link_static_attr` (idempotent: same inputs, same bits)."""
    setattr(data, ATTR, link_static_attr(data))
