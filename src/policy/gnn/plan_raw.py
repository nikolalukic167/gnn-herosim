"""raw_plan_v1: the partial plan as raw graph facts instead of engineered columns.

The partial_state contracts summarise the committed plan into per-candidate columns
(exchange seconds to placed partners, committed service, log backlog, ...). This contract
writes only what the plan *is*, onto every bipartite edge row (both directions):

  col 0  committed   1 if the edge's task is committed to the edge's platform, else 0
  col 1  count       number of committed tasks on the edge's platform

Nothing is derived from physics: no seconds, no bytes, no route. A model that wants the
exchange cost of a placement has to learn it: which partner sits where, through the peer
edges, and which platforms share a node, through the same-node edges the encoder adds
under ``plan_raw``. The MP-OFF twin sees only its own edge's two columns.

Because the facts live on every edge and feed message passing, the encode is recomputed
for every distinct committed set (``make_partial_state_score_fn``'s shared-encode shortcut
is invalid here). One closure serves the trainer's losses, the validation decoder and the
live decoder, as with the partial-state columns.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Tuple

import torch
from torch import Tensor

from src.policy.gnn.partial_state_edges import candidate_edge_rows

PLAN_RAW_DIM = 2
PLAN_RAW_CONTRACT = "raw_plan_v1"
PLAN_RAW_ENV = "GNN_PLAN_RAW"


def _edge_platforms(data: Any) -> Tuple[List[int], Dict[int, int]]:
    """(platform position of every edge row, forward row -> reverse row)."""
    n_tasks = int(data.n_tasks)
    n_platforms = int(data.n_platforms)
    src = data.edge_index[0].tolist()
    dst = data.edge_index[1].tolist()
    plat: List[int] = []
    for s, d in zip(src, dst):
        if s < n_tasks <= d < n_tasks + n_platforms:
            plat.append(d - n_tasks)
        elif d < n_tasks <= s < n_tasks + n_platforms:
            plat.append(s - n_tasks)
        else:
            raise ValueError(
                f"FAIL LOUD: raw_plan_v1 expects edge_index to hold task<->platform rows only; "
                f"row ({s}, {d}) is neither (n_tasks={n_tasks}, n_platforms={n_platforms})"
            )
    where = {(s, d): r for r, (s, d) in enumerate(zip(src, dst))}
    reverse: Dict[int, int] = {}
    for rows in candidate_edge_rows(data).values():
        for r in rows:
            back = where.get((dst[r], src[r]))
            if back is None:
                raise ValueError(f"FAIL LOUD: raw_plan_v1: forward edge row {r} has no reverse row")
            reverse[r] = back
    return plat, reverse


def plan_raw_edge_attr(data: Any, committed: Mapping[int, Any]) -> Tensor:
    """The [E, 2] raw-plan block for ``committed`` = {task_idx: placement tuple}."""
    rows = candidate_edge_rows(data)
    memo = getattr(data, "_plan_raw_layout", None)
    if memo is None:
        plat, reverse = _edge_platforms(data)
        placements = data.task_logit_to_placement
        index_of = [
            {tuple(int(v) for v in c): k for k, c in enumerate(placements[t])}
            for t in range(int(data.n_tasks))
        ]
        memo = (torch.as_tensor(plat, dtype=torch.long), reverse, index_of)
        setattr(data, "_plan_raw_layout", memo)
    plat_t, reverse, index_of = memo
    n_edges = int(data.edge_index.size(1))
    attr = torch.zeros((n_edges, PLAN_RAW_DIM), dtype=torch.float32)
    count = torch.zeros(int(data.n_platforms), dtype=torch.float32)
    for j, placement in committed.items():
        key = tuple(int(v) for v in placement)
        k = index_of[int(j)].get(key)
        if k is None:
            raise KeyError(f"raw_plan_v1: task {j} committed to {key}, not one of its candidates")
        r = rows[int(j)][k]
        attr[r, 0] = 1.0
        attr[reverse[r], 0] = 1.0
        count[plat_t[r]] += 1.0
    attr[:, 1] = count[plat_t]
    return attr.to(data.edge_index.device)


def make_plan_raw_score_fn(model: Any, data: Any) -> Callable[[int, Mapping[int, Any]], Tensor]:
    """``(task_idx, committed) -> logits[task_idx]``, re-encoding per distinct committed set."""
    if not getattr(model, "plan_raw", False):
        raise ValueError("make_plan_raw_score_fn: the model was not built with plan_raw")
    if int(getattr(model, "partial_state_edge_dim", 0)) != PLAN_RAW_DIM:
        raise ValueError(
            f"make_plan_raw_score_fn: model.partial_state_edge_dim="
            f"{int(getattr(model, 'partial_state_edge_dim', 0))}, raw_plan_v1 emits {PLAN_RAW_DIM}"
        )
    cache: Dict[Tuple[Any, ...], List[Tensor]] = {}

    def score(task_idx: int, committed: Mapping[int, Any]) -> Tensor:
        key = tuple(sorted((int(j), tuple(int(v) for v in p)) for j, p in committed.items()))
        logits = cache.get(key)
        if logits is None:
            data.partial_state_edge_attr = plan_raw_edge_attr(data, committed)
            task_emb, platform_emb = model._encode(data)
            logits = model._score(task_emb, platform_emb, data)
            cache[key] = logits
        return logits[int(task_idx)]

    return score
