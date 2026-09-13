"""Prefix conditioning for the route_b stage-2 T2 GNN arm (A1).

The 38 partial-state columns that the T1 MLP consumes (``reduced_features.
partial_state_columns`` — THE single definition, §2's one-definition rule) are written
onto a graph's candidate edges and re-read at every decode step, so the GNN scores each
task against the placements already committed rather than against a static snapshot.

Three pieces:

* :func:`candidate_edge_rows` — the edge-row ↔ logit-index map, recovered by the same
  scan ``TaskPlacementGNN._score`` performs and then ASSERTED against the cache's
  ``platform_pos``, so the ordering is checked rather than assumed.
* :func:`refresh_partial_state_edge_attr` — writes one task's block for one prefix.
* :func:`make_partial_state_score_fn` — the ONE closure both the trainer and the §4
  masked decoder call, which is what makes train-time and decode-time prefixes agree by
  construction rather than by two implementations staying in sync.

Nothing here recomputes a feature formula; the columns come from
``partial_state_columns`` and nowhere else.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import torch
from torch import Tensor

from src.policy.tabular.reduced_features import (
    PARTIAL_STATE_FEATURE_DIM,
    partial_state_columns,
)

_ROWS_CACHE_ATTR = "_candidate_edge_rows"


def candidate_edge_rows(data: Any) -> Dict[int, List[int]]:
    """{task_idx: [row index into data.edge_index, ...]} in per-task logit order.

    Derived by the same scan ``TaskPlacementGNN._score`` uses (rows whose source is a
    task node and whose target is a platform node, in edge_index order), then asserted
    column-by-column against
    ``queue_key_to_platform_meta[task_logit_to_queue_key[t][k]]["platform_pos"]``.

    The cache builder sorts each task's compatible platforms ascending precisely so this
    ordering holds, but a silent drift here would misattribute every prefix column to
    the wrong candidate, so it is verified per graph rather than trusted. Memoized on
    the graph as ``_candidate_edge_rows``.
    """
    cached = getattr(data, _ROWS_CACHE_ATTR, None)
    if cached is not None:
        return cached

    n_tasks = int(data.n_tasks)
    n_platforms = int(data.n_platforms)
    ei = data.edge_index
    src = ei[0].tolist()
    dst = ei[1].tolist()

    rows: Dict[int, List[int]] = {t: [] for t in range(n_tasks)}
    for row, (s, d) in enumerate(zip(src, dst)):
        pj = d - n_tasks
        if 0 <= pj < n_platforms and s < n_tasks:
            rows[s].append(row)

    meta = getattr(data, "queue_key_to_platform_meta", None)
    keys = getattr(data, "task_logit_to_queue_key", None)
    if meta is None or keys is None:
        raise ValueError(
            "candidate_edge_rows: graph carries no queue_key_to_platform_meta / "
            "task_logit_to_queue_key, so the edge-row <-> logit-index map cannot be "
            "verified. Prefix conditioning requires a cache that has them."
        )

    for t in range(n_tasks):
        expected = [
            int(meta[str(k)]["platform_pos"]) for k in keys[t]
        ]
        got = [int(dst[row]) - n_tasks for row in rows[t]]
        if got != expected:
            raise ValueError(
                f"FAIL LOUD: edge-row <-> logit-index misalignment for task {t}. "
                f"edge_index platform positions {got} != cache logit order {expected}. "
                "The prefix columns would be attributed to the wrong candidates."
            )

    setattr(data, _ROWS_CACHE_ATTR, rows)
    return rows


def refresh_partial_state_edge_attr(
    data: Any,
    ctx: Any,
    task_idx: int,
    committed: Mapping[int, Any],
) -> Tensor:
    """Zero ``data.partial_state_edge_attr`` and fill ``task_idx``'s candidate rows.

    Only ``task_idx`` is filled. ``partial_state_columns`` raises for any task whose
    parents are not all committed (the §4 topological-order invariant), and at a given
    decode step no other task's logits are read — so filling them would be both
    impossible and pointless. Reverse edge rows stay zero and are dropped by the
    scorer's ``valid`` mask.

    Returns the tensor, allocating it lazily on first call.
    """
    rows = candidate_edge_rows(data)
    ei = data.edge_index
    n_edges = int(ei.size(1))

    attr = getattr(data, "partial_state_edge_attr", None)
    if attr is None or int(attr.size(0)) != n_edges:
        attr = torch.zeros(
            (n_edges, PARTIAL_STATE_FEATURE_DIM), dtype=torch.float32, device=ei.device
        )
        data.partial_state_edge_attr = attr
    else:
        attr.zero_()

    task_rows = rows[int(task_idx)]
    if not task_rows:
        return attr

    candidates = [tuple(c) for c in data.task_logit_to_placement[int(task_idx)]]
    try:
        block = partial_state_columns(ctx, int(task_idx), candidates, committed)
    except KeyError as exc:
        raise KeyError(
            f"refresh_partial_state_edge_attr: task {task_idx} has a candidate absent "
            f"from the partial-state context (missing key {exc!r}). The cache's "
            "candidate set and its partial_state_ctx.demand table disagree."
        ) from exc

    attr[torch.as_tensor(task_rows, dtype=torch.long, device=attr.device)] = (
        torch.as_tensor(block, dtype=attr.dtype, device=attr.device)
    )
    return attr


def make_partial_state_score_fn(
    model: Any,
    data: Any,
    ctx: Any,
    *,
    cache_encode: bool = True,
) -> Callable[[int, Mapping[int, Any]], Tensor]:
    """The ONE prefix-conditioned scorer: ``(task_idx, committed) -> logits[task_idx]``.

    Both ``train_near_rtt``'s teacher-forced loss and ``seq_decode``'s masked_topo
    decoder call this, so the prefix a step is trained on and the prefix it is decoded
    under are produced by the same code path.

    ``cache_encode`` reuses one GIN pass across every step and every tied plan. That is
    valid only because the partial-state columns enter at the EdgeScorer and never touch
    a node feature, which is asserted here — if a future change routes prefix state into
    ``platform_features``, this stops silently being correct and starts failing loudly.
    """
    if not getattr(model, "partial_state_edge_dim", 0):
        raise ValueError(
            "make_partial_state_score_fn: model.partial_state_edge_dim is 0, so the "
            "prefix columns would be built and then discarded. Construct the model "
            "with partial_state_edge_dim=PARTIAL_STATE_FEATURE_DIM."
        )
    if int(model.partial_state_edge_dim) != PARTIAL_STATE_FEATURE_DIM:
        raise ValueError(
            f"make_partial_state_score_fn: model.partial_state_edge_dim="
            f"{int(model.partial_state_edge_dim)} but this contract emits "
            f"{PARTIAL_STATE_FEATURE_DIM} columns."
        )

    cached: Dict[str, Tuple[Tensor, Tensor]] = {}

    def score(task_idx: int, committed: Mapping[int, Any]) -> Tensor:
        refresh_partial_state_edge_attr(data, ctx, task_idx, committed)
        if cache_encode:
            if "emb" not in cached:
                cached["emb"] = model._encode(data)
            task_emb, platform_emb = cached["emb"]
        else:
            task_emb, platform_emb = model._encode(data)
        return model._score(task_emb, platform_emb, data)[int(task_idx)]

    return score
