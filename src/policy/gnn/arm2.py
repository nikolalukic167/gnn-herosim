"""uncut_groups_s0_v1 Arm 2: committed-state re-encode (B1) and the group-anchor head (B2).

B1. Placements already made in a big group change how its open partners are scored. Three inputs carry
the committed set into the node encoding: a per-task ``committed`` flag, and per platform the count of
tasks committed on it and the seconds of exchange the open tasks would pay to the committed peers if
they ran on that platform's node (``peer_committed_mass``, log1p seconds as partial_state_v5 columns 7-8).
A committed task keeps only its edge to the chosen platform in message passing. ``GNN_REENCODE_EVERY``
re-runs ``TaskPlacementGNN._encode`` after every m commits; ``inf`` (the default) is today's single
encode and adds nothing.

B2. A group (a connected component of the batch's peer graph) picks one node before its tasks are placed.
The head scores the group's candidate nodes from the first encode; the argmax (label in training) enters
every member's scorer as a 0/1 ``on_anchor_node`` edge column.

Everything is ordered by id, so no set or dict order depends on PYTHONHASHSEED. The closure
:func:`make_arm2_score_fn` is the one both the trainer and the decoders call.
"""

from __future__ import annotations

import math
import os
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from src.policy.gnn.partial_state_edges import candidate_edge_rows
from src.policy.gnn.plan_raw import _edge_platforms
from src.policy.tabular.reduced_features import partial_state_columns, partial_state_feature_dim

REENCODE_ENV = "GNN_REENCODE_EVERY"
GROUP_ANCHOR_ENV = "GNN_GROUP_ANCHOR"
ANCHOR_CE_WEIGHT_ENV = "NEAR_RTT_ANCHOR_CE_WEIGHT"
RANDOM_PREFIX_K_ENV = "NEAR_RTT_RANDOM_PREFIX_K"
MAX_REENCODES = 16
SMALL_BATCH = 10
COMMITTED_PLATFORM_DIM = 2
ANCHOR_SCALARS = 2
ARM2_ATTRS = ("arm2_task_committed", "arm2_platform_state", "arm2_edge_keep")


def parse_reencode_every(raw: Optional[str]) -> str:
    """``inf`` (off), ``auto`` or a positive integer, as the string the sidecar stores."""
    value = (raw if raw is not None else os.environ.get(REENCODE_ENV, "")).strip().lower()
    if value in ("", "inf", "off", "0"):
        return "inf"
    if value == "auto":
        return "auto"
    try:
        m = int(value)
    except ValueError:
        m = 0
    if m < 1:
        raise ValueError(
            f"FAIL LOUD: {REENCODE_ENV}={value!r} is not inf, auto or a positive integer"
        )
    return str(m)


def reencode_period(setting: str, n_tasks: int) -> Optional[int]:
    """Commits between encodes: None for ``inf``; ``auto`` is 1 for n <= 10, else ceil(n / 16)."""
    if setting == "inf":
        return None
    if setting == "auto":
        return 1 if n_tasks <= SMALL_BATCH else max(1, math.ceil(n_tasks / MAX_REENCODES))
    return int(setting)


def reencodes_for_batch(setting: str, n_tasks: int) -> int:
    """Encodes a full decode of ``n_tasks`` commits runs (the first one included)."""
    m = reencode_period(setting, n_tasks)
    return 1 if m is None else 1 + (n_tasks - 1) // m


def peer_groups(data: Any) -> List[List[int]]:
    """Connected components of the in-batch peer graph with at least two tasks, sorted by id."""
    n_tasks = int(data.n_tasks)
    ei = getattr(data, "peer_edge_index", None)
    if ei is None:
        raise ValueError("FAIL LOUD: the group anchor needs peer_edge_index on the graph")
    parent = list(range(n_tasks))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    if ei.numel() > 0:
        for a, b in zip(ei[0].tolist(), ei[1].tolist()):
            ra, rb = find(int(a)), find(int(b))
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    comps: Dict[int, List[int]] = {}
    for t in range(n_tasks):
        comps.setdefault(find(t), []).append(t)
    return sorted((sorted(c) for c in comps.values() if len(c) > 1), key=lambda c: c[0])


class Arm2Layout:
    """Static per-graph facts the closure reuses at every step (memoised on the graph)."""

    def __init__(self, data: Any, ctx: Any, *, anchor: bool, committed_inputs: bool) -> None:
        n_tasks, n_platforms = int(data.n_tasks), int(data.n_platforms)
        self.n_tasks, self.n_platforms = n_tasks, n_platforms
        self.rows = candidate_edge_rows(data)
        plat, reverse = _edge_platforms(data)
        self.plat = np.asarray(plat, dtype=np.int64)
        self.reverse = reverse
        placements = data.task_logit_to_placement
        self.placements = {t: [tuple(int(v) for v in c) for c in placements[t]] for t in range(n_tasks)}
        self.index_of = [{c: k for k, c in enumerate(self.placements[t])} for t in range(n_tasks)]
        self.pos_node: Dict[int, int] = {}
        for t in range(n_tasks):
            for k, cand in enumerate(self.placements[t]):
                self.pos_node[int(self.plat[self.rows[t][k]])] = cand[0]
        self.n_edges = int(data.edge_index.size(1))
        self.device = data.edge_index.device

        self.peer_b = None
        self.pull = None
        if committed_inputs and ctx is not None and ctx.peer_pairs and ctx.node_exchange:
            self.peer_b = np.zeros((n_tasks, n_tasks), dtype=np.float64)
            for (i, j), b in ctx.peer_pairs.items():
                if 0 <= int(i) < n_tasks and 0 <= int(j) < n_tasks:
                    self.peer_b[int(i), int(j)] = float(b)
            nodes = sorted(set(self.pos_node.values()))
            self.node_ids = nodes
            nidx = {n: a for a, n in enumerate(nodes)}
            size = len(nodes)
            pb = np.zeros((size, size), dtype=np.float64)
            lat = np.zeros((size, size), dtype=np.float64)
            for frm in nodes:
                for to in nodes:
                    key = (frm, to) if ctx.both_directions else (to, frm)
                    rate, latency = ctx.node_exchange[key]
                    pb[nidx[frm], nidx[to]] = rate
                    lat[nidx[frm], nidx[to]] = latency if rate > 0.0 else 0.0
            self.pull = (nidx, pb, lat)

        self.groups: List[List[int]] = []
        self.group_of: Dict[int, int] = {}
        self.group_nodes: List[List[int]] = []
        self.group_node_pos: List[List[List[int]]] = []
        self.group_scalars: List[Tensor] = []
        if anchor:
            if ctx is None or not ctx.service_s or not ctx.backlog_s:
                raise ValueError(
                    "FAIL LOUD: the group anchor needs backlog_s and service_s in the partial-state "
                    "context (a partial_state_v4/v5 cache or live graph)"
                )
            self.groups = peer_groups(data)
            for g, members in enumerate(self.groups):
                nodes: Dict[int, Dict[int, None]] = {}
                share: Dict[int, int] = {}
                cost: Dict[int, float] = {}
                for t in members:
                    self.group_of[t] = g
                    seen = set()
                    best: Dict[int, float] = {}
                    for k, cand in enumerate(self.placements[t]):
                        node = cand[0]
                        nodes.setdefault(node, {})[int(self.plat[self.rows[t][k]])] = None
                        seen.add(node)
                        c = float(ctx.backlog_s[cand]) + float(ctx.service_s[(t, cand)])
                        best[node] = min(best.get(node, math.inf), c)
                    for node in seen:
                        share[node] = share.get(node, 0) + 1
                        cost[node] = cost.get(node, 0.0) + best[node]
                order = sorted(nodes)
                self.group_nodes.append(order)
                self.group_node_pos.append([sorted(nodes[n]) for n in order])
                self.group_scalars.append(torch.tensor(
                    [[share[n] / len(members), math.log1p(cost[n])] for n in order],
                    dtype=torch.float32, device=self.device,
                ))

    def committed_state(self, snapshot: Mapping[int, Any]) -> Tuple[Tensor, Tensor, Tensor]:
        """(task flag [T], platform state [P, 2], edge keep mask [E]) for ``snapshot``."""
        flag = np.zeros(self.n_tasks, dtype=np.float32)
        count = np.zeros(self.n_platforms, dtype=np.float64)
        keep = np.ones(self.n_edges, dtype=bool)
        committed_nodes: Dict[int, int] = {}
        for j in sorted(snapshot):
            key = tuple(int(v) for v in snapshot[j])
            k = self.index_of[int(j)].get(key)
            if k is None:
                raise KeyError(f"arm2: task {j} committed to {key}, not one of its candidates")
            flag[int(j)] = 1.0
            count[int(self.plat[self.rows[int(j)][k]])] += 1.0
            committed_nodes[int(j)] = key[0]
            for r_k, r in enumerate(self.rows[int(j)]):
                if r_k != k:
                    keep[r] = False
                    keep[self.reverse[r]] = False
        mass = np.zeros(self.n_platforms, dtype=np.float64)
        if self.peer_b is not None and committed_nodes:
            nidx, pb, lat = self.pull
            open_mask = 1.0 - flag.astype(np.float64)
            js = sorted(committed_nodes)
            bytes_j = open_mask @ self.peer_b[:, js]
            pairs_j = open_mask @ (self.peer_b[:, js] > 0.0)
            per_node = np.zeros(pb.shape[1], dtype=np.float64)
            for a, j in enumerate(js):
                src = nidx[committed_nodes[j]]
                per_node += bytes_j[a] * pb[src] + pairs_j[a] * lat[src]
            for pos, node in self.pos_node.items():
                mass[pos] = per_node[nidx[node]]
        state = np.stack([np.log1p(count), np.log1p(mass)], axis=1).astype(np.float32)
        dev = self.device
        return (torch.as_tensor(flag, device=dev), torch.as_tensor(state, device=dev),
                torch.as_tensor(keep, device=dev))


def _layout(data: Any, ctx: Any, model: Any) -> Arm2Layout:
    key = (bool(model.group_anchor), bool(model.committed_inputs))
    memo = getattr(data, "_arm2_layout", None)
    if memo is None or memo[0] != key:
        memo = (key, Arm2Layout(data, ctx, anchor=key[0], committed_inputs=key[1]))
        setattr(data, "_arm2_layout", memo)
    return memo[1]


def layout_for(model: Any, data: Any, ctx: Any) -> Arm2Layout:
    return _layout(data, ctx, model)


def random_topological_orders(
    n_tasks: int, dag_parents: Mapping[int, Sequence[int]], k: int, seed: int, tag: str
) -> List[List[int]]:
    """``k`` seeded random commit orders that respect the DAG (each step draws uniformly from the ready tasks).

    Seeded by ``(seed, tag, i)`` through crc32, never ``hash()``, so a rerun draws the same orders.
    """
    import random
    import zlib

    orders: List[List[int]] = []
    for i in range(int(k)):
        rng = random.Random(zlib.crc32(f"{int(seed)}|{tag}|{i}".encode()))
        placed: set = set()
        order: List[int] = []
        while len(order) < n_tasks:
            ready = [t for t in range(n_tasks)
                     if t not in placed and all(int(p) in placed for p in (dag_parents.get(t) or ()))]
            if not ready:
                raise ValueError("FAIL LOUD: random_topological_orders: the batch DAG has a cycle")
            t = ready[rng.randrange(len(ready))]
            placed.add(t)
            order.append(t)
        orders.append(order)
    return orders


def set_committed_state(data: Any, layout: Arm2Layout, snapshot: Mapping[int, Any]) -> None:
    flag, state, keep = layout.committed_state(snapshot)
    data.arm2_task_committed, data.arm2_platform_state, data.arm2_edge_keep = flag, state, keep


def clear_committed_state(data: Any) -> None:
    for name in ARM2_ATTRS:
        if getattr(data, name, None) is not None:
            setattr(data, name, None)


def anchor_logits(model: Any, layout: Arm2Layout, task_emb: Tensor, platform_emb: Tensor) -> List[Tensor]:
    """Per group, the head's logit over its candidate nodes (ascending node id)."""
    out: List[Tensor] = []
    for g, members in enumerate(layout.groups):
        gemb = task_emb[torch.as_tensor(members, device=task_emb.device)].mean(dim=0)
        nembs = torch.stack([
            platform_emb[torch.as_tensor(p, device=platform_emb.device)].mean(dim=0)
            for p in layout.group_node_pos[g]
        ])
        feats = torch.cat([
            gemb.unsqueeze(0).expand(nembs.size(0), -1), nembs,
            layout.group_scalars[g].to(nembs.device, dtype=nembs.dtype),
        ], dim=-1)
        out.append(model.anchor_head(feats).squeeze(-1))
    return out


def anchor_label(layout: Arm2Layout, plan_nodes: Mapping[int, int]) -> List[int]:
    """Per group, the node holding the most group tasks in the label plan (ties: lowest node id)."""
    labels: List[int] = []
    for members in layout.groups:
        counts: Dict[int, int] = {}
        for t in members:
            counts[plan_nodes[t]] = counts.get(plan_nodes[t], 0) + 1
        labels.append(min(counts, key=lambda n: (-counts[n], n)))
    return labels


def make_arm2_score_fn(
    model: Any,
    data: Any,
    ctx: Any,
    *,
    teacher_anchor: Optional[Sequence[int]] = None,
) -> Callable[[int, Mapping[int, Any]], Tensor]:
    """``(task_idx, committed) -> logits[task_idx]`` for a model with B1 and/or B2.

    The encode used at ``len(committed) = n`` sees the first ``(n // m) * m`` commits of the dict, in
    insertion order, so train (label order) and decode (commit order) agree. ``teacher_anchor`` is the
    label node per group, used in place of the head's argmax. The returned callable carries
    ``.anchor_logits()`` and ``.encode_count`` for the loss and the decision-charge meter.
    """
    expected = partial_state_feature_dim(getattr(ctx, "contract", None))
    if int(model.partial_state_edge_dim) != expected + (1 if model.group_anchor else 0):
        raise ValueError(
            f"make_arm2_score_fn: model.partial_state_edge_dim={int(model.partial_state_edge_dim)} but the "
            f"contract emits {expected} columns (+1 with the group anchor)"
        )
    layout = _layout(data, ctx, model)
    period = reencode_period(model.reencode_every, layout.n_tasks) if model.committed_inputs else None
    encodes: Dict[Tuple[Any, ...], Tuple[Tensor, Tensor]] = {}
    anchors: Dict[str, Any] = {}

    def encode(snapshot: Mapping[int, Any]) -> Tuple[Tensor, Tensor]:
        key = tuple(sorted((int(j), tuple(int(v) for v in p)) for j, p in snapshot.items()))
        cached = encodes.get(key)
        if cached is None:
            if model.committed_inputs:
                set_committed_state(data, layout, snapshot)
            cached = model._encode(data)
            encodes[key] = cached
        return cached

    def anchor_nodes() -> List[int]:
        if "nodes" not in anchors:
            task_emb, platform_emb = encode({})
            logits = anchor_logits(model, layout, task_emb, platform_emb)
            anchors["logits"] = logits
            if teacher_anchor is not None:
                anchors["nodes"] = [int(n) for n in teacher_anchor]
            else:
                nodes = []
                for g, lg in enumerate(logits):
                    vals = [float(v) for v in lg.detach().tolist()]
                    nodes.append(layout.group_nodes[g][vals.index(max(vals))])
                anchors["nodes"] = nodes
        return anchors["nodes"]

    base_attr: Dict[str, Tensor] = {}

    def score(task_idx: int, committed: Mapping[int, Any]) -> Tensor:
        c0 = 0 if period is None else (len(committed) // period) * period
        snapshot = dict(list(committed.items())[:c0])
        task_emb, platform_emb = encode(snapshot)
        task_idx = int(task_idx)
        candidates = layout.placements[task_idx]
        rows = layout.rows[task_idx]
        attr = base_attr.get("attr")
        width = int(model.partial_state_edge_dim)
        if attr is None:
            attr = torch.zeros((layout.n_edges, width), dtype=torch.float32, device=layout.device)
            base_attr["attr"] = attr
        else:
            attr.zero_()
        if rows:
            block = partial_state_columns(ctx, task_idx, candidates, committed)
            idx = torch.as_tensor(rows, dtype=torch.long, device=attr.device)
            attr[idx, :expected] = torch.as_tensor(block, dtype=attr.dtype, device=attr.device)
            if model.group_anchor and task_idx in layout.group_of:
                node = anchor_nodes()[layout.group_of[task_idx]]
                attr[idx, expected] = torch.as_tensor(
                    [1.0 if c[0] == node else 0.0 for c in candidates], dtype=attr.dtype, device=attr.device)
        data.partial_state_edge_attr = attr
        return model._score(task_emb, platform_emb, data)[task_idx]

    def anchor_logits_fn() -> List[Tensor]:
        anchor_nodes()
        return anchors["logits"]

    score.anchor_logits = anchor_logits_fn
    score.anchor_layout = layout
    score.encode_count = lambda: len(encodes)
    return score
