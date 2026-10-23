"""live_link_features_v1, variant (b): the bipartite conv sees the link columns of the COMMITTED PREFIX and the graph is re-encoded for every committed set.

`TaskPlacementGNN(mp_link_edge_conv=True)` appends E1-E3 (placement/link_features.py) to the conv's edge attributes, on every bipartite row (forward and reverse). E1-E3 of task t's candidate depend on
t's committed partners and on the flows the committed batch-mates will run, with t's own placement left out (the refine convention), so the attribute changes with the committed set and so does the encode.

  link_rows(LS, Cm, ND)          [S, R, 3]  E1-E3 of every (task, candidate) row under S committed sets: Cm[s, j] = 1 iff task j is committed in state s, ND[s, j] = the node index of its placement.
                                 The sparse twin of batched_tf._link_block (partner lists instead of an all-pairs mask), in torch, float64: the one implementation of the conv block at serving (CPU) and in training (CPU or GPU).
  link_edge_block(data, ctx, committed)   [E, 3] one committed set -> edge rows (reverse rows copy the forward ones)
  make_link_conv_score_fn(model, data, ctx)   `(task_idx, committed) -> logits`, re-encoding once per distinct committed set; what the decoder, the validation decode and the live scheduler call
  link_conv_loss(model, data, ctx, plans, order, full_context_weight, device)   the batched teacher-forced loss: every distinct prefix / full-context state of the graph is one disjoint copy of the graph
                                 in ONE big graph, encoded in one call (one PyG-style batch), then scored in one call (batched_tf.BatchedTeacherForced supplies the states and the scorer columns)

The scorer columns still come from `partial_state_columns` (the single definition); only the conv block is new here. Serving stays on CPU; training may run on a GPU, where scatter-adds are not bitwise deterministic
(the determinism test covers the CPU path)."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data

from src.placement.link_features import LINK_EDGE_DIM
from src.policy.gnn.batched_tf import BatchedTeacherForced, _Static
from src.policy.gnn.partial_state_edges import candidate_edge_rows, refresh_partial_state_edge_attr
from src.policy.tabular import reduced_features as rf

ATTR = "link_edge_conv_attr"
MIB = 1024 ** 2
_MEMO = "_link_conv_static"
# a chunk of states is processed in tensors of at most this many elements per intermediate
_CHUNK_ELEMS = 24_000_000


class LinkStatic:
    """Everything about a graph the conv block needs that does not depend on a committed set: the batched_tf statics, the partner lists, the pair list, the route tables and the edge-row maps."""

    def __init__(self, data: Any, ctx: Any) -> None:
        if ctx.contract not in rf.LINK_CONTRACTS:
            raise ValueError(f"FAIL LOUD: the link conv block is defined for {sorted(rf.LINK_CONTRACTS)}, not {ctx.contract!r}")
        if any(ctx.parents.get(t) for t in ctx.parents):
            raise ValueError("FAIL LOUD: the link conv block supports peer corpora without DAG parents (commit order = ascending task index)")
        S = self.S = _Static(data, ctx)
        self.n, self.R = S.n, S.R
        n, R = S.n, S.R
        deg = (S.Pm > 0).sum(1)
        self.D = D = int(max(deg.max(), 1))
        pj = np.zeros((R, D), dtype=np.int64)
        pb = np.zeros((R, D))
        pm = np.zeros((R, D), dtype=bool)
        for t in range(n):
            js = np.nonzero(S.Pm[t] > 0)[0]  # ascending: the definition's partner order
            a, c = int(S.start[t]), int(S.C[t])
            for r in range(a, a + c):
                pj[r, :len(js)] = js
                pb[r, :len(js)] = S.Bm[t, js]
                pm[r, :len(js)] = True
        self.pj, self.pb, self.pm = torch.from_numpy(pj), torch.from_numpy(pb), torch.from_numpy(pm)
        L = S.link
        self.pu, self.pv = torch.from_numpy(L["pu"]), torch.from_numpy(L["pv"])
        self.hopl, self.hopbw = torch.from_numpy(L["HOPL"]), torch.from_numpy(L["HOPBW"])
        self.base = torch.from_numpy(L["base"])
        self.PB, self.LATE = torch.from_numpy(S.PB), torch.from_numpy(S.LATE)
        self.L, self.H = int(L["L"]), int(L["H"])
        self.rn, self.rt = torch.from_numpy(S.row_node), torch.from_numpy(S.rt)
        self.start = S.start
        # edge rows: forward row of (task, logit k) and its reverse row
        from src.policy.gnn.plan_raw import _edge_platforms

        _plat, reverse = _edge_platforms(data)
        fwd = np.asarray(S.edge_rows, dtype=np.int64)
        self.fwd = torch.from_numpy(fwd)
        self.rev = torch.from_numpy(np.asarray([reverse[int(r)] for r in fwd], dtype=np.int64))
        self.n_edges = int(data.edge_index.size(1))
        place = data.task_logit_to_placement
        self.index_of = [{tuple(int(v) for v in c): k for k, c in enumerate(place[t])} for t in range(n)]
        self._dev: Dict[str, Dict[str, Tensor]] = {}

    def on(self, device: torch.device) -> Dict[str, Tensor]:
        key = str(device)
        if key not in self._dev:
            names = ("pj", "pb", "pm", "pu", "pv", "hopl", "hopbw", "base", "PB", "LATE", "rn", "rt", "fwd", "rev")
            self._dev[key] = {k: getattr(self, k).to(device) for k in names}
        return self._dev[key]

    def placement_arrays(self, committed: Mapping[int, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """(Cm [n], ND [n]) of one committed set {task: placement tuple}."""
        cm = np.zeros(self.n)
        nd = np.zeros(self.n, dtype=np.int64)
        for j, p in committed.items():
            key = tuple(int(v) for v in p)
            k = self.index_of[int(j)].get(key)
            if k is None:
                raise KeyError(f"link conv: task {j} committed to {key}, not one of its candidates")
            cm[int(j)] = 1.0
            nd[int(j)] = int(self.S.row_node[int(self.start[int(j)]) + k])
        return cm, nd


def static_of(data: Any, ctx: Any) -> LinkStatic:
    memo = getattr(data, _MEMO, None)
    if memo is None:
        memo = LinkStatic(data, ctx)
        setattr(data, _MEMO, memo)
    return memo


def link_rows(LS: LinkStatic, Cm: Tensor, ND: Tensor) -> Tensor:
    """E1-E3 of every candidate row under each of Sc committed sets -> [Sc, R, 3] float64. Cm [Sc, n] 0/1 float64, ND [Sc, n] long (any valid node index for an uncommitted task)."""
    dev = Cm.device
    T = LS.on(dev)
    Sc, n = Cm.shape
    R, D, H, L = LS.R, LS.D, LS.H, LS.L
    u, v = T["pu"], T["pv"]
    K = int(u.numel())
    sidx = torch.arange(Sc, device=dev)
    if K:
        nu, nv = ND[:, u], ND[:, v]
        act = Cm[:, u] * Cm[:, v] * (nu != nv).to(Cm.dtype)  # [Sc, K]: both ends committed, on different nodes -> one flow v -> u
        hl = T["hopl"][nv, nu]  # [Sc, K, H]
        w = act[:, :, None] * (hl >= 0).to(Cm.dtype)
        idx = hl.clamp(min=0)
        pend = torch.zeros(Sc, L, dtype=Cm.dtype, device=dev)
        pend.scatter_add_(1, idx.reshape(Sc, -1), w.reshape(Sc, -1))
        corr = torch.zeros(Sc, n * L, dtype=Cm.dtype, device=dev)  # the flows a task's OWN placement adds are not in its prefix
        for end in (u, v):
            corr.scatter_add_(1, (end[None, :, None] * L + idx).reshape(Sc, -1), w.reshape(Sc, -1))
        counts_t = T["base"][None, None, :] + pend[:, None, :] - corr.view(Sc, n, L)  # [Sc, n, L]
    else:
        counts_t = T["base"][None, None, :].expand(Sc, n, L)
    pj, pb, pm, rn, rt = T["pj"], T["pb"], T["pm"], T["rn"], T["rt"]
    ndj = ND[:, pj]  # [Sc, R, D]
    use = (Cm[:, pj] > 0) & pm[None] & (ndj != rn[None, :, None])
    hl2 = T["hopl"][ndj, rn[None, :, None]]  # [Sc, R, D, H]: the route partner j's node -> this row's node
    valid = hl2 >= 0
    cnt = counts_t[sidx[:, None, None, None], rt[None, :, None, None], hl2.clamp(min=0)]
    act_h = torch.where(valid, cnt, torch.zeros((), dtype=cnt.dtype, device=dev))
    anyact = (act_h > 0).any(-1)
    bw = T["hopbw"][ndj, rn[None, :, None]]
    inf = torch.full((), float("inf"), dtype=cnt.dtype, device=dev)
    rate = torch.where(valid, bw * MIB / (act_h + 1.0), inf).amin(-1)
    pbj = T["PB"][ndj, rn[None, :, None]]
    lat = T["LATE"][ndj, rn[None, :, None]]
    b = pb[None]
    solo_j = b * pbj + lat
    shared_j = torch.where(anyact, b / rate + lat, solo_j)
    zero = torch.zeros((), dtype=cnt.dtype, device=dev)
    shared = torch.zeros(Sc, R, dtype=cnt.dtype, device=dev)
    solo = torch.zeros(Sc, R, dtype=cnt.dtype, device=dev)
    for d in range(D):  # ascending partner index, sequential: the definition's float order
        shared = shared + torch.where(use[..., d], shared_j[..., d], zero)
        solo = solo + torch.where(use[..., d], solo_j[..., d], zero)
    busiest = torch.where(use[..., None], act_h, zero).amax(dim=(-1, -2))
    e1 = torch.log1p(shared)
    ratio = torch.where(solo > 0, shared / torch.where(solo > 0, solo, torch.ones_like(solo)), torch.ones_like(solo))
    e2 = torch.where((solo > 0) & (shared > solo), torch.log(ratio), zero)
    e3 = torch.log1p(busiest)
    return torch.stack([e1, e2, e3], dim=-1)


def rows_to_edges(LS: LinkStatic, rows: Tensor) -> Tensor:
    """[Sc, R, 3] candidate rows -> [Sc, E, 3] float32 edge rows: forward rows get the block, reverse rows copy it."""
    T = LS.on(rows.device)
    out = torch.zeros(rows.shape[0], LS.n_edges, LINK_EDGE_DIM, dtype=torch.float32, device=rows.device)
    r32 = rows.to(torch.float32)
    out[:, T["fwd"]] = r32
    out[:, T["rev"]] = r32
    return out


def link_edge_block(data: Any, ctx: Any, committed: Mapping[int, Any], device: Optional[torch.device] = None) -> Tensor:
    """The conv block [E, 3] float32 of one committed set."""
    LS = static_of(data, ctx)
    dev = device or data.edge_index.device
    cm, nd = LS.placement_arrays(committed)
    with torch.no_grad():
        rows = link_rows(LS, torch.from_numpy(cm)[None].to(dev), torch.from_numpy(nd)[None].to(dev))
        return rows_to_edges(LS, rows)[0]


def make_link_conv_score_fn(model: Any, data: Any, ctx: Any, task_rows_only: bool = False):
    """`(task_idx, committed) -> logits[task_idx]`: the conv block of the committed set, one encode per DISTINCT committed set, the scorer columns from `partial_state_columns` for the scored task."""
    if not getattr(model, "mp_link_edge_conv", False):
        raise ValueError("make_link_conv_score_fn: the model was not built with mp_link_edge_conv")
    if getattr(model, "plan_raw", False) or getattr(model, "committed_inputs", False) or getattr(model, "group_anchor", False):
        raise ValueError("FAIL LOUD: mp_link_edge_conv does not combine with plan_raw / the Arm 2 re-encode / the group anchor")
    expected = rf.partial_state_feature_dim(getattr(ctx, "contract", None))
    if int(getattr(model, "partial_state_edge_dim", 0)) != expected:
        raise ValueError(f"make_link_conv_score_fn: model.partial_state_edge_dim={int(getattr(model, 'partial_state_edge_dim', 0))}, contract {ctx.contract!r} emits {expected}")
    LS = static_of(data, ctx)
    dev = data.edge_index.device
    enc: Dict[Tuple[Any, ...], Tuple[Tensor, Tensor]] = {}
    rows_state: Dict[str, Any] = {}

    def score(task_idx: int, committed: Mapping[int, Any]) -> Tensor:
        key = tuple(sorted((int(j), tuple(int(v) for v in p)) for j, p in committed.items()))
        emb = enc.get(key)
        if emb is None:
            setattr(data, ATTR, link_edge_block(data, ctx, committed, dev))
            emb = enc[key] = model._encode(data)
        task_emb, platform_emb = emb
        refresh_partial_state_edge_attr(data, ctx, task_idx, committed)
        if task_rows_only:
            from src.policy.gnn.partial_state_edges import _score_task_rows

            return _score_task_rows(model, task_emb, platform_emb, data, int(task_idx), rows_state)
        return model._score(task_emb, platform_emb, data)[int(task_idx)]

    return score


# ---------------------------------------------------------------------------------------------------------------------------- training: one big graph of S disjoint copies

def replicate_graph(data: Any, copies: int, conv_attr: Tensor) -> Data:
    """`copies` disjoint copies of the graph as ONE graph whose task block is [copy0 tasks ... copyS-1 tasks] and platform block [copy0 platforms ...], so `model._encode` runs unchanged.
    `conv_attr` [copies, E, 3] becomes the big graph's link_edge_conv_attr."""
    n_t, n_p = int(data.n_tasks), int(data.n_platforms)
    S = int(copies)
    dev = data.edge_index.device

    def remap(idx: Tensor) -> Tensor:  # node ids of one copy -> the big graph's, for all copies; [2, S*E]
        s = torch.arange(S, device=dev).repeat_interleave(idx.size(1))
        flat = idx.repeat(1, S)
        is_task = flat < n_t
        new = torch.where(is_task, flat + s * n_t, S * n_t + s * n_p + (flat - n_t))
        return new

    big = Data()
    big.n_tasks, big.n_platforms = S * n_t, S * n_p
    big.task_features = data.task_features.repeat(S, 1)
    big.platform_features = data.platform_features.repeat(S, 1)
    onehot = getattr(data, "task_type_onehot4", None)
    if onehot is not None:
        big.task_type_onehot4 = onehot.repeat(S, 1)
    big.edge_index = remap(data.edge_index)
    big.edge_attr = data.edge_attr.repeat(S, 1)
    setattr(big, ATTR, conv_attr.reshape(S * conv_attr.shape[1], LINK_EDGE_DIM))
    pei = getattr(data, "peer_edge_index", None)
    if pei is not None and pei.numel() > 0:
        s = torch.arange(S, device=dev).repeat_interleave(pei.size(1))
        big.peer_edge_index = pei.to(dev).repeat(1, S) + s * n_t
        big.peer_edge_attr = data.peer_edge_attr.to(dev).repeat(S, 1)
    elif pei is not None:
        big.peer_edge_index, big.peer_edge_attr = pei, data.peer_edge_attr
    nei = getattr(data, "node_edge_index", None)
    if nei is not None and nei.numel() > 0:
        big.node_edge_index = remap(nei.to(dev))
    dei = getattr(data, "dag_edge_index", None)
    if dei is not None and dei.numel() > 0:
        s = torch.arange(S, device=dev).repeat_interleave(dei.size(1))
        big.dag_edge_index = dei.to(dev).repeat(1, S) + s * n_t
    return big


def state_arrays(b: BatchedTeacherForced) -> Tuple[np.ndarray, np.ndarray]:
    """(Cm [S, n], ND [S, n]) of the batched teacher-forced object's distinct states: the committed indicator and the node of every task under the state's plan."""
    S = b.S
    n = S.n
    pos = np.empty(n, dtype=np.int64)
    pos[b.order] = np.arange(n)
    Cm = np.zeros((len(b.states), n))
    ND = np.zeros((len(b.states), n), dtype=np.int64)
    for s, (ns, k, t) in enumerate(b.states):
        if ns == 0:
            Cm[s] = (pos < pos[t]).astype(np.float64)
        else:
            Cm[s] = 1.0
            Cm[s, t] = 0.0
        ND[s] = S.row_node[S.start + b.plans[k]]
    return Cm, ND


def state_log_softmax_per_state(b: BatchedTeacherForced, model: Any, data: Any, task_emb: Tensor, platform_emb: Tensor) -> Tensor:
    """`b.state_log_softmax` with a separate encode per state: the state's rows read the embeddings of its own copy."""
    import torch.nn.functional as F

    S = b.S
    dev = task_emb.device
    n_t, n_p = int(data.n_tasks), int(data.n_platforms)
    eidx = torch.as_tensor(S.edge_rows[b.row_in_R], device=dev)
    ei = data.edge_index.to(dev)
    ti, pj = ei[0, eidx], ei[1, eidx] - n_t
    seg = torch.as_tensor(b.seg, device=dev)
    parts = [task_emb[seg * n_t + ti], platform_emb[seg * n_p + pj]]
    if hasattr(data, "edge_attr") and data.edge_attr.numel() > 0:
        parts.append(data.edge_attr.to(dev)[eidx].to(task_emb.dtype))
    parts.append(b.ps.to(dev, dtype=task_emb.dtype))
    x = torch.cat(parts, dim=-1)
    sc = model.edge_scorer
    logits = sc.fc2(sc.dropout(F.relu(sc.fc1(x)))).squeeze(-1)
    col = torch.as_tensor(b.col, device=dev)
    pad = torch.full((len(b.counts), b.cmax), float("-inf"), device=dev, dtype=logits.dtype)
    pad = pad.index_put((seg, col), logits)
    return F.log_softmax(pad, dim=-1)


def _encode_copies(model: Any, data: Any, attr: Tensor, edge_budget: int) -> Tuple[Tensor, Tensor]:
    """The task and platform embeddings of every copy, [S * n_tasks, d] and [S * n_platforms, d] in copy order. Copies are encoded in chunks of at most `edge_budget` edges (the copies are disjoint, so a chunk
    encodes exactly what the whole would), and under autograd each chunk runs inside torch.utils.checkpoint: only the chunk's inputs and outputs are kept, its activations are recomputed in the backward pass
    (the RNG state is restored, so a dropout mask is the one the forward drew). That bounds the memory by one chunk instead of all 2T states of the largest group."""
    from torch.utils.checkpoint import checkpoint

    S = int(attr.shape[0])
    per_copy = int(data.edge_index.size(1)) + int(getattr(data, "peer_edge_index", torch.empty(2, 0)).size(1)) + int(getattr(data, "node_edge_index", torch.empty(2, 0)).size(1))
    step = max(1, min(S, int(edge_budget // max(per_copy, 1))))
    tes: List[Tensor] = []
    pes: List[Tensor] = []
    for a in range(0, S, step):
        chunk = attr[a:a + step]

        def run(chunk=chunk):
            return model._encode(replicate_graph(data, int(chunk.shape[0]), chunk))

        if torch.is_grad_enabled():
            te, pe = checkpoint(run, use_reentrant=False)
        else:
            te, pe = run()
        tes.append(te)
        pes.append(pe)
    return torch.cat(tes, 0), torch.cat(pes, 0)


_E_STATS_N = [0]


def _debug_transform_e(LS: LinkStatic, data: Any, ctx: Any, attr: Tensor) -> Tensor:
    """Diagnostics of the E instability (env only, never set in a registered run): LINK_CONV_E_STATS=1 prints per-column statistics of the forward rows over the first 40 graphs; LINK_CONV_E_SCALE=a,b,c divides
    the columns; LINK_CONV_E_MODE=zero zeroes E, =static replaces it with the state-independent S1-S3 block."""
    import os

    if os.environ.get("LINK_CONV_E_STATS") == "1" and _E_STATS_N[0] < 40:
        _E_STATS_N[0] += 1
        fwd = attr[:, LS.on(attr.device)["fwd"]].reshape(-1, LINK_EDGE_DIM)
        q = torch.quantile(fwd[:: max(1, fwd.shape[0] // 200000)], torch.tensor([0.5, 0.99], device=fwd.device), dim=0)
        print(f"[e-stats] graph={getattr(data, 'dataset_id', '?')} states={attr.shape[0]} mean={fwd.mean(0).tolist()} std={fwd.std(0).tolist()} p50={q[0].tolist()} p99={q[1].tolist()} max={fwd.max(0).values.tolist()}", flush=True)
    sc = os.environ.get("LINK_CONV_E_SCALE")
    if sc:
        attr = attr / torch.tensor([float(v) for v in sc.split(",")], dtype=attr.dtype, device=attr.device)
    mode = os.environ.get("LINK_CONV_E_MODE", "")
    if mode == "zero":
        attr = torch.zeros_like(attr)
    elif mode == "static":
        from src.policy.gnn.link_static import link_static_attr_from_ctx

        attr = link_static_attr_from_ctx(data, ctx).to(attr.device, attr.dtype)[None].expand_as(attr).contiguous()
    elif mode:
        raise ValueError(f"LINK_CONV_E_MODE={mode!r}")
    return attr


def link_conv_loss(model: Any, data: Any, ctx: Any, plans: Sequence[Sequence[int]], order: Sequence[int], full_context_weight: float, device: torch.device,
                   edge_budget: int = 300_000) -> Tensor:
    """The any-of-K teacher-forced CE of the re-encode variant: -logsumexp_k sum_t log p(plan_k[t] | prefix state), plus `full_context_weight` times the same over the full-context states.
    Each distinct state is encoded under ITS committed set's conv block, as a disjoint copy of the graph, in chunks of `edge_budget` edges with checkpointing (`_encode_copies`)."""
    from src.policy.gnn.batched_tf import _unsupported

    _unsupported(model, ctx)
    if not getattr(model, "mp_link_edge_conv", False):
        raise ValueError("link_conv_loss: the model was not built with mp_link_edge_conv")
    b = BatchedTeacherForced(data, ctx, plans, order, with_full=full_context_weight > 0.0)
    LS = static_of(data, ctx)
    Cm, ND = state_arrays(b)
    n_states = Cm.shape[0]
    rows = []
    step = max(1, int(_CHUNK_ELEMS // max(LS.R * LS.D * LS.H, 1)))
    with torch.no_grad():
        for a in range(0, n_states, step):
            rows.append(link_rows(LS, torch.from_numpy(Cm[a:a + step]).to(device), torch.from_numpy(ND[a:a + step]).to(device)))
        attr = _debug_transform_e(LS, data, ctx, rows_to_edges(LS, torch.cat(rows, 0)))
    task_emb, platform_emb = _encode_copies(model, data, attr, edge_budget)
    lp = state_log_softmax_per_state(b, model, data, task_emb, platform_emb)
    loss = -torch.logsumexp(b.step_logps(lp, "prefix").sum(dim=1), dim=0)
    if full_context_weight > 0.0:
        loss = loss + full_context_weight * -torch.logsumexp(b.step_logps(lp, "full").sum(dim=1), dim=0)
    return loss


def equivalence_report(model: Any, graphs: Sequence[Any], loss_fn: Any, device: torch.device, n_graphs: int, seq_max_tasks: int, small_budget: int = 60_000) -> None:
    """Real-data check of the re-encode loss (NEAR_RTT_LINK_CONV_EQUIV=1 in the trainer): per sampled graph (every k-th by size, plus the largest ten) the loss and its gradient with the default chunk budget against a
    small budget, against one unchunked call where it fits, and against the sequential closure for graphs up to `seq_max_tasks`. `loss_fn(model, data, seq)` returns the scalar loss; eval mode, dropout off."""
    import time

    model.eval()
    order = sorted(range(len(graphs)), key=lambda i: int(graphs[i].n_tasks))
    step = max(1, len(order) // max(1, n_graphs - 10))
    picks = order[::step] + order[-10:]
    defaults = link_conv_loss.__defaults__

    def run(data, budget, seq):
        link_conv_loss.__defaults__ = (budget,)
        model.zero_grad(set_to_none=True)
        loss = loss_fn(model, data, seq)
        loss.backward()
        g = torch.cat([p.grad.detach().flatten() if p.grad is not None else torch.zeros(p.numel(), device=device) for p in model.parameters()])
        return float(loss.detach()), g

    try:
        for i in picks:
            data = graphs[i]
            nt = int(data.n_tasks)
            t0 = time.time()
            ref, gref = run(data, defaults[0], False)
            row = f"[equiv] graph={getattr(data, 'dataset_id', '?')} n_tasks={nt} ce={ref:.6g} ce_per_task={ref / max(nt, 1):.4g} gnorm={float(gref.norm()):.4g}"
            for name, budget, seq in (("small_budget", small_budget, False), ("unchunked", 10 ** 12, False), ("sequential", defaults[0], True)):
                if seq and nt > seq_max_tasks:
                    continue
                if name == "unchunked" and nt > 80:
                    continue
                v, g = run(data, budget, seq)
                row += f" | {name}: dce={v - ref:+.3g} grad_rel={float((g - gref).norm() / (gref.norm() + 1e-12)):.3g}"
            print(row + f" ({time.time() - t0:.1f}s)", flush=True)
    finally:
        link_conv_loss.__defaults__ = defaults
