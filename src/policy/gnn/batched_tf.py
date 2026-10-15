"""uncut_groups_s0_v1 Arm 1b: the batched teacher-forced loss (NEAR_RTT_BATCHED_TF=1).

Arm 1's loss (``train_near_rtt.loss_tied_teacher_forced_ce``) walks every task of a group in Python: per step it rebuilds the
partial-state block (``partial_state_columns``), refills ``partial_state_edge_attr`` and scores that task's 3-10 rows. Under teacher
forcing the label plans are known up front, so every committed prefix is known before the walk starts and nothing is sequential:

  * the partial-state block of every (plan, task) state is built in ONE vectorised pass (``_Static`` + ``_blocks``), from a mask
    ``M[t, j]`` saying which tasks are committed in task t's state (strict prefix of the commit order, or "all others" for the
    full-context term);
  * every distinct state's candidate rows go through the edge scorer in ONE call, padded with -inf to the widest candidate set
    before ``log_softmax``;
  * the tied-plan CE and the full-context term are gathered from those log-probs exactly as the sequential loss composes them:
    ``-logsumexp_k sum_t logp`` (+ ``w`` times the same over the full-context states).

What differs from the sequential loss (declared in the sidecar as ``batched_teacher_forced`` and in the lineage node):
  1. Dropout. The sequential loop draws a fresh dropout mask over the whole graph's edge tensor at every step. Here the
     edge scorer's dropout draws one mask over the batch of distinct-state rows. Same distribution (independent per row, a
     state scored once however many plans share it), a different random stream. With dropout off (eval mode) the two agree to
     float rounding.
  2. Float summation order inside the features and the logit sums (matrix products instead of dict loops). Agreement is ~1e-6
     relative, not bit for bit.
Everything else is the sequential definition: strict prefix per state (a task's own commit is excluded), each tied plan keeps its
own prefix (the dedupe key is the full (task, committed set with each placement)), the encode is prefix-free (asserted), the
loss composition is unchanged, and the same cap / base-load / peer / load-seconds / linkrank columns are produced.

Supported: ``TaskPlacementGNN`` with the fc1 / dropout / fc2 EdgeScorer and ``TaskPlacementMLPSame`` (its own ``net`` on the same flat
per-edge vector; LayerNorm is per row, so no state mixes with another), ``partial_state_v5`` with load seconds. Anything else
(Arm 2 re-encode or group anchor, plan_raw, another contract, random-prefix orders) raises.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from src.policy.gnn.partial_state_edges import candidate_edge_rows
from src.policy.tabular import reduced_features as rf

_EPS = rf._PARTIAL_STATE_EPS


def _is_mlp_same(model: Any) -> bool:
    return getattr(model, "arm_kind", None) == "mlp_same" and isinstance(getattr(model, "net", None), torch.nn.Sequential)


def _unsupported(model: Any, ctx: Any) -> None:
    for flag in ("plan_raw", "committed_inputs", "group_anchor"):
        if getattr(model, flag, False):
            raise ValueError(f"NEAR_RTT_BATCHED_TF: {flag} models are not supported (Arm 1 TaskPlacementGNN and TaskPlacementMLPSame only)")
    if not _is_mlp_same(model):
        sc = getattr(model, "edge_scorer", None)
        if sc is None or not all(hasattr(sc, a) for a in ("fc1", "fc2", "dropout")):
            raise ValueError("NEAR_RTT_BATCHED_TF needs the fc1 / dropout / fc2 EdgeScorer of TaskPlacementGNN, or TaskPlacementMLPSame")
    if not getattr(model, "partial_state_edge_dim", 0):
        raise ValueError("NEAR_RTT_BATCHED_TF is defined for the prefix-conditioned scorer (partial_state_edge_dim > 0)")
    if ctx.contract not in rf.BOTH_DIRECTION_CONTRACTS or ctx.contract not in rf.PEER_BLOCK_CONTRACTS:
        raise ValueError(f"NEAR_RTT_BATCHED_TF is built for partial_state_v5, not {ctx.contract!r}")
    if ctx.contract in rf.KRANK_ONEHOT_CONTRACTS:
        raise ValueError("NEAR_RTT_BATCHED_TF: one-hot krank contracts are not supported")
    if any(ctx.parents.get(t) for t in ctx.parents):
        raise ValueError("NEAR_RTT_BATCHED_TF: DAG corpora (parent edges) are not supported; the peer block owns columns 7-9")
    if int(model.partial_state_edge_dim) != rf.partial_state_feature_dim(ctx.contract):
        raise ValueError("NEAR_RTT_BATCHED_TF: model.partial_state_edge_dim disagrees with the contract width")


class _Static:
    """Everything about a graph that does not depend on a plan or a prefix, over R = sum_t C_t candidate rows in
    (task, logit-index) order."""

    def __init__(self, data: Any, ctx: Any) -> None:
        self.n = n = int(data.n_tasks)
        place = data.task_logit_to_placement
        self.C = np.array([len(place[t]) for t in range(n)], dtype=np.int64)
        if (self.C <= 0).any():
            raise ValueError("NEAR_RTT_BATCHED_TF: a task has no candidates")
        self.start = np.concatenate([[0], np.cumsum(self.C)[:-1]]).astype(np.int64)
        self.R = R = int(self.C.sum())
        self.rt = np.repeat(np.arange(n), self.C)
        er = candidate_edge_rows(data)  # asserts edge row <-> logit order
        self.edge_rows = np.array([r for t in range(n) for r in er[t]], dtype=np.int64)
        cands = [tuple(c) for t in range(n) for c in place[t]]
        self.cands = cands

        nodes = set(ctx.node_of.values())
        for lst in ctx.cand_nodes.values():
            nodes.update(lst)
        for a, b in ctx.node_exchange:
            nodes.add(a)
            nodes.add(b)
        self.nodes = sorted(nodes)
        nidx = {v: i for i, v in enumerate(self.nodes)}
        self.nidx = nidx
        V = len(self.nodes)

        self.row_node = np.array([nidx[ctx.node_of[c]] for c in cands], dtype=np.int64)
        keys: Dict[Any, int] = {}
        self.row_key = np.array([keys.setdefault(c, len(keys)) for c in cands], dtype=np.int64)
        self.row_d = np.array([float(ctx.demand[(int(self.rt[i]), c)]) for i, c in enumerate(cands)], dtype=np.float64)
        self.row_svc = np.array(
            [float(ctx.service_s[(int(self.rt[i]), c)]) if ctx.load_seconds else 0.0 for i, c in enumerate(cands)],
            dtype=np.float64)
        self.types = np.array([int(ctx.task_type_index[t]) for t in range(n)], dtype=np.int64)
        if ((self.types < 0) | (self.types >= rf.KRANK_TYPES)).any():
            raise ValueError("NEAR_RTT_BATCHED_TF: type index out of range")
        caps = np.array([float(ctx.node_caps.get(ctx.node_of[c], math.inf)) for c in cands], dtype=np.float64)
        self.cap_ok = np.isfinite(caps) & (caps > 0.0)
        self.cap = np.where(self.cap_ok, caps, 1.0)
        base = {v: float(ctx.base_load.get(v, 0.0)) for v in self.nodes}
        self.row_base = np.array([base[ctx.node_of[c]] for c in cands], dtype=np.float64)

        # exchange tables: xchg(frm, to, b) = b * PB[frm, to] + LATE[frm, to]
        PB = np.zeros((V, V)); LATE = np.zeros((V, V)); HAS = np.zeros((V, V), dtype=bool)
        for (a, b), (pb, lat) in ctx.node_exchange.items():
            PB[nidx[a], nidx[b]] = pb
            LATE[nidx[a], nidx[b]] = lat if pb > 0.0 else 0.0
            HAS[nidx[a], nidx[b]] = True
        if not HAS.all():
            raise ValueError("NEAR_RTT_BATCHED_TF: node_exchange does not cover every node pair; the batched builder needs a complete table")
        self.PB, self.LATE = PB, LATE

        Bm = np.zeros((n, n)); Pm = np.zeros((n, n))
        for (i, j), b in ctx.peer_pairs.items():
            if i == j:
                raise ValueError("NEAR_RTT_BATCHED_TF: a self peer pair")
            Bm[i, j] = b
            Pm[i, j] = 1.0
        self.Bm, self.Pm = Bm, Pm
        self.rBr, self.rPr = Bm[self.rt], Pm[self.rt]

        # peer-mass lookahead: mean over task j's candidate nodes of the exchange to / from this row's node
        self.peer_mass = bool(ctx.peer_mass)
        PBm = np.zeros((n, V)); LAm = np.zeros((n, V)); PBp = np.zeros((n, V)); LAp = np.zeros((n, V))
        has_nodes = np.zeros(n)
        for j in range(n):
            nj = [nidx[x] for x in (ctx.cand_nodes.get(j) or ())]
            if nj:
                has_nodes[j] = 1.0
                PBm[j] = PB[nj].mean(0); LAm[j] = LATE[nj].mean(0)  # mean_nj xchg(nj -> v)
                PBp[j] = PB[:, nj].mean(1); LAp[j] = LATE[:, nj].mean(1)  # mean_nj xchg(v -> nj)
        rn = self.row_node
        Pme = self.rPr * has_nodes[None, :]
        self.MM = Pme * (self.rBr * PBm[:, rn].T + LAm[:, rn].T)
        self.MMp = Pme * (self.rBr * PBp[:, rn].T + LAp[:, rn].T)

        # rank / link statics
        n_rank = len(ctx.node_rank)
        self.template = np.zeros((R, rf.partial_state_feature_dim(ctx.contract)), dtype=np.float64)
        for i, c in enumerate(cands):
            r = int(ctx.node_rank[ctx.node_of[c]])
            if not (0 <= r < n_rank):
                raise ValueError(f"NEAR_RTT_BATCHED_TF: node rank {r} outside the {n_rank} ranked nodes")
            slot = rf.PARTIAL_STATE_BASE_DIM + int(self.types[self.rt[i]]) * rf.KRANK_V3_SCALARS
            self.template[i, slot] = (r / (n_rank - 1)) if n_rank > 1 else 0.0
            self.template[i, slot + 1] = 1.0 / n_rank
        self.linkrank_base = rf.PARTIAL_STATE_BASE_DIM + rf.krank_feature_dim(ctx.contract)
        self.load_base = self.linkrank_base + rf.LINKRANK_FEATURE_DIM
        self.lkidx: Dict[str, int] = {}
        row_links: List[List[int]] = []
        for i, c in enumerate(cands):
            row_links.append([self.lkidx.setdefault(lk, len(self.lkidx))
                              for lk in ctx.ingress_links.get((int(self.rt[i]), ctx.node_of[c]), ())])
        for lst in ctx.ingress_links.values():  # links of the plan's own commits may lie outside the rows' lists
            for lk in lst:
                self.lkidx.setdefault(lk, len(self.lkidx))
        Lmax = max([len(x) for x in row_links] + [1])
        self.RL = np.zeros((R, Lmax), dtype=np.int64)
        self.RLv = np.zeros((R, Lmax), dtype=bool)
        self.RLc = np.zeros((R, Lmax), dtype=bool)
        core = ctx.core_links
        inv = {v: k for k, v in self.lkidx.items()}
        for i, lst in enumerate(row_links):
            for q, li in enumerate(lst):
                self.RL[i, q] = li
                self.RLv[i, q] = True
                self.RLc[i, q] = inv[li] in core
        self.L = len(self.lkidx)
        self.row_backlog = np.zeros(R)
        if ctx.load_seconds:
            for i, c in enumerate(cands):
                if c not in ctx.backlog_s:
                    raise ValueError(f"NEAR_RTT_BATCHED_TF: no backlog for candidate {c!r}")
                self.row_backlog[i] = float(ctx.backlog_s[c])
        self.ctx_load_seconds = bool(ctx.load_seconds)


def _blocks(S: _Static, ctx: Any, data: Any, plan: Sequence[int], M: np.ndarray) -> np.ndarray:
    """The partial-state block of every row, each task's rows under its own state: ``M[t, j] = 1`` iff task j is committed in
    task t's state (at the plan's placement). Returns [R, D] float32. Mirrors ``reduced_features.partial_state_columns``."""
    n, R = S.n, S.R
    p = np.asarray(plan, dtype=np.int64)
    if p.shape != (n,) or (p < 0).any() or (p >= S.C).any():
        raise ValueError("NEAR_RTT_BATCHED_TF: a tied plan's logit index is out of range")
    prow = S.start + p
    nd_plan, key_plan = S.row_node[prow], S.row_key[prow]
    d_plan, svc_plan = S.row_d[prow], S.row_svc[prow]
    rt, rn = S.rt, S.row_node
    out = S.template.copy()

    Mr = M[rt]  # [R, n]
    A = Mr * (nd_plan[None, :] == rn[:, None])
    Tone = np.zeros((n, rf.KRANK_TYPES)); Tone[np.arange(n), S.types] = 1.0
    out[:, 0:rf.KRANK_TYPES] = A @ Tone
    ld = S.row_base + A @ d_plan
    ok = S.cap_ok
    out[:, 4] = np.where(ok, ld / S.cap, 0.0)
    out[:, 5] = np.where(ok, (S.cap - ld - S.row_d) / S.cap, 1.0)
    out[:, 6] = np.where(ok & (ld + S.row_d > S.cap + _EPS), 1.0, 0.0)

    # peer block (pull = partner node -> this row's node; push = this row's node -> partner node)
    PBl, LEl = S.PB[nd_plan[None, :], rn[:, None]], S.LATE[nd_plan[None, :], rn[:, None]]
    PBu, LEu = S.PB[rn[:, None], nd_plan[None, :]], S.LATE[rn[:, None], nd_plan[None, :]]
    use = Mr * S.rPr
    cx = (use * (S.rBr * PBl + LEl)).sum(1)
    px = (use * (S.rBr * PBu + LEu)).sum(1)
    if S.peer_mass:
        notc = 1.0 - Mr
        mass = (notc * S.MM).sum(1)
        pmass = (notc * S.MMp).sum(1)
    else:
        mass = pmass = np.zeros(R)
    out[:, 7] = np.log1p(cx)
    out[:, 8] = np.log1p(mass)
    out[:, 9] = 0.0
    out[:, S.load_base + rf.LOAD_SECONDS_DIM + 0] = np.log1p(px)
    out[:, S.load_base + rf.LOAD_SECONDS_DIM + 1] = np.log1p(pmass)

    # linkrank: how many committed tasks already use each link of this row's ingress route
    Linc = np.zeros((n, max(S.L, 1)))
    for j in range(n):
        for lk in ctx.ingress_links.get((j, ctx.node_of[S.cands[int(prow[j])]]), ()):
            Linc[j, S.lkidx[lk]] += 1.0
    Cz = M @ Linc
    cnt = Cz[rt[:, None], S.RL]  # [R, Lmax]
    v, vc = S.RLv, S.RLv & S.RLc
    out[:, S.linkrank_base + 0] = np.where(v, cnt + 1.0, 0.0).max(1)
    out[:, S.linkrank_base + 1] = (v & (cnt >= 1.0)).sum(1)
    out[:, S.linkrank_base + 2] = np.where(vc, cnt + 1.0, 0.0).max(1)
    out[:, S.linkrank_base + 3] = (vc & (cnt >= 1.0)).sum(1)

    if S.ctx_load_seconds:
        # service already committed to the SAME replica: execution + I/O + the pull each committed mate pays its committed partners
        Ex = S.Pm * (S.Bm * S.PB[nd_plan[None, :], nd_plan[:, None]] + S.LATE[nd_plan[None, :], nd_plan[:, None]])  # [mate j, partner i]
        W = M * (svc_plan[None, :] + M @ Ex.T)  # [t, j]
        cs = (W[rt] * (key_plan[None, :] == S.row_key[:, None])).sum(1)
        out[:, S.load_base + 0] = np.log1p(S.row_backlog)
        out[:, S.load_base + 1] = np.log1p(cs)
        out[:, S.load_base + 2] = np.log1p(S.row_backlog + cs)
    return out.astype(np.float32)


class BatchedTeacherForced:
    """All distinct teacher-forced states of one graph, ready to score in one call.

    ``prefix_state[k, t]`` / ``full_state[k, t]`` index the state whose logits give plan k's task t in the prefix walk / in the
    full-context ("every other task committed") term. States are deduped on the FULL (task, committed set with placements),
    per namespace, never across plans' different prefixes.
    """

    def __init__(self, data: Any, ctx: Any, plans: Sequence[Sequence[int]], order: Sequence[int], with_full: bool) -> None:
        S = self.S = _Static(data, ctx)
        n = S.n
        self.plans = np.asarray([[int(v) for v in pl] for pl in plans], dtype=np.int64)
        K = self.K = int(self.plans.shape[0])
        if self.plans.shape[1] != n:
            raise ValueError("NEAR_RTT_BATCHED_TF: a tied plan does not cover every task")
        order = [int(t) for t in order]
        if sorted(order) != list(range(n)):
            raise ValueError("NEAR_RTT_BATCHED_TF: the commit order is not a permutation of the tasks")
        pos = np.empty(n, dtype=np.int64); pos[order] = np.arange(n)
        self.order = order
        Mpre = (pos[None, :] < pos[:, None]).astype(np.float64)  # strict prefix: a task's own commit is excluded
        Mfull = 1.0 - np.eye(n)
        rng = np.random.RandomState(0x5EED)
        z = rng.randint(1, 2 ** 62, size=(n, int(S.C.max())), dtype=np.int64).astype(np.uint64)
        self.state_of: Dict[str, np.ndarray] = {}
        feats: List[Tuple[int, int, int]] = []  # (namespace, plan, task) of the first plan owning each distinct state
        for ns, (name, on) in enumerate((("prefix", True), ("full", with_full))):
            sid = np.full((K, n), -1, dtype=np.int64)
            if on:
                zp = z[np.arange(n)[None, :], self.plans]  # [K, n]
                if name == "prefix":
                    acc = np.bitwise_xor.accumulate(zp[:, order], axis=1)
                    pre = np.zeros_like(acc); pre[:, 1:] = acc[:, :-1]
                    xor = np.empty_like(zp); xor[:, order] = pre
                else:
                    xor = np.bitwise_xor.reduce(zp, axis=1)[:, None] ^ zp
                seen: Dict[Tuple[int, int], int] = {}
                for k in range(K):
                    for t in range(n):
                        key = (t, int(xor[k, t]))
                        s = seen.get(key)
                        if s is None:
                            s = seen[key] = len(feats)
                            feats.append((ns, k, t))
                        sid[k, t] = s
            self.state_of[name] = sid
        self.states = feats
        S_ = len(feats)
        by_plan: Dict[Tuple[int, int], np.ndarray] = {}
        for ns, M in ((0, Mpre), (1, Mfull)):
            for k in sorted({k for (nn, k, _) in feats if nn == ns}):
                by_plan[(ns, k)] = _blocks(S, ctx, data, self.plans[k], M)
        cnt = np.array([S.C[t] for (_, _, t) in feats], dtype=np.int64)
        self.counts = cnt
        self.offs = np.concatenate([[0], np.cumsum(cnt)[:-1]]).astype(np.int64)
        tot = int(cnt.sum())
        self.row_in_R = np.empty(tot, dtype=np.int64)
        ps = np.empty((tot, S.template.shape[1]), dtype=np.float32)
        for s, (ns, k, t) in enumerate(feats):
            a, c = int(S.start[t]), int(S.C[t])
            self.row_in_R[self.offs[s]:self.offs[s] + c] = np.arange(a, a + c)
            ps[self.offs[s]:self.offs[s] + c] = by_plan[(ns, k)][a:a + c]
        self.ps = torch.from_numpy(ps)
        self.seg = np.repeat(np.arange(S_), cnt)
        self.col = np.arange(tot) - np.repeat(self.offs, cnt)
        self.cmax = int(cnt.max())

    def state_log_softmax(self, model: Any, data: Any, task_emb: Tensor, platform_emb: Tensor, ps: Tensor = None) -> Tensor:
        """log_softmax over each distinct state's candidates, [S, Cmax] (-inf padded). One edge-scorer call over all their rows.
        A state is shared by every plan that reaches it; the plans differ in which candidate they pick there (``step_logps``)."""
        S = self.S
        dev = task_emb.device
        eidx = torch.as_tensor(S.edge_rows[self.row_in_R], device=dev)
        n_tasks = int(data.n_tasks)
        ei = data.edge_index.to(dev)
        ti, pj = ei[0, eidx], ei[1, eidx] - n_tasks
        parts = [task_emb[ti], platform_emb[pj]]
        if hasattr(data, "edge_attr") and data.edge_attr.numel() > 0:
            parts.append(data.edge_attr.to(dev)[eidx].to(task_emb.dtype))
        parts.append((self.ps if ps is None else ps).to(dev, dtype=task_emb.dtype))
        x = torch.cat(parts, dim=-1)
        if _is_mlp_same(model):
            logits = model.net(x).squeeze(-1)  # LayerNorm is per row: no state mixes with another
        else:
            logits = model.edge_scorer.fc2(model.edge_scorer.dropout(F.relu(model.edge_scorer.fc1(x)))).squeeze(-1)
        seg = torch.as_tensor(self.seg, device=dev)
        col = torch.as_tensor(self.col, device=dev)
        pad = torch.full((len(self.counts), self.cmax), float("-inf"), device=dev, dtype=logits.dtype)
        pad = pad.index_put((seg, col), logits)
        return F.log_softmax(pad, dim=-1)

    def step_logps(self, lp: Tensor, name: str) -> Tensor:
        """[K, n] log p(plan_k[t] | its state) for the "prefix" or "full" states."""
        dev = lp.device
        sid = torch.as_tensor(self.state_of[name], device=dev)
        if bool((sid < 0).any()):
            raise ValueError(f"NEAR_RTT_BATCHED_TF: the {name} states were not built")
        return lp[sid, torch.as_tensor(self.plans, device=dev)]


def batched_tf_loss(model: Any, data: Any, ctx: Any, plans: Sequence[Sequence[int]], order: Sequence[int],
                    full_context_weight: float, device: torch.device) -> Tensor:
    """-log sum_k prod_t p(plan_k[t] | strict prefix) + w * (-log sum_k prod_t p(plan_k[t] | all others)), batched."""
    _unsupported(model, ctx)
    b = BatchedTeacherForced(data, ctx, plans, order, with_full=full_context_weight > 0.0)
    task_emb, platform_emb = model._encode(data)  # prefix-free: the partial-state block never enters the encode
    lp = b.state_log_softmax(model, data, task_emb, platform_emb)
    loss = -torch.logsumexp(b.step_logps(lp, "prefix").sum(dim=1), dim=0)
    if full_context_weight > 0.0:
        loss = loss + full_context_weight * -torch.logsumexp(b.step_logps(lp, "full").sum(dim=1), dim=0)
    return loss
