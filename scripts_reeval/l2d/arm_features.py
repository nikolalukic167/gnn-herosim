"""literature_reeval_v1 -- the four arms, expressed purely as *inputs* to the unmodified L2D policy.

Upstream ``ActorCritic`` = GIN (K=2 sum-aggregation layers over the disjunctive graph) -> mean
readout -> actor MLP over [h_candidate, h_graph]. Its neighbourhood is *typed and bounded*: the env
gives each operation at most three in-neighbours -- itself, its job predecessor, and its machine
predecessor (``JSSP_Env.step``: ``adj[action, action-1]``, ``adj[action, precd]``, ``adj[succd, action]``).
Two hops of that is a fixed 7-slot tuple, so "everything message passing can see" is a finite
hand-built vector. That is the T1 contract this program used on HeROsim, applied here.

Arms (only ``adj`` and ``fea`` differ; architecture, PPO, seeds, budget are identical):

  gnn      -- upstream exactly: real adjacency, raw 2-dim node features (LB/1000, finished).
  mpoff    -- identity adjacency, raw features. Per-node MLP + the same mean readout (DeepSets);
              the analogue of this program's MP-OFF arm.
  mlp_t1   -- identity adjacency; features = the typed 2-hop receptive field the GIN aggregates
              over (7 slots x 2 raw feats + 6 presence flags = 20 dims). Same information as `gnn`,
              no learned message passing.
  mlp_t1x  -- mlp_t1 + 7 hand scalars a scheduler would compute (duration, work remaining, ops
              remaining, machine remaining work, machine ready time, earliest start, global LB max).
              The analogue of the hetdem/krank count columns.

The global mean readout is kept in every arm on purpose: the ablated quantity is *neighbourhood*
message passing, not permutation-invariant global context.
"""
from __future__ import annotations

import numpy as np
import torch

ARMS = ("gnn", "mpoff", "mlp_t1", "mlp_t1x")
RAW_DIM = 2
T1_DIM = 20
T1X_DIM = T1_DIM + 7

INPUT_DIM = {"gnn": RAW_DIM, "mpoff": RAW_DIM, "mlp_t1": T1_DIM, "mlp_t1x": T1X_DIM}
USES_REAL_ADJ = {"gnn": True, "mpoff": False, "mlp_t1": False, "mlp_t1x": False}

_ET_NORM = 1000.0  # upstream configs.et_normalize_coef
_DUR_HIGH = 99.0   # upstream configs.high


def identity_adj(n: int, device) -> torch.Tensor:
    idx = torch.arange(n, device=device)
    return torch.sparse_coo_tensor(
        torch.stack((idx, idx)), torch.ones(n, dtype=torch.float32, device=device), (n, n)
    ).coalesce()


def _predecessors(adj: np.ndarray, n_j: int, n_m: int):
    """(job_pred, mach_pred) index arrays, -1 where absent. Fails loudly if the adjacency ever
    stops being 'self + job pred + <=1 machine pred' -- the T1 tuple would then be wrong."""
    n = n_j * n_m
    ids = np.arange(n)
    col = ids % n_m
    job_pred = np.where(col > 0, ids - 1, -1)
    a = (adj > 0)
    rest = a.copy()
    rest[ids, ids] = False
    rest[ids[col > 0], job_pred[col > 0]] = False
    cnt = rest.sum(axis=1)
    if cnt.max() > 1:
        bad = int(np.argmax(cnt))
        raise AssertionError(
            f"operation {bad} has {cnt[bad]} in-neighbours beyond self/job-pred; "
            f"T1 receptive-field tuple assumes <=1 machine predecessor"
        )
    mach_pred = np.where(cnt == 1, rest.argmax(axis=1), -1)
    return job_pred, mach_pred


def typed_receptive_field(adj: np.ndarray, fea: np.ndarray, n_j: int, n_m: int) -> np.ndarray:
    """20-dim per-op vector: raw feats of [self, jp, mp, jp.jp, jp.mp, mp.jp, mp.mp] + 6 presence flags."""
    jp, mp = _predecessors(adj, n_j, n_m)

    def hop(idx_from: np.ndarray, table: np.ndarray) -> np.ndarray:
        out = np.full_like(idx_from, -1)
        ok = idx_from >= 0
        out[ok] = table[idx_from[ok]]
        return out

    slots = [np.arange(n_j * n_m), jp, mp, hop(jp, jp), hop(jp, mp), hop(mp, jp), hop(mp, mp)]
    parts = []
    flags = []
    for k, s in enumerate(slots):
        present = s >= 0
        f = np.zeros((s.shape[0], fea.shape[1]), dtype=np.float32)
        f[present] = fea[s[present]]
        parts.append(f)
        if k > 0:
            flags.append(present.astype(np.float32)[:, None])
    return np.concatenate(parts + flags, axis=1)


def domain_scalars(env) -> np.ndarray:
    """7 hand scalars per op from public env state (dur, finished_mark, temp1, m, LBs)."""
    n_j, n_m = env.number_of_jobs, env.number_of_machines
    dur = env.dur.astype(np.float32)
    unfinished = 1.0 - env.finished_mark.astype(np.float32)
    rem = dur * unfinished
    wkr = np.cumsum(rem[:, ::-1], axis=1)[:, ::-1]                         # work remaining from op c onward
    ops_rem = np.tile((n_m - np.arange(n_m)) / n_m, (n_j, 1)).astype(np.float32)
    mach = env.m.astype(np.int64) - 1                                       # machine id per op, 0-based
    mach_rem = np.zeros(n_m, dtype=np.float32)
    np.add.at(mach_rem, mach.ravel(), rem.ravel())
    end = env.temp1.astype(np.float32)                                      # end time of scheduled ops, 0 else
    mach_ready = np.zeros(n_m, dtype=np.float32)
    np.maximum.at(mach_ready, mach.ravel(), end.ravel())
    job_ready = np.concatenate([np.zeros((n_j, 1), np.float32), end[:, :-1]], axis=1)
    est_start = np.maximum(job_ready, mach_ready[mach])
    lb_max = np.full((n_j, n_m), float(env.LBs.max()), dtype=np.float32)
    cols = [dur / _DUR_HIGH, wkr / (_DUR_HIGH * n_m), ops_rem, mach_rem[mach] / (_DUR_HIGH * n_j),
            mach_ready[mach] / _ET_NORM, est_start / _ET_NORM, lb_max / _ET_NORM]
    return np.stack([c.reshape(-1) for c in cols], axis=1).astype(np.float32)


def build_inputs(arm: str, env, adj: np.ndarray, fea: np.ndarray, device):
    """Return (adj_tensor_sparse, fea_tensor) for one env state under the given arm."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; choose from {ARMS}")
    n = adj.shape[0]
    if arm == "gnn":
        adj_t = torch.from_numpy(np.copy(adj)).to(device).to_sparse()
        fea_np = fea
    else:
        adj_t = identity_adj(n, device)
        if arm == "mpoff":
            fea_np = fea
        else:
            fea_np = typed_receptive_field(adj, fea, env.number_of_jobs, env.number_of_machines)
            if arm == "mlp_t1x":
                fea_np = np.concatenate([fea_np, domain_scalars(env)], axis=1)
    fea_t = torch.from_numpy(np.ascontiguousarray(fea_np, dtype=np.float32)).to(device)
    if fea_t.shape[1] != INPUT_DIM[arm]:
        raise AssertionError(f"{arm}: built {fea_t.shape[1]} dims, contract says {INPUT_DIM[arm]}")
    return adj_t, fea_t
