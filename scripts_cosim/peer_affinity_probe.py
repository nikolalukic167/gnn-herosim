#!/usr/bin/env python3
"""peer_affinity_v1 -- Phase 0 paper screen (registered 2026-09-09). NO SIMULATION IS RUN.

Question: does a cost indexed by PAIRS of task instances -- an instance-specific,
continuous exchange volume x_ij charged as transfer(node(i), node(j)) at task i's input
stage, with no commit order and a binding per-node capacity -- give a k-task batch joint
structure that (a) per-machine count columns do not repair, (b) a sequential greedy with an
exact prefix cannot recover, and (c) a hand-built peer-mass lookahead cannot recover either?
Only (a) AND (b) AND (c) leave anything for a task<->task graph to compute
(docs/lineages/peer_affinity_v1.md, "Registered bars").

Everything is built from stored arm_b0 datasets (data-locality OFF, the additive control
arm of route_b_pilot_v1):

  base       c_i(p) = the stored duration of the real task the paper task copies, on
             placement p, at that placement's EARLIEST dispatch time in the sweep (a child's
             duration varies with dispatch time only -- Amendment A1; a parallel batch is
             dispatched at t = 0). FAILS LOUD if it varies at a fixed dispatch time.
  sharing    q_p * cnt_p (cnt_p - 1) / 2 for tasks sharing a platform (count-shaped by
             design -- the --allow-non-unique-replicas physics the simulated screen will
             carry, so the count competitor has something real to repair).
  exchange   sum_{i<j} x_ij * X(node(pi_i), node(pi_j)), X = hops * bytes / (bottleneck *
             1024^2) + network_maps latency, exactly _payload_transfer_time +
             _dependency_transfer_time's charge; 0 when co-located.
  capacity   cap_n = alpha * max demand any candidate places on n (score_route_b_contention
             .Dataset.node_caps, alpha_max); demand_i(p) = U(0.5,2) * memoryRequirements.

Every plan of the n_cand^k product is enumerated in numpy; bars are read on the
alpha-feasible subset; fits use the full product as fit rows (the scorer's convention).

Usage:
  peer_affinity_probe.py --out simulation_data/peer_affinity_probe.json
  peer_affinity_probe.py --cells 'k8c3' --limit-sources 3 --out /tmp/smoke.json
  peer_affinity_probe.py --emit-dir simulation_data/peer_affinity_paper_k8 --stages screen emit crosscheck ...
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts_cosim"))
import score_route_b_contention as S  # noqa: E402

CORPUS_DEFAULT = ROOT / "simulation_data/gnn_datasets_dag4_route_b_pilot_v1_arm_b0"
TT = S.load_task_types(ROOT / "data/nofs-ids/task-types.json")
MB = 1e6
X_SCALES_MB = [50.0, 200.0, 800.0]
ALPHAS = [1.5, 2.0]  # 4-task-equivalent tightness; the cap uses alpha_k = alpha * k / 4 (Amendment A2)
SHAPES = [(8, 3), (8, 4), (10, 3), (10, 4), (12, 3)]
PARTNERS = [2, 3]
DEMAND_SPREAD = (0.5, 2.0)
X_LOG10_SPREAD = 1.0
EPS = 1e-9
SOURCE_STRIDE = 6
SOURCE_COUNT = 34
MIN_SCORED = 17  # a cell with fewer scored datasets is UNREADABLE, never a pass (Amendment A2)
CHUNK = 100_000
FIT_GUARD = 2  # rows >= FIT_GUARD * n_params or the fit is refused (docs/lessons.md 2026-08-27)
COUNT_COMPETITOR = "v1"  # "v1" = the registered paper-screen block; "v2" adds per-(platform, type) counts + squares

# Registered bars -- docs/lineages/peer_affinity_v1.md. Keep in sync with the node.
BARS = {
    "S0a_r2": 0.999, "S0a_regret_gt1_frac": 0.02,
    "S0b_r2": 0.999, "S0b_regret_gt1_frac": 0.02,
    "B0_cap_binds_frac": 0.50,
    "B1_median_pct": 5.0,
    "B2_median_repair": 0.5,
    "B3_median_pct": 5.0,
    "B4_median_closure": 0.5,
    "B5_median_pct": 2.0,
    "C1_median_pct": 5.0,
}


def regret_pct(v: float, opt: float) -> float:
    return 100.0 * (v - opt) / opt


def median(xs: Sequence[float]) -> float:
    xs = sorted(x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x)))
    n = len(xs)
    if n == 0:
        return float("nan")
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def cell_name(x_mb: float, alpha: float, k: int, n_cand: int, partners: int) -> str:
    return f"x{int(x_mb)}_a{alpha}_k{k}c{n_cand}_p{partners}"


# ----------------------------------------------------------------------------- source corpus
def load_rows_with_times(ds_dir: Path):
    rows = []
    with open(ds_dir / "placements/placements.jsonl") as fh:
        for i, line in enumerate(fh):
            r = json.loads(line)
            tt = r.get("task_times")
            if not tt:
                raise RuntimeError(f"{ds_dir}: row {i} has no task_times -- the source sweep must "
                                   "have run with HEROSIM_RETAIN_TASK_TIMES=1")
            plan = {int(k): (int(v[0]), int(v[1])) for k, v in r["placement_plan"].items()}
            rows.append((plan, float(r["rtt"]),
                         {int(a): (float(c) - float(b), float(b)) for a, b, c in tt}))
    return rows


class Source:
    """One stored arm_b0 dataset: per-(task, placement) base cost, candidate sets, and the
    server-node exchange matrices, all taken from the files the simulator wrote."""

    def __init__(self, ds_dir: Path):
        self.ds_dir = ds_dir
        self.ds = S.Dataset(ds_dir, TT, "rtt")
        rows = load_rows_with_times(ds_dir)
        # Base cost rule (Amendment A1 in the node, 2026-09-09): a child's stored duration
        # varies with (a) its DISPATCH time (the platform's warm queue drains while the parent
        # runs) and (b) a sibling on the SAME NODE at the same time (node-level storage
        # serialisation -- node-indexed, count-shaped, the one-integer channel of the record).
        # A parallel batch is dispatched at t = 0 and the sharing term below carries (b), so
        # the paper cost of (task, placement) is the UNCONTENDED duration at that placement's
        # EARLIEST dispatch time in the sweep (the minimum over those rows). Guard: a root task
        # (dispatched at t = 0, alone) must not vary at all.
        acc: Dict[Tuple[int, Tuple[int, int]], List[Tuple[float, float]]] = {}
        max_sum_gap = 0.0
        for plan, rtt, dur in rows:
            max_sum_gap = max(max_sum_gap, abs(sum(d for d, _ in dur.values()) - rtt) / max(rtt, 1e-12))
            for t, p in plan.items():
                acc.setdefault((t, p), []).append(dur[t])
        if max_sum_gap > 1e-6:
            raise RuntimeError(f"{ds_dir}: rtt is not the sum of task durations (rel gap "
                               f"{max_sum_gap:.3e}) -- the paper base cost would not be the label")
        self.cost: Dict[Tuple[int, Tuple[int, int]], float] = {}
        self.max_spread = 0.0          # residual spread at the earliest dispatch time (reported)
        self.max_spread_any = 0.0      # spread over ALL dispatch times (the DAG timing effect)
        roots = {t for t in range(len(self.ds.task_type_names)) if not any(c == t for _p, c in self.ds.dag_edges)}
        for key, vals in acc.items():
            d_all = [d for d, _ in vals]
            self.max_spread_any = max(self.max_spread_any, (max(d_all) - min(d_all)) / max(sum(d_all) / len(d_all), 1e-12))
            d0 = min(disp for _, disp in vals)
            sel = [d for d, disp in vals if abs(disp - d0) <= 1e-9]
            m = min(sel)  # the UNCONTENDED duration: a sibling on the same node only ever adds time
            spread = (max(sel) - m) / max(m, 1e-12)
            self.max_spread = max(self.max_spread, spread)
            if key[0] in roots and spread > 1e-6:
                raise RuntimeError(f"{ds_dir}: ROOT task {key[0]} on {key[1]} varies at dispatch t=0 "
                                   f"(rel spread {spread:.3e}) -- nothing can explain that; refusing")
            self.cost[key] = m
        self.task_ids = sorted({t for t, _p in acc})
        self.cands = {t: sorted(p for (tt, p) in acc if tt == t) for t in self.task_ids}
        self.nodes = sorted({self.ds.node_of(p) for (_t, p) in acc})
        self.node_index = {n: i for i, n in enumerate(self.nodes)}
        n = len(self.nodes)
        self.per_byte = np.zeros((n, n))
        self.latency = np.zeros((n, n))
        self.max_asymmetry = 0.0
        for a in range(n):
            for b in range(n):
                if a == b:
                    continue
                hops, bneck, lat = self.ds.route_metrics(self.nodes[a], self.nodes[b])
                self.per_byte[a, b] = hops / (bneck * 1024 * 1024)
                self.latency[a, b] = lat
        for a in range(n):
            for b in range(a + 1, n):
                for M in (self.per_byte, self.latency):
                    self.max_asymmetry = max(self.max_asymmetry, abs(M[a, b] - M[b, a]) / max(M[a, b], M[b, a], 1e-12))
                    M[a, b] = M[b, a] = 0.5 * (M[a, b] + M[b, a])

    def platform_type(self, p: Tuple[int, int]) -> str:
        return self.ds.platform_map[p[1]][1]


# ------------------------------------------------------------------------------ paper dataset
class Paper:
    """k paper tasks resampled from a Source, with per-instance demand and a peer matrix.
    The peer draw is stored as a multiplier of x_scale, so one draw serves every x_scale cell."""

    def __init__(self, src: Source, k: int, n_cand: int, partners: int, seed: int):
        self.src, self.k, self.n_cand, self.partners, self.seed = src, k, n_cand, partners, seed
        rng = random.Random(seed)
        eligible = [t for t in src.task_ids if len(src.cands[t]) >= n_cand]
        if not eligible:
            raise RuntimeError(f"{src.ds_dir}: no task has >= {n_cand} candidates")
        # Candidates are drawn ONCE per real task (Amendment A2): every paper copy of a real
        # task shares its candidate set, so co-residents of one type are interchangeable
        # up to their per-instance demand -- the count-shaped base the registration names.
        cand_of: Dict[int, List[Tuple[int, int]]] = {}
        for t in eligible:
            pool = list(src.cands[t])
            rng.shuffle(pool)
            chosen, seen = [], set()
            for p in pool:  # distinct nodes first, so the exchange matrix has something to vary
                nd = src.ds.node_of(p)
                if nd not in seen and len(chosen) < n_cand:
                    chosen.append(p); seen.add(nd)
            for p in pool:
                if len(chosen) < n_cand and p not in chosen:
                    chosen.append(p)
            cand_of[t] = chosen
        self.real: List[int] = []
        self.cands: List[List[Tuple[int, int]]] = []
        self.scale: List[float] = []
        for _i in range(k):
            t = rng.choice(eligible)
            self.real.append(t); self.cands.append(list(cand_of[t]))
            self.scale.append(rng.uniform(*DEMAND_SPREAD))
        pairs: Dict[Tuple[int, int], float] = {}
        for i in range(k):
            for j in rng.sample([j for j in range(k) if j != i], partners):
                key = (min(i, j), max(i, j))
                if key not in pairs:
                    pairs[key] = 10.0 ** rng.uniform(-X_LOG10_SPREAD, X_LOG10_SPREAD)
        self.pairs = sorted(pairs.items())
        # dense arrays (k x n_cand)
        self.types = [src.ds.task_type_names[t] for t in self.real]
        self.type_vocab = sorted(set(self.types))
        self.type_idx = np.array([self.type_vocab.index(tp) for tp in self.types])
        self.sources = [src.ds.task_sources[t] for t in self.real]
        self.plats = sorted({p for cs in self.cands for p in cs})
        self.plat_index = {p: i for i, p in enumerate(self.plats)}
        self.cost = np.array([[src.cost[(t, p)] for p in cs] for t, cs in zip(self.real, self.cands)])
        self.node = np.array([[src.node_index[src.ds.node_of(p)] for p in cs] for cs in self.cands])
        self.plat = np.array([[self.plat_index[p] for p in cs] for cs in self.cands])
        self.demand = np.array([[s * float(TT[tp]["memoryRequirements"][src.platform_type(p)]) for p in cs]
                                for s, tp, cs in zip(self.scale, self.types, self.cands)])
        for tp, cs in zip(self.types, self.cands):
            for p in cs:
                if src.platform_type(p) not in TT[tp]["memoryRequirements"]:
                    raise RuntimeError(f"no memoryRequirements[{tp}][{src.platform_type(p)}]")
        # sharing coefficient per platform: mean base cost of the tasks that list it
        q = {}
        for i in range(k):
            for c, p in enumerate(self.cands[i]):
                q.setdefault(p, []).append(self.cost[i, c])
        self.q = np.array([sum(q[p]) / len(q[p]) for p in self.plats])
        self.n_nodes = len(src.nodes)
        self.n_plat = len(self.plats)
        self.n_types = len(self.type_vocab)
        # peer-mass lookahead per (i, c): sum_j x_ij * mean_{c'} X(node(i,c), node(j,c'))
        pb, pl = self.src.per_byte, self.src.latency
        self.pm_b = np.zeros((k, n_cand)); self.pm_l = np.zeros((k, n_cand))
        for (i, j), mult in self.pairs:
            for a, b in ((i, j), (j, i)):
                self.pm_b[a] += mult * pb[self.node[a][:, None], self.node[b][None, :]].mean(1)
                self.pm_l[a] += pl[self.node[a][:, None], self.node[b][None, :]].mean(1)

    # --- enumeration -------------------------------------------------------------------
    def enumerate(self) -> None:
        k, m = self.k, self.n_cand
        N = m ** k
        idx = np.arange(N, dtype=np.int64)
        P = np.empty((N, k), dtype=np.int8)
        for i in range(k):
            P[:, i] = (idx // (m ** (k - 1 - i))) % m
        self.P = P
        ar = np.arange(k)
        self.NODE = self.node[ar, P].astype(np.int8)
        self.PLAT = self.plat[ar, P].astype(np.int16)
        DEM = self.demand[ar, P]
        self.BASE = self.cost[ar, P].sum(1)
        self.cnt = np.zeros((N, self.n_plat), dtype=np.int8)
        for p in range(self.n_plat):
            self.cnt[:, p] = (self.PLAT == p).sum(1)
        c = self.cnt.astype(np.float64)
        self.SHARE = (c * (c - 1) / 2) @ self.q
        self.load = np.zeros((N, self.n_nodes))
        self.nodecnt = np.zeros((N, self.n_nodes), dtype=np.int8)
        self.min_single = np.full((N, self.n_nodes), np.inf)
        for n in range(self.n_nodes):
            mask = self.NODE == n
            self.load[:, n] = (DEM * mask).sum(1)
            self.nodecnt[:, n] = mask.sum(1)
            self.min_single[:, n] = np.where(mask, DEM, np.inf).min(1)
        self.max_demand_on_node = np.array([self.demand[self.node == n].max() if (self.node == n).any() else 0.0
                                            for n in range(self.n_nodes)])
        self.QB = np.zeros(N); self.QL = np.zeros(N)
        for (i, j), mult in self.pairs:
            self.QB += mult * self.src.per_byte[self.NODE[:, i], self.NODE[:, j]]
            self.QL += self.src.latency[self.NODE[:, i], self.NODE[:, j]]
        # count-oracle stratum key: sorted multiset of (platform, type) ids
        self.KEY = np.sort(self.PLAT.astype(np.int32) * self.n_types + self.type_idx[None, :], axis=1)
        self.collision_free = (self.cnt.max(1) == 1)
        self.N = N

    def plan_index(self, plan: Sequence[int]) -> int:
        idx = 0
        for c in plan:
            idx = idx * self.n_cand + int(c)
        return idx

    def caps(self, alpha: float) -> np.ndarray:
        # absent (all-zero) nodes are uncapped -- the scorer's convention
        return np.where(self.max_demand_on_node > 0, alpha * self.max_demand_on_node, np.inf)

    def total(self, x_bytes: float) -> np.ndarray:
        if x_bytes == 0.0:
            return self.BASE + self.SHARE
        return self.BASE + self.SHARE + x_bytes * self.QB + self.QL

    def exchange(self, x_bytes: float) -> np.ndarray:
        return x_bytes * self.QB + self.QL

    # --- design columns ---------------------------------------------------------------
    def indicator_cols(self, sel: np.ndarray) -> np.ndarray:
        offsets = getattr(self, "cand_offsets", None)
        if offsets is None:
            offsets = [i * self.n_cand for i in range(self.k)]
        width = offsets[-1] + (len(self.cands[-1]) if hasattr(self, "cands") else self.n_cand)
        X = np.zeros((len(sel), width), dtype=np.float64)
        rows = np.arange(len(sel))
        for i in range(self.k):
            X[rows, offsets[i] + self.P[sel, i]] = 1.0
        return X

    def count_cols(self, sel: np.ndarray, cap: np.ndarray) -> np.ndarray:
        cnt = self.cnt[sel].astype(np.float64)
        nodecnt = self.nodecnt[sel].astype(np.float64)
        load = self.load[sel]
        cols = [np.maximum(nodecnt - 1, 0).sum(1, keepdims=True)]                         # 1int
        kint = np.zeros((len(sel), self.n_nodes * self.n_types))
        for i in range(self.k):
            np.add.at(kint, (np.arange(len(sel)), self.NODE[sel, i].astype(int) * self.n_types + self.type_idx[i]), 1.0)
        cols.append(kint)                                                                 # kint
        cols.append(cnt * (cnt - 1) / 2)                                                  # per-platform pairs
        hd_quad = np.zeros((len(sel), self.n_types))                                      # hetdem
        for i in range(self.k):
            hd_quad[:, self.type_idx[i]] += load[np.arange(len(sel)), self.NODE[sel, i].astype(int)]
        cols.append(hd_quad)
        with np.errstate(divide="ignore", invalid="ignore"):
            cols.append(np.where(np.isfinite(cap), load * load / cap, 0.0).sum(1, keepdims=True))
        cols.append((load * (load > cap + EPS)).sum(1, keepdims=True))
        ms = np.where(np.isfinite(self.min_single[sel]), self.min_single[sel], 0.0)
        cols.append(np.where(nodecnt >= 2, load - ms, 0.0).sum(1, keepdims=True))
        cols.append((load * load).sum(1, keepdims=True))
        if COUNT_COMPETITOR == "v2":
            # Count competitor v2 (2026-09-10, simulated screens): per-(platform, type) counts,
            # their squares and the platform occupancy square -- the sufficient statistics of
            # any symmetric function of a platform's co-resident multiset up to second order.
            # The simulator's serialisation on a shared platform is such a function; v1's
            # node x type + platform-pair columns left the CONTROL arm at R^2 0.9989.
            pt = np.zeros((len(sel), self.n_plat * self.n_types))
            for i in range(self.k):
                np.add.at(pt, (np.arange(len(sel)), self.PLAT[sel, i].astype(int) * self.n_types + self.type_idx[i]), 1.0)
            cols.append(pt)
            cols.append(pt * pt)
            cols.append(cnt * cnt)
        return np.concatenate(cols, axis=1)

    def peer_mass_col(self, sel: np.ndarray, x_bytes: float) -> np.ndarray:
        pm = x_bytes * self.pm_b + self.pm_l
        return pm[np.arange(self.k), self.P[sel]].sum(1, keepdims=True)

    def n_count_cols(self) -> int:
        base = 1 + self.n_nodes * self.n_types + self.n_plat + self.n_types + 4
        return base + (2 * self.n_plat * self.n_types + self.n_plat if COUNT_COMPETITOR == "v2" else 0)


# ------------------------------------------------------------------------------------ fitting
TIE_REL = 1e-9


def fit_argmin(paper: Paper, y: np.ndarray, fit_rows: np.ndarray, read_rows: np.ndarray,
               blocks: Sequence[str], cap: Optional[np.ndarray] = None, x_bytes: float = 0.0):
    """Chunked least squares of y on the requested column blocks over fit_rows.

    Returns (tie_positions into read_rows, R^2 on fit_rows, n_params) or None when the 2x
    guard refuses. `tie_positions` is every read row whose prediction is within TIE_REL of
    the minimum: an indicator fit scores plans that swap interchangeable copies IDENTICALLY,
    so a single argmin is a tie-break artifact (score_route_b_contention's r_exact_band
    lesson, 2026-08-27). Callers read the MEAN true cost over the tie set (the fair reading)
    and may report the optimistic / pessimistic ends."""
    def cols(sel):
        parts = [np.ones((len(sel), 1)), paper.indicator_cols(sel)]
        if "count" in blocks:
            parts.append(paper.count_cols(sel, cap))
        if "peer" in blocks:
            parts.append(paper.peer_mass_col(sel, x_bytes))
        return np.concatenate(parts, axis=1)
    n_params = cols(fit_rows[:1]).shape[1]
    if len(fit_rows) < FIT_GUARD * n_params:
        return None
    G = np.zeros((n_params, n_params)); b = np.zeros(n_params)
    for s in range(0, len(fit_rows), CHUNK):
        sel = fit_rows[s:s + CHUNK]
        X = cols(sel); yy = y[sel]
        G += X.T @ X; b += X.T @ yy
    beta = np.linalg.lstsq(G, b, rcond=None)[0]
    ss_res = 0.0
    yf = y[fit_rows]; ss_tot = float(((yf - yf.mean()) ** 2).sum())
    for s in range(0, len(fit_rows), CHUNK):
        sel = fit_rows[s:s + CHUNK]
        r = y[sel] - cols(sel) @ beta
        ss_res += float(r @ r)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-18)
    preds = np.concatenate([cols(read_rows[s:s + CHUNK]) @ beta for s in range(0, len(read_rows), CHUNK)])
    pmin = float(preds.min())
    ties = np.nonzero(preds <= pmin + TIE_REL * max(abs(pmin), 1.0))[0]
    return ties, float(r2), int(n_params)


def tie_regret(y_read: np.ndarray, ties: np.ndarray, opt: float) -> Dict[str, float]:
    vals = y_read[ties]
    return {"mean_tied": regret_pct(float(vals.mean()), opt), "optimistic": regret_pct(float(vals.min()), opt),
            "pessimistic": regret_pct(float(vals.max()), opt), "n_tied": int(len(ties))}


# ------------------------------------------------------------------------------------- greedy
def greedy(paper: Paper, x_bytes: float, cap: np.ndarray, order: Sequence[int], lookahead: bool
           ) -> Optional[List[int]]:
    """Sequential greedy on the TRUE marginals with an exact prefix. Committed peers' exchange
    is charged exactly; uncommitted peers contribute nothing (B3) or their peer-mass
    expectation (B5). Capacity-masked; returns None when no feasible candidate remains."""
    k = paper.k
    plan = [-1] * k
    load = np.zeros(paper.n_nodes); pcnt = np.zeros(paper.n_plat)
    partners = {i: [] for i in range(k)}
    for (i, j), mult in paper.pairs:
        partners[i].append((j, mult)); partners[j].append((i, mult))
    pm = x_bytes * paper.pm_b + paper.pm_l
    for i in order:
        best, best_c = np.inf, -1
        for c in range(paper.n_cand):
            n = int(paper.node[i, c]); p = int(paper.plat[i, c])
            if load[n] + paper.demand[i, c] > cap[n] + EPS:
                continue
            m = paper.cost[i, c] + paper.q[p] * pcnt[p]
            for j, mult in partners[i]:
                if plan[j] >= 0:
                    nj = int(paper.node[j, plan[j]])
                    m += x_bytes * mult * paper.src.per_byte[n, nj] + paper.src.latency[n, nj]
            if lookahead:
                # pm already sums over ALL partners; subtract the committed ones' share
                la = pm[i, c]
                for j, mult in partners[i]:
                    if plan[j] >= 0:
                        la -= (x_bytes * mult * paper.src.per_byte[n, paper.node[j]].mean()
                               + paper.src.latency[n, paper.node[j]].mean())
                m += la
            if m < best - 1e-12 or (abs(m - best) <= 1e-12 and (n, p) < (int(paper.node[i, best_c]), int(paper.plat[i, best_c]))):
                best, best_c = m, c
        if best_c < 0:
            return None
        plan[i] = best_c
        load[int(paper.node[i, best_c])] += paper.demand[i, best_c]
        pcnt[int(paper.plat[i, best_c])] += 1
    return plan


# ------------------------------------------------------------------------------------ screen
def regret_pct(v: float, opt: float) -> float:
    return 100.0 * (v - opt) / opt


def alpha_k(alpha4: float, k: int) -> float:
    """Equal-tightness cap for k tasks: alpha_k = alpha4 * k / 4 (route_b s9d / dag_fabric_contention_v1
    used alpha8 = 4.0 for two 4-task instances at alpha4 = 2.0). Calibrated 2026-09-09 on feasibility
    counts only: at the unscaled alpha4 half or more of the 34 sources have NO feasible plan at k >= 8."""
    return alpha4 * k / 4.0


def screen_dataset(paper: Paper, x_mb: float, alpha4: float, rng_orders: random.Random) -> dict:
    x = x_mb * MB
    alpha = alpha_k(alpha4, paper.k)
    cap = paper.caps(alpha)
    y = paper.total(x)
    y0 = paper.total(0.0)
    feas_mask = (paper.load <= cap[None, :] + EPS).all(1)
    feas = np.nonzero(feas_mask)[0]
    all_rows = np.arange(paper.N)
    out: dict = {"ds": paper.src.ds_dir.name, "seed": paper.seed, "n_rows": int(paper.N), "alpha_k": alpha,
                 "n_feasible": int(len(feas)), "n_nodes": paper.n_nodes, "n_plat": paper.n_plat,
                 "n_pairs": len(paper.pairs)}
    if len(feas) == 0:
        out["no_feasible_rows"] = True
        return out
    opt = int(feas[np.argmin(y[feas])]); opt_val = float(y[opt])
    if opt_val <= 0:
        raise RuntimeError(f"{paper.src.ds_dir}: non-positive optimum {opt_val}")
    out["opt_val"] = opt_val
    out["qap_share_at_opt"] = float(paper.exchange(x)[opt] / opt_val)
    # B0 -- does the cap bind?
    out["B0_cap_binds"] = bool(not feas_mask[int(np.argmin(y))])
    # S0a -- collision-free feasible plans, x == 0, pure additive
    cf = np.nonzero(feas_mask & paper.collision_free)[0]
    out["n_collision_free"] = int(len(cf))
    fit = fit_argmin(paper, y0, cf, cf, ("ind",)) if len(cf) else None
    if fit is not None:
        o0 = float(y0[cf].min())
        out["S0a_r2"] = fit[1]; out["S0a_regret_pct"] = tie_regret(y0[cf], fit[0], o0)["mean_tied"]
    # S0b -- full feasible set, x == 0, additive + counts
    fit = fit_argmin(paper, y0, all_rows, feas, ("ind", "count"), cap)
    if fit is not None:
        o0 = float(y0[feas].min())
        out["S0b_r2"] = fit[1]; out["S0b_regret_pct"] = tie_regret(y0[feas], fit[0], o0)["mean_tied"]
        out["S0b_n_params"] = fit[2]
    # B1 -- pointwise-only fit on the real target
    fit = fit_argmin(paper, y, all_rows, feas, ("ind",))
    if fit is None:
        out["fit_refused"] = True
        return out
    band = tie_regret(y[feas], fit[0], opt_val)
    out["B1_regret_pct"] = band["mean_tied"]; out["B1_band"] = band; out["additive_r2"] = fit[1]
    # B2 -- + count columns
    fit = fit_argmin(paper, y, all_rows, feas, ("ind", "count"), cap)
    band = tie_regret(y[feas], fit[0], opt_val)
    out["B2_regret_pct"] = band["mean_tied"]; out["B2_band"] = band; out["count_r2"] = fit[1]
    out["B2_n_params"] = fit[2]
    out["B2_repair"] = (max(0.0, (out["B1_regret_pct"] - out["B2_regret_pct"]) / out["B1_regret_pct"])
                        if out["B1_regret_pct"] > 0 else None)
    # B4 -- + peer-mass lookahead column
    fit = fit_argmin(paper, y, all_rows, feas, ("ind", "count", "peer"), cap, x)
    band = tie_regret(y[feas], fit[0], opt_val)
    out["B4_regret_pct"] = band["mean_tied"]; out["B4_band"] = band; out["peer_r2"] = fit[1]
    out["B4_closure"] = (max(0.0, (out["B1_regret_pct"] - out["B4_regret_pct"]) / out["B1_regret_pct"])
                         if out["B1_regret_pct"] > 0 else None)
    # B3 / B5 -- greedy on true marginals with exact prefix, id order (registered) + others
    id_order = list(range(paper.k))
    strength = {i: 0.0 for i in range(paper.k)}
    for (i, j), mult in paper.pairs:
        strength[i] += mult; strength[j] += mult
    big_first = sorted(range(paper.k), key=lambda i: -strength[i])
    rand_orders = [rng_orders.sample(range(paper.k), paper.k) for _ in range(8)]
    for tag, look in (("B3", False), ("B5", True)):
        plan = greedy(paper, x, cap, id_order, look)
        out[f"{tag}_stuck"] = plan is None
        if plan is not None:
            out[f"{tag}_regret_pct"] = regret_pct(float(y[paper.plan_index(plan)]), opt_val)
        # ordering split (reported, no bar): the deterministic hand rule "largest exchange
        # mass first" is a decoder choice any arm could make; best-of-8 random orders is an
        # ORACLE over orderings (needs the true plan cost to pick) -- a search, not a scorer.
        pb = greedy(paper, x, cap, big_first, look)
        out[f"{tag}_bigfirst_regret_pct"] = None if pb is None else regret_pct(float(y[paper.plan_index(pb)]), opt_val)
        rr = []
        for order in rand_orders:
            p2 = greedy(paper, x, cap, order, look)
            if p2 is not None:
                rr.append(regret_pct(float(y[paper.plan_index(p2)]), opt_val))
        out[f"{tag}_best_random8_regret_pct"] = min(rr) if rr else None
        alts = rr + [v for v in (out[f"{tag}_bigfirst_regret_pct"], out.get(f"{tag}_regret_pct")) if v is not None]
        out[f"{tag}_best_order_regret_pct"] = min(alts) if alts else None
    # C1 -- count oracle: random feasible plan inside the optimum's (platform, type) stratum
    same = feas[(paper.KEY[feas] == paper.KEY[opt]).all(1)]
    out["C1_stratum_size"] = int(len(same))
    out["C1_regret_pct"] = regret_pct(float(y[same].mean()), opt_val)
    out["C1_max_regret_pct"] = regret_pct(float(y[same].max()), opt_val)
    out["spread"] = "VOID (k > n_server_nodes)" if paper.k > paper.n_nodes else "n/a"
    return out


def aggregate(cell: str, per: List[dict]) -> dict:
    ok = [d for d in per if "B1_regret_pct" in d]
    n = len(ok)
    agg = {"cell": cell, "n_datasets": len(per), "n_scored": n,
           "n_no_feasible": sum(d.get("no_feasible_rows", False) for d in per),
           "n_fit_refused": sum(d.get("fit_refused", False) for d in per)}
    agg["readable"] = n >= MIN_SCORED
    if n == 0:
        agg["pass"] = False
        return agg
    agg["alpha_k"] = ok[0]["alpha_k"]
    frac = lambda key, thr: sum(d.get(key, float("nan")) > thr for d in ok if d.get(key) is not None) / n  # noqa: E731
    agg.update({
        "S0a_r2_median": median([d.get("S0a_r2") for d in ok]),
        "S0a_regret_gt1_frac": frac("S0a_regret_pct", 1.0),
        "S0b_r2_median": median([d.get("S0b_r2") for d in ok]),
        "S0b_regret_gt1_frac": frac("S0b_regret_pct", 1.0),
        "B0_cap_binds_frac": sum(d["B0_cap_binds"] for d in ok) / n,
        "B1_median_pct": median([d["B1_regret_pct"] for d in ok]),
        "B1_gt5_frac": frac("B1_regret_pct", 5.0),
        "B2_median_pct": median([d["B2_regret_pct"] for d in ok]),
        "B2_median_repair": median([d["B2_repair"] for d in ok]),
        "B3_median_pct": median([d.get("B3_regret_pct") for d in ok]),
        "B3_stuck_frac": sum(d["B3_stuck"] for d in ok) / n,
        "B3_best_order_median_pct": median([d.get("B3_best_order_regret_pct") for d in ok]),
        "B3_bigfirst_median_pct": median([d.get("B3_bigfirst_regret_pct") for d in ok]),
        "B3_best_random8_median_pct": median([d.get("B3_best_random8_regret_pct") for d in ok]),
        "B5_bigfirst_median_pct": median([d.get("B5_bigfirst_regret_pct") for d in ok]),
        "B5_best_random8_median_pct": median([d.get("B5_best_random8_regret_pct") for d in ok]),
        "B4_median_pct": median([d["B4_regret_pct"] for d in ok]),
        "B4_median_closure": median([d["B4_closure"] for d in ok]),
        "B5_median_pct": median([d.get("B5_regret_pct") for d in ok]),
        "B5_stuck_frac": sum(d["B5_stuck"] for d in ok) / n,
        "B5_best_order_median_pct": median([d.get("B5_best_order_regret_pct") for d in ok]),
        "C1_median_pct": median([d["C1_regret_pct"] for d in ok]),
        "C1_stratum_size_median": median([d["C1_stratum_size"] for d in ok]),
        "qap_share_at_opt_median": median([d["qap_share_at_opt"] for d in ok]),
        "additive_r2_median": median([d["additive_r2"] for d in ok]),
        "count_r2_median": median([d["count_r2"] for d in ok]),
        "n_feasible_median": median([d["n_feasible"] for d in ok]),
        "rows_over_2x_params_min": min(d["n_rows"] / (2 * d["B2_n_params"]) for d in ok),
        "spread": "VOID (k > n_server_nodes)" if all(d.get("spread", "").startswith("VOID") for d in ok) else "reported",
    })
    checks = {
        "S0a": agg["S0a_r2_median"] >= BARS["S0a_r2"] and agg["S0a_regret_gt1_frac"] <= BARS["S0a_regret_gt1_frac"],
        "S0b": agg["S0b_r2_median"] >= BARS["S0b_r2"] and agg["S0b_regret_gt1_frac"] <= BARS["S0b_regret_gt1_frac"],
        "B0": agg["B0_cap_binds_frac"] >= BARS["B0_cap_binds_frac"],
        "B1": agg["B1_median_pct"] >= BARS["B1_median_pct"],
        "B2": agg["B2_median_repair"] < BARS["B2_median_repair"],
        "B3": agg["B3_median_pct"] >= BARS["B3_median_pct"],
        "B4": agg["B4_median_closure"] < BARS["B4_median_closure"],
        "B5": agg["B5_median_pct"] >= BARS["B5_median_pct"],
        "C1": agg["C1_median_pct"] >= BARS["C1_median_pct"],
    }
    agg["checks"] = {k: bool(v) for k, v in checks.items()}
    agg["pass"] = bool(agg["readable"] and all(checks.values()))
    agg["hand_lookahead_suffices"] = bool(checks["B3"] and not checks["B5"])
    return agg


# ---------------------------------------------------------------------------- simulated sweep
class Simulated(Paper):
    """The same instance object built from a SIMULATED dataset written by
    generate_gnn_datasets_fast.py --grid peer_affinity_screen (plan §A2): k single-task events,
    per-event demand_scale, a top-level peer_exchange table, and a placements.jsonl that is the
    FULL Cartesian product of the tasks' candidate replicas (--allow-non-unique-replicas).

    `total(x)` is a table lookup: x > 0 -> the treated corpus's rtt, x == 0 -> the paired
    control corpus's rtt (same seed, HEROSIM_PEER_EXCHANGE unset). The exchange term is
    recomputed from the stored files with the physics' own formula and must agree with the
    retained `peer_exchange_total` on every row (Amendment A4's physics-agreement check).
    """

    def __init__(self, treated_dir: Path, control_dir: Optional[Path]):
        self.treated_dir, self.control_dir = treated_dir, control_dir
        self.ds = S.Dataset(treated_dir, TT, "rtt")
        ds = self.ds
        w = json.load(open(treated_dir / "workload.json"))
        self.k = len(ds.task_type_names)
        if any(len(ev["application"]["dag"]) != 1 for ev in w["events"]):
            raise RuntimeError(f"{treated_dir}: the simulated screen expects single-task events")
        self.types = list(ds.task_type_names)
        self.type_vocab = sorted(set(self.types))
        self.type_idx = np.array([self.type_vocab.index(tp) for tp in self.types])
        self.sources = list(ds.task_sources)
        self.scale = list(ds.demand_scales)
        # candidates per task = the placements the sweep actually enumerated for it
        cand_sets: Dict[int, set] = {}
        for plan, _v in ds.rows:
            for t, pl in plan.items():
                cand_sets.setdefault(t, set()).add(pl)
        self.cands = [sorted(cand_sets[i]) for i in range(self.k)]
        self.cand_offsets = list(np.cumsum([0] + [len(c) for c in self.cands[:-1]]))
        self.n_cand = max(len(c) for c in self.cands)
        expected_rows = int(np.prod([len(c) for c in self.cands]))
        if len(ds.rows) != expected_rows:
            raise RuntimeError(f"{treated_dir}: {len(ds.rows)} rows but the candidate product is "
                               f"{expected_rows} -- not a full sweep; refusing")
        # nodes / platforms / exchange matrices from the stored infrastructure
        self.src = self  # the exchange matrices live here (Paper reads src.per_byte / src.latency)
        self.nodes = sorted({ds.node_of(pl) for cs in self.cands for pl in cs})
        self.node_index = {n: i for i, n in enumerate(self.nodes)}
        n = len(self.nodes)
        self.per_byte = np.zeros((n, n)); self.latency = np.zeros((n, n)); self.max_asymmetry = 0.0
        for a in range(n):
            for b in range(n):
                if a != b:
                    hops, bneck, lat = ds.route_metrics(self.nodes[a], self.nodes[b])
                    self.per_byte[a, b] = hops / (bneck * 1024 * 1024); self.latency[a, b] = lat
        for a in range(n):
            for b in range(a + 1, n):
                for M in (self.per_byte, self.latency):
                    self.max_asymmetry = max(self.max_asymmetry, abs(M[a, b] - M[b, a]) / max(M[a, b], M[b, a], 1e-12))
                    M[a, b] = M[b, a] = 0.5 * (M[a, b] + M[b, a])
        self.plats = sorted({pl for cs in self.cands for pl in cs})
        self.plat_index = {pl: i for i, pl in enumerate(self.plats)}
        # peer table in bytes (multiplier convention of Paper: mult * x with x == 1 byte)
        triples = w.get("peer_exchange") or []
        if not triples:
            raise RuntimeError(f"{treated_dir}: workload has no peer_exchange table")
        pairs: Dict[Tuple[int, int], float] = {}
        for i, j, b in triples:
            pairs[(min(int(i), int(j)), max(int(i), int(j)))] = float(b)
        self.pairs = sorted(pairs.items())
        self.partners = None
        self.seed = None
        pad = self.n_cand
        def padded(fn, fill):
            arr = np.full((self.k, pad), fill, dtype=float)
            for i, cs in enumerate(self.cands):
                for c, pl in enumerate(cs):
                    arr[i, c] = fn(i, pl)
            return arr
        self.node = padded(lambda i, pl: self.node_index[ds.node_of(pl)], -1).astype(int)
        self.plat = padded(lambda i, pl: self.plat_index[pl], -1).astype(int)
        self.demand = padded(lambda i, pl: ds.demand[(i, pl)], 0.0)
        self.cost = np.zeros((self.k, pad))  # unused for a table instance
        self.q = np.zeros(len(self.plats))
        self.n_nodes, self.n_plat, self.n_types = n, len(self.plats), len(self.type_vocab)
        # table
        self.N = expected_rows
        radix = [len(c) for c in self.cands]
        self._radix = radix
        def index_of(plan):
            idx = 0
            for i in range(self.k):
                idx = idx * radix[i] + self.cands[i].index(plan[i])
            return idx
        self.Y = np.full(self.N, np.nan)
        self.EX_RET = np.full(self.N, np.nan)
        with open(treated_dir / "placements/placements.jsonl") as fh:
            for line in fh:
                r = json.loads(line)
                plan = {int(t): (int(v[0]), int(v[1])) for t, v in r["placement_plan"].items()}
                idx = index_of(plan)
                self.Y[idx] = float(r["rtt"])
                if "peer_exchange_total" in r:
                    self.EX_RET[idx] = float(r["peer_exchange_total"])
        if np.isnan(self.Y).any():
            raise RuntimeError(f"{treated_dir}: sweep does not cover the full product")
        self.Y0 = None
        if control_dir is not None:
            cds = S.Dataset(control_dir, TT, "rtt")
            self.Y0 = np.full(self.N, np.nan)
            for plan, v in cds.rows:
                self.Y0[index_of(plan)] = v
            if np.isnan(self.Y0).any():
                raise RuntimeError(f"{control_dir}: control sweep does not cover the treated product")
        # pm columns with the ACTUAL bytes (x == 1 -> pm_b carries bytes already)
        self.pm_b = np.zeros((self.k, pad)); self.pm_l = np.zeros((self.k, pad))
        for (i, j), mult in self.pairs:
            for a, b in ((i, j), (j, i)):
                na = self.node[a, :len(self.cands[a])]; nb = self.node[b, :len(self.cands[b])]
                self.pm_b[a, :len(na)] += mult * self.per_byte[na[:, None], nb[None, :]].mean(1)
                self.pm_l[a, :len(na)] += self.latency[na[:, None], nb[None, :]].mean(1)

    def enumerate(self) -> None:
        k = self.k
        N = self.N
        idx = np.arange(N, dtype=np.int64)
        P = np.empty((N, k), dtype=np.int8)
        rem = idx.copy()
        for i in reversed(range(k)):
            P[:, i] = rem % self._radix[i]
            rem //= self._radix[i]
        self.P = P
        ar = np.arange(k)
        self.NODE = self.node[ar, P].astype(np.int8)
        self.PLAT = self.plat[ar, P].astype(np.int16)
        DEM = self.demand[ar, P]
        self.cnt = np.zeros((N, self.n_plat), dtype=np.int8)
        for p in range(self.n_plat):
            self.cnt[:, p] = (self.PLAT == p).sum(1)
        self.load = np.zeros((N, self.n_nodes)); self.nodecnt = np.zeros((N, self.n_nodes), dtype=np.int8)
        self.min_single = np.full((N, self.n_nodes), np.inf)
        for n in range(self.n_nodes):
            mask = self.NODE == n
            self.load[:, n] = (DEM * mask).sum(1); self.nodecnt[:, n] = mask.sum(1)
            self.min_single[:, n] = np.where(mask, DEM, np.inf).min(1)
        self.max_demand_on_node = np.array([self.demand[self.node == n].max() if (self.node == n).any() else 0.0
                                            for n in range(self.n_nodes)])
        # exchange recomputed with the physics' formula, charged on BOTH ends of every pair
        self.EX = np.zeros(N)
        for (i, j), mult in self.pairs:
            self.EX += 2.0 * (mult * self.per_byte[self.NODE[:, i], self.NODE[:, j]] + self.latency[self.NODE[:, i], self.NODE[:, j]])
        self.KEY = np.sort(self.PLAT.astype(np.int32) * self.n_types + self.type_idx[None, :], axis=1)
        self.collision_free = (self.cnt.max(1) == 1)
        self.BASE = self.Y - self.EX; self.SHARE = np.zeros(N); self.QB = np.zeros(N); self.QL = np.zeros(N)

    def plan_index(self, plan: Sequence[int]) -> int:
        idx = 0
        for i, c in enumerate(plan):
            idx = idx * self._radix[i] + int(c)
        return idx

    def total(self, x_bytes: float) -> np.ndarray:
        if x_bytes == 0.0:
            if self.Y0 is None:
                raise RuntimeError("no control corpus paired; S0 cannot be read")
            return self.Y0
        return self.Y

    def exchange(self, x_bytes: float) -> np.ndarray:
        return self.EX

    def peer_mass_col(self, sel: np.ndarray, x_bytes: float) -> np.ndarray:
        pm = self.pm_b + self.pm_l
        return pm[np.arange(self.k), self.P[sel]].sum(1, keepdims=True)

    def physics_agreement(self) -> dict:
        ok = ~np.isnan(self.EX_RET)
        if not ok.any():
            return {"rows_with_retained_total": 0, "max_rel_gap": None}
        gap = np.abs(self.EX[ok] - self.EX_RET[ok]) / np.maximum(1.0, np.abs(self.EX_RET[ok]))
        diff = self.Y - self.Y0 if self.Y0 is not None else None
        return {"rows_with_retained_total": int(ok.sum()), "max_rel_gap": float(gap.max()),
                "exchange_share_of_treated_minus_control_median": (
                    float(np.median(self.EX[diff > 0] / diff[diff > 0])) if diff is not None and (diff > 0).any() else None)}


def greedy_table(inst: Simulated, y: np.ndarray, cap: np.ndarray, order: Sequence[int], mode: str
                 ) -> Optional[List[int]]:
    """Sequential greedy on the TRUE TABLE with an exact prefix (Amendment A4's analogues of
    B3/B5 for a simulated sweep, where no analytic marginal exists):
      mode == "reference": score candidate c for task i by the row where the prefix and
        (i, c) are fixed and every uncommitted task sits on its lowest-id candidate -- the
        committed peers' exchange and sharing are charged exactly, the future is a fixed
        default (B3').
      mode == "expected": score by the MEAN over all completions of the uncommitted tasks
        (B5', the table's own peer-mass lookahead).
    Capacity-masked on the committed load; None when no feasible candidate remains."""
    k = inst.k
    plan = [-1] * k
    load = np.zeros(inst.n_nodes)
    for i in order:
        best, best_c = np.inf, -1
        committed = [t for t in range(k) if plan[t] >= 0]
        for c in range(len(inst.cands[i])):
            n = int(inst.node[i, c])
            if load[n] + inst.demand[i, c] > cap[n] + EPS:
                continue
            mask = np.ones(inst.N, dtype=bool)
            for t in committed:
                mask &= inst.P[:, t] == plan[t]
            mask &= inst.P[:, i] == c
            if mode == "reference":
                for t in range(k):
                    if plan[t] < 0 and t != i:
                        mask &= inst.P[:, t] == 0
                vals = y[mask]
                m = float(vals[0]) if len(vals) else np.inf
            else:
                m = float(y[mask].mean()) if mask.any() else np.inf
            if m < best - 1e-12 or (abs(m - best) <= 1e-12 and (n, int(inst.plat[i, c])) < (int(inst.node[i, best_c]), int(inst.plat[i, best_c]))):
                best, best_c = m, c
        if best_c < 0:
            return None
        plan[i] = best_c
        load[int(inst.node[i, best_c])] += inst.demand[i, best_c]
    return plan


def screen_simulated(inst: Simulated, alpha4: float, rng_orders: random.Random) -> dict:
    """Same statistics as screen_dataset on a simulated pair; B3/B5 are greedy_table analogues."""
    alpha = alpha_k(alpha4, inst.k)
    cap = inst.caps(alpha)
    y = inst.Y
    feas_mask = (inst.load <= cap[None, :] + EPS).all(1)
    feas = np.nonzero(feas_mask)[0]
    all_rows = np.arange(inst.N)
    out: dict = {"ds": inst.treated_dir.name, "seed": None, "n_rows": int(inst.N), "alpha_k": alpha,
                 "n_feasible": int(len(feas)), "n_nodes": inst.n_nodes, "n_plat": inst.n_plat,
                 "n_pairs": len(inst.pairs), "cands_per_task": [len(c) for c in inst.cands],
                 "physics_agreement": inst.physics_agreement()}
    if len(feas) == 0:
        out["no_feasible_rows"] = True
        return out
    opt = int(feas[np.argmin(y[feas])]); opt_val = float(y[opt])
    out["opt_val"] = opt_val
    out["qap_share_at_opt"] = float(inst.EX[opt] / opt_val)
    out["B0_cap_binds"] = bool(not feas_mask[int(np.argmin(y))])
    if inst.Y0 is not None:
        y0 = inst.Y0
        cf = np.nonzero(feas_mask & inst.collision_free)[0]
        out["n_collision_free"] = int(len(cf))
        fit = fit_argmin(inst, y0, cf, cf, ("ind",)) if len(cf) else None
        if fit is not None:
            o0 = float(y0[cf].min())
            out["S0a_r2"] = fit[1]; out["S0a_regret_pct"] = tie_regret(y0[cf], fit[0], o0)["mean_tied"]
        fit = fit_argmin(inst, y0, all_rows, feas, ("ind", "count"), cap)
        if fit is not None:
            o0 = float(y0[feas].min())
            out["S0b_r2"] = fit[1]; out["S0b_regret_pct"] = tie_regret(y0[feas], fit[0], o0)["mean_tied"]
            out["S0b_n_params"] = fit[2]
    fit = fit_argmin(inst, y, all_rows, feas, ("ind",))
    if fit is None:
        out["fit_refused"] = True
        return out
    band = tie_regret(y[feas], fit[0], opt_val)
    out["B1_regret_pct"] = band["mean_tied"]; out["B1_band"] = band; out["additive_r2"] = fit[1]
    fit = fit_argmin(inst, y, all_rows, feas, ("ind", "count"), cap)
    band = tie_regret(y[feas], fit[0], opt_val)
    out["B2_regret_pct"] = band["mean_tied"]; out["B2_band"] = band; out["count_r2"] = fit[1]; out["B2_n_params"] = fit[2]
    out["B2_repair"] = (max(0.0, (out["B1_regret_pct"] - out["B2_regret_pct"]) / out["B1_regret_pct"])
                        if out["B1_regret_pct"] > 0 else None)
    fit = fit_argmin(inst, y, all_rows, feas, ("ind", "count", "peer"), cap, 1.0)
    band = tie_regret(y[feas], fit[0], opt_val)
    out["B4_regret_pct"] = band["mean_tied"]; out["B4_band"] = band; out["peer_r2"] = fit[1]
    out["B4_closure"] = (max(0.0, (out["B1_regret_pct"] - out["B4_regret_pct"]) / out["B1_regret_pct"])
                         if out["B1_regret_pct"] > 0 else None)
    id_order = list(range(inst.k))
    strength = {i: 0.0 for i in range(inst.k)}
    for (i, j), mult in inst.pairs:
        strength[i] += mult; strength[j] += mult
    big_first = sorted(range(inst.k), key=lambda i: -strength[i])
    rand_orders = [rng_orders.sample(range(inst.k), inst.k) for _ in range(8)]
    for tag, mode in (("B3", "reference"), ("B5", "expected")):
        plan = greedy_table(inst, y, cap, id_order, mode)
        out[f"{tag}_stuck"] = plan is None
        if plan is not None:
            out[f"{tag}_regret_pct"] = regret_pct(float(y[inst.plan_index(plan)]), opt_val)
        pb = greedy_table(inst, y, cap, big_first, mode)
        out[f"{tag}_bigfirst_regret_pct"] = None if pb is None else regret_pct(float(y[inst.plan_index(pb)]), opt_val)
        rr = []
        for order in rand_orders:
            p2 = greedy_table(inst, y, cap, order, mode)
            if p2 is not None:
                rr.append(regret_pct(float(y[inst.plan_index(p2)]), opt_val))
        out[f"{tag}_best_random8_regret_pct"] = min(rr) if rr else None
        alts = rr + [v for v in (out[f"{tag}_bigfirst_regret_pct"], out.get(f"{tag}_regret_pct")) if v is not None]
        out[f"{tag}_best_order_regret_pct"] = min(alts) if alts else None
    same = feas[(inst.KEY[feas] == inst.KEY[opt]).all(1)]
    out["C1_stratum_size"] = int(len(same))
    out["C1_regret_pct"] = regret_pct(float(y[same].mean()), opt_val)
    out["C1_max_regret_pct"] = regret_pct(float(y[same].max()), opt_val)
    out["spread"] = "VOID (k > n_server_nodes)" if inst.k > inst.n_nodes else "n/a"
    return out


def run_simulated(treated: Path, control: Optional[Path], alphas: Sequence[float], out_path: Path,
                  limit: Optional[int] = None) -> dict:
    dirs = sorted(p for p in treated.glob("ds_*") if (p / "placements/placements.jsonl").exists())
    if limit:
        dirs = dirs[:limit]
    if not dirs:
        raise RuntimeError(f"{treated}: no datasets")
    report = {"treated": str(treated), "control": str(control) if control else None, "bars": BARS,
              "count_competitor": COUNT_COMPETITOR,
              "cells": {}, "per_dataset": {}, "skipped": []}
    per = {a: [] for a in alphas}
    t0 = time.time()
    for d in dirs:
        cdir = (control / d.name) if control else None
        if cdir is not None and not (cdir / "placements/placements.jsonl").exists():
            report["skipped"].append({"ds": d.name, "reason": "no control dataset"})
            continue
        try:
            inst = Simulated(d, cdir)
        except RuntimeError as exc:
            report["skipped"].append({"ds": d.name, "reason": str(exc)})
            continue
        inst.enumerate()
        for a in alphas:
            per[a].append(screen_simulated(inst, a, random.Random(hash((d.name, a)) & 0xFFFF if False else (int(d.name.split("_")[-1]) * 7919 + int(a * 10)))))
        del inst
    print(f"[probe] simulated: {len(dirs) - len(report['skipped'])} datasets x {len(alphas)} alphas in {time.time() - t0:.0f}s; "
          f"skipped {len(report['skipped'])}", flush=True)
    for a in alphas:
        cell = f"sim_a{a}_{treated.name}"
        report["per_dataset"][cell] = per[a]
        agg = aggregate(cell, per[a])
        pa = [d["physics_agreement"]["max_rel_gap"] for d in per[a] if d.get("physics_agreement", {}).get("max_rel_gap") is not None]
        agg["physics_agreement_max_rel_gap"] = max(pa) if pa else None
        agg["exchange_share_of_diff_median"] = median([d["physics_agreement"].get("exchange_share_of_treated_minus_control_median") for d in per[a]])
        agg["cands_per_task_median"] = median([float(np.median(d["cands_per_task"])) for d in per[a]])
        report["cells"][cell] = agg
        print(f"[probe] {cell}: pass={agg.get('pass')} " + " ".join(f"{kk}={'Y' if v else 'n'}" for kk, v in agg.get("checks", {}).items())
              + f" B1={agg.get('B1_median_pct', float('nan')):.2f} B2rep={agg.get('B2_median_repair', float('nan')):.2f} "
                f"B3={agg.get('B3_median_pct', float('nan')):.2f} B5={agg.get('B5_median_pct', float('nan')):.2f} C1={agg.get('C1_median_pct', float('nan')):.2f} "
                f"phys_gap={agg['physics_agreement_max_rel_gap']} scored={agg['n_scored']}", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=1, default=str))
    return report


# --------------------------------------------------------------------------------------- emit
def emit_dataset(paper: Paper, x_mb: float, alpha: float, out_dir: Path) -> None:
    """Write the paper dataset in the on-disk co-sim format so the existing instruments
    (separability_diagnostic, score_route_b_contention) read the same rows."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "placements").mkdir(exist_ok=True)
    for name in ("infrastructure.json", "space_with_network.json"):
        shutil.copy(paper.src.ds_dir / name, out_dir / name)
    events = []
    for i in range(paper.k):
        events.append({"timestamp": 0.0,
                       "application": {"name": f"nofs-{paper.types[i]}", "dag": {paper.types[i]: []},
                                       "demand_scale": {paper.types[i]: paper.scale[i]}},
                       "qos": {"name": "medium", "maxDurationDeviation": 15},
                       "node_name": paper.sources[i]})
    x = x_mb * MB
    workload = {"rps": 1, "duration": 1, "events": events,
                "peer_exchange": [[i, j, mult * x] for (i, j), mult in paper.pairs],
                "paper_model": {"source": str(paper.src.ds_dir), "seed": paper.seed, "k": paper.k,
                                "n_cand": paper.n_cand, "partners": paper.partners, "x_scale_mb": x_mb,
                                "alpha": alpha, "sharing_q": {str(p): float(q) for p, q in zip(paper.plats, paper.q)}}}
    (out_dir / "workload.json").write_text(json.dumps(workload))
    y = paper.total(x)
    with open(out_dir / "placements/placements.jsonl", "w") as fh:
        for r in range(paper.N):
            plan = {str(i): [int(paper.cands[i][paper.P[r, i]][0]), int(paper.cands[i][paper.P[r, i]][1])]
                    for i in range(paper.k)}
            fh.write(json.dumps({"placement_plan": plan, "rtt": float(y[r])}, separators=(",", ":")) + "\n")
    (out_dir / "placement_metadata.json").write_text(json.dumps({
        "num_placements": int(paper.N), "completed": int(paper.N), "rows_written": int(paper.N),
        "timed_out": 0, "worker_failed": 0, "worker_exception": 0, "early_terminated": False,
        "timeout_per_placement_s": None, "sweep_complete": True, "paper_model": True}))


def crosscheck_dataset(paper: Paper, x_mb: float, alpha: float, ds_dir: Path) -> dict:
    import separability_diagnostic as D  # noqa: E402
    x = x_mb * MB
    y = paper.total(x)
    all_rows = np.arange(paper.N)
    # probe's own unconstrained full-sweep statistics (what the diagnostic computes)
    fit = fit_argmin(paper, y, all_rows, all_rows, ("ind",))
    opt_u = float(y.min())
    band = tie_regret(y, fit[0], opt_u)
    mine = {"additive_r2": fit[1], "additive_choice_regret_rel": band["mean_tied"] / 100.0}
    band_add = (band["optimistic"] / 100.0, band["pessimistic"] / 100.0)
    # one-integer repair = additive + node-occupancy excess only
    def cols_1int(sel):
        return np.concatenate([np.ones((len(sel), 1)), paper.indicator_cols(sel),
                               np.maximum(paper.nodecnt[sel].astype(float) - 1, 0).sum(1, keepdims=True)], axis=1)
    G = np.zeros((cols_1int(all_rows[:1]).shape[1],) * 2); b = np.zeros(G.shape[0])
    for s in range(0, paper.N, CHUNK):
        sel = all_rows[s:s + CHUNK]; X = cols_1int(sel); G += X.T @ X; b += X.T @ y[sel]
    beta = np.linalg.lstsq(G, b, rcond=None)[0]
    pred = np.concatenate([cols_1int(all_rows[s:s + CHUNK]) @ beta for s in range(0, paper.N, CHUNK)])
    pmin = float(pred.min())
    ties1 = np.nonzero(pred <= pmin + TIE_REL * max(abs(pmin), 1.0))[0]
    band1 = tie_regret(y, ties1, opt_u)
    aug = band1["mean_tied"] / 100.0
    mine["one_integer_repair_frac"] = (max(0.0, (mine["additive_choice_regret_rel"] - aug) / mine["additive_choice_regret_rel"])
                                       if mine["additive_choice_regret_rel"] > 0 else None)
    band_aug = (band1["optimistic"] / 100.0, band1["pessimistic"] / 100.0)
    theirs_full = D.analyze_dataset(ds_dir)
    m4 = theirs_full.get("m4") or {}
    if m4.get("degenerate", True):
        raise RuntimeError(f"{ds_dir}: separability_diagnostic refused the fit: {m4}")
    theirs = {k: m4.get(k) for k in list(mine) + ["additive_plus_collision_choice_regret_rel"]}
    ds = S.Dataset(ds_dir, TT, "rtt")
    caps = ds.node_caps(alpha)
    my_caps = {paper.src.nodes[n]: float(c) for n, c in enumerate(paper.caps(alpha)) if math.isfinite(c)}
    n_feas_theirs = sum(ds.plan_feasible(p, caps) for p, _v in ds.rows)
    n_feas_mine = int((paper.load <= paper.caps(alpha)[None, :] + EPS).all(1).sum())
    caps_agree = (set(caps) == set(my_caps) and all(abs(caps[n] - my_caps[n]) <= 1e-9 * max(1.0, caps[n]) for n in caps))
    # R^2 must agree to 1e-6; the diagnostic's single-argmin regrets must fall inside the
    # probe's tie band (its argmin is one member of the tie set, chosen by row order).
    def within(v, lo, hi):
        return v is not None and lo - 1e-9 <= v <= hi + 1e-9
    agree = {"additive_r2": abs(mine["additive_r2"] - theirs["additive_r2"]) <= 1e-6,
             "additive_choice_regret_rel": within(theirs["additive_choice_regret_rel"], *band_add)}
    t_aug = theirs.get("additive_plus_collision_choice_regret_rel")
    agree["one_integer_repair_frac"] = ((theirs["one_integer_repair_frac"] is None and mine["one_integer_repair_frac"] is None)
                                        or within(t_aug, *band_aug)
                                        or (theirs["one_integer_repair_frac"] is None and band_add[0] <= 0.0))
    mine["bands"] = {"additive": band_add, "one_integer": band_aug}
    return {"ds": ds_dir.name, "probe": mine, "diagnostic": theirs, "agree": agree,
            "caps_agree": bool(caps_agree), "n_feasible_probe": n_feas_mine, "n_feasible_scorer": n_feas_theirs,
            "ok": bool(all(agree.values()) and caps_agree and n_feas_mine == n_feas_theirs)}


# --------------------------------------------------------------------------------------- main
def select_sources(corpus: Path, limit: Optional[int]) -> List[Path]:
    dirs = sorted(p for p in corpus.glob("ds_*") if (p / "placements/placements.jsonl").exists())
    if not dirs:
        raise RuntimeError(f"{corpus}: no datasets with placements.jsonl")
    chosen = dirs[::SOURCE_STRIDE][:SOURCE_COUNT]
    return chosen[:limit] if limit else chosen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=CORPUS_DEFAULT)
    ap.add_argument("--cells", default="", help="comma-separated substrings; a cell runs if it contains any")
    ap.add_argument("--limit-sources", type=int, default=None)
    ap.add_argument("--paper-seed-base", type=int, default=7000)
    ap.add_argument("--stages", nargs="+", default=["screen"], choices=["screen", "emit", "crosscheck"])
    ap.add_argument("--emit-dir", type=Path, default=ROOT / "simulation_data/peer_affinity_paper_k8")
    ap.add_argument("--emit-cells", default="x200_a2.0_k8c3_p2,x200_a2.0_k8c4_p2")
    ap.add_argument("--from-simulated", type=Path, default=None,
                    help="treated corpus dir (HEROSIM_PEER_EXCHANGE=1); reads the registered bars on the simulated sweep")
    ap.add_argument("--control", type=Path, default=None, help="paired control corpus (flag unset, same seeds)")
    ap.add_argument("--alphas", default="1.5,2.0", help="alpha_4 values for --from-simulated")
    ap.add_argument("--count-competitor", choices=["v1", "v2"], default=None,
                    help="count column block; default v1 for the paper screen, v2 for --from-simulated")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    global COUNT_COMPETITOR
    COUNT_COMPETITOR = args.count_competitor or ("v2" if args.from_simulated is not None else "v1")
    if args.from_simulated is not None:
        run_simulated(args.from_simulated, args.control, [float(a) for a in args.alphas.split(",")],
                      args.out, args.limit_sources)
        return 0
    filters = [f for f in args.cells.split(",") if f]
    emit_cells = set(args.emit_cells.split(","))
    sources = select_sources(args.corpus, args.limit_sources)
    t0 = time.time()
    loaded = [Source(d) for d in sources]
    print(f"[probe] {len(loaded)} sources loaded in {time.time() - t0:.1f}s; residual duration spread at "
          f"earliest dispatch {max(s.max_spread for s in loaded):.2e} (over all dispatch times "
          f"{max(s.max_spread_any for s in loaded):.2e}), max route asymmetry "
          f"{max(s.max_asymmetry for s in loaded):.2e}", flush=True)
    report = {"corpus": str(args.corpus), "sources": [str(s.ds_dir.name) for s in loaded],
              "source_duration_spread_at_earliest_dispatch": {s.ds_dir.name: s.max_spread for s in loaded},
              "source_duration_spread_any_dispatch": {s.ds_dir.name: s.max_spread_any for s in loaded},
              "bars": BARS, "paper_seed_base": args.paper_seed_base, "cells": {}, "per_dataset": {},
              "crosscheck": {}}
    for (k, n_cand) in SHAPES:
        for partners in PARTNERS:
            cells = [(x, a, cell_name(x, a, k, n_cand, partners)) for x in X_SCALES_MB for a in ALPHAS]
            if filters:
                cells = [c for c in cells if any(f in c[2] for f in filters)]
            if not cells:
                continue
            per = {c[2]: [] for c in cells}
            checks = {c[2]: [] for c in cells}
            t1 = time.time()
            # one paper dataset in memory at a time: a 1M-row instance is ~250 MB of arrays
            for si, src in enumerate(loaded):
                paper = Paper(src, k, n_cand, partners, args.paper_seed_base + 100 * si + 10 * partners + n_cand)
                paper.enumerate()
                for x, a, cell in cells:
                    if "screen" in args.stages:
                        per[cell].append(screen_dataset(paper, x, a, random.Random(paper.seed ^ 0x5EED)))
                    if cell in emit_cells and ("emit" in args.stages or "crosscheck" in args.stages):
                        d = args.emit_dir / cell / f"{paper.src.ds_dir.name}_s{paper.seed}"
                        if "emit" in args.stages:
                            emit_dataset(paper, x, alpha_k(a, k), d)
                        if "crosscheck" in args.stages:
                            checks[cell].append(crosscheck_dataset(paper, x, alpha_k(a, k), d))
                del paper
            print(f"[probe] k={k} c={n_cand} p={partners}: {len(loaded)} paper datasets x {len(cells)} cells "
                  f"({n_cand ** k} rows each) in {time.time() - t1:.0f}s", flush=True)
            for x, a, cell in cells:
                if per[cell]:
                    report["per_dataset"][cell] = per[cell]
                    agg = aggregate(cell, per[cell])
                    report["cells"][cell] = agg
                    print(f"[probe] {cell}: pass={agg['pass']} " + " ".join(
                        f"{kk}={'Y' if v else 'n'}" for kk, v in agg.get("checks", {}).items())
                        + f" B1={agg.get('B1_median_pct', float('nan')):.2f} B3={agg.get('B3_median_pct', float('nan')):.2f}"
                          f" B5={agg.get('B5_median_pct', float('nan')):.2f} C1={agg.get('C1_median_pct', float('nan')):.2f}"
                          f" nofeas={agg.get('n_no_feasible')} readable={agg.get('readable')}", flush=True)
                if checks[cell]:
                    report["crosscheck"][cell] = {"n": len(checks[cell]), "n_ok": sum(c["ok"] for c in checks[cell]),
                                                  "per_dataset": checks[cell]}
                    print(f"[probe] crosscheck {cell}: {sum(c['ok'] for c in checks[cell])}/{len(checks[cell])} agree", flush=True)
            # checkpoint the report after every shape so a killed run keeps its finished cells
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=1, default=str))
    passing = [c for c, a in report["cells"].items() if a.get("pass")]
    report["decision"] = {"go": bool(passing), "passing_cells": passing,
                          "hand_lookahead_suffices_cells": [c for c, a in report["cells"].items()
                                                            if a.get("hand_lookahead_suffices")]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str))
    brief = {"decision": report["decision"],
             "cells": {c: {k: v for k, v in a.items() if k != "checks"} for c, a in report["cells"].items()},
             "crosscheck": {c: {k: v for k, v in r.items() if k != "per_dataset"} for c, r in report["crosscheck"].items()}}
    print(json.dumps(brief, indent=1, default=str))
    print(f"[probe] wrote {args.out} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
