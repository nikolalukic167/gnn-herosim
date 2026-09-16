"""partial_state_v3 -- the registered read. Every bar below is a module constant.

Registered in docs/lineages/partial_state_v3.md before any cache was built. The reads are
pure functions over small summaries so they can be tested without a cluster:

  P0  instrument   v3 cache == v2 cache on the untouched columns; rank recoverable
  P1  offline tie  v3 vs v2 held-out regret, paired by training seed (ordering only)
  P2  live tie     v3 vs v2 at 6 servers, paired by training seed (blocking for P3's reading)
  P3  PRIMARY      does a 6-server checkpoint survive 12 / 24 / 80 servers, live
  P4  the prize    conditional on P3 = GENERALISES
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from scripts_cosim.queue_range_v1_read import wilcoxon_p

# --- registered bars -------------------------------------------------------------------
P0_MAX_COLUMN_DIFF = 0.0        # untouched columns must be bit-identical between v2 and v3
P0_MIN_RANK_RECOVERY = 1.0      # fraction of edges whose v2 rank is recovered from v3
P0_MIN_DATASETS = 516           # the whole corpus, not a sample

P1_TIE_PP = 1.0                 # |median delta| in held-out regret pp
P1_ALPHA = 0.05
P1_MIN_SEEDS = 12

P2_TIE_PCT = 5.0                # |median relative difference| in mean elapsed, %
P2_ALPHA = 0.05
P2_MIN_SEEDS = 12
P2_MIN_CELLS = 2                # of 3
P2_CELLS = ("cell_s7901_f4000_pg16", "cell_s9001_f4000_pg16", "cell_s9002_f4000_pg16")
# queue_range_v1's `plain` medians the v2 control arm must reproduce to three decimals.
P2_V2_CONTROL_MEDIANS = {"cell_s7901_f4000_pg16": 54.819,
                         "cell_s9001_f4000_pg16": 22.350,
                         "cell_s9002_f4000_pg16": 65.401}
P2_CONTROL_TOL = 0.0005

P3_RUNGS: Sequence[tuple] = (   # (tag, servers, workload, arrivals/s)
    ("R0", 6, "drainable_f4000_n50000", 0.460),
    ("R1", 12, "drainable_f2000_n50000", 0.921),
    ("R2", 24, "drainable_f1000_n50000", 1.842),
    ("R3", 80, "drainable_f300_n50000", 6.139),
)
P3_TOPOLOGY_SEEDS = (9001, 9002, 9003, 9005)
P3_CHECKPOINT_SEEDS = (1, 2, 4, 5)
P3_MIN_CELLS = 3
P3_TOL = 0.10                   # d_r <= d_R0 + P3_TOL on every scaled rung
P3_MAX_WALLCLOCK_MIN = 45
P3_EXPECTED = "DEGRADES"        # registered expectation, written down before the data

P4_MIN_IMPROVEMENT = 0.05       # d_R3 <= d_R0 - 0.05, monotone

PH2_SERVERS = 12
PH2_MAX_CANDIDATES = 5.0
PH2_N_DATASETS = 200

# --- verdict strings -------------------------------------------------------------------
V_P0_PASS, V_P0_FAIL = "INSTRUMENT-PASS", "NOT-A-REPRESENTATION-CHANGE"
V_TIE, V_COSTS, V_HELPS = "TIE", "ENCODING-COSTS", "ENCODING-HELPS"
V_P2_COSTS = "REPRESENTATION-COSTS-LIVE"
V_P2_CONTROL_FAIL = "V2-PATH-MOVED"
V_STILL_PINNED = "STILL-PINNED"
V_GENERALISES, V_DEGRADES = "GENERALISES", "DEGRADES"
V_UNAFFORDABLE = "RUNG-UNAFFORDABLE"
V_SCALE_HELPS, V_SCALE_NO = "SCALE-HELPS", "SCALE-DOES-NOT-HELP"
V_UNREADABLE = "UNREADABLE"


def median(xs: Sequence[float]) -> Optional[float]:
    v = sorted(float(x) for x in xs)
    if not v:
        return None
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


# --- P0 --------------------------------------------------------------------------------

def p0_dataset(v2_block: np.ndarray, v3_block: np.ndarray, *, base_dim: int, v2_krank_dim: int,
               v3_krank_dim: int, types: int, link_dim: int) -> dict:
    """One dataset's partial-state edge blocks under both contracts (n_edges x dim each).

    Layout: [base_dim | krank | link_dim]. The base and link columns must agree exactly; the
    v2 one-hot rank must be recoverable from v3's (rank_frac, inv_n) pair in the task's own
    type slot.
    """
    v2 = np.asarray(v2_block, dtype=np.float64)
    v3 = np.asarray(v3_block, dtype=np.float64)
    if v2.shape[0] != v3.shape[0]:
        raise ValueError(f"edge count differs: v2 {v2.shape[0]} vs v3 {v3.shape[0]}")
    if v2.shape[1] != base_dim + v2_krank_dim + link_dim:
        raise ValueError(f"v2 width {v2.shape[1]} != {base_dim}+{v2_krank_dim}+{link_dim}")
    if v3.shape[1] != base_dim + v3_krank_dim + link_dim:
        raise ValueError(f"v3 width {v3.shape[1]} != {base_dim}+{v3_krank_dim}+{link_dim}")

    base_diff = float(np.max(np.abs(v2[:, :base_dim] - v3[:, :base_dim]))) if v2.size else 0.0
    link_diff = float(np.max(np.abs(v2[:, -link_dim:] - v3[:, -link_dim:]))) if (v2.size and link_dim) else 0.0

    k2 = v2[:, base_dim:base_dim + v2_krank_dim]
    k3 = v3[:, base_dim:base_dim + v3_krank_dim]
    recovered = 0
    n = v2.shape[0]
    for i in range(n):
        hot = np.flatnonzero(k2[i] > 0.5)
        if hot.size != 1:
            raise ValueError(f"edge {i}: v2 krank block is not one-hot ({hot.size} set)")
        r2, k = divmod(int(hot[0]), types)
        rank_frac, inv_n = float(k3[i, k * 2]), float(k3[i, k * 2 + 1])
        others = np.delete(k3[i], [k * 2, k * 2 + 1])
        if inv_n <= 0.0 or np.any(others != 0.0):
            continue
        n_nodes = int(round(1.0 / inv_n))
        r3 = 0 if n_nodes <= 1 else int(round(rank_frac * (n_nodes - 1)))
        if r3 == r2:
            recovered += 1
    return {"n_edges": n, "base_diff": base_diff, "link_diff": link_diff,
            "rank_recovery": (recovered / n) if n else 1.0}


def read_p0(per_dataset: Sequence[dict]) -> dict:
    n = len(per_dataset)
    if n < P0_MIN_DATASETS:
        return {"verdict": V_UNREADABLE, "n_datasets": n,
                "reason": f"{n} datasets < P0_MIN_DATASETS={P0_MIN_DATASETS}"}
    max_col = max(max(d["base_diff"], d["link_diff"]) for d in per_dataset)
    min_rec = min(d["rank_recovery"] for d in per_dataset)
    ok = max_col <= P0_MAX_COLUMN_DIFF and min_rec >= P0_MIN_RANK_RECOVERY
    return {"verdict": V_P0_PASS if ok else V_P0_FAIL, "n_datasets": n,
            "max_column_diff": max_col, "min_rank_recovery": min_rec}


# --- P1 / P2: paired ties --------------------------------------------------------------

def paired_tie(v3: Mapping[int, float], v2: Mapping[int, float], *, tol: float, alpha: float,
               min_seeds: int, relative: bool, lower_is_better: bool = True) -> dict:
    """Paired by training seed. `relative` reads (v3 - v2) / v2 in %, else v3 - v2 (pp)."""
    seeds = sorted(set(v3) & set(v2))
    if len(seeds) < min_seeds:
        return {"verdict": V_UNREADABLE, "n": len(seeds),
                "reason": f"{len(seeds)} paired seeds < {min_seeds}"}
    diffs = []
    for s in seeds:
        d = float(v3[s]) - float(v2[s])
        diffs.append(100.0 * d / float(v2[s]) if relative else d)
    med = median(diffs)
    p = wilcoxon_p(diffs)
    tie = abs(med) <= tol or p >= alpha
    if tie:
        verdict = V_TIE
    else:
        worse = (med > 0) if lower_is_better else (med < 0)
        verdict = V_COSTS if worse else V_HELPS
    return {"verdict": verdict, "n": len(seeds), "median": med, "p": p,
            "v3_ahead": sum((d < 0) == lower_is_better for d in diffs if d != 0)}


def read_p1(v3_regret: Mapping[int, float], v2_regret: Mapping[int, float]) -> dict:
    return paired_tie(v3_regret, v2_regret, tol=P1_TIE_PP, alpha=P1_ALPHA,
                      min_seeds=P1_MIN_SEEDS, relative=False)


def read_p2(per_cell_v3: Mapping[str, Mapping[int, float]],
            per_cell_v2: Mapping[str, Mapping[int, float]],
            v2_control_medians: Mapping[str, float]) -> dict:
    """per_cell_*: cell -> {training seed -> mean elapsed}. v2_control_medians: the v2
    control arm's median elapsed per cell, read from THIS job."""
    control = {}
    for cell, want in P2_V2_CONTROL_MEDIANS.items():
        got = v2_control_medians.get(cell)
        control[cell] = (got is not None and abs(float(got) - want) <= P2_CONTROL_TOL, got, want)
    if not all(ok for ok, _, _ in control.values()):
        return {"verdict": V_P2_CONTROL_FAIL, "control": control,
                "reason": "the v2 serving path did not reproduce queue_range_v1's plain medians"}
    cells = {}
    for cell in P2_CELLS:
        if cell in per_cell_v3 and cell in per_cell_v2:
            cells[cell] = paired_tie(per_cell_v3[cell], per_cell_v2[cell], tol=P2_TIE_PCT,
                                     alpha=P2_ALPHA, min_seeds=P2_MIN_SEEDS, relative=True)
    ties = sum(1 for r in cells.values() if r["verdict"] == V_TIE)
    readable = sum(1 for r in cells.values() if r["verdict"] != V_UNREADABLE)
    if readable < P2_MIN_CELLS:
        verdict = V_UNREADABLE
    elif ties >= P2_MIN_CELLS:
        verdict = V_TIE
    else:
        verdict = V_P2_COSTS if any(r["verdict"] == V_COSTS for r in cells.values()) else V_HELPS
    return {"verdict": verdict, "cells": cells, "control": control, "ties": ties}


# --- P3 / P4 -----------------------------------------------------------------------------

def cell_deficit(arm_elapsed: Mapping[int, float], reactive_elapsed: float) -> float:
    """d = median over checkpoint seeds of (arm / reactive) - 1."""
    if reactive_elapsed <= 0:
        raise ValueError("reactive elapsed must be positive")
    return float(median([float(v) / float(reactive_elapsed) - 1.0 for v in arm_elapsed.values()]))


def read_p3(rungs: Mapping[str, dict]) -> dict:
    """rungs: tag -> {"cells": {cell: {"reactive": s, "gnn": {seed: s}, "mpoff": {seed: s},
    "completed": {"gnn": k, "mpoff": k}, "expected": {"gnn": k, "mpoff": k}}},
    "wallclock_min": [..]}. Hung cells are omitted by the caller (attrition), not failed arms:
    an arm that STARTED and died on the representation is a completed < expected count."""
    out = {"expected": P3_EXPECTED, "rungs": {}}
    for tag, servers, _, rate in P3_RUNGS:
        r = rungs.get(tag)
        if not r or not r.get("cells"):
            out["rungs"][tag] = {"servers": servers, "readable": False, "reason": "no cells"}
            continue
        cells = r["cells"]
        incomplete = [(c, arm) for c, v in cells.items() for arm in ("gnn", "mpoff")
                      if v["completed"].get(arm, 0) < v["expected"].get(arm, 0)]
        d = {arm: median([cell_deficit(v[arm], v["reactive"]) for v in cells.values() if v.get(arm)])
             for arm in ("gnn", "mpoff")}
        wall = median(r.get("wallclock_min", []) or [0.0])
        out["rungs"][tag] = {"servers": servers, "arrivals_per_s": rate, "n_cells": len(cells),
                             "readable": len(cells) >= P3_MIN_CELLS, "incomplete_arms": incomplete,
                             "d_gnn": d["gnn"], "d_mpoff": d["mpoff"], "wallclock_min": wall}
    scaled = [t for t, *_ in P3_RUNGS[1:]]
    # P3-a: a representation that still cannot be served at scale closes the lineage first.
    pinned = [(t, out["rungs"][t]["incomplete_arms"]) for t in scaled
              if out["rungs"][t].get("readable") and out["rungs"][t]["incomplete_arms"]]
    if pinned:
        return {**out, "verdict": V_STILL_PINNED, "pinned": pinned}
    r0 = out["rungs"]["R0"]
    readable = [t for t in scaled if out["rungs"][t].get("readable")]
    if not r0.get("readable") or not readable:
        return {**out, "verdict": V_UNREADABLE,
                "reason": "R0 or every scaled rung is below P3_MIN_CELLS"}
    r3 = out["rungs"]["R3"]
    if r3.get("readable") and r3["wallclock_min"] > P3_MAX_WALLCLOCK_MIN:
        out["r3_affordable"] = False
    breaks = [t for t in readable if out["rungs"][t]["d_gnn"] > r0["d_gnn"] + P3_TOL]
    out["verdict"] = V_DEGRADES if breaks else V_GENERALISES
    out["first_break"] = breaks[0] if breaks else None
    out["readable_scaled"] = readable
    return out


def read_p4(p3: dict) -> dict:
    if p3.get("verdict") != V_GENERALISES:
        return {"verdict": "NOT-READ", "reason": f"P3 = {p3.get('verdict')}"}
    tags = ["R0"] + list(p3["readable_scaled"])
    ds = [p3["rungs"][t]["d_gnn"] for t in tags]
    monotone = all(b <= a for a, b in zip(ds, ds[1:]))
    improved = ds[-1] <= ds[0] - P4_MIN_IMPROVEMENT
    return {"verdict": V_SCALE_HELPS if (monotone and improved) else V_SCALE_NO,
            "d": dict(zip(tags, ds)), "monotone": monotone, "improved": improved}
