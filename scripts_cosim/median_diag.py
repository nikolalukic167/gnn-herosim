#!/usr/bin/env python3
"""Median-latency diagnostic (DESCRIPTIVE, existing cells only; no new cell, no verdict re-read).

Where does the FL gnn's slower median task come from, against the in-flight bar `cd_ib0_inflightfull_k100`?
Per task (paired by task id, bucketed by the bar's groupSize, tasks_bsz col 3), the cells record dispatch, done, ioEnd and computeStart, so the
latency splits exactly into three spans:
    A = ioEnd - dispatch       scheduler wait + batch assembly + decision charge + cold start / pull / init + ingress + input and peer transfer
    L = computeStart - ioEnd   lock wait (tasks_lock col 3)
    P = done - computeStart    compute + output
Cold start, execution, communications, peer exchange, network latency and decision time are per CELL means (summary.json), not per task.
A learned arm's per-task values are the mean of its seeds 1 and 2 (as hybrid_screen.py)."""
import json, os, re, sys
import numpy as np

U = "/share/nikola.lukic/uncut_groups"; C = f"{U}/comb"; OUT = sys.argv[1]
TOPOS = range(19537, 19549); WINDOWS = ("g0", "g1", "g2", "g3"); RUNGS = ("moderate", "heavy")
ARMS = {"gnn": ("cells_comb", "ra_gnn_eng_combfl_nosplit_k100", (1, 2)), "mlp": ("cells_comb", "ra_mlp_same_combfl_nosplit_k100", (1, 2)),
        "bar": ("cells_bar_g8", "cd_ib0_inflightfull_k100", (0,))}
BK = (("le10", 0, 10), ("11_50", 11, 50), ("51_100", 51, 100), ("101_256", 101, 256))
T_OPEN = 360.0
QS = (10, 25, 50, 75, 90, 99)
CELL_KEYS = ("cold_start_pct", "averageColdStartTime", "averageInitializationTime", "averagePullTime", "averageExecutionTime", "averageComputeTime",
             "averageCommunicationsTime", "averageNetworkLatency", "averageQueueTime", "averageWaitTime", "totalPeerExchangeTime", "totalPeerRendezvousWait", "averageElapsedTime")


def path(arm, t, w, r, s):
    d, kind, _ = ARMS[arm]
    return f"{C}/{d}/cc40s{t}__{w}{r}__{kind}_s{s}"


def load_cell(arm, t, w, r):
    """(summary dicts per seed, bsz arrays per seed, lock arrays per seed)"""
    sm, bz, lk = [], [], []
    for s in ARMS[arm][2]:
        p = path(arm, t, w, r, s)
        sm.append(json.load(open(p + ".summary.json"))); z, k = np.load(p + ".tasks_bsz.npy"), np.load(p + ".tasks_lock.npy")
        if not (z[:, 0] == k[:, 0]).all() or np.isnan(k).any():
            raise SystemExit(f"FAIL LOUD: {p}: tasks_bsz / tasks_lock rows differ or are unstamped")
        o = np.argsort(z[:, 0], kind="stable")      # rows are in dispatch order, which differs per arm and seed: pair by task id
        if len(np.unique(z[:, 0])) != len(z):
            raise SystemExit(f"FAIL LOUD: {p}: duplicate task ids")
        bz.append(z[o]); lk.append(k[o])
    return sm, bz, lk


def spans(bz, lk):
    """per-task A, L, P, total, dispatch (seed-mean for several seeds), keyed by the row order; ids returned for pairing"""
    ids = bz[0][:, 0]
    for b in bz[1:]:
        if not (b[:, 0] == ids).all():
            raise SystemExit("FAIL LOUD: task ids differ between seeds")
    A = np.mean([l[:, 1] - b[:, 1] for b, l in zip(bz, lk)], axis=0)
    L = np.mean([l[:, 3] for l in lk], axis=0)
    tot = np.mean([b[:, 2] - b[:, 1] for b in bz], axis=0)
    disp = np.mean([b[:, 1] for b in bz], axis=0)
    return ids, A, L, tot - A - L, tot, disp


def cell_vals(summaries):
    out = {}
    for k in CELL_KEYS:
        out[k] = float(np.mean([s[k] for s in summaries if s.get(k) is not None]))
    out["placement_wait_mean"] = float(np.mean([s["placement_wait"]["mean"] for s in summaries]))
    tr = [sum(v["transfers"] for v in s["peerExchangeByAccessClass"].values()) for s in summaries]
    sec = [sum(v["seconds"] for v in s["peerExchangeByAccessClass"].values()) for s in summaries]
    out["cross_node_transfers"] = float(np.mean(tr)); out["peer_seconds_per_transfer"] = float(np.mean(sec) / np.mean(tr))
    out["pairs_in_batch"] = float(np.mean([s["schedulerCounters"].get("prefix_pairs_in_batch", 0) for s in summaries]))
    out["peers_outside_batch"] = float(np.mean([s["schedulerCounters"].get("prefix_peers_outside_batch", 0) for s in summaries]))
    dt = [s["decisionTiming"] for s in summaries]
    out["decision_ms_per_call_le10"] = float(np.mean([1000 * d["by_bucket"]["le10"]["total_s"] / max(1, d["by_bucket"]["le10"]["calls"]) for d in dt]))
    out["plan_evals_le10"] = float(np.mean([d["by_bucket"]["le10"]["plan_evals"] for d in dt]))
    rs = [s.get("replica_count_series") or {} for s in summaries]
    vals = []
    for x in rs:
        for k, v in x.items():
            if k != "t" and isinstance(v, list) and v and all(isinstance(y, (int, float)) for y in v):
                vals.append(float(np.mean(v)))
    out["replica_count_mean"] = float(np.mean(vals)) if vals else None
    return out


def agg(rung):
    acc = {a: {b[0]: {"A": [], "L": [], "P": [], "T": [], "D": []} for b in BK} for a in ARMS}
    cells = {a: [] for a in ARMS}
    for t in TOPOS:
        for w in WINDOWS:
            data = {a: load_cell(a, t, w, rung) for a in ARMS}
            sp = {a: spans(data[a][1], data[a][2]) for a in ARMS}
            ref = sp["bar"][0]
            for a in ARMS:
                if not (sp[a][0] == ref).all():
                    raise SystemExit(f"FAIL LOUD: ids differ {a} {t} {w} {rung}")
            gs = data["bar"][1][0][:, 3].astype(int)
            for a in ARMS:
                cells[a].append(cell_vals(data[a][0]))
                for name, lo, hi in BK:
                    m = (gs >= lo) & (gs <= hi)
                    for key, v in zip("ALPTD", sp[a][1:]):
                        acc[a][name][key].append(v[m].astype(np.float32))
    return {a: {b: {k: np.concatenate(v) for k, v in d.items()} for b, d in acc[a].items()} for a in ARMS}, cells


def block(d, mask_fn):
    """component table for one arm over the tasks passing mask_fn(D)"""
    m = mask_fn(d["D"])
    A, L, P, T = (d[k][m] for k in "ALPT")
    lo, hi = np.percentile(T, 45), np.percentile(T, 55)
    near = (T >= lo) & (T <= hi)
    return {"n": int(m.sum()),
            "mean": {"A": float(A.mean()), "L": float(L.mean()), "P": float(P.mean()), "total": float(T.mean())},
            "p50": {"A": float(np.median(A)), "L": float(np.median(L)), "P": float(np.median(P)), "total": float(np.median(T))},
            "typical_task_mean(total in own p45-p55)": {"A": float(A[near].mean()), "L": float(L[near].mean()), "P": float(P[near].mean()), "total": float(T[near].mean())},
            "total_quantiles": {f"p{q}": float(np.percentile(T, q)) for q in QS},
            "share_lock_wait_gt_10ms": float((L > 0.01).mean())}


def paired(g, b, mask_fn):
    m = mask_fn(b["D"])
    out = {}
    for k, nm in (("T", "total"), ("A", "A_dispatch_to_ioEnd"), ("L", "L_lock_wait"), ("P", "P_compute_post")):
        dlt = g[k][m].astype(np.float64) - b[k][m]
        out[nm] = {"median_delta": float(np.median(dlt)), "mean_delta": float(dlt.mean()), "share_gnn_slower": float((dlt > 1e-9).mean()), "share_gnn_faster": float((dlt < -1e-9).mean())}
    return out


def cellsum(cells):
    keys = cells[0].keys()
    return {k: (float(np.mean([c[k] for c in cells])) if cells[0][k] is not None else None) for k in keys}


def main():
    res = {}
    lines = []
    for r in RUNGS:
        acc, cells = agg(r)
        res[r] = {"buckets": {}, "cells": {}}
        res[r]["cells"] = {a: cellsum(cells[a]) for a in ARMS}
        res[r]["cells_paired_vs_bar"] = {a: {k: float(np.median([c[k] / cb[k] for c, cb in zip(cells[a], cells["bar"]) if cb[k]])) for k in ("cold_start_pct", "cross_node_transfers", "peer_seconds_per_transfer", "totalPeerExchangeTime", "averageCommunicationsTime", "placement_wait_mean")} for a in ("gnn", "mlp")}
        for name, _, _ in BK:
            res[r]["buckets"][name] = {}
            for fname, fn in (("all", lambda D: np.ones(len(D), bool)), ("after360", lambda D: D >= T_OPEN)):
                blocks = {a: block(acc[a][name], fn) for a in ARMS}
                pr = {a: paired(acc[a][name], acc["bar"][name], fn) for a in ("gnn", "mlp")}
                res[r]["buckets"][name][fname] = {"arms": blocks, "paired_vs_bar": pr}
                for a in ("gnn", "bar"):
                    b = blocks[a]
                    lines.append(f"{r:8s} {name:8s} {fname:8s} {a:4s} n={b['n']:8d} p50 tot {b['p50']['total']:.3f} (A {b['p50']['A']:.3f} L {b['p50']['L']:.3f} P {b['p50']['P']:.3f})  mean tot {b['mean']['total']:.3f} (A {b['mean']['A']:.3f} L {b['mean']['L']:.3f} P {b['mean']['P']:.3f})  typical (A {b['typical_task_mean(total in own p45-p55)']['A']:.3f} L {b['typical_task_mean(total in own p45-p55)']['L']:.3f} P {b['typical_task_mean(total in own p45-p55)']['P']:.3f})  q " + " ".join(f"{k}={v:.2f}" for k, v in b["total_quantiles"].items()))
                p = pr["gnn"]
                lines.append(f"{r:8s} {name:8s} {fname:8s} gnn-bar median delta: tot {p['total']['median_delta']:+.3f} A {p['A_dispatch_to_ioEnd']['median_delta']:+.3f} L {p['L_lock_wait']['median_delta']:+.3f} P {p['P_compute_post']['median_delta']:+.3f}   mean delta: tot {p['total']['mean_delta']:+.3f} A {p['A_dispatch_to_ioEnd']['mean_delta']:+.3f} L {p['L_lock_wait']['mean_delta']:+.3f} P {p['P_compute_post']['mean_delta']:+.3f}   gnn slower on {100 * p['total']['share_gnn_slower']:.1f}% of tasks")
        lines.append(f"{r} CELL MEANS (mean over 48 cells): " + json.dumps({a: {k: (round(v, 4) if v is not None else None) for k, v in res[r]['cells'][a].items()} for a in ARMS}))
        lines.append(f"{r} PAIRED PER-CELL RATIO vs bar (median over 48 cells): " + json.dumps({a: {k: round(v, 4) for k, v in d.items()} for a, d in res[r]["cells_paired_vs_bar"].items()}))
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f"{OUT}/median_diag.json", "w"), indent=1)
    open(f"{OUT}/median_diag.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


main()
