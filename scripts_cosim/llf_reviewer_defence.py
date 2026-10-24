#!/usr/bin/env python3
"""Reviewer-defence analysis on the ALREADY-READ sealed-12 cells of live_link_features_v1 (descriptive; no label, no new cell).

Cells: the parent confirmation's grid, read through the parent reader's own loader (`llf_confirm_read.load`: pins, 24 servers, the expected grid and nothing else) from the
view `d_conf_primaryA` (static e40 gnn / wide / same-input MLP from d_conf_final; the bar from d_conf). Windows g0 and g1 only: no g2 / g3 cell of these arms exists (and none is run
here). Statistic and runaway rule are the parent's (`confirm_rulings.json`): per topology the median over the windows of the paired % (a learned cell = the mean over its seeds),
then the median over the 12 topologies, the faster count and the exact two-sided Wilcoxon; "all" = every cell, "no_runaway" = a pair is dropped when either cell is a runaway
(> 10 x the arm's median in the rung, any seed flags the cell; the flag is computed on the whole-run mean, as in the parent, and re-used for every statistic below).

  1. Tail. p50 / p95 / p99 of the per-task latency (completion - arrival, `tasks.npy`) and of the per-request completion time of fan-out requests (workload `peer_group`: latest member
     completion minus first member arrival; `tasks_bsz.npy` columns, as the 2026-10-22 request-level read), gnn vs bar, vs wide MLP, vs same-input MLP.
  2. Decision charge. The share of the mean latency that is charged decision time: `schedulerCounters.sched_felt_task_seconds / num_tasks` (the seconds every task felt of the scheduler's
     charged decision time; the charge sits in the placement wait) over `averageElapsedTime`, per arm and rung. Then the "free decisions" counterfactual: each cell's whole-run mean minus its
     own felt decision seconds per task, and gnn vs wide / vs bar / vs mlp recomputed on it. FIRST-ORDER only: it removes the delay a task is charged and not the second-order effect of
     the charge (later placements, queues and link sharing that the delay moves).
  3. Mean-latency decomposition per arm and rung (seconds per task): placement wait (of which decision) , queue, cold start, pull, init, execution, peer exchange, rendezvous wait,
     communications, network latency; the residual to the mean is reported (the simulator's components overlap: compute time contains the exchange, cold start, pull and execution).

  llf_reviewer_defence.py --cells D_CONF_PRIMARYA --inputs IN --pin static=02668ecf,refs=0ec7cdcb,bar=0ec7cdcb --rulings confirm_rulings.json --out-dir DIR
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import statistics as st
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llf_confirm_read as R  # noqa: E402

PCTS = (50, 95, 99)
SHOW = ("gnn", "wide", "mlp", "bar")
PAIRS = (("gnn", "bar"), ("gnn", "wide"), ("gnn", "mlp"))
COMPONENTS = ("placement_wait", "decision_felt", "placement_wait_other", "queue", "cold_start", "pull", "init", "execution", "exchange", "rendezvous", "communications", "network_latency",
              "compute", "lock_wait")
NAME = R.NAME


def _pcts(v: np.ndarray, pre: str) -> Dict[str, float]:
    return {f"{pre}_p{p}": float(np.percentile(v, p)) for p in PCTS}


class Workloads:
    def __init__(self, inputs: str):
        self.inputs, self.cache = inputs, {}

    def get(self, rung: str, w: str):
        if (rung, w) not in self.cache:
            ev = json.load(open(f"{self.inputs}/wf1_{rung}/wl/grounded_{w}_n50000.json"))["events"]
            ts = np.array([e["timestamp"] for e in ev])
            g = np.array([e["peer_group"] for e in ev])
            _u, inv, cnt = np.unique(g, return_inverse=True, return_counts=True)
            self.cache[(rung, w)] = (ts, inv, cnt)
        return self.cache[(rung, w)]


def cell_extras(stem: str, s: Dict[str, Any], wl: Workloads, rung: str, w: str) -> Tuple[Dict[str, float], Dict[str, float]]:
    """(statistics added to the grid value, decomposition) of one cell."""
    f32 = np.load(stem + ".tasks.npy").astype(np.float64)
    b = np.load(stem + ".tasks_bsz.npy").astype(np.float64)
    n = int(s["num_tasks"])
    if len(f32) != n or len(b) != n:
        raise SystemExit(f"FAIL LOUD: {stem}: sidecar rows {len(f32)}/{len(b)} != num_tasks {n}")
    el = f32[:, 1] - f32[:, 0]
    if abs(el.mean() - s["averageElapsedTime"]) > 1e-6 * max(1.0, s["averageElapsedTime"]):
        raise SystemExit(f"FAIL LOUD: {stem}: mean(col1 - col0) {el.mean()} is not averageElapsedTime {s['averageElapsedTime']}")
    ts, inv, cnt = wl.get(rung, w)
    ids = b[:, 0].astype(int)
    if not np.array_equal(np.sort(ids), np.arange(n)):
        raise SystemExit(f"FAIL LOUD: {stem}: task ids are not a permutation of the workload events")
    inv, size = inv[ids], cnt[inv[ids]]
    if not np.array_equal(b[:, 3].astype(int), size):
        raise SystemExit(f"FAIL LOUD: {stem}: groupSize != the workload's group size")
    arr, done = b[:, 1], b[:, 2]
    first = np.full(len(cnt), np.inf)
    last = np.full(len(cnt), -np.inf)
    np.minimum.at(first, inv, arr)
    np.maximum.at(last, inv, done)
    req = (last - first)[cnt >= 2]
    felt = float((s.get("schedulerCounters") or {}).get("sched_felt_task_seconds", 0.0)) / n
    stats = {**_pcts(el, "t"), **_pcts(req, "r"), "whole_free": float(s["averageElapsedTime"]) - felt, "decision_felt": felt}
    pw = s["placement_wait"]["mean"]
    dec = {"placement_wait": float(pw), "decision_felt": felt, "placement_wait_other": float(pw) - felt, "queue": float(s["averageQueueTime"]),
           "cold_start": float(s["averageColdStartTime"]), "pull": float(s["averagePullTime"]), "init": float(s["averageInitializationTime"]),
           "execution": float(s["averageExecutionTime"]), "exchange": float(s["totalPeerExchangeTime"]) / n, "rendezvous": float(s["totalPeerRendezvousWait"]) / n,
           "communications": float(s["averageCommunicationsTime"]), "network_latency": float(s["averageNetworkLatency"]), "compute": float(s["averageComputeTime"]),
           "lock_wait": float((s.get("lock_wait") or {}).get("mean", 0.0)), "mean": float(s["averageElapsedTime"])}
    return stats, dec


def augment(V: Dict[Tuple, Dict[str, float]], root: str, inputs: str) -> Dict[Tuple, Dict[str, float]]:
    """Add the extra statistics to the reader's grid values and return the decomposition per cell. The cells are the same files the reader loaded."""
    group_kinds = {R.ARMS[a]: a for a in R.ARMS}
    wl = Workloads(inputs)
    D: Dict[Tuple, Dict[str, float]] = {}
    for f in sorted(glob.glob(os.path.join(root, "*", "cc40s*.summary.json"))):
        m = NAME.match(os.path.basename(f))
        key = (m.group(4), int(m.group(5)), int(m.group(1)), m.group(2), m.group(3))
        if key not in V or group_kinds[key[0]] not in SHOW:
            continue
        s = json.load(open(f))
        stats, dec = cell_extras(f[:-len(".summary.json")], s, wl, m.group(3), m.group(2))
        V[key].update(stats)
        D[key] = dec
    return D


def clean(row: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in row.items() if k != "per_topology"} | {"per_topology": {str(t): v for t, v in row["per_topology"].items()}}


def tail_block(g: R.Grid) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for scope, pre in (("task", "t"), ("request_fanout", "r")):
        for p in PCTS:
            for a, b in PAIRS:
                for rung in R.RUNGS:
                    for reading in ("all", "no_runaway"):
                        out[f"{scope}|p{p}|{a}_vs_{b}|{rung}|{reading}"] = clean(g.contrast(a, b, rung, f"{pre}_p{p}", reading))
    return out


def free_block(g: R.Grid) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for stat in ("whole", "whole_free"):
        for a, b in PAIRS:
            for rung in R.RUNGS:
                for reading in ("all", "no_runaway"):
                    out[f"{stat}|{a}_vs_{b}|{rung}|{reading}"] = clean(g.contrast(a, b, rung, stat, reading))
    return out


def share_block(V: Dict[Tuple, Dict[str, float]], D: Dict[Tuple, Dict[str, float]], g: R.Grid) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for a in SHOW:
        for rung in R.RUNGS:
            keys = [(R.ARMS[a], s, t, w, rung) for s in g.seeds(a) for t in g.topos for w in R.WINDOWS]
            felt = [D[k]["decision_felt"] for k in keys]
            mean = [D[k]["mean"] for k in keys]
            share = [d / m for d, m in zip(felt, mean)]
            ok = [i for i, k in enumerate(keys) if not g.seed_flagged(a, k[1], k[2], k[3], rung, "whole")]
            out[f"{a}|{rung}"] = {"cells": len(keys), "felt_s_per_task_mean": st.fmean(felt), "felt_s_per_task_median": st.median(felt), "mean_latency_s": st.fmean(mean),
                                  "share_of_mean_latency_pooled": sum(felt) / sum(mean), "share_median_cell": st.median(share), "share_p90_cell": sorted(share)[int(0.9 * len(share))],
                                  "cells_without_runaway": len(ok), "share_pooled_without_runaway": (sum(felt[i] for i in ok) / sum(mean[i] for i in ok)) if ok else None}
    return out


def decomposition_block(D: Dict[Tuple, Dict[str, float]], g: R.Grid) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for a in SHOW:
        for rung in R.RUNGS:
            keys = [(R.ARMS[a], s, t, w, rung) for s in g.seeds(a) for t in g.topos for w in R.WINDOWS]
            ok = [k for k in keys if not g.seed_flagged(a, k[1], k[2], k[3], rung, "whole")]
            row: Dict[str, Any] = {"cells": len(keys), "cells_without_runaway": len(ok)}
            for name, ks in (("all_cells_mean", keys), ("no_runaway_mean", ok)):
                if not ks:
                    continue
                comp = {c: st.fmean(D[k][c] for k in ks) for c in COMPONENTS + ("mean",)}
                comp["residual_mean_minus_sum_of(placement_wait,queue,compute,communications,network_latency)"] = comp["mean"] - (comp["placement_wait"] + comp["queue"] + comp["compute"] + comp["communications"] + comp["network_latency"])
                row[name] = comp
            row["all_cells_median"] = {c: st.median(D[k][c] for k in keys) for c in COMPONENTS + ("mean",)}
            out[f"{a}|{rung}"] = row
    return out


def summary(res: Dict[str, Any]) -> str:
    f = lambda x: "n/a" if x is None else f"{x:+.2f}"
    T, F, S = res["tail"], res["free_decisions"], res["decision_share"]
    line = lambda d: f"{f(d['median_pct'])} % ({d['faster']}/{d['n']}, p {('%.3g' % d['p']) if d['p'] is not None else 'n/a'})"
    out = ["reviewer-defence analysis, sealed-12, windows g0+g1 (no g2/g3 cell of these arms exists), seeds 1-3, per-topology median over windows then median over 12 topologies, exact Wilcoxon;"
           " 'no runaway' = pairs with a flagged cell dropped"]
    for rung in R.RUNGS:
        out.append(f"{rung}: mean gnn vs bar {line(F['whole|gnn_vs_bar|%s|no_runaway' % rung])} no-runaway, {line(F['whole|gnn_vs_bar|%s|all' % rung])} all; vs wide {line(F['whole|gnn_vs_wide|%s|no_runaway' % rung])} / {line(F['whole|gnn_vs_wide|%s|all' % rung])}"
                   f"; free decisions: vs bar {line(F['whole_free|gnn_vs_bar|%s|no_runaway' % rung])}, vs wide {line(F['whole_free|gnn_vs_wide|%s|no_runaway' % rung])} (no runaway)")
        out.append(f"{rung}: task p50/p95/p99 gnn vs wide (no runaway) {line(T['task|p50|gnn_vs_wide|%s|no_runaway' % rung])} / {line(T['task|p95|gnn_vs_wide|%s|no_runaway' % rung])} / {line(T['task|p99|gnn_vs_wide|%s|no_runaway' % rung])};"
                   f" vs bar {line(T['task|p50|gnn_vs_bar|%s|no_runaway' % rung])} / {line(T['task|p95|gnn_vs_bar|%s|no_runaway' % rung])} / {line(T['task|p99|gnn_vs_bar|%s|no_runaway' % rung])}")
        out.append(f"{rung}: request (fan-out) p50/p99 gnn vs bar (no runaway) {line(T['request_fanout|p50|gnn_vs_bar|%s|no_runaway' % rung])} / {line(T['request_fanout|p99|gnn_vs_bar|%s|no_runaway' % rung])}; vs wide {line(T['request_fanout|p50|gnn_vs_wide|%s|no_runaway' % rung])} / {line(T['request_fanout|p99|gnn_vs_wide|%s|no_runaway' % rung])}")
        out.append(f"{rung}: charged decision time, median cell share of the mean latency (felt ms/task): " + ", ".join(f"{a} {100 * S[a + '|' + rung]['share_median_cell']:.2f} % ({1000 * S[a + '|' + rung]['felt_s_per_task_mean']:.0f})" for a in SHOW))
    Dc = res["decomposition"]
    out.append("decomposition, s/task, mean over cells without runaway (gnn / wide / mlp / bar): " + "; ".join(
        f"{rung} mean " + " / ".join(f"{Dc[a + '|' + rung]['no_runaway_mean']['mean']:.2f}" for a in SHOW) + ", exchange " + " / ".join(f"{Dc[a + '|' + rung]['no_runaway_mean']['exchange']:.2f}" for a in SHOW)
        + ", decision " + " / ".join(f"{Dc[a + '|' + rung]['no_runaway_mean']['decision_felt']:.3f}" for a in SHOW) for rung in R.RUNGS))
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", required=True)
    ap.add_argument("--inputs", required=True, help="the IN dir with wf1_<rung>/wl/grounded_<w>_n50000.json (the workloads are identical across the confirmation's IN dirs)")
    ap.add_argument("--pin", required=True)
    ap.add_argument("--rulings", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    sha = hashlib.sha256(",".join(map(str, R.IDS)).encode()).hexdigest()
    if sha != R.IDS_SHA256:
        raise SystemExit("FAIL LOUD: the id list is not the sealed one")
    rul = R.check_rulings(json.load(open(a.rulings)))
    V, pins, pins_by_group = R.load(a.cells, R.IDS, a.pin)
    D = augment(V, a.cells, a.inputs)
    g = R.Grid(V, R.IDS, rul)
    res = {"family": "llf_cd_confirm_v1 reviewer-defence analysis on the sealed-12 (descriptive, no label)", "topologies": list(R.IDS), "windows": list(R.WINDOWS), "ids_sha256": sha,
           "pins_by_group": pins_by_group, "rulings": rul, "arms": {k: R.ARMS[k] for k in SHOW}, "cells_read": len(D), "tail": tail_block(g), "decision_share": share_block(V, D, g),
           "free_decisions": free_block(g), "decomposition": decomposition_block(D, g), "runaway_flags_whole": [r for r in g.runaways() if r["stat"] == "whole" and r["arm"] in SHOW],
           "scripts_sha256": hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()}
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, "reviewer_defence.json")
    with open(out + ".tmp", "w") as fh:
        json.dump(res, fh, indent=1, default=str)
    os.replace(out + ".tmp", out)
    txt = summary(res)
    open(os.path.join(a.out_dir, "summary.txt"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
