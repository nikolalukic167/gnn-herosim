#!/usr/bin/env python3
"""warm_surge_s0_v1 Phase 0a admissibility read (docs/lineages/warm_surge_s0_v1.md). Counts first, then per (rung, k) the cells that pass all of

  reactive queue share <= 0.80 (unknown is not a pass)      the summary's whole-run effective_queue_share (a post-switch share is not in the summary)
  cd backlog not growing                                     mean in-system count over the last quarter of [t_s, last arrival] <= 1.5 x the second quarter (cd .tasks.npy)
  cd end time - last arrival <= 120 s ; requestFailures == 0 (both arms)

A condition is admitted when >= 10 of 12 cells pass. Reactive's own end - last arrival and backlog ratio are printed, descriptive.

  warm_surge_0a_read.py --gate <dir> --topos 19420,...,19431 [--t-switch 1200] [--out read.json]
"""
import argparse
import json
import os
import sys

import numpy as np

RUNGS = ("heavy", "moderate")
KS = (2, 3)
SHARE_MAX, GROW_MAX, TAIL_MAX, NEED = 0.80, 1.5, 120.0, 10


def load(gate, topo, rung, k, arm):
    stem = os.path.join(gate, f"cc40s{topo}__g0{rung}k{k}__{arm}_s0")
    p = stem + ".summary.json"
    return (json.load(open(p)) if os.path.exists(p) else None), stem + ".tasks.npy"


def in_system_ratio(npy, t_s, last):
    r = np.load(npy).astype(np.float64)
    edges = np.linspace(t_s, last, 5)
    grid = np.linspace(t_s, last, 400)
    cnt = np.array([np.count_nonzero((r[:, 0] <= t) & (r[:, 1] > t)) for t in grid])
    q = [cnt[(grid >= edges[i]) & (grid <= edges[i + 1])].mean() for i in range(4)]
    return q[3] / q[1] if q[1] > 0 else None, q


def cell(gate, topo, rung, k, t_s):
    cd, cd_npy = load(gate, topo, rung, k, "cd")
    rx, rx_npy = load(gate, topo, rung, k, "reactive")
    if cd is None or rx is None:
        return {"topo": topo, "pass": False, "why": ["missing summary"]}
    last = float(cd["arrival_end"]["last_arrival_s"])
    ratio, q = in_system_ratio(cd_npy, t_s, last)
    rx_ratio, _ = in_system_ratio(rx_npy, t_s, last)
    share = rx.get("effective_queue_share")
    tail = float(cd["endTime"]) - last
    why = []
    if share is None or share > SHARE_MAX:
        why.append(f"reactive effective_queue_share {share}")
    if ratio is None or ratio > GROW_MAX:
        why.append(f"cd backlog ratio {ratio}")
    if tail > TAIL_MAX:
        why.append(f"cd end - last arrival {tail:.0f}s")
    if cd["requestFailures"] or rx["requestFailures"]:
        why.append(f"requestFailures cd {cd['requestFailures']} reactive {rx['requestFailures']}")
    return {"topo": topo, "pass": not why, "why": why, "reactive_effective_queue_share": share, "cd_backlog_ratio": ratio,
            "cd_tail_s": tail, "reactive_tail_s": float(rx["endTime"]) - last, "reactive_backlog_ratio": rx_ratio,
            "cd_mean_ms": cd["averageElapsedTime"], "reactive_mean_ms": rx["averageElapsedTime"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate", required=True); ap.add_argument("--topos", required=True)
    ap.add_argument("--t-switch", type=float, default=1200.0); ap.add_argument("--out")
    a = ap.parse_args()
    topos = [int(x) for x in a.topos.split(",")]
    n = len([f for f in os.listdir(a.gate) if f.endswith(".summary.json")])
    nf = len([f for f in os.listdir(a.gate) if f.endswith(".failed.json")])
    print(f"{n} summaries, {nf} failed (expected {len(topos) * len(RUNGS) * len(KS) * 2})")
    out = {"summaries": n, "failed": nf, "conditions": {}}
    for rung in RUNGS:
        for k in KS:
            cells = [cell(a.gate, t, rung, k, a.t_switch) for t in topos]
            npass = sum(c["pass"] for c in cells)
            out["conditions"][f"{rung}x{k}"] = {"pass": npass, "admitted": npass >= NEED, "cells": cells}
            print(f"{rung} x{k}: {npass}/{len(topos)} pass -> {'ADMITTED' if npass >= NEED else 'INADMISSIBLE'}")
            for c in cells:
                f = lambda v, p=2: "-" if v is None else f"{v:.{p}f}"
                print(f"   {c['topo']} {'ok ' if c['pass'] else 'NO '} share {f(c.get('reactive_effective_queue_share'))} cd ratio {f(c.get('cd_backlog_ratio'))} "
                      f"cd tail {f(c.get('cd_tail_s'), 0)}s | rx tail {f(c.get('reactive_tail_s'), 0)}s rx ratio {f(c.get('reactive_backlog_ratio'))} "
                      f"| cd {f(c.get('cd_mean_ms'))} rx {f(c.get('reactive_mean_ms'))} {'; '.join(c['why'])}")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
