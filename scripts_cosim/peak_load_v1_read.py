#!/usr/bin/env python3
"""peak_load_v1 -- every strategy at peak-hour load, 1.5x the mean rate (Azure's hourly peak/mean ~1.6,
Shahrad et al. ATC'20 Fig. 4), on two workloads.

  peak_load_v1_read.py a --ladderjit <dir> --fill <dir> --selection selected.json [--out read.json]
  peak_load_v1_read.py b --gate <groundedx15 dir> --selection selected.json [--out read.json]

(a) the w0 x1.5 draws d1-d4 of burst_ladder_v1 Amendment 1 (reactive, CD, cd_inflight, self-predict, the GNN)
    completed with random, one-pass and Decima; a witness reruns CD and the GNN seed 1 on two cells.
(b) the Alibaba-grounded windows g0-g3 at x1.5, every arm.

Per topology: the median over (window, seed) of the paired % arm vs reference on the same window; exact two-sided
Wilcoxon over topologies; a pair with a missing run is dropped by name; < 8 topologies reads DESIGN-SHORT. No
admissibility filter -- peak hour is the regime; reactive's queue share is reported per cell.
Labels: CONFIRMED (median <= -5 %, p < 0.05) / DIRECTION-ONLY (median < 0, p < 0.05) / REF-FASTER / NOT-SEPARATED.
Primary: the GNN (xs1load_selfref) vs CD.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from statistics import median
from typing import Dict, List, Sequence

from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fresh_topo_burst_v1_read as F  # noqa: E402

GNN = "xs1load_selfref"
LEARNED = ("xs1load_selfref", "xs1load", "gnnedge0", "mpoff")
SEEDS = (1, 2, 3, 4)
BAR, ALPHA, MIN_TOPOLOGIES = 5.0, 0.05, 8
A_WINDOWS = tuple(f"w0x15d{k}" for k in (1, 2, 3, 4))
A_RULES = ("cd", "cd_inflight", "selfpredict", "reactive", "random", "batched", "decima")
B_WINDOWS = tuple(f"g{i}x15" for i in range(4))
B_RULES = ("cd", "selfpredict", "reactive", "random", "batched", "decima")


def label(med: float, p: float) -> str:
    if p < ALPHA and med <= -BAR:
        return "CONFIRMED"
    if p < ALPHA and med < 0:
        return "DIRECTION-ONLY"
    if p < ALPHA and med > 0:
        return "REF-FASTER"
    return "NOT-SEPARATED"


def contrast(s: Dict, topos: Sequence[int], windows: Sequence[str], arm: str, ref: str) -> dict:
    arm_seeds = SEEDS if arm in LEARNED else (0,)
    same_seed = arm in LEARNED and ref in LEARNED
    per, dropped = {}, {}
    for t in topos:
        ds, miss = [], []
        for w in windows:
            for sd in arm_seeds:
                a, b = s.get((t, w, arm, sd)), s.get((t, w, ref, sd if same_seed else 0))
                if a is None or b is None:
                    miss.append(f"{w}/s{sd}")
                    continue
                ds.append(F._pct(F._el(a), F._el(b)))
        if miss:
            dropped[str(t)] = miss
        if ds:
            per[t] = median(ds)
    out = {"arm": arm, "ref": ref, "n_topologies": len(per), "dropped": dropped}
    if len(per) < MIN_TOPOLOGIES:
        return dict(out, label="DESIGN-SHORT")
    xs = [per[t] for t in sorted(per)]
    p = float(wilcoxon(xs, method="exact").pvalue) if any(xs) else 1.0
    med = median(xs)
    return dict(out, median_pct=med, p=p, faster=sum(x < 0 for x in xs), label=label(med, p),
                per_topology={str(t): per[t] for t in sorted(per)})


def shares(s: Dict, topos: Sequence[int], windows: Sequence[str]) -> Dict[str, float]:
    out = {}
    for t in topos:
        for w in windows:
            r = s.get((t, w, "reactive", 0))
            if r:
                out[f"{t}/{w}"] = float(r["averageQueueTime"]) / float(r["averageElapsedTime"])
    return out


def read_a(ladderjit: str, fill: str, topos: Sequence[int]) -> dict:
    old, new = F._summaries(ladderjit), F._summaries(fill)
    wit = {}
    for k, r in new.items():
        if k[2] in ("cd", GNN) and k[1] == "w0x15d1" and k[0] in (9119, 9420) and (k[2] == "cd" or k[3] == 1):
            ref = old.get(k)
            wit[f"{k[0]}/{k[2]}"] = {"rerun": F._el(r), "ladderjit": F._el(ref) if ref else None,
                                    "equal": ref is not None and F._el(r) == F._el(ref)}
    s = dict(old)
    s.update({k: v for k, v in new.items() if k[2] in ("random", "batched", "decima")})
    res = {"W": {"cells": wit, "pass": len(wit) == 4 and all(v["equal"] for v in wit.values())},
           "primary": contrast(s, topos, A_WINDOWS, GNN, "cd"),
           "vs": {r: contrast(s, topos, A_WINDOWS, GNN, r) for r in A_RULES},
           "rules_vs_reactive": {r: contrast(s, topos, A_WINDOWS, r, "reactive") for r in A_RULES if r != "reactive"},
           "reactive_queue_share": shares(s, topos, A_WINDOWS)}
    return res


def read_b(gate: str, topos: Sequence[int]) -> dict:
    s = F._summaries(gate)
    return {"primary": contrast(s, topos, B_WINDOWS, GNN, "cd"),
            "vs": {f"{a}_vs_{r}": contrast(s, topos, B_WINDOWS, a, r) for a in LEARNED for r in B_RULES},
            "mp_twin": contrast(s, topos, B_WINDOWS, "gnnedge0", "mpoff"),
            "rules_vs_reactive": {r: contrast(s, topos, B_WINDOWS, r, "reactive") for r in B_RULES if r != "reactive"},
            "reactive_queue_share": shares(s, topos, B_WINDOWS)}


def _line(name: str, c: dict) -> str:
    return (f"{name:34s} {c.get('median_pct', float('nan')):+7.2f} %  p={c.get('p', float('nan')):.4f}  "
            f"{c.get('faster')}/{c['n_topologies']}  {c['label']}")


def main(argv: List[str] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("part", choices=("a", "b"))
    ap.add_argument("--ladderjit")
    ap.add_argument("--fill")
    ap.add_argument("--gate")
    ap.add_argument("--selection", required=True)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    topos = json.load(open(a.selection))["topologies"]
    if a.part == "a":
        if not (a.ladderjit and a.fill):
            ap.error("part a needs --ladderjit and --fill")
        res = read_a(a.ladderjit, a.fill, topos)
        print(f"witness pass: {res['W']['pass']}", file=sys.stderr)
    else:
        if not a.gate:
            ap.error("part b needs --gate")
        res = read_b(a.gate, topos)
        print(_line("MP twin gnnedge0 vs mpoff", res["mp_twin"]), file=sys.stderr)
    print(_line(f"PRIMARY {GNN} vs cd", res["primary"]), file=sys.stderr)
    for k, c in res["vs"].items():
        print(_line(f"{k}", c), file=sys.stderr)
    for k, c in res["rules_vs_reactive"].items():
        print(_line(f"{k}_vs_reactive", c), file=sys.stderr)
    sh = list(res["reactive_queue_share"].values())
    if sh:
        print(f"reactive queue share: median {median(sh):.2f}, range {min(sh):.2f}-{max(sh):.2f}, n={len(sh)}",
              file=sys.stderr)
    if a.out:
        with open(a.out + ".partial", "w") as fh:
            json.dump(res, fh, indent=1)
        os.replace(a.out + ".partial", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
