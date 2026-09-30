#!/usr/bin/env python3
"""peak_controls_v1 -- the two controls for peak_load_v1's ladder result, on the same x2 / x3 / x5 grounded cells.

  peak_controls_v1_read.py --ladder <groundedladder dir> --ctl <peakctl dir> [--mlp <peakmlp dir>]
                           --selection selected.json [--out read.json]

C1 (objective): CD with the co-sim label's queueing externality added to its score, at the label's rate (cdext) and
   at the rung's offered rate (cdextr). Does the GNN still beat search once search optimises the same objective?
C2 (model class): xs1mpoff_selfref, the MP-OFF twin of the GNN (same corpus, split, labels, features, self-refine).
   Does the graph model beat a pointwise scorer with identical information? Paired on the checkpoint seed.

Statistic as peak_load_v1: per topology the median over (window, seed) of the paired % arm vs reference on the same
window; exact two-sided Wilcoxon over topologies; labels CONFIRMED / DIRECTION-ONLY / REF-FASTER / NOT-SEPARATED.
Holm over the three rungs within each contrast family. Witness: CD rerun at this commit on two x3 cells must equal
the groundedladder runs to the digit.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from statistics import mean, median
from typing import Dict, List, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fresh_topo_burst_v1_read as F  # noqa: E402
import peak_load_v1_read as P  # noqa: E402
from peak_load_v1_read import contrast, label  # noqa: E402

GNN = "xs1load_selfref"
MLP = "xs1mpoff_selfref"
P.LEARNED = tuple(P.LEARNED) + (MLP,)  # contrast() pairs learned arms on the checkpoint seed
RUNGS = ("x20", "x30", "x50")
LEARNED = (GNN, MLP)
FAMILIES = {
    "gnn_vs_cdext": (GNN, "cdext"),
    "gnn_vs_cdextr": (GNN, "cdextr"),
    "gnn_vs_mlp": (GNN, MLP),
    "mlp_vs_cd": (MLP, "cd"),
    "cdext_vs_cd": ("cdext", "cd"),
    "cdextr_vs_cd": ("cdextr", "cd"),
    "gnn_vs_cd": (GNN, "cd"),
}
DESCRIBE = (GNN, MLP, "cd", "cdext", "cdextr", "batched", "decima", "reactive", "random", "selfpredict")
WITNESS = ((9119, "g0x30"), (9420, "g0x30"))


def holm(ps: Dict[str, float]) -> Dict[str, float]:
    order = sorted(ps, key=lambda k: ps[k])
    out, run = {}, 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(order) - i) * ps[k]))
        out[k] = run
    return out


def describe(s: Dict, topos: Sequence[int], windows: Sequence[str], arm: str) -> dict:
    rows = [r for (t, w, k, _sd), r in s.items() if k == arm and t in topos and w in windows]
    if not rows:
        return {"n": 0}
    el = [F._el(r) for r in rows]
    q = [float(r["averageQueueTime"]) for r in rows]
    ex = [float(r.get("totalPeerExchangeTime") or 0.0) / float(r["num_tasks"]) for r in rows]
    return {"n": len(rows), "mean_latency": mean(el), "median_latency": median(el),
            "mean_queue": mean(q), "median_queue": median(q),
            "mean_exchange_per_task": mean(ex), "median_exchange_per_task": median(ex)}


def per_seed(s: Dict, topos: Sequence[int], windows: Sequence[str], arm: str, ref: str) -> dict:
    out = {}
    for sd in (1, 2, 3, 4):
        sub = {k: v for k, v in s.items() if k[2] != arm or k[3] == sd}
        c = contrast(sub, topos, windows, arm, ref)
        out[str(sd)] = {k: c.get(k) for k in ("median_pct", "p", "faster", "n_topologies", "label")}
    return out


def read(ladder: str, ctl: str, mlp: str, topos: Sequence[int]) -> dict:
    base, new = F._summaries(ladder), F._summaries(ctl)
    wit = {}
    for t, w in WITNESS:
        a, b = new.get((t, w, "cd", 0)), base.get((t, w, "cd", 0))
        wit[f"{t}/{w}"] = {"rerun": F._el(a) if a else None, "groundedladder": F._el(b) if b else None,
                           "equal": a is not None and b is not None and F._el(a) == F._el(b)}
    s = dict(base)
    s.update({k: v for k, v in new.items() if k[2] in ("cdext", "cdextr")})
    if mlp:
        s.update({k: v for k, v in F._summaries(mlp).items() if k[2] == MLP})
    res = {"witness": {"cells": wit, "pass": all(v["equal"] for v in wit.values())}, "rungs": {}, "holm": {}}
    for r in RUNGS:
        ws = tuple(f"g{i}{r}" for i in range(4))
        res["rungs"][r] = {
            "contrasts": {name: contrast(s, topos, ws, a, b) for name, (a, b) in FAMILIES.items()
                          if any(k[2] == a for k in s) and any(k[2] == b for k in s)},
            "describe": {arm: describe(s, topos, ws, arm) for arm in DESCRIBE},
        }
        if any(k[2] == MLP for k in s):
            res["rungs"][r]["mlp_per_seed_vs_cd"] = per_seed(s, topos, ws, MLP, "cd")
    for name in FAMILIES:
        ps = {r: res["rungs"][r]["contrasts"][name]["p"] for r in RUNGS
              if "p" in res["rungs"][r]["contrasts"].get(name, {})}
        if len(ps) == len(RUNGS):
            adj = holm(ps)
            res["holm"][name] = {r: {"p": ps[r], "p_holm": adj[r],
                                     "label_holm": label(res["rungs"][r]["contrasts"][name]["median_pct"], adj[r])}
                                 for r in RUNGS}
    return res


def main(argv: List[str] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ladder", required=True)
    ap.add_argument("--ctl", required=True)
    ap.add_argument("--mlp")
    ap.add_argument("--selection", required=True)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    topos = json.load(open(a.selection))["topologies"]
    res = read(a.ladder, a.ctl, a.mlp, topos)
    print(f"witness pass: {res['witness']['pass']}  {res['witness']['cells']}", file=sys.stderr)
    for r in RUNGS:
        print(f"--- {r}", file=sys.stderr)
        for name, c in res["rungs"][r]["contrasts"].items():
            ph = (res["holm"].get(name) or {}).get(r, {}).get("p_holm", math.nan)
            print(f"{name:16s} {c.get('median_pct', math.nan):+7.2f} %  p={c.get('p', math.nan):.4f}  "
                  f"holm={ph:.4f}  {c.get('faster')}/{c['n_topologies']}  {c['label']}", file=sys.stderr)
        for arm, d in res["rungs"][r]["describe"].items():
            if d["n"]:
                print(f"   {arm:18s} n={d['n']:3d} mean={d['mean_latency']:8.2f} median={d['median_latency']:8.2f} "
                      f"queue={d['median_queue']:8.2f} exch={d['median_exchange_per_task']:.3f}", file=sys.stderr)
    if a.out:
        with open(a.out + ".partial", "w") as fh:
            json.dump(res, fh, indent=1)
        os.replace(a.out + ".partial", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
