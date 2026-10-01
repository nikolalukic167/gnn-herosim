#!/usr/bin/env python3
"""peak_controls_v1 Amendment 1 -- the MP-OFF twin on the bursty w0 draws, against the GNN's own w0 results.

  peak_controls_v1_w0_read.py --ladderjit <dir> --capacity <dir> --x11confirm <dir> --x15fill <dir> --w0mlp <dir>
                              --selection selected.json [--out read.json]

Per rung (x1.5: draws w0x15d1-4, GNN seeds 1-4 from ladderjit; x1.1: draws w0x11d1-4, GNN seed 1 from capacity and
seeds 2-4 from x11confirm): GNN vs MLP (paired on the seed), MLP vs CD, GNN vs CD, MLP vs every other rule present.
Statistic, labels and Holm over the two rungs as peak_controls_v1_read. Witness: the GNN seed 1 rerun on 9119 / 9420
w0x15d1 must equal ladderjit to the digit.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fresh_topo_burst_v1_read as F  # noqa: E402
from peak_controls_v1_read import GNN, MLP, describe, holm  # noqa: E402
from peak_load_v1_read import contrast, label  # noqa: E402

RUNGS = ("x15", "x11")
RULES = ("cd", "cd_inflight", "selfpredict", "reactive", "random", "batched", "decima")


def main(argv: List[str] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("ladderjit", "capacity", "x11confirm", "x15fill", "w0mlp", "selection"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    topos = json.load(open(a.selection))["topologies"]
    base = F._summaries(a.ladderjit)
    s = dict(base)
    s.update(F._summaries(a.capacity))
    s.update(F._summaries(a.x11confirm))
    s.update({k: v for k, v in F._summaries(a.x15fill).items() if k[2] in ("random", "batched", "decima")})
    new = F._summaries(a.w0mlp)
    wit = {}
    for t in (9119, 9420):
        r, b = new.get((t, "w0x15d1", GNN, 1)), base.get((t, "w0x15d1", GNN, 1))
        wit[str(t)] = {"rerun": F._el(r) if r else None, "ladderjit": F._el(b) if b else None,
                       "equal": bool(r and b and F._el(r) == F._el(b))}
    s.update({k: v for k, v in new.items() if k[2] == MLP})
    fams = {"gnn_vs_mlp": (GNN, MLP), "gnn_vs_cd": (GNN, "cd")}
    fams.update({f"mlp_vs_{r}": (MLP, r) for r in RULES})
    res = {"witness": {"cells": wit, "pass": all(v["equal"] for v in wit.values())}, "rungs": {}, "holm": {}}
    for r in RUNGS:
        ws = tuple(f"w0{r}d{k}" for k in (1, 2, 3, 4))
        res["rungs"][r] = {"contrasts": {n: contrast(s, topos, ws, x, y) for n, (x, y) in fams.items()
                                         if any(k[2] == y and k[1] in ws for k in s)},
                           "describe": {arm: describe(s, topos, ws, arm) for arm in (GNN, MLP, "cd", "reactive")}}
    for n in fams:
        ps = {r: res["rungs"][r]["contrasts"].get(n, {}).get("p") for r in RUNGS}
        ps = {r: p for r, p in ps.items() if p is not None}
        if ps:
            adj = holm(ps)
            res["holm"][n] = {r: {"p": ps[r], "p_holm": adj[r],
                                  "label_holm": label(res["rungs"][r]["contrasts"][n]["median_pct"], adj[r])}
                              for r in ps}
    print(f"witness pass: {res['witness']['pass']} {wit}", file=sys.stderr)
    for r in RUNGS:
        print(f"--- {r}", file=sys.stderr)
        for n, c in res["rungs"][r]["contrasts"].items():
            ph = (res["holm"].get(n) or {}).get(r, {}).get("p_holm", math.nan)
            print(f"{n:18s} {c.get('median_pct', math.nan):+7.2f} %  p={c.get('p', math.nan):.4f}  holm={ph:.4f}  "
                  f"{c.get('faster')}/{c['n_topologies']}  {c['label']}", file=sys.stderr)
        for arm, d in res["rungs"][r]["describe"].items():
            if d["n"]:
                print(f"   {arm:18s} n={d['n']:3d} mean={d['mean_latency']:8.2f} queue={d['mean_queue']:8.2f}",
                      file=sys.stderr)
    if a.out:
        with open(a.out + ".partial", "w") as fh:
            json.dump(res, fh, indent=1)
        os.replace(a.out + ".partial", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
