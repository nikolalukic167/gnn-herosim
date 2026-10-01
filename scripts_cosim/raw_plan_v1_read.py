#!/usr/bin/env python3
"""raw_plan_v1 -- the learned scorer with the raw plan instead of engineered plan-context columns.

  raw_plan_v1_read.py --ladder <groundedladder> --ctl <peakctl> --mlp <peakmlp> --raw <rawgnn> [<rawmlp> ...]
                      --selection selected.json [--out read.json]

R1: rawgnn vs rawmlp (does message passing recover the plan context the pointwise twin cannot see?).
R2: rawgnn and rawmlp vs CD and vs cdextr (does the raw-plan scorer still beat search?).
R3: rawgnn vs xs1load and rawmlp vs xs1mpoff (what the engineered context was worth to each class).
Statistic, labels and Holm over the three rungs as peak_controls_v1_read.
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
import peak_load_v1_read as P  # noqa: E402
from peak_controls_v1_read import GNN, MLP, RUNGS, describe, holm  # noqa: E402
from peak_load_v1_read import contrast, label  # noqa: E402

RGNN, RMLP = "rawgnn_selfref", "rawmlp_selfref"
P.LEARNED = tuple(P.LEARNED) + (MLP, RGNN, RMLP)
FAMILIES = {
    "rawgnn_vs_rawmlp": (RGNN, RMLP),
    "rawgnn_vs_cd": (RGNN, "cd"), "rawmlp_vs_cd": (RMLP, "cd"),
    "rawgnn_vs_cdextr": (RGNN, "cdextr"), "rawmlp_vs_cdextr": (RMLP, "cdextr"),
    "rawgnn_vs_cdext": (RGNN, "cdext"), "rawmlp_vs_cdext": (RMLP, "cdext"),
    "rawgnn_vs_xs1load": (RGNN, GNN), "rawmlp_vs_xs1mpoff": (RMLP, MLP),
    "rawgnn_vs_batched": (RGNN, "batched"), "rawgnn_vs_decima": (RGNN, "decima"),
    "rawgnn_vs_reactive": (RGNN, "reactive"), "rawgnn_vs_random": (RGNN, "random"),
    "rawmlp_vs_batched": (RMLP, "batched"), "rawmlp_vs_decima": (RMLP, "decima"),
    "rawmlp_vs_reactive": (RMLP, "reactive"), "rawmlp_vs_random": (RMLP, "random"),
}


def read(dirs: List[str], topos: List[int]) -> dict:
    s = {}
    for d in dirs:
        s.update(F._summaries(d))
    res = {"rungs": {}, "holm": {}}
    for r in RUNGS:
        ws = tuple(f"g{i}{r}" for i in range(4))
        res["rungs"][r] = {
            "contrasts": {n: contrast(s, topos, ws, a, b) for n, (a, b) in FAMILIES.items()},
            "describe": {arm: describe(s, topos, ws, arm) for arm in (RGNN, RMLP, GNN, MLP, "cd", "cdextr")},
        }
    for n in FAMILIES:
        ps = {r: res["rungs"][r]["contrasts"][n].get("p") for r in RUNGS}
        if all(p is not None for p in ps.values()):
            adj = holm(ps)
            res["holm"][n] = {r: {"p": ps[r], "p_holm": adj[r],
                                  "label_holm": label(res["rungs"][r]["contrasts"][n]["median_pct"], adj[r])}
                              for r in RUNGS}
    return res


def main(argv: List[str] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("ladder", "ctl", "mlp", "selection"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--raw", nargs="+", required=True)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    topos = json.load(open(a.selection))["topologies"]
    res = read([a.ladder, a.ctl, a.mlp, *a.raw], topos)
    for r in RUNGS:
        print(f"--- {r}", file=sys.stderr)
        for n, c in res["rungs"][r]["contrasts"].items():
            ph = (res["holm"].get(n) or {}).get(r, {}).get("p_holm", math.nan)
            print(f"{n:20s} {c.get('median_pct', math.nan):+7.2f} %  p={c.get('p', math.nan):.4f}  "
                  f"holm={ph:.4f}  {c.get('faster')}/{c['n_topologies']}  {c['label']}", file=sys.stderr)
        for arm, d in res["rungs"][r]["describe"].items():
            if d["n"]:
                print(f"   {arm:18s} n={d['n']:3d} mean={d['mean_latency']:8.2f} queue={d['mean_queue']:8.2f} "
                      f"exch={d['mean_exchange_per_task']:.3f}", file=sys.stderr)
    if a.out:
        with open(a.out + ".partial", "w") as fh:
            json.dump(res, fh, indent=1)
        os.replace(a.out + ".partial", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
