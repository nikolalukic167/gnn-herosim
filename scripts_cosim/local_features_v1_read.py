#!/usr/bin/env python3
"""local_features_v1 reader (docs/lineages/local_features_v1.md).

  local_features_v1_read.py --gate <lf1 gate dir> --refs <small_batch_confirm_v1 gate dir> --selection selected.json [--out]

Statistic as peak_load_v1: per topology, the median paired % over (window, seed); learned arms pair on the seed, rules on
their single run. Median over the 19 topologies, exact two-sided Wilcoxon over topologies.

Primary family (Holm across all 12): lf1gnn vs {cd, cdextr, lf1mlp, sb1mpoff} x {x20, x30, x50}. Verdict, signed before data:
- GNN-BEATS-ALL         the same >= 2 rungs at which all four are CONFIRMED (median <= -5 %, Holm p < 0.05), and no
                        reference Holm-confirmed faster at any rung;
- GNN-BEATS-SAME-INPUT  cd, cdextr and lf1mlp CONFIRMED at the same >= 2 rungs, sb1mpoff not, nothing faster except
                        possibly sb1mpoff;
- NO-WIN                anything else.
Descriptive: lf1gnn vs lf1twin (no-conv twin), sb1load, xs1load; lf1mlp and lf1twin vs cd.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fresh_topo_burst_v1_read as F  # noqa: E402
import peak_load_v1_read as P  # noqa: E402
from peak_controls_v1_read import RUNGS, describe, holm  # noqa: E402
from peak_load_v1_read import contrast, label  # noqa: E402

GNN, TWIN, MLP = "lf1gnn_selfref", "lf1twin_selfref", "lf1mlp_selfref"
SB1, SB1T, XS1 = "sb1load_selfref", "sb1mpoff_selfref", "xs1load_selfref"
P.LEARNED = tuple(P.LEARNED) + (GNN, TWIN, MLP, SB1, SB1T, XS1)
REFS = {"cd": "cd", "cdextr": "cdextr", "mlp": MLP, "sb1mpoff": SB1T}
EIGHT = tuple(range(1, 9))


def _windows(r: str) -> tuple:
    return tuple(f"g{i}{r}" for i in range(4))


def _c(seeds: tuple, s: Dict, topos: List[int], ws: tuple, a: str, b: str) -> dict:
    old = P.SEEDS
    P.SEEDS = seeds
    try:
        return contrast(s, topos, ws, a, b)
    finally:
        P.SEEDS = old


def read(s: Dict, topos: List[int]) -> dict:
    tests, ps = {}, {}
    for r in RUNGS:
        for name, ref in REFS.items():
            c = _c(EIGHT, s, topos, _windows(r), GNN, ref)
            tests[f"{r}/{name}"] = c
            if c.get("p") is not None:
                ps[f"{r}/{name}"] = c["p"]
    complete = len(ps) == len(RUNGS) * len(REFS)
    adj = holm(ps) if complete else {}
    for k, c in tests.items():
        if k in adj:
            c["p_holm"] = adj[k]
            c["label_holm"] = label(c["median_pct"], adj[k])
    if not complete:
        verdict = "DESIGN-SHORT"
    else:
        conf = lambda r, n: tests[f"{r}/{n}"]["label_holm"] == "CONFIRMED"  # noqa: E731
        faster = [k for k, c in tests.items() if c["label_holm"] == "REF-FASTER"]
        same = [r for r in RUNGS if all(conf(r, n) for n in ("cd", "cdextr", "mlp"))]
        everything = [r for r in same if conf(r, "sb1mpoff")]
        if len(everything) >= 2 and not faster:
            verdict = "GNN-BEATS-ALL"
        elif len(same) >= 2 and all(k.endswith("/sb1mpoff") for k in faster):
            verdict = "GNN-BEATS-SAME-INPUT"
        else:
            verdict = "NO-WIN"
    desc = {}
    for r in RUNGS:
        ws = _windows(r)
        desc[r] = {
            "lf1gnn_vs_lf1twin": _c(EIGHT, s, topos, ws, GNN, TWIN),
            "lf1gnn_vs_sb1load": _c(EIGHT, s, topos, ws, GNN, SB1),
            "lf1gnn_vs_xs1load_s1to4": _c((1, 2, 3, 4), s, topos, ws, GNN, XS1),
            "lf1mlp_vs_cd": _c(EIGHT, s, topos, ws, MLP, "cd"),
            "lf1twin_vs_cd": _c(EIGHT, s, topos, ws, TWIN, "cd"),
            "arms": {a: describe(s, topos, ws, a) for a in (GNN, TWIN, MLP, SB1, SB1T, "cd", "cdextr")},
        }
    return {"lineage": "local_features_v1", "topologies": topos, "tests": tests, "verdict": verdict, "descriptive": desc}


def main(argv: List[str] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate", required=True)
    ap.add_argument("--refs", required=True)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    topos = json.load(open(a.selection))["topologies"]
    s = F._summaries(a.refs)
    s.update(F._summaries(a.gate))
    res = read(s, topos)
    for k, c in res["tests"].items():
        print(f"{k:14s} {c.get('median_pct', math.nan):+8.2f} %  p={c.get('p', math.nan):.4f}  "
              f"holm={c.get('p_holm', math.nan):.4f}  {c.get('faster')}/{c['n_topologies']}  "
              f"{c.get('label_holm', c['label'])}", file=sys.stderr)
    for r, d in res["descriptive"].items():
        for n, c in d.items():
            if n != "arms":
                print(f"[desc] {r} {n:26s} {c.get('median_pct', math.nan):+8.2f} %  p={c.get('p', math.nan):.4f}  "
                      f"{c.get('faster')}/{c['n_topologies']}", file=sys.stderr)
    print(f"VERDICT: {res['verdict']}", file=sys.stderr)
    if a.out:
        with open(a.out + ".partial", "w") as fh:
            json.dump(res, fh, indent=1)
        os.replace(a.out + ".partial", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
