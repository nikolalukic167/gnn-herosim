#!/usr/bin/env python3
"""small_batch_confirm_v1 reader (docs/lineages/small_batch_confirm_v1.md).

  small_batch_confirm_v1_read.py --gate <confirm gate dir> --selection selected.json [--out read.json]

Statistic as peak_load_v1: per topology, the median paired % over (window, seed); learned arms pair on the seed, rules on
their single run. Then the median over the 19 topologies and an exact two-sided Wilcoxon over topologies.

Primary family (Holm across all 9): sb1load vs {cd, cdextr, sb1mpoff} x {x20, x30, x50}.
Verdict, signed before data:
- GNN-BEATS-ALL   the same >= 2 rungs where vs cd and vs cdextr are CONFIRMED (median <= -5 %, Holm p < 0.05) and vs
                  sb1mpoff is Holm p < 0.05 with median < 0 (any magnitude; the magnitude is quoted with it), and no
                  reference is Holm-confirmed faster at any rung;
- SEARCH-WIN-NOT-MP  vs cd and cdextr CONFIRMED at the same >= 2 rungs, the twin clause fails, no reference faster;
- NO-WIN          anything else.
Descriptive only: sb1mpoff vs cd / cdextr, sb1load vs xs1load (seeds 1-4), sb1load vs reactive.
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

GNN, TWIN, OLD = "sb1load_selfref", "sb1mpoff_selfref", "xs1load_selfref"
P.LEARNED = tuple(P.LEARNED) + (GNN, TWIN, OLD)
REFS = {"cd": "cd", "cdextr": "cdextr", "twin": TWIN}


def _windows(r: str) -> tuple:
    return tuple(f"g{i}{r}" for i in range(4))


def _with_seeds(seeds: tuple, fn, *args):
    old = P.SEEDS
    P.SEEDS = seeds
    try:
        return fn(*args)
    finally:
        P.SEEDS = old


def read(s: Dict, topos: List[int]) -> dict:
    eight = tuple(range(1, 9))
    tests, ps = {}, {}
    for r in RUNGS:
        for name, ref in REFS.items():
            c = _with_seeds(eight, contrast, s, topos, _windows(r), GNN, ref)
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
        search = [r for r in RUNGS if all(tests[f"{r}/{n}"]["label_holm"] == "CONFIRMED" for n in ("cd", "cdextr"))]
        twin_ok = [r for r in search if tests[f"{r}/twin"]["label_holm"] in ("CONFIRMED", "DIRECTION-ONLY")]
        ref_faster = [k for k, c in tests.items() if c["label_holm"] == "REF-FASTER"]
        if ref_faster or len(search) < 2:
            verdict = "NO-WIN"
        elif len(twin_ok) >= 2:
            verdict = "GNN-BEATS-ALL"
        else:
            verdict = "SEARCH-WIN-NOT-MP"
    desc = {}
    for r in RUNGS:
        ws = _windows(r)
        desc[r] = {
            "sb1mpoff_vs_cd": _with_seeds(eight, contrast, s, topos, ws, TWIN, "cd"),
            "sb1mpoff_vs_cdextr": _with_seeds(eight, contrast, s, topos, ws, TWIN, "cdextr"),
            "sb1load_vs_reactive": _with_seeds(eight, contrast, s, topos, ws, GNN, "reactive"),
            "sb1load_vs_xs1load_s1to4": _with_seeds((1, 2, 3, 4), contrast, s, topos, ws, GNN, OLD),
            "arms": {a: describe(s, topos, ws, a) for a in (GNN, TWIN, OLD, "cd", "cdextr", "reactive")},
        }
    return {"lineage": "small_batch_confirm_v1", "topologies": topos, "tests": tests, "verdict": verdict,
            "descriptive": desc}


def main(argv: List[str] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate", required=True)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    topos = json.load(open(a.selection))["topologies"]
    res = read(F._summaries(a.gate), topos)
    for k, c in res["tests"].items():
        print(f"{k:12s} {c.get('median_pct', math.nan):+8.2f} %  p={c.get('p', math.nan):.4f}  "
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
