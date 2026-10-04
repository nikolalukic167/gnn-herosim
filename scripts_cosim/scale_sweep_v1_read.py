#!/usr/bin/env python3
"""scale_sweep_v1 reader (docs/lineages/scale_sweep_v1.md) -- exploratory, development topologies, descriptive only.

  scale_sweep_v1_read.py --gate <scale gate dir> --base <dev gate dirs: groundedladder peakctl sb1dev> --selection <dev selected.json> [--out]

For each condition (base = the existing x3 / x5 ladder runs; m2, m4 = ~2x / ~4x tasks per batch; s12 = 12 servers) and
rung: GNN vs its MP-OFF twin for both engineered pairs (xs1load/xs1mpoff trained on 10-task groups, sb1load/sb1mpoff on
live-sized ones), and each learned arm vs CD and cdextr. Per-topology median over (window, seed), median over the 12
topologies, exact Wilcoxon, no Holm. The question it answers: does the GNN-vs-twin margin grow with tasks per batch or
cluster size? Any condition where it does is a candidate for a registered confirmation on fresh topologies.
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
from peak_load_v1_read import contrast  # noqa: E402

ARMS = ("xs1load_selfref", "xs1mpoff_selfref", "sb1load_selfref", "sb1mpoff_selfref")
P.LEARNED = tuple(P.LEARNED) + ARMS
PAIRS = [("xs1load_selfref", "xs1mpoff_selfref"), ("sb1load_selfref", "sb1mpoff_selfref")] + \
        [(a, r) for a in ARMS for r in ("cd", "cdextr")]


def main(argv: List[str] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate", required=True)
    ap.add_argument("--base", nargs="+", required=True)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    topos = json.load(open(a.selection))["topologies"]
    s = {}
    for d in a.base + [a.gate]:
        s.update(F._summaries(d))
    res = {}
    for cond in ("", "m2", "m4", "s12"):
        for r in ("x30", "x50"):
            ws = tuple(f"g{i}{r}{cond}" for i in range(4))
            key = f"{cond or 'base'}/{r}"
            res[key] = {f"{x}_vs_{y}": contrast(s, topos, ws, x, y) for x, y in PAIRS}
            for n, c in res[key].items():
                print(f"{key:9s} {n:34s} {c.get('median_pct', math.nan):+8.2f} %  p={c.get('p', math.nan):.4f}  "
                      f"{c.get('faster')}/{c['n_topologies']}", file=sys.stderr)
    if a.out:
        with open(a.out + ".partial", "w") as fh:
            json.dump(res, fh, indent=1)
        os.replace(a.out + ".partial", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
