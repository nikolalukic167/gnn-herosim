#!/usr/bin/env python3
"""aggr_sum_v1 reader: sum vs mean aggregation in the bipartite convs, on small_batch_confirm_v1's 19 topologies.

  aggr_sum_v1_read.py --gate <aggr_sum_v1 gate dir> [--out]

sb1sum vs sb1load and lf1sum vs lf1gnn (8 seeds, paired on seed), and each sum arm vs CD, cdextr, reactive and the six
rules; per topology the median paired % over (window, seed), median over the 19 topologies, exact Wilcoxon, Holm over
every test printed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fresh_topo_burst_v1_read as F  # noqa: E402
import local_features_v1_read as L  # noqa: E402
import peak_load_v1_read as P  # noqa: E402
from peak_controls_v1_read import RUNGS, describe, holm  # noqa: E402
from peak_load_v1_read import label  # noqa: E402

D = "/home/nikola.lukic/gnn-herosim/simulation_data"
SB1SUM, LF1SUM = "sb1sum_selfref", "lf1sum_selfref"
P.LEARNED = tuple(P.LEARNED) + (SB1SUM, LF1SUM)
REFS = ("cd", "cdextr", "reactive", "random", "drain", "locality", "decima", "batched", "selfpredict")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    s = {}
    for d in (f"{D}/small_batch_confirm_v1/gate", f"{D}/local_features_v1/ref_retry", f"{D}/local_features_v1/gate",
              f"{D}/rule_baselines_v1/gate", a.gate):
        s.update(F._summaries(d))
    topos = json.load(open(f"{D}/small_batch_confirm_v1/inputs/selected.json"))["topologies"]
    tests = {}
    for r in RUNGS:
        ws = L._windows(r)
        tests[f"{r}/sb1sum_vs_sb1load"] = L._c(L.EIGHT, s, topos, ws, SB1SUM, L.SB1)
        tests[f"{r}/lf1sum_vs_lf1gnn"] = L._c(L.EIGHT, s, topos, ws, LF1SUM, L.GNN)
        tests[f"{r}/lf1sum_vs_lf1mlp"] = L._c(L.EIGHT, s, topos, ws, LF1SUM, L.MLP)
        for arm, n in ((SB1SUM, "sb1sum"), (LF1SUM, "lf1sum")):
            for k in REFS:
                tests[f"{r}/{n}_vs_{k}"] = L._c(L.EIGHT, s, topos, ws, arm, k)
    ps = {k: c["p"] for k, c in tests.items() if c.get("p") is not None}
    adj = holm(ps)
    for k, c in tests.items():
        if k in adj:
            c["p_holm"], c["label_holm"] = adj[k], label(c["median_pct"], adj[k])
        print(f"{k:28s} {c.get('median_pct', math.nan):+8.2f} %  holm={c.get('p_holm', math.nan):.4f}  "
              f"{c.get('faster')}/{c['n_topologies']}  {c.get('label_holm', c['label'])}", file=sys.stderr)
    arms = {r: {x: describe(s, topos, L._windows(r), x) for x in (SB1SUM, L.SB1, LF1SUM, L.GNN, L.MLP, "cd")} for r in RUNGS}
    for r, d in arms.items():
        print(r, "  ".join(f"{x.replace('_selfref', '')}:{v.get('median_latency', math.nan):.0f}" for x, v in d.items()),
              file=sys.stderr)
    if a.out:
        with open(a.out + ".partial", "w") as fh:
            json.dump({"topologies": topos, "tests": tests, "arms": arms}, fh, indent=1)
        os.replace(a.out + ".partial", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
