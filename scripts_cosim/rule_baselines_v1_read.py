#!/usr/bin/env python3
"""rule_baselines_v1 reader (docs/lineages/rule_baselines_v1.md).

  rule_baselines_v1_read.py --gate <rb1 gate dir> --lf1 <local_features_v1 gate dir> --selection selected.json [--out]

lf1gnn (8 seeds) vs each of six hand rules (1 run) x {x20, x30, x50} = 18 tests, Holm across all 18; per topology the
median paired % over (window, seed), median over the 19 topologies, exact two-sided Wilcoxon (peak_load_v1_read).
Labels: CONFIRMED (median <= -5 %, Holm p < 0.05) / DIRECTION-ONLY / REF-FASTER / NOT-SEPARATED. Every label is
reported; there is no aggregate verdict. Descriptive: lf1mlp vs each rule.
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
import local_features_v1_read as L  # noqa: E402
from peak_controls_v1_read import RUNGS, describe, holm  # noqa: E402
from peak_load_v1_read import label  # noqa: E402

RULES = ("random", "drain", "locality", "decima", "batched", "selfpredict")


def read(s, topos: List[int]) -> dict:
    tests = {f"{r}/{k}": L._c(L.EIGHT, s, topos, L._windows(r), L.GNN, k) for r in RUNGS for k in RULES}
    ps = {k: c["p"] for k, c in tests.items() if c.get("p") is not None}
    adj = holm(ps) if len(ps) == len(tests) else {}
    for k, c in tests.items():
        if k in adj:
            c["p_holm"], c["label_holm"] = adj[k], label(c["median_pct"], adj[k])
    desc = {f"{r}/{k}": L._c(L.EIGHT, s, topos, L._windows(r), L.MLP, k) for r in RUNGS for k in RULES}
    arms = {r: {a: describe(s, topos, L._windows(r), a) for a in (L.GNN, L.MLP) + RULES} for r in RUNGS}
    return {"lineage": "rule_baselines_v1", "topologies": topos, "complete": bool(adj), "tests": tests,
            "lf1mlp_vs_rules": desc, "arms": arms}


def main(argv: List[str] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate", required=True)
    ap.add_argument("--lf1", required=True)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    topos = json.load(open(a.selection))["topologies"]
    s = F._summaries(a.lf1)
    s.update(F._summaries(a.gate))
    res = read(s, topos)
    for k, c in res["tests"].items():
        print(f"{k:18s} {c.get('median_pct', math.nan):+8.2f} %  holm={c.get('p_holm', math.nan):.4f}  "
              f"{c.get('faster')}/{c['n_topologies']}  {c.get('label_holm', c['label'])}  dropped_topos={len(c['dropped'])}",
              file=sys.stderr)
    for k, c in res["lf1mlp_vs_rules"].items():
        print(f"[mlp] {k:18s} {c.get('median_pct', math.nan):+8.2f} %  {c.get('faster')}/{c['n_topologies']}", file=sys.stderr)
    print(f"complete: {res['complete']}", file=sys.stderr)
    if a.out:
        with open(a.out + ".partial", "w") as fh:
            json.dump(res, fh, indent=1)
        os.replace(a.out + ".partial", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
