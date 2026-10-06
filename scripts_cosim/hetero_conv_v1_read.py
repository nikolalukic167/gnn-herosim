#!/usr/bin/env python3
"""hetero_conv_v1 reader (docs/lineages/hetero_conv_v1.md).

  hetero_conv_v1_read.py --gate <het1 gate dir> --refs <small_batch_confirm_v1 gate> <local_features_v1 gate> [retry dirs]
                         --selection selected.json [--failed DIRS] [--out]

Statistic as local_features_v1: per topology, the median paired % over (window, seed); learned arms pair on the seed,
rules on their single run. Median over the 19 topologies, exact two-sided Wilcoxon over topologies.

Primary family (Holm across all 9): lf1het vs {lf1twin (its MP-OFF twin), lf1gnn (plain convs), cd} x {x20, x30, x50}.
Verdict, first that holds:
- HETERO-BEATS-CD-AND-TWIN  the same >= 2 rungs at which vs cd and vs lf1twin are both CONFIRMED (median <= -5 %,
                            Holm p < 0.05), and no reference Holm-confirmed faster at any rung;
- HETERO-HELPS              vs lf1gnn CONFIRMED at >= 2 rungs and lf1gnn Holm-confirmed faster at none;
- NO-GAIN                   anything else;
- INCOMPLETE                a primary test short of 19 topologies, or a primary run missing without a failure record
                            from the one rerun at 3x the timeout (--failed dirs).
Descriptive, outside the verdict: lf1het vs lf1mlp, cdextr, sb1load, sb1mpoff; lf1gnn vs cd.
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
from local_features_v1_read import EIGHT, _c, _windows, failed_keys  # noqa: E402
from peak_controls_v1_read import RUNGS, describe, holm  # noqa: E402
from peak_load_v1_read import label  # noqa: E402

HET, GNN, TWIN, MLP = "lf1het_selfref", "lf1gnn_selfref", "lf1twin_selfref", "lf1mlp_selfref"
SB1, SB1T = "sb1load_selfref", "sb1mpoff_selfref"
P.LEARNED = tuple(P.LEARNED) + (HET,)
REFS = {"twin": TWIN, "gnn": GNN, "cd": "cd"}


def unexplained(s: Dict, topos: List[int], failed: set) -> List[str]:
    want = [(a, sd) for a in (HET, GNN, TWIN) for sd in EIGHT] + [("cd", 0)]
    return [f"{t}/{w}/{k}/s{sd}" for t in topos for r in RUNGS for w in _windows(r) for k, sd in want
            if (t, w, k, sd) not in s and (t, w, k, sd) not in failed]


def read(s: Dict, topos: List[int], failed: set = frozenset()) -> dict:
    tests, ps = {}, {}
    for r in RUNGS:
        for name, ref in REFS.items():
            c = _c(EIGHT, s, topos, _windows(r), HET, ref)
            tests[f"{r}/{name}"] = c
            if c.get("p") is not None:
                ps[f"{r}/{name}"] = c["p"]
    complete = len(ps) == len(RUNGS) * len(REFS)
    adj = holm(ps) if complete else {}
    for k, c in tests.items():
        if k in adj:
            c["p_holm"] = adj[k]
            c["label_holm"] = label(c["median_pct"], adj[k])
    missing = unexplained(s, topos, failed)
    if not complete:
        verdict = "DESIGN-SHORT"
    elif any(c["n_topologies"] != len(topos) for c in tests.values()) or missing:
        verdict = "INCOMPLETE"
    else:
        lab = lambda r, n: tests[f"{r}/{n}"]["label_holm"]  # noqa: E731
        faster = [k for k, c in tests.items() if c["label_holm"] == "REF-FASTER"]
        win = [r for r in RUNGS if lab(r, "cd") == "CONFIRMED" and lab(r, "twin") == "CONFIRMED"]
        helps = [r for r in RUNGS if lab(r, "gnn") == "CONFIRMED"]
        if len(win) >= 2 and not faster:
            verdict = "HETERO-BEATS-CD-AND-TWIN"
        elif len(helps) >= 2 and not any(lab(r, "gnn") == "REF-FASTER" for r in RUNGS):
            verdict = "HETERO-HELPS"
        else:
            verdict = "NO-GAIN"
    desc = {}
    for r in RUNGS:
        ws = _windows(r)
        desc[r] = {
            "lf1het_vs_lf1mlp": _c(EIGHT, s, topos, ws, HET, MLP),
            "lf1het_vs_cdextr": _c(EIGHT, s, topos, ws, HET, "cdextr"),
            "lf1het_vs_sb1load": _c(EIGHT, s, topos, ws, HET, SB1),
            "lf1het_vs_sb1mpoff": _c(EIGHT, s, topos, ws, HET, SB1T),
            "lf1gnn_vs_cd": _c(EIGHT, s, topos, ws, GNN, "cd"),
            "arms": {a: describe(s, topos, ws, a) for a in (HET, GNN, TWIN, MLP, SB1, "cd")},
        }
    return {"lineage": "hetero_conv_v1", "topologies": topos, "tests": tests, "verdict": verdict, "descriptive": desc,
            "unexplained_missing": missing, "failed_twice": sorted(f"{t}/{w}/{k}/s{sd}" for t, w, k, sd in failed)}


def main(argv: List[str] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate", required=True)
    ap.add_argument("--refs", nargs="+", required=True, help="later dirs override earlier ones")
    ap.add_argument("--failed", nargs="*", default=[], help="dirs whose *.failed.json record the 3x-timeout rerun")
    ap.add_argument("--selection", required=True)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    topos = json.load(open(a.selection))["topologies"]
    s = {}
    for d in a.refs + [a.gate]:
        s.update(F._summaries(d))
    res = read(s, topos, failed_keys(a.failed))
    for k, c in res["tests"].items():
        print(f"{k:12s} {c.get('median_pct', math.nan):+8.2f} %  p={c.get('p', math.nan):.4f}  "
              f"holm={c.get('p_holm', math.nan):.4f}  {c.get('faster')}/{c['n_topologies']}  "
              f"{c.get('label_holm', c['label'])}", file=sys.stderr)
    for r, d in res["descriptive"].items():
        for n, c in d.items():
            if n != "arms":
                print(f"[desc] {r} {n:20s} {c.get('median_pct', math.nan):+8.2f} %  p={c.get('p', math.nan):.4f}  "
                      f"{c.get('faster')}/{c['n_topologies']}", file=sys.stderr)
    print(f"verdict: {res['verdict']}  (unexplained missing: {len(res['unexplained_missing'])})", file=sys.stderr)
    if a.out:
        tmp = a.out + ".partial"
        with open(tmp, "w") as fh:
            json.dump(res, fh, indent=1, default=str)
        os.replace(tmp, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
