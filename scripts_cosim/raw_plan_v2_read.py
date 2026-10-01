#!/usr/bin/env python3
"""raw_plan_v2 reader (docs/lineages/raw_plan_v2.md).

  raw_plan_v2_read.py dev  --dirs <groundedladder> <peakctl> <rawgnn> <rawmlp> <rawE> <rawS> <rawES> <rawStwin>
                           --selection selected.json [--out dev_read.json]
  raw_plan_v2_read.py conf --dirs <phase C output dirs...> --selection <phase C selected.json>
                           --selected rawE|rawS|rawES [--out conf_read.json]

Statistic as peak_load_v1: per topology, the median paired % over (window, seed). Learned arms are paired on the
seed, rules on their single run. Then the median over topologies and an exact two-sided Wilcoxon over topologies.

dev:
- descriptive contrasts, labelled development;
- the pre-registered selection: the convolution arm with the lowest median paired % vs cdextr, pooled over the
  three rungs (per topology the median over every (rung, window, seed) pair).

conf:
- the primary family {selected vs cd, cdextr, its twin, rawmlp} x {x20, x30, x50}, Holm across all 12;
- verdict RAW-GNN-WIN iff there are the same >= 2 rungs at which all four comparisons are CONFIRMED, and no
  reference is Holm-confirmed faster at any rung.
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

RAWGNN, RAWMLP = "rawgnn_selfref", "rawmlp_selfref"
E, S, ES, TWIN = "rawE_selfref", "rawS_selfref", "rawES_selfref", "rawStwin_selfref"
CONV_ARMS = (E, S, ES)
TWIN_OF = {E: RAWMLP, S: TWIN, ES: TWIN}
P.LEARNED = tuple(P.LEARNED) + ("xs1mpoff_selfref", RAWGNN, RAWMLP, E, S, ES, TWIN)
REFS = ("cd", "cdextr")


def _windows(r: str) -> tuple:
    return tuple(f"g{i}{r}" for i in range(4))


def _summaries(dirs: List[str]) -> Dict:
    s: Dict = {}
    for d in dirs:
        s.update(F._summaries(d))
    return s


def read_dev(s: Dict, topos: List[int]) -> dict:
    fam = {}
    for a in CONV_ARMS:
        fam[f"{a}_vs_rawgnn"] = (a, RAWGNN)
        fam[f"{a}_vs_twin"] = (a, TWIN_OF[a])
        fam[f"{a}_vs_rawmlp"] = (a, RAWMLP)
        for r in REFS:
            fam[f"{a}_vs_{r}"] = (a, r)
    fam.update({"rawES_vs_rawS": (ES, S), "rawES_vs_rawE": (ES, E),
                "rawStwin_vs_rawmlp": (TWIN, RAWMLP), "rawStwin_vs_cd": (TWIN, "cd"),
                "rawStwin_vs_cdextr": (TWIN, "cdextr")})
    res = {"phase": "development (descriptive; these topologies informed the proposal)", "rungs": {}}
    for r in RUNGS:
        ws = _windows(r)
        res["rungs"][r] = {
            "contrasts": {n: contrast(s, topos, ws, a, b) for n, (a, b) in fam.items()},
            "describe": {arm: describe(s, topos, ws, arm) for arm in (E, S, ES, TWIN, RAWGNN, RAWMLP, "cd", "cdextr")},
        }
    pooled = tuple(w for r in RUNGS for w in _windows(r))
    sel = {a: contrast(s, topos, pooled, a, "cdextr") for a in CONV_ARMS}
    ok = {a: c for a, c in sel.items() if "median_pct" in c}
    if len(ok) != len(CONV_ARMS):
        res["selection"] = {"pooled_vs_cdextr": sel, "selected": None,
                            "reason": "DESIGN-SHORT: an arm lacks enough topologies"}
    else:
        best = min(ok, key=lambda a: ok[a]["median_pct"])
        res["selection"] = {"pooled_vs_cdextr": {a: ok[a]["median_pct"] for a in CONV_ARMS},
                            "selected": best, "twin": TWIN_OF[best]}
    return res


def read_conf(s: Dict, topos: List[int], selected: str) -> dict:
    refs = {"cd": "cd", "cdextr": "cdextr", "twin": TWIN_OF[selected], "rawmlp": RAWMLP}
    tests, ps = {}, {}
    for r in RUNGS:
        for name, ref in refs.items():
            c = contrast(s, topos, _windows(r), selected, ref)
            tests[f"{r}/{name}"] = c
            if c.get("p") is not None:
                ps[f"{r}/{name}"] = c["p"]
    complete = len(ps) == len(RUNGS) * len(refs)
    adj = holm(ps) if complete else {}
    for k, c in tests.items():
        if k in adj:
            c["p_holm"] = adj[k]
            c["label_holm"] = label(c["median_pct"], adj[k])
    if not complete:
        verdict = "DESIGN-SHORT"
    else:
        win_rungs = [r for r in RUNGS if all(tests[f"{r}/{n}"]["label_holm"] == "CONFIRMED" for n in refs)]
        ref_faster = [k for k, c in tests.items() if c["label_holm"] == "REF-FASTER"]
        verdict = "RAW-GNN-WIN" if len(win_rungs) >= 2 and not ref_faster else "NO-WIN"
    describe_ = {r: {arm: describe(s, topos, _windows(r), arm) for arm in (selected, refs["twin"], RAWMLP, "cd", "cdextr")}
                 for r in RUNGS}
    return {"phase": "confirmation", "selected": selected, "references": refs, "tests": tests,
            "verdict": verdict, "describe": describe_}


def main(argv: List[str] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("dev", "conf"))
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--selected", choices=("rawE", "rawS", "rawES"))
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    topos = json.load(open(a.selection))["topologies"]
    s = _summaries(a.dirs)
    if a.mode == "dev":
        res = read_dev(s, topos)
        for r in RUNGS:
            print(f"--- {r} (development)", file=sys.stderr)
            for n, c in res["rungs"][r]["contrasts"].items():
                print(f"{n:28s} {c.get('median_pct', math.nan):+8.2f} %  p={c.get('p', math.nan):.4f}  "
                      f"{c.get('faster')}/{c['n_topologies']}  {c['label']}", file=sys.stderr)
        print(f"selection: {res['selection']}", file=sys.stderr)
    else:
        if not a.selected:
            ap.error("conf needs --selected")
        res = read_conf(s, topos, f"{a.selected}_selfref")
        for k, c in res["tests"].items():
            print(f"{k:16s} {c.get('median_pct', math.nan):+8.2f} %  p={c.get('p', math.nan):.4f}  "
                  f"holm={c.get('p_holm', math.nan):.4f}  {c.get('faster')}/{c['n_topologies']}  "
                  f"{c.get('label_holm', c['label'])}", file=sys.stderr)
        print(f"VERDICT: {res['verdict']}", file=sys.stderr)
    if a.out:
        with open(a.out + ".partial", "w") as fh:
            json.dump(res, fh, indent=1)
        os.replace(a.out + ".partial", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
