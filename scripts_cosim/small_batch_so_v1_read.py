#!/usr/bin/env python3
"""small_batch_so_v1 reader (docs/lineages/small_batch_so_v1.md).

  small_batch_so_v1_read.py --gate <so1 gate dir> --refs <client_local_v1/gate_so_server> --selection selected.json
                            [--failed DIRS] [--out]

Statistic as local_features_v1, seeds 1-4: per topology, the median paired % over (window, seed); learned arms pair on the
seed, rules on their single run. Median over the 19 topologies, exact two-sided Wilcoxon over topologies.

Primary family (Holm across 6): so1load vs {cd, so1mpoff} x {x20, x30, x50}. Verdict, first that holds:
- BEATS-CD    vs cd CONFIRMED (median <= -5 %, Holm p < 0.05) at >= 2 rungs and cd Holm-confirmed faster at none
              (mp_win_rungs: the rungs at which vs so1mpoff is CONFIRMED too, reported beside it);
- NO-WIN      anything else;
- INCOMPLETE  a primary test short of 19 topologies, or a primary run missing without a failure record (--failed).
Descriptive: so1load vs zero-shot sb1load, reactive, locality, batched, selfpredict.

Amendment 1 (2026-10-06, before any so1lf* training): a second family, Holm across its own 6 tests, read when the runs
exist: so1lfgnn (lf1gnn's recipe) vs {cd, so1lfmlp (its same-input MLP)} x rungs. LF-BEATS-MLP-AND-CD iff the same >= 2
rungs have both CONFIRMED and no reference is Holm-confirmed faster anywhere; else NO-WIN (INCOMPLETE as above).
Descriptive: so1lfgnn vs zero-shot lf1gnn, so1lfmlp vs zero-shot lf1mlp, so1lfgnn vs so1load.
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
from local_features_v1_read import _c, _windows, failed_keys  # noqa: E402
from peak_controls_v1_read import RUNGS, describe, holm  # noqa: E402
from peak_load_v1_read import label  # noqa: E402

ARM, TWIN, ZS = "so1load_selfref", "so1mpoff_selfref", "sb1load_selfref"
LF, LFM = "so1lfgnn_selfref", "so1lfmlp_selfref"
P.LEARNED = tuple(P.LEARNED) + (ARM, TWIN, ZS, "sb1mpoff_selfref", LF, LFM, "lf1gnn_selfref", "lf1mlp_selfref")
REFS = {"cd": "cd", "twin": TWIN}
EIGHT = (1, 2, 3, 4)  # this lineage's seeds; the name is the helper's


def unexplained(s: Dict, topos: List[int], failed: set) -> List[str]:
    want = [(a, sd) for a in (ARM, TWIN) for sd in EIGHT] + [("cd", 0)]
    return [f"{t}/{w}/{k}/s{sd}" for t in topos for r in RUNGS for w in _windows(r) for k, sd in want
            if (t, w, k, sd) not in s and (t, w, k, sd) not in failed]


def read_lf(s: Dict, topos: List[int], failed: set) -> dict:
    """Amendment 1: so1lfgnn vs {cd, so1lfmlp}, Holm across 6."""
    if not any(k[2] == LF for k in s):
        return {"verdict": "NOT-RUN"}
    tests, ps = {}, {}
    for r in RUNGS:
        for name, ref in (("cd", "cd"), ("mlp", LFM)):
            c = _c(EIGHT, s, topos, _windows(r), LF, ref)
            tests[f"{r}/{name}"] = c
            if c.get("p") is not None:
                ps[f"{r}/{name}"] = c["p"]
    adj = holm(ps) if len(ps) == 6 else {}
    for k, c in tests.items():
        if k in adj:
            c["p_holm"], c["label_holm"] = adj[k], label(c["median_pct"], adj[k])
    want = [(a, sd) for a in (LF, LFM) for sd in EIGHT] + [("cd", 0)]
    missing = [f"{t}/{w}/{k}/s{sd}" for t in topos for r in RUNGS for w in _windows(r) for k, sd in want
               if (t, w, k, sd) not in s and (t, w, k, sd) not in failed]
    if len(ps) != 6:
        verdict = "DESIGN-SHORT"
    elif any(c["n_topologies"] != len(topos) for c in tests.values()) or missing:
        verdict = "INCOMPLETE"
    else:
        win = [r for r in RUNGS if tests[f"{r}/cd"]["label_holm"] == tests[f"{r}/mlp"]["label_holm"] == "CONFIRMED"]
        faster = any(c["label_holm"] == "REF-FASTER" for c in tests.values())
        verdict = "LF-BEATS-MLP-AND-CD" if len(win) >= 2 and not faster else "NO-WIN"
    desc = {r: {"so1lfgnn_vs_zeroshot_lf1gnn": _c(EIGHT, s, topos, _windows(r), LF, "lf1gnn_selfref"),
                "so1lfmlp_vs_zeroshot_lf1mlp": _c(EIGHT, s, topos, _windows(r), LFM, "lf1mlp_selfref"),
                "so1lfgnn_vs_so1load": _c(EIGHT, s, topos, _windows(r), LF, ARM)} for r in RUNGS}
    return {"tests": tests, "verdict": verdict, "unexplained_missing": missing, "descriptive": desc}


def read(s: Dict, topos: List[int], failed: set = frozenset()) -> dict:
    tests, ps = {}, {}
    for r in RUNGS:
        for name, ref in REFS.items():
            c = _c(EIGHT, s, topos, _windows(r), ARM, ref)
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
    mp_win = []
    if not complete:
        verdict = "DESIGN-SHORT"
    elif any(c["n_topologies"] != len(topos) for c in tests.values()) or missing:
        verdict = "INCOMPLETE"
    else:
        lab = lambda r, n: tests[f"{r}/{n}"]["label_holm"]  # noqa: E731
        win = [r for r in RUNGS if lab(r, "cd") == "CONFIRMED"]
        verdict = "BEATS-CD" if len(win) >= 2 and not any(lab(r, "cd") == "REF-FASTER" for r in RUNGS) else "NO-WIN"
        mp_win = [r for r in win if lab(r, "twin") == "CONFIRMED"]
    desc = {}
    for r in RUNGS:
        ws = _windows(r)
        desc[r] = {
            "so1load_vs_zeroshot_sb1load": _c(EIGHT, s, topos, ws, ARM, ZS),
            "so1load_vs_reactive": _c(EIGHT, s, topos, ws, ARM, "reactive"),
            "so1load_vs_locality": _c(EIGHT, s, topos, ws, ARM, "locality"),
            "so1load_vs_batched": _c(EIGHT, s, topos, ws, ARM, "batched"),
            "so1load_vs_selfpredict": _c(EIGHT, s, topos, ws, ARM, "selfpredict"),
            "arms": {a: describe(s, topos, ws, a) for a in (ARM, TWIN, ZS, "cd", "reactive")},
        }
    return {"lineage": "small_batch_so_v1", "topologies": topos, "tests": tests, "verdict": verdict, "mp_win_rungs": mp_win,
            "amendment1": read_lf(s, topos, failed),
            "descriptive": desc,
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
                print(f"[desc] {r} {n:28s} {c.get('median_pct', math.nan):+8.2f} %  p={c.get('p', math.nan):.4f}  "
                      f"{c.get('faster')}/{c['n_topologies']}", file=sys.stderr)
    print(f"verdict: {res['verdict']}  (unexplained missing: {len(res['unexplained_missing'])})", file=sys.stderr)
    a1 = res["amendment1"]
    for k, c in (a1.get("tests") or {}).items():
        print(f"[A1] {k:10s} {c.get('median_pct', math.nan):+8.2f} %  p={c.get('p', math.nan):.4f}  "
              f"holm={c.get('p_holm', math.nan):.4f}  {c.get('faster')}/{c['n_topologies']}  {c.get('label_holm', c['label'])}",
              file=sys.stderr)
    print(f"amendment 1 verdict: {a1['verdict']}", file=sys.stderr)
    if a.out:
        tmp = a.out + ".partial"
        with open(tmp, "w") as fh:
            json.dump(res, fh, indent=1, default=str)
        os.replace(tmp, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
