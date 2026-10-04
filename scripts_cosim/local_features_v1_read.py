#!/usr/bin/env python3
"""local_features_v1 reader (docs/lineages/local_features_v1.md).

  local_features_v1_read.py --gate <lf1 gate dir> --refs <small_batch_confirm_v1 gate dir> --selection selected.json [--out]

Statistic as peak_load_v1: per topology, the median paired % over (window, seed); learned arms pair on the seed, rules on
their single run. Median over the 19 topologies, exact two-sided Wilcoxon over topologies.

Primary family (Holm across all 9; Amendment 1, 2026-10-04, before any gate run): lf1gnn vs {cd, cdextr, lf1mlp} x
{x20, x30, x50}. Verdict:
- GNN-BEATS-MLP-AND-CD  the same >= 2 rungs at which all three are CONFIRMED (median <= -5 %, Holm p < 0.05), and no
                        reference Holm-confirmed faster at any rung;
- NO-WIN                anything else;
- INCOMPLETE            (Amendment 2) a primary test short of 19 topologies, or a primary run missing without a failure
                        record from the one rerun at 3x the timeout (local_features_v1_retry.sbatch; --failed dirs).
Robustness, outside the verdict (Amendment 2): the original 12-test verdict (with sb1mpoff); every lf1gnn seed alone vs
CD, cdextr and the median-over-seeds MLP; a seed-pairing-free contrast vs the MLP (median over seeds per window).
Descriptive, outside the verdict: lf1gnn vs sb1mpoff (engineered pointwise scorer), lf1twin (no-conv twin), sb1load,
xs1load; lf1mlp and lf1twin vs cd.
"""
from __future__ import annotations

import argparse
import glob
import re
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
from scipy.stats import wilcoxon  # noqa: E402
from statistics import median  # noqa: E402

GNN, TWIN, MLP = "lf1gnn_selfref", "lf1twin_selfref", "lf1mlp_selfref"
SB1, SB1T, XS1 = "sb1load_selfref", "sb1mpoff_selfref", "xs1load_selfref"
P.LEARNED = tuple(P.LEARNED) + (GNN, TWIN, MLP, SB1, SB1T, XS1)
REFS = {"cd": "cd", "cdextr": "cdextr", "mlp": MLP}
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


def unpaired(s: Dict, topos: List[int], ws: tuple, arm: str, arm_seeds: tuple, ref: str, ref_seeds: tuple) -> dict:
    """Per (topology, window): % of the arm's median-over-seeds latency vs the reference's; no seed-to-seed pairing."""
    per, dropped = {}, {}
    for t in topos:
        ds = []
        for w in ws:
            a = [s.get((t, w, arm, sd)) for sd in arm_seeds]
            b = [s.get((t, w, ref, sd)) for sd in ref_seeds]
            if None in a or None in b:
                dropped.setdefault(str(t), []).append(w)
                continue
            ds.append(F._pct(median(F._el(r) for r in a), median(F._el(r) for r in b)))
        if ds:
            per[t] = median(ds)
    xs = [per[t] for t in sorted(per)]
    out = {"arm": arm, "ref": ref, "n_topologies": len(per), "dropped": dropped}
    if len(xs) < P.MIN_TOPOLOGIES:
        return dict(out, label="DESIGN-SHORT")
    p = float(wilcoxon(xs, method="exact").pvalue) if any(xs) else 1.0
    return dict(out, median_pct=median(xs), p=p, faster=sum(x < 0 for x in xs), label=label(median(xs), p))


def verdict_of(tests: Dict, refs) -> str:
    ps = {k: c["p"] for k, c in tests.items() if c.get("p") is not None}
    if len(ps) != len(tests):
        return "DESIGN-SHORT"
    adj = holm(ps)
    lab = {k: label(c["median_pct"], adj[k]) for k, c in tests.items()}
    faster = [k for k, v in lab.items() if v == "REF-FASTER"]
    win = [r for r in RUNGS if all(lab[f"{r}/{n}"] == "CONFIRMED" for n in refs)]
    return "WIN" if len(win) >= 2 and not faster else "NO-WIN"


def robustness(s: Dict, topos: List[int], tests: Dict) -> dict:
    orig = dict(tests)
    for r in RUNGS:
        orig[f"{r}/sb1mpoff"] = _c(EIGHT, s, topos, _windows(r), GNN, SB1T)
    full = verdict_of(orig, ("cd", "cdextr", "mlp", "sb1mpoff"))
    same = verdict_of(tests, tuple(REFS))
    registered = "GNN-BEATS-ALL" if full == "WIN" else ("GNN-BEATS-SAME-INPUT" if same == "WIN" else "NO-WIN")
    seeds = {}
    for sd in EIGHT:
        row = {}
        for r in RUNGS:
            ws = _windows(r)
            row[r] = {"cd": _c((sd,), s, topos, ws, GNN, "cd").get("median_pct"),
                      "cdextr": _c((sd,), s, topos, ws, GNN, "cdextr").get("median_pct"),
                      "mlp_median_seed": unpaired(s, topos, ws, GNN, (sd,), MLP, EIGHT).get("median_pct")}
        seeds[f"s{sd}"] = row
    return {"registered_12_test_verdict": registered,
            "per_seed_median_pct": seeds,
            "mlp_unpaired": {r: unpaired(s, topos, _windows(r), GNN, EIGHT, MLP, EIGHT) for r in RUNGS}}


def failed_keys(dirs: List[str]) -> set:
    out = set()
    for d in dirs:
        for f in glob.glob(os.path.join(d, "*.failed.json")):
            m = re.match(r"cc40s(\d+)__(\w+?)__(\w+)_s(\d+)\.failed\.json$", os.path.basename(f))
            if m:
                out.add((int(m[1]), m[2], m[3], int(m[4])))
    return out


def unexplained(s: Dict, topos: List[int], failed: set) -> List[str]:
    want = [(GNN, sd) for sd in EIGHT] + [(MLP, sd) for sd in EIGHT] + [("cd", 0), ("cdextr", 0)]
    return [f"{t}/{w}/{k}/s{sd}" for t in topos for r in RUNGS for w in _windows(r) for k, sd in want
            if (t, w, k, sd) not in s and (t, w, k, sd) not in failed]


def read(s: Dict, topos: List[int], failed: set = frozenset()) -> dict:
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
    elif any(c["n_topologies"] != len(topos) for c in tests.values()) or unexplained(s, topos, failed):
        verdict = "INCOMPLETE"
    else:
        faster = [k for k, c in tests.items() if c["label_holm"] == "REF-FASTER"]
        win = [r for r in RUNGS if all(tests[f"{r}/{n}"]["label_holm"] == "CONFIRMED" for n in REFS)]
        verdict = "GNN-BEATS-MLP-AND-CD" if len(win) >= 2 and not faster else "NO-WIN"
    desc = {}
    for r in RUNGS:
        ws = _windows(r)
        desc[r] = {
            "lf1gnn_vs_sb1mpoff": _c(EIGHT, s, topos, ws, GNN, SB1T),
            "lf1gnn_vs_lf1twin": _c(EIGHT, s, topos, ws, GNN, TWIN),
            "lf1gnn_vs_sb1load": _c(EIGHT, s, topos, ws, GNN, SB1),
            "lf1gnn_vs_xs1load_s1to4": _c((1, 2, 3, 4), s, topos, ws, GNN, XS1),
            "lf1mlp_vs_cd": _c(EIGHT, s, topos, ws, MLP, "cd"),
            "lf1twin_vs_cd": _c(EIGHT, s, topos, ws, TWIN, "cd"),
            "arms": {a: describe(s, topos, ws, a) for a in (GNN, TWIN, MLP, SB1, SB1T, "cd", "cdextr")},
        }
    return {"lineage": "local_features_v1", "topologies": topos, "tests": tests, "verdict": verdict, "descriptive": desc,
            "unexplained_missing": unexplained(s, topos, failed),
            "failed_twice": sorted(f"{t}/{w}/{k}/s{sd}" for t, w, k, sd in failed),
            "robustness": robustness(s, topos, tests) if complete else {}}


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
        print(f"{k:14s} {c.get('median_pct', math.nan):+8.2f} %  p={c.get('p', math.nan):.4f}  "
              f"holm={c.get('p_holm', math.nan):.4f}  {c.get('faster')}/{c['n_topologies']}  "
              f"{c.get('label_holm', c['label'])}", file=sys.stderr)
    for r, d in res["descriptive"].items():
        for n, c in d.items():
            if n != "arms":
                print(f"[desc] {r} {n:26s} {c.get('median_pct', math.nan):+8.2f} %  p={c.get('p', math.nan):.4f}  "
                      f"{c.get('faster')}/{c['n_topologies']}", file=sys.stderr)
    rb = res["robustness"]
    if rb:
        print(f"[robust] registered 12-test verdict: {rb['registered_12_test_verdict']}", file=sys.stderr)
        for r, c in rb["mlp_unpaired"].items():
            print(f"[robust] {r} vs mlp unpaired {c.get('median_pct', math.nan):+8.2f} %  p={c.get('p', math.nan):.4f}",
                  file=sys.stderr)
        for sd, row in rb["per_seed_median_pct"].items():
            print(f"[robust] {sd} " + "  ".join(f"{r}:" + "/".join(f"{v:+.1f}" if v is not None else "nan"
                  for v in d.values()) for r, d in row.items()) + "   (cd/cdextr/mlp %)", file=sys.stderr)
    print(f"[runs] failed twice: {len(res['failed_twice'])}  missing without a record: {len(res['unexplained_missing'])}",
          file=sys.stderr)
    print(f"VERDICT: {res['verdict']}", file=sys.stderr)
    if a.out:
        with open(a.out + ".partial", "w") as fh:
            json.dump(res, fh, indent=1)
        os.replace(a.out + ".partial", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
