"""scale_160_v1 live-gate reader (docs/lineages/scale_160_v1.md, "Gate" / "Primary family" / "Win").

Statistic as r1_attribution_v1_read.py: per topology the median over (window, seed) of the paired % of an arm's mean latency
against CD on the same (topology, window, seed-or-seed-0); then the median over topologies and an exact two-sided Wilcoxon.
  * primary family, Holm over 4: {ra_gnn_eng, ra_gnn_eng_physmp} vs CD x {moderate, heavy};
  * label per arm, from its two rungs (the registered wording "anything less is reported by label"):
      CD-FASTER      CD is Holm-confirmed faster (median > 0, Holm p < .05) at some rung;
      WIN            else median <= -5 % with Holm p < .05 at both rungs;
      DIRECTION-ONLY else median < 0 at both rungs;
      NOT-SEPARATED  otherwise;
  * descriptive, outside the family: self-predict and CD<-GNN (ra_gnn_eng_cdapply) vs CD, per-task decision cost, the spin counters
    (deferrals, reachability scale-up failures, averageWaitTime) and the cells whose deferrals exceed 5x CD's.
A cell that failed drops out of that arm's tests only; the read is repeated without every topology that has a failure (`sensitivity`).

  scale_160_v1_read.py --gate <dir> [<dir> ...] [--out read.json]
"""
import argparse
import json
import signal
import statistics as st
import sys
from typing import Dict, List, Optional

import r1_attribution_v1_read as R
import r1a_gate_spin as SP

RUNGS = ("moderate", "heavy")
ARMS = ("ra_gnn_eng", "ra_gnn_eng_physmp")
DESCRIPTIVE = ("selfpredict", "ra_gnn_eng_cdapply")
BAR_PCT, ALPHA = -5.0, 0.05


def label(tests: Dict[str, dict]) -> str:
    """tests: rung -> contrast with holm_p."""
    cs = [tests[r] for r in RUNGS]
    if any(c["median_pct"] is None for c in cs):
        return "NOT-SEPARATED"
    if any(c["median_pct"] > 0 and c["holm_p"] < ALPHA for c in cs):
        return "CD-FASTER"
    if all(c["median_pct"] <= BAR_PCT and c["holm_p"] < ALPHA for c in cs):
        return "WIN"
    if all(c["median_pct"] < 0 for c in cs):
        return "DIRECTION-ONLY"
    return "NOT-SEPARATED"


def family(cells: dict, topos: List[int]) -> dict:
    tests = {(a, r): R.contrast(cells, a, "cd", r, topos) for a in ARMS for r in RUNGS}
    adj = R.holm({f"{a}|{r}": c["p"] for (a, r), c in tests.items()})  # a missing arm counts as p = 1
    for (a, r), c in tests.items():
        c["holm_p"] = adj[f"{a}|{r}"]
    return {"holm_over": len(tests),
            "tests": {f"{a}|{r}": c for (a, r), c in tests.items()},
            "labels": {a: label({r: tests[(a, r)] for r in RUNGS}) for a in ARMS},
            "descriptive": {f"{a}|{r}": R.contrast(cells, a, "cd", r, topos) for a in DESCRIPTIVE for r in RUNGS}}


def timing(cells: dict) -> dict:
    out = {}
    for kind in sorted({k[0] for k in cells}):
        for rung in RUNGS:
            xs = [(s.get("decisionTiming") or {}).get("per_task_median_s") for k, s in cells.items() if k[0] == kind and k[4] == rung]
            xs = [x for x in xs if x is not None]
            out[f"{kind}|{rung}"] = {"cells": len(xs), "per_task_median_us": 1e6 * st.median(xs) if xs else None}
    return out


def spin(cells: dict) -> dict:
    out = {}
    for kind in sorted({k[0] for k in cells}):
        for rung in RUNGS:
            cs = {k: s for k, s in cells.items() if k[0] == kind and k[4] == rung}
            if not cs:
                continue
            flagged = 0
            for (_, seed, topo, win, r), s in cs.items():
                d, ref = SP.deferred(s), cells.get(("cd", 0, topo, win, rung))
                dc = SP.deferred(ref) if ref else None
                flagged += int(kind != "cd" and d is not None and dc is not None and d > SP.FLAG_X * max(dc, 1.0))
            out[f"{kind}|{rung}"] = {"cells": len(cs), "deferred_median": SP.med(SP.deferred(s) for s in cs.values()),
                                     "reach_fail_median": SP.med(SP.reach_failures(s) for s in cs.values()),
                                     "avg_wait_median_s": SP.med(float(s["averageWaitTime"]) for s in cs.values()),
                                     "cells_deferrals_over_5x_cd": flagged}
    return out


def read(gate) -> dict:
    cells, failed = R.load(gate)
    cells = {k: v for k, v in cells.items() if k[4] in RUNGS}
    failed = {k: v for k, v in failed.items() if k[4] in RUNGS}
    topos = sorted({k[2] for k in list(cells) + list(failed)})
    bad = sorted({k[2] for k in failed})
    sens = {k: v for k, v in cells.items() if k[2] not in bad}
    return {"gate": gate, "topologies": topos, "n_summaries": len(cells), "n_failed": len(failed),
            "arms": R.arm_table(cells, failed, {}), "main": family(cells, topos), "timing": timing(cells), "spin": spin(cells),
            "sensitivity": {"excluded_topologies": bad, "family": family(sens, [t for t in topos if t not in bad])}}


def print_report(r: dict) -> None:
    f = lambda x, n=4: "-" if x is None else f"{x:.{n}f}"
    print(f"{r['n_summaries']} summaries, {r['n_failed']} failed, {len(r['topologies'])} topologies")
    for name, fam in (("main", r["main"]), ("sensitivity", r["sensitivity"]["family"])):
        print(f"-- primary family ({name}, Holm over {fam['holm_over']}); labels {fam['labels']}")
        for k, c in {**fam["tests"], **fam["descriptive"]}.items():
            print(f"  {k:34s} median {('-' if c['median_pct'] is None else format(c['median_pct'], '+.2f') + ' %'):>10s} n={c['n_topologies']:2d} "
                  f"wins={c['wins']} p={f(c['p'])} holm={f(c.get('holm_p'))}")
    print("-- decision cost (median us per task) and spin")
    for k, t in r["timing"].items():
        s = r["spin"].get(k, {})
        print(f"  {k:34s} {f(t['per_task_median_us'], 0):>8s} us  deferred {SP.fmt(s.get('deferred_median'))}  reach-fail {SP.fmt(s.get('reach_fail_median'))}  "
              f"wait {f(s.get('avg_wait_median_s'))}  >5x CD {s.get('cells_deferrals_over_5x_cd')}")


def main() -> int:
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    r = read(a.gate)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(r, fh, indent=1)
    print_report(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
