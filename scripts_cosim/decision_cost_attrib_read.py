"""decision_cost_s0_v1 attribution read (Amendment 5): is the cdxapply win at k = 1 the GNN seed's or the expansion's?

Cells (attrib dir): cdxapply_k100, cd_ib0_k100, cd_expand_k100 x 12 topologies x g0,g1 x moderate,heavy = 144. The twin-seed + expansion control does not exist in the harness (not built).
Step 0, a STOP on mismatch: cdxapply_k100 and cd_ib0_k100 whole-run mean latency must reproduce the grid's k = 1 cells (grid dir) to 1e-9 relative.
Unit and statistic as in decision_cost_read.py (topology; median over windows of the paired %; median over topologies; sign count; exact Wilcoxon; negative = first arm faster).
Metrics: whole-run mean latency; mean latency of tasks arriving after 360 s; after the 5 % dispatch point (descriptive: summary `steady_state`).
Holm over {cdxapply vs cd_expand, cdxapply vs cd_ib0} x {moderate, heavy} per metric.
Labels per rung (whole-run): cdxapply vs cd_expand median >= -1 % (cd_expand within 1 pp or faster) -> SEARCH-WIN-NOT-GNN; median <= -2 % and Holm p < .05 -> GNN-SEED-WIN; otherwise UNRESOLVED. MP is not assignable: no twin-seed control.
Collapse count per arm and rung: cell mean > 3 x the same cell's cd mean (grid, k = 1), or end - last arrival > 120 s.

  decision_cost_attrib_read.py --attrib <dir> --grid <dir> [--out read.json]
"""
import argparse
import json
import statistics as st
import sys

import r1_attribution_v1_read as R
from decision_cost_read import TOPOS, WINDOWS, RUNGS, split_kind

NEW = ("ra_gnn_eng_cdxapply", "cd_ib0", "cd_expand")
METRICS = ("whole_run", "after_360s", "after_p5_dispatch")


def load_dir(d):
    raw, failed = R.load(d)
    cells = {}
    for (kind, seed, topo, win, rung), v in raw.items():
        arm, k = split_kind(kind)
        cells[(arm, k, topo, win, rung)] = v
    return cells, len(failed)


def metric(cell, m):
    if m == "whole_run":
        return float(cell["averageElapsedTime"])
    ss = (cell.get("steady_state") or {}).get(m)
    return float(ss["mean"]) if ss and ss.get("mean") is not None else None


def paired(cells, a, b, rung, m):
    out = {}
    for t in TOPOS:
        v = []
        for w in WINDOWS:
            ca, cb = cells.get((a, 1.0, t, w, rung)), cells.get((b, 1.0, t, w, rung))
            if ca is None or cb is None:
                continue
            x, y = metric(ca, m), metric(cb, m)
            if x is None or y is None:
                continue
            v.append(100.0 * (x - y) / y)
        if v:
            out[t] = st.median(v)
    return out


def contrast(cells, a, b, rung, m):
    per = paired(cells, a, b, rung, m)
    if not per:
        return {"n_topologies": 0, "median_pct": None, "p": None, "wins": None}
    v = list(per.values())
    return {"n_topologies": len(v), "median_pct": st.median(v), "p": R.exact_wilcoxon(v), "wins": sum(1 for x in v if x < 0),
            "per_topology": {str(t): round(x, 2) for t, x in sorted(per.items())}}


def reproduction(att, grid):
    worst, n, missing = 0.0, 0, []
    for arm in ("ra_gnn_eng_cdxapply", "cd_ib0"):
        for (a, k, t, w, r), c in att.items():
            if a != arm:
                continue
            g = grid.get((a, 1.0, t, w, r))
            if g is None:
                missing.append((a, t, w, r))
                continue
            x, y = float(c["averageElapsedTime"]), float(g["averageElapsedTime"])
            worst = max(worst, abs(x - y) / y)
            n += 1
    return {"compared": n, "max_rel_diff": worst, "grid_cell_missing": missing, "ok": n > 0 and worst <= 1e-9 and not missing}


def collapse(att, grid):
    out = {}
    for arm in NEW:
        for r in RUNGS:
            n = bad = 0
            for (a, k, t, w, rr), c in att.items():
                if a != arm or rr != r:
                    continue
                n += 1
                cd = grid.get(("cd", 1.0, t, w, r))
                ae = c.get("arrival_end") or {}
                slow = cd is not None and float(c["averageElapsedTime"]) > 3.0 * float(cd["averageElapsedTime"])
                late = ae.get("end_time_s") is not None and ae.get("last_arrival_s") is not None and float(ae["end_time_s"]) - float(ae["last_arrival_s"]) > 120.0
                bad += 1 if (slow or late) else 0
            out[f"{arm}|{r}"] = {"cells": n, "collapsed": bad}
    return out


def read(att_dir, grid_dir):
    att, att_failed = load_dir(att_dir)
    grid, _ = load_dir(grid_dir)
    rep = reproduction(att, grid)
    res = {"reproduction": rep, "failed": att_failed, "cells": {arm: sum(1 for k in att if k[0] == arm) for arm in NEW}}
    if not rep["ok"]:
        res["verdict"] = "STOP: whole-run k = 1 values do not reproduce the grid"
        return res
    tests = {m: {f"vs {ref}|{r}": contrast(att, "ra_gnn_eng_cdxapply", ref, r, m) for ref in ("cd_expand", "cd_ib0") for r in RUNGS} for m in METRICS}
    for m in METRICS:
        adj = R.holm({k: c["p"] for k, c in tests[m].items()})
        for k, c in tests[m].items():
            c["holm_p"] = adj[k]
    lab = {}
    for r in RUNGS:
        c = tests["whole_run"][f"vs cd_expand|{r}"]
        if c["median_pct"] is None:
            lab[r] = "INCOMPLETE"
        elif c["median_pct"] >= -1.0:
            lab[r] = "SEARCH-WIN-NOT-GNN"
        elif c["median_pct"] <= -2.0 and c["holm_p"] is not None and c["holm_p"] < 0.05:
            lab[r] = "GNN-SEED-WIN"
        else:
            lab[r] = "UNRESOLVED"
    res.update(tests=tests, labels=lab, collapse=collapse(att, grid),
               arm_vs_arm_cd_expand_vs_cd_ib0={m: {r: contrast(att, "cd_expand", "cd_ib0", r, m) for r in RUNGS} for m in METRICS},
               twin_seed_control="does not exist in the harness (ra_twin_eng_cdxapply is not a registered kind); not built; MP is not assignable")
    return res


def print_report(r):
    f = lambda x: "-" if x is None else f"{x:.4f}"
    print("-- cells", r["cells"], "failed", r["failed"])
    print("-- reproduction vs grid k = 1:", r["reproduction"])
    if "tests" not in r:
        print("--", r["verdict"])
        return
    for m, ts in r["tests"].items():
        print(f"== {m}")
        for k, c in ts.items():
            mm = "-" if c["median_pct"] is None else format(c["median_pct"], "+.2f") + " %"
            print(f"  cdxapply {k:24s} median {mm:>9s} n={c['n_topologies']} wins={c['wins']} p={f(c['p'])} holm={f(c['holm_p'])}")
    print("== cd_expand vs cd_ib0 (descriptive)")
    for m, d in r["arm_vs_arm_cd_expand_vs_cd_ib0"].items():
        for rg, c in d.items():
            mm = "-" if c["median_pct"] is None else format(c["median_pct"], "+.2f") + " %"
            print(f"  {m:18s} {rg:9s} median {mm:>9s} wins={c['wins']} p={f(c['p'])}")
    print("-- collapse (mean > 3 x cd, or end - last arrival > 120 s)", r["collapse"])
    print("-- labels (whole-run, per rung)", r["labels"])
    print("--", r["twin_seed_control"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attrib", required=True)
    ap.add_argument("--grid", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    r = read(a.attrib, a.grid)
    if a.out:
        json.dump(r, open(a.out, "w"), indent=1)
    print_report(r)
    return 0 if "tests" in r else 2


if __name__ == "__main__":
    sys.exit(main())
