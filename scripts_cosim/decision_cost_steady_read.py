"""decision_cost_s0_v1 steady-state follow-up (Amendment 4): cd_exactS_ib0, ra_gnn_eng_selfsearch, ra_gnn_eng_nosplit at k = 1, 48 cells each at pin 95202b74.
Step 0, a STOP on mismatch: whole-run mean latency reproduces the grid's k = 1 cells to 1e-9 relative.
Then each arm vs cd_expand and vs cd_ib0 (the attribution dir, pin 0fb7c4b8, same cells as the grid k = 1 for cd_ib0), whole-run / after 360 s / after the 5 % dispatch point,
paired % per topology (median over windows), median over topologies, sign count, exact Wilcoxon, Holm over the arm x ref x rung family per metric. Descriptive, no bar.

  decision_cost_steady_read.py --steady <dir> --attrib <dir> --grid <dir> [--out read.json]
"""
import argparse
import json
import sys

import r1_attribution_v1_read as R
from decision_cost_read import RUNGS, TOPOS, WINDOWS
from decision_cost_attrib_read import METRICS, contrast, load_dir

ARMS = ("cd_exactS_ib0", "ra_gnn_eng_selfsearch", "ra_gnn_eng_nosplit")
REFS = ("cd_expand", "cd_ib0")


def read(steady_dir, attrib_dir, grid_dir):
    st_, failed = load_dir(steady_dir)
    at, _ = load_dir(attrib_dir)
    grid, _ = load_dir(grid_dir)
    worst, n, missing = 0.0, 0, []
    for (a, k, t, w, r), c in st_.items():
        g = grid.get((a, 1.0, t, w, r))
        if g is None:
            missing.append((a, t, w, r))
            continue
        x, y = float(c["averageElapsedTime"]), float(g["averageElapsedTime"])
        worst = max(worst, abs(x - y) / y)
        n += 1
    rep = {"compared": n, "max_rel_diff": worst, "grid_cell_missing": missing, "ok": n > 0 and worst <= 1e-9 and not missing}
    res = {"reproduction": rep, "failed": failed, "cells": {a: sum(1 for k in st_ if k[0] == a) for a in ARMS}}
    if not rep["ok"]:
        res["verdict"] = "STOP: whole-run k = 1 values do not reproduce the grid"
        return res
    cells = dict(at)
    cells.update(st_)
    tests = {m: {f"{a} vs {ref}|{r}": contrast(cells, a, ref, r, m) for a in ARMS for ref in REFS for r in RUNGS} for m in METRICS}
    for m in METRICS:
        adj = R.holm({k: c["p"] for k, c in tests[m].items()})
        for k, c in tests[m].items():
            c["holm_p"] = adj[k]
    res["tests"] = tests
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steady", required=True)
    ap.add_argument("--attrib", required=True)
    ap.add_argument("--grid", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    r = read(a.steady, a.attrib, a.grid)
    if a.out:
        json.dump(r, open(a.out, "w"), indent=1)
    f = lambda x: "-" if x is None else f"{x:.4f}"
    print("-- cells", r["cells"], "failed", r["failed"], "\n-- reproduction", r["reproduction"])
    if "tests" not in r:
        print("--", r["verdict"])
        return 2
    for m, ts in r["tests"].items():
        print(f"== {m}")
        for k, c in ts.items():
            mm = "-" if c["median_pct"] is None else format(c["median_pct"], "+.2f") + " %"
            print(f"  {k:46s} median {mm:>9s} n={c['n_topologies']} wins={c['wins']} p={f(c['p'])} holm={f(c['holm_p'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
