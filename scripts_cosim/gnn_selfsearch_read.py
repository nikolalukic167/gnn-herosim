"""gnn_selfsearch diagnostic reader (docs/lineages/accel_nosplit_v1.md). Written before the data lands.

Arm ra_gnn_eng_selfsearch (seed 1) on 16301-16312, g0, both rungs. Per rung: the median over topologies of the paired % of its mean latency against
cd (rerun at the arm's pin), ra_gnn_eng_nosplit (same job), cd_exactS and ra_gnn_eng_cdxapply (reused from other pins, descriptive), the sign count
(wins = topologies where the arm is faster) and an exact two-sided Wilcoxon. Negative = the arm is faster. Trigger (signed): both rungs <= -5 % vs cd.
Counts first: summaries and failures per arm and rung, then exact vs ascent batches, plan-changed share and decision time.
A topology outside 16301-16312 is refused. Optional --cd-ref compares the rerun cd cell by cell against an earlier cd directory (identity).

  gnn_selfsearch_read.py --gate <dir> [<dir> ...] [--cd-ref <dir>] [--out read.json]
"""
import argparse
import json
import os
import signal
import statistics as st
import sys

import r1_attribution_v1_read as R

ARM = "ra_gnn_eng_selfsearch"
REFS = ("cd", "ra_gnn_eng_nosplit", "cd_exactS", "ra_gnn_eng_cdxapply")
RUNGS = ("moderate", "heavy")
TOPO_RANGE = (16301, 16312)
BAR_PCT = -5.0


def check_topologies(keys):
    topos = sorted({k[2] for k in keys})
    out = [t for t in topos if not TOPO_RANGE[0] <= t <= TOPO_RANGE[1]]
    if out:
        raise SystemExit(f"FAIL LOUD: topologies outside {TOPO_RANGE[0]}-{TOPO_RANGE[1]}: {out}")
    return topos


def search_counts(cells, rung):
    cs = [s["schedulerCounters"] for k, s in cells.items() if k[0] == ARM and k[4] == rung]
    tot = lambda n: sum(int(c.get(n) or 0) for c in cs)
    b = tot("ss_batches")
    dt = [s["decisionTiming"] for k, s in cells.items() if k[0] == ARM and k[4] == rung]
    return {"cells": len(cs), "batches": b, "exact_batches": tot("ss_exact_batches"), "ascent_batches": tot("ss_ascent_batches"),
            "slate_declared_batches": tot("slate_declared_batches"), "changed_batch_share": tot("ss_changed_batches") / b if b else None,
            "changed_task_share": tot("ss_changed_tasks") / tot("ss_tasks") if tot("ss_tasks") else None,
            "plans_scored": tot("ss_scored"), "decision_per_task_median_s": st.median(d["per_task_median_s"] for d in dt) if dt else None,
            "decision_per_task_mean_s": st.mean(d["per_task_mean_s"] for d in dt) if dt else None}


def cd_identity(cells, ref_cells):
    """The rerun cd against the earlier cd cells: every one should match on mean latency."""
    bad, n = [], 0
    for k, s in cells.items():
        if k[0] != "cd":
            continue
        r = ref_cells.get(k)
        if r is None:
            continue
        n += 1
        if r["averageElapsedTime"] != s["averageElapsedTime"] or r["num_tasks"] != s["num_tasks"]:
            bad.append(list(k))
    return {"compared": n, "different": len(bad), "different_cells": bad[:10]}


def read(gate, cd_ref=None):
    cells, failed = R.load(gate)
    cells = {k: v for k, v in cells.items() if k[4] in RUNGS}
    failed = {k: v for k, v in failed.items() if k[4] in RUNGS}
    topos = check_topologies(list(cells) + list(failed))
    out = {"gate": gate, "topologies": topos, "n_summaries": len(cells), "n_failed": len(failed),
           "arm_table": R.arm_table(cells, failed, {}), "search": {r: search_counts(cells, r) for r in RUNGS},
           "contrasts": {f"{ARM} vs {b}|{r}": R.contrast(cells, ARM, b, r, topos) for b in REFS for r in RUNGS}}
    out["trigger"] = all((out["contrasts"][f"{ARM} vs cd|{r}"]["median_pct"] or 0) <= BAR_PCT for r in RUNGS)
    if cd_ref:
        ref_cells, _ = R.load(cd_ref)
        out["cd_identity"] = cd_identity(cells, ref_cells)
    return out


def print_report(r):
    f = lambda x, n=4: "-" if x is None else f"{x:.{n}f}"
    print(f"{r['n_summaries']} summaries, {r['n_failed']} failed, {len(r['topologies'])} topologies")
    print("-- cells per arm and rung (finished/failed)")
    for k, a in r["arm_table"].items():
        print(f"  {k:36s} {a['finished']}/{a['failed']}" + (f"  failed why {a['failed_why']}" if a["failed"] else ""))
    print("-- search counts (selfsearch)")
    for rung, c in r["search"].items():
        print(f"  {rung}: {c}")
    if "cd_identity" in r:
        print(f"-- cd rerun vs earlier cd: {r['cd_identity']}")
    print("-- paired % of selfsearch vs reference (negative = selfsearch faster); wins = topologies where it is faster")
    for k, c in r["contrasts"].items():
        m = "-" if c["median_pct"] is None else format(c["median_pct"], "+.2f") + " %"
        print(f"  {k:44s} median {m:>10s} n={c['n_topologies']:2d} wins={c['wins']} p={f(c['p'])}")
    print(f"-- trigger (both rungs <= {BAR_PCT} % vs cd): {r['trigger']}")


def main():
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate", nargs="+", required=True)
    ap.add_argument("--cd-ref", nargs="+", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    r = read(a.gate, a.cd_ref)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(r, fh, indent=1)
    print_report(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
