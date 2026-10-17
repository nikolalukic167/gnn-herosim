#!/usr/bin/env python3
"""uncut_groups_s0_v1 reference baselines read (amendment 2026-10-17): DESCRIPTIVE rows against the cd_ib0_k100 bar, no label, no family.

  uncut_baselines_read.py --tools <checkout at 557d84bf> --cells DIR[,DIR...] --bar DIR --row name:kind:seeds [--row ...] [--out x.json]

The statistic is the frozen reader's (uncut_arm_gate_read.contrast at 557d84bf, imported): per topology the median over windows of 100 (arm - bar) / bar,
the arm's cell mean the mean over its seeds, whole run and after 360 s, then the median and mean over topologies and the topologies ahead of n. A cell with
a seed missing is missing for the arm; the failed cells (watchdog kills) of a row are listed, never dropped silently.
Kinds are the cell names without the `_k100` suffix (cd_ib0 for the bar, reactive, random, ra_mlp_same_arm1_nosplit ...).
"""
import argparse
import json
import os
import statistics as st
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tools", required=True)
    ap.add_argument("--cells", required=True)
    ap.add_argument("--bar", required=True)
    ap.add_argument("--row", action="append", required=True)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    sys.path.insert(0, os.path.join(a.tools, "scripts_cosim"))
    import uncut_arm_gate_read as R

    cells, failed = R.load(a.cells + "," + a.bar, "_k100")
    topos = sorted({k[2] for k in cells if k[0] == "cd_ib0"})
    if len(topos) != 12:
        raise SystemExit(f"FAIL LOUD: the bar has {len(topos)} topologies, not 12")
    out = {"label": "DESCRIPTIVE reference rows (no label, no family)", "bar": "cd_ib0_k100 (869828)", "topologies": topos, "rows": {}}
    print("DESCRIPTIVE reference rows vs cd_ib0_k100 (869828): per-topology median over 4 windows of % (arm - bar)/bar, then median / mean over topologies, topologies ahead of n; negative = faster than the bar")
    for spec in a.row:
        name, kind, seeds = spec.split(":")
        sd = [int(x) for x in seeds.split(",")]
        bad = sorted({(k[3], k[4], k[1]) for k in failed if k[0] == kind})
        out["rows"][name] = {"kind": kind, "seeds": sd, "failed_cells": [f"{w}{r}_s{s}@{t}" for (w, r, s), t in ((x, k[2]) for x in bad for k in failed if k[0] == kind and (k[3], k[4], k[1]) == x)], "cells": {}}
        for r in R.RUNGS:
            for s in R.STATS:
                c = R.contrast(cells, kind, sd, "cd_ib0", [0], r, topos, s)
                v = list(c["per_topology"].values())
                out["rows"][name]["cells"][f"{r}/{s}"] = {"median_pct": c["median_pct"], "mean_pct": st.mean(v) if v else None, "ahead": c["faster"], "n_topologies": c["n_topologies"], "n_cells": c["n_cells"], "p_unadjusted": c["p"], "per_topology": c["per_topology"]}
                m = f"{c['median_pct']:+.2f}" if c["median_pct"] is not None else "n/a"
                print(f"  {name:28s} {r:8s} {s:8s} median {m} %  mean {st.mean(v):+.2f} %  ahead {c['faster']}/{c['n_topologies']}  p {c['p']:.4g}  cells {c['n_cells']}" if v else f"  {name} {r} {s}: no cells")
        tails = [c.get("endTime", 0.0) - (c.get("arrival_end") or {}).get("last_arrival_s", 0.0) for k, c in cells.items() if k[0] == kind]
        out["rows"][name]["median_seconds_after_last_arrival"] = st.median(tails) if tails else None
        print(f"  {name}: median (end of run - last arrival) {st.median(tails):.0f} s over {len(tails)} cells" if tails else f"  {name}: no cells")
        nf = len([k for k in failed if k[0] == kind])
        print(f"  {name}: failed cells (watchdog) {nf}")
        out["rows"][name]["n_failed"] = nf
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
