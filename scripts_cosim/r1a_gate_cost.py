#!/usr/bin/env python3
"""r1_attribution_v1 live gate: wall time per cell per arm from a dry run, projected to the full grid.

Full grid per arm and rung = topologies x windows (x seeds for learned and seeded-CD arms). The projection is
sum(cells x median dry-run wall) / (concurrent jobs x cells in parallel per job); failed (watchdog / timeout) cells cost their wall time too.

  r1a_gate_cost.py --dry <dir> [--topologies 19 --windows 4 --seeds 2 --jobs 44 --par 60] [--out cost.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics as st
from collections import defaultdict

NAME = re.compile(r"cc40s(\d+)__(g\d)(light|moderate|heavy)__(.+)_s(\d+)$")
CLASSICAL = ("cd", "cd_declared", "locality", "batched", "selfpredict", "reactive")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry", required=True)
    ap.add_argument("--topologies", type=int, default=19)
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--jobs", type=int, default=44)
    ap.add_argument("--par", type=int, default=60)
    ap.add_argument("--out")
    a = ap.parse_args()
    wall = defaultdict(list)
    for f in glob.glob(os.path.join(a.dry, "*.summary.json")) + glob.glob(os.path.join(a.dry, "*.failed.json")):
        m = NAME.match(os.path.basename(f).rsplit(".", 2)[0])
        if m:
            w = json.load(open(f)).get("wallclock_s")
            if w is not None:
                wall[(m[4], m[3])].append(float(w))
    rows, total = [], 0.0
    for (kind, rung), ws in sorted(wall.items()):
        cells = a.topologies * a.windows * (1 if kind in CLASSICAL else a.seeds)
        med = st.median(ws)
        rows.append({"arm": kind, "rung": rung, "dry_cells": len(ws), "median_wall_s": med, "max_wall_s": max(ws), "full_cells": cells,
                     "cell_hours": cells * med / 3600})
        total += cells * med
    slots = a.jobs * a.par
    out = {"rows": rows, "total_cell_hours": total / 3600, "jobs": a.jobs, "par": a.par,
           "wall_hours_at_jobs": total / slots / 3600, "wall_hours_at_half_jobs": total / (slots / 2) / 3600,
           "note": "dry-run wall under the dry run's own load; the full gate runs --par cells per node"}
    for r in rows:
        print(f"{r['arm']:24s} {r['rung']:9s} dry n={r['dry_cells']} median {r['median_wall_s']:7.0f} s max {r['max_wall_s']:7.0f} s  "
              f"full {r['full_cells']:4d} cells = {r['cell_hours']:7.1f} cell-h")
    print(f"total {out['total_cell_hours']:.0f} cell-hours; {a.jobs} jobs x {a.par} parallel -> {out['wall_hours_at_jobs']:.1f} h; half the jobs -> {out['wall_hours_at_half_jobs']:.1f} h")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
