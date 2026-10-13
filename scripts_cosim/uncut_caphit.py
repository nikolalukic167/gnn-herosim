#!/usr/bin/env python3
"""uncut_groups_s0_v1: how often does cd_exactS's exact search hit its 100,000-plan cap, read from the k = 0 hand cells' own scheduler counters (no reconstruction).

The scheduler counts, per cell: `pg_exact_batches` (batches of >= 2 tasks whose slate product fitted under the cap and was enumerated), `pg_exact_fallbacks` (batches that
exceeded the cap and ran the expansion instead = cap hits), `pg_exact_plans` (plans enumerated in the exact batches). The summary carries the cap hits and the exact
batches only as per-cell totals, NOT by served-size bucket; the served batch counts by bucket are `pgb_whole_*` / `pgb_split_*`. So the bucket-wise cap-hit count is not
answerable from a summary, and this reader reports totals per rung (and window) beside the served batch counts and says so.

  uncut_caphit.py CELLS_DIR [--arms cd_exactS_ib0,cd_exactS,cd_exactS_ib0_prune] [--json OUT]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

BUCKETS = ("le10", "11_50", "51_100", "gt100")
NAME = re.compile(r"cc40s(\d+)__(g\d)(heavy|moderate)__(.+)_s(\d+)\.summary\.json$")


def cell_row(summary: dict) -> dict:
    sc = summary["schedulerCounters"]
    for k in ("pg_exact_batches", "pg_exact_fallbacks", "pg_exact_plans"):
        if k not in sc:
            raise KeyError(f"FAIL LOUD: schedulerCounters has no {k}")
    served = {b: int(sc.get(f"pgb_whole_{b}", 0)) + int(sc.get(f"pgb_split_{b}", 0)) for b in BUCKETS}
    return {"exact_batches": int(sc["pg_exact_batches"]), "cap_hits": int(sc["pg_exact_fallbacks"]), "exact_plans": int(sc["pg_exact_plans"]), "served": served,
            "max_batch": int(sc.get("pgb_max_batch", 0))}


def pool(rows: list) -> dict:
    if not rows:
        return {"cells": 0}
    eb, ch = sum(r["exact_batches"] for r in rows), sum(r["cap_hits"] for r in rows)
    served = {b: sum(r["served"][b] for r in rows) for b in BUCKETS}
    return {"cells": len(rows), "exact_batches": eb, "cap_hits": ch, "cap_hit_share_of_scored_batches": ch / (eb + ch) if eb + ch else None,
            "plans_per_exact_batch": sum(r["exact_plans"] for r in rows) / eb if eb else None, "served_batches": served,
            "served_batches_gt_7_lower_bound_note": "a cap hit needs a slate product above 100,000 = 5^n for n >= 8 at top-K 5; the summary has no bucket for 8-10",
            "cap_hits_per_served_batch_gt10": ch / (served["11_50"] + served["51_100"] + served["gt100"]) if sum(served[b] for b in BUCKETS[1:]) else None,
            "max_batch_seen": max(r["max_batch"] for r in rows)}


def read_dir(cells: Path, arms: list) -> dict:
    out: dict = {}
    failed = defaultdict(int)
    for f in sorted(glob.glob(str(cells / "*.failed.json"))):
        m = re.search(r"cc40s\d+__g\d(heavy|moderate)__(.+)_s\d+\.failed\.json$", os.path.basename(f))
        if m and m.group(2) in arms:
            failed[(m.group(2), m.group(1))] += 1
    by = defaultdict(list)
    for f in sorted(glob.glob(str(cells / "*.summary.json"))):
        m = NAME.search(os.path.basename(f))
        if not m or m.group(4) not in arms:
            continue
        by[(m.group(4), m.group(3), m.group(2))].append(cell_row(json.load(open(f))))
    for arm in arms:
        out[arm] = {}
        for rung in ("heavy", "moderate"):
            rows = [r for w in ("g0", "g1", "g2", "g3") for r in by.get((arm, rung, w), [])]
            out[arm][rung] = {"pooled": pool(rows), "failed_cells": failed.get((arm, rung), 0),
                              "by_window": {w: pool(by.get((arm, rung, w), [])) for w in ("g0", "g1", "g2", "g3")}}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cells")
    ap.add_argument("--arms", default="cd_exactS_ib0,cd_exactS,cd_exactS_ib0_prune")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    res = read_dir(Path(a.cells), a.arms.split(","))
    for arm, d in res.items():
        for rung, x in d.items():
            p = x["pooled"]
            if p["cells"]:
                print(f"{arm:24s} {rung:8s} cells {p['cells']:3d} (+{x['failed_cells']} failed)  exact {p['exact_batches']:7d}  cap hits {p['cap_hits']:6d}  "
                      f"share {100 * p['cap_hit_share_of_scored_batches']:.1f}%  plans/exact batch {p['plans_per_exact_batch']:.0f}  served {p['served_batches']}")
            else:
                print(f"{arm:24s} {rung:8s} no finished cell (+{x['failed_cells']} failed)")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
