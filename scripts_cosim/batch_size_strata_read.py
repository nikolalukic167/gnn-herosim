#!/usr/bin/env python3
"""Does the live learned-vs-CD gap depend on the live batch size? (docs/lineages/raw_plan_v2.md, pipeline review)

  batch_size_strata_read.py --dirs <gate dirs> --arm rawgnn_selfref --ref cd [--out read.json]

Training batches are all 10 tasks; live batches are smaller (tasks / prefix_batches from the arm's own schedulerCounters).
Per rung, per (topology, window) cell: mean batch size of the learned arm (mean over seeds) against the paired % of the
arm's mean latency vs the reference. Reports Spearman rho and the gap of the lower vs upper half of cells by batch size.
Descriptive, observational: cells differ in more than batch size, so this orders the work and does not close it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from statistics import mean, median

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fresh_topo_burst_v1_read as F  # noqa: E402


def rank(xs):
    o = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    for k, i in enumerate(o):
        r[i] = float(k)
    return r


def spearman(a, b):
    ra, rb = rank(a), rank(b)
    ma, mb = mean(ra), mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--topos", nargs="*", type=int)
    ap.add_argument("--out")
    a = ap.parse_args()
    s = {}
    for d in a.dirs:
        s.update(F._summaries(d))
    cells = defaultdict(lambda: defaultdict(list))
    for (t, w, kind, seed), r in s.items():
        if a.topos and t not in a.topos:
            continue
        if kind == a.arm:
            c = r.get("schedulerCounters") or {}
            b = c.get("prefix_batches")
            if not b:
                raise SystemExit(f"FAIL LOUD: {t} {w} {kind} s{seed} has no prefix_batches")
            cells[(t, w)]["bs"].append(r["num_tasks"] / b)
            cells[(t, w)]["arm"].append(float(r["averageElapsedTime"]))
        elif kind == a.ref:
            cells[(t, w)]["ref"].append(float(r["averageElapsedTime"]))
    res = {}
    for rung in ("x20", "x30", "x50"):
        rows = []
        for (t, w), v in sorted(cells.items()):
            if not w.endswith(rung) or not v["arm"] or not v["ref"]:
                continue
            rows.append({"topology": t, "window": w, "batch": mean(v["bs"]),
                         "pct": 100.0 * (mean(v["arm"]) - mean(v["ref"])) / mean(v["ref"])})
        if len(rows) < 8:
            res[rung] = {"n": len(rows)}
            continue
        rows.sort(key=lambda r: r["batch"])
        h = len(rows) // 2
        res[rung] = {"n": len(rows), "batch_min": rows[0]["batch"], "batch_max": rows[-1]["batch"],
                     "batch_median": median(r["batch"] for r in rows),
                     "spearman_batch_vs_pct": spearman([r["batch"] for r in rows], [r["pct"] for r in rows]),
                     "median_pct_small_half": median(r["pct"] for r in rows[:h]),
                     "median_pct_large_half": median(r["pct"] for r in rows[h:]), "rows": rows}
        x = res[rung]
        print(f"{rung}: n={x['n']} batch {x['batch_min']:.2f}..{x['batch_max']:.2f} (median {x['batch_median']:.2f}) "
              f"rho={x['spearman_batch_vs_pct']:+.2f}  gap small-batch half {x['median_pct_small_half']:+.1f} % "
              f"vs large-batch half {x['median_pct_large_half']:+.1f} %", file=sys.stderr)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
