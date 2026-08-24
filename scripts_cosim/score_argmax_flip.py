#!/usr/bin/env python3
"""
Does the identity of the makespan's argmax branch flip across plans?

Follow-up diagnostic to score_makespan_vs_sum.py, separating two things the
disagreement fraction cannot: "sum and max pick different plans" versus "the max
is a joint property of the plan at all". If the worst branch is the same task in
(nearly) every plan, then max_b T_b = T_{b*} for a fixed b* and the makespan
objective collapses to a single-task problem - pointwise-learnable regardless of
how often it disagrees with the sum. Non-decomposability requires the argmax to
move with the plan. Both extremes degenerate: identical branches make max track
sum; one dominant branch fixes the argmax.

Per dataset: argmax branch of a plan = task with the largest makespan
contribution (done - min dispatched). flip_rate = 1 - modal argmax frequency
across plans (K=4 tasks => range [0, 0.75]).

Registered decision rule (fixed 2026-08-24, before this script ran):
  reopen M3 for width scaling ONLY IF mean flip_rate >= 0.25 AND flip_rate
  correlates with disagreement across the 200 datasets; otherwise M3 closes
  permanently. Expectation registered alongside: flat.

Usage:
  pipenv run python scripts_cosim/score_argmax_flip.py \
      --base-dir simulation_data/gnn_datasets_4tasks_m3_makespan_pilot \
      [--report PATH] [--rel-tol 1e-9]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional

from score_makespan_vs_sum import load_rows, score_rows, is_spread


def argmax_branch(row: dict) -> int:
    tt = row["task_times"]
    base = min(t[1] for t in tt)
    return max(tt, key=lambda t: t[2] - base)[0]


def flip_rate(rows: List[dict]) -> Optional[dict]:
    if len(rows) < 2:
        return None
    counts = Counter(argmax_branch(r) for r in rows)
    modal_task, modal_n = counts.most_common(1)[0]
    return {
        "n_plans": len(rows),
        "flip_rate": 1.0 - modal_n / len(rows),
        "modal_argmax_task": int(modal_task),
        "n_distinct_argmax_tasks": len(counts),
    }


def pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-dir", required=True, nargs="+")
    ap.add_argument("--report", default=None)
    ap.add_argument("--rel-tol", type=float, default=1e-9)
    args = ap.parse_args()

    ds_records = []
    for base in args.base_dir:
        base_path = Path(base)
        ds_dirs = sorted(base_path.glob("ds_*"))
        if not ds_dirs:
            raise RuntimeError(f"No ds_* under {base_path}")
        for ds in ds_dirs:
            rows = load_rows(ds / "placements" / "placements.jsonl")
            full_gate = score_rows(rows, args.rel_tol)
            if full_gate is None:
                continue
            spread_rows = [r for r in rows if is_spread(r)]
            rec = {
                "dataset": str(ds),
                "full_flip": flip_rate(rows),
                "spread_flip": flip_rate(spread_rows),
                "full_disagree": full_gate["disagree"],
                "full_regret": full_gate["makespan_regret_of_sum_argmin"],
            }
            spread_gate = score_rows(spread_rows, args.rel_tol)
            rec["spread_disagree"] = spread_gate["disagree"] if spread_gate else None
            rec["spread_regret"] = (
                spread_gate["makespan_regret_of_sum_argmin"] if spread_gate else None
            )
            ds_records.append(rec)

    flips = [r["full_flip"]["flip_rate"] for r in ds_records]
    regrets = [r["full_regret"] for r in ds_records]
    disagree_flags = [1.0 if r["full_disagree"] else 0.0 for r in ds_records]
    mean_flip = sum(flips) / len(flips)
    spread_flips = [
        r["spread_flip"]["flip_rate"] for r in ds_records if r["spread_flip"]
    ]
    dis_group = [f for f, d in zip(flips, disagree_flags) if d]
    agr_group = [f for f, d in zip(flips, disagree_flags) if not d]

    agg = {
        "n_datasets": len(ds_records),
        "mean_flip_rate": mean_flip,
        "median_flip_rate": sorted(flips)[len(flips) // 2],
        "mean_flip_rate_spread": sum(spread_flips) / len(spread_flips)
        if spread_flips
        else None,
        "frac_flip_zero": sum(1 for f in flips if f == 0.0) / len(flips),
        "frac_flip_ge_0.25": sum(1 for f in flips if f >= 0.25) / len(flips),
        "corr_flip_vs_regret": pearson(flips, regrets),
        "corr_flip_vs_disagree": pearson(flips, disagree_flags),
        "mean_flip_disagreeing": sum(dis_group) / len(dis_group) if dis_group else None,
        "mean_flip_agreeing": sum(agr_group) / len(agr_group) if agr_group else None,
    }
    report = {
        "registered_rule": {
            "reopen_iff": "mean flip_rate >= 0.25 AND flip correlates with disagreement",
            "registered_expectation": "flat",
        },
        "aggregate": agg,
        "datasets": ds_records,
    }

    fires = mean_flip >= 0.25
    print(f"datasets: {agg['n_datasets']}")
    print(
        f"mean flip_rate: {mean_flip:.4f} (median {agg['median_flip_rate']:.4f}, "
        f"spread {agg['mean_flip_rate_spread']:.4f}) "
        f"[threshold 0.25: {'MET' if fires else 'NOT met'}]"
    )
    print(
        f"flip==0 in {agg['frac_flip_zero']:.1%} of datasets; "
        f"flip>=0.25 in {agg['frac_flip_ge_0.25']:.1%}"
    )
    print(
        f"corr(flip, makespan-regret) = {agg['corr_flip_vs_regret']}, "
        f"corr(flip, disagree) = {agg['corr_flip_vs_disagree']}"
    )
    print(
        f"mean flip | disagreeing = {agg['mean_flip_disagreeing']}, "
        f"| agreeing = {agg['mean_flip_agreeing']}"
    )
    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=1)
        print(f"report written: {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
