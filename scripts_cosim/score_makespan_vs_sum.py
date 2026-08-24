#!/usr/bin/env python3
"""
Re-score a placement sweep under a per-batch makespan instead of the sum.

The 4 tasks of every co-sim dataset are already a fan-out of width 4: concurrent,
independent, jointly placed. The recorded objective (`rtt` = sum of per-task
elapsed) is what makes the target branch-decomposable; a makespan
(max task done - min task dispatched) over the *same* simulated times is the
cheapest possible test of the M3 hypothesis - min-sum and min-max assignment are
different problems, so if the argmins disagree often enough, a max-composed
objective has non-pointwise structure without any new physics.

Requires sweeps generated with HEROSIM_RETAIN_TASK_TIMES=1 (rows carry
`task_times`: [[task_id, dispatched, done], ...]). Fails loud on rows without it.

Registered decision rule (written before the pilot ran, 2026-08-24):
  - full-sweep disagreement fraction >= 0.10, or
  - spread-plans (all tasks on distinct nodes) disagreement fraction >= 0.05
  where a dataset "disagrees" iff NO sum-optimal plan is makespan-optimal
  (conservative tie handling: ties broken in favor of agreement).

Usage:
  pipenv run python scripts_cosim/score_makespan_vs_sum.py \
      --base-dir simulation_data/gnn_datasets_4tasks_m3_makespan_pilot \
      [--report PATH] [--rel-tol 1e-9]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_rows(jsonl_path: Path) -> List[dict]:
    rows = []
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "task_times" not in row:
                raise RuntimeError(
                    f"{jsonl_path}:{i + 1} has no task_times - this sweep was not "
                    "generated with HEROSIM_RETAIN_TASK_TIMES=1 and cannot be "
                    "re-scored under a makespan"
                )
            rows.append(row)
    return rows


def makespan(row: dict) -> float:
    tt = row["task_times"]
    if not tt:
        raise RuntimeError("empty task_times row")
    done = max(t[2] for t in tt)
    dispatched = min(t[1] for t in tt)
    return done - dispatched


def sum_elapsed(row: dict) -> float:
    return sum(t[2] - t[1] for t in row["task_times"])


def is_spread(row: dict) -> bool:
    nodes = [v[0] for v in row["placement_plan"].values()]
    return len(set(nodes)) == len(nodes)


def score_rows(rows: List[dict], rel_tol: float) -> Optional[dict]:
    """One dataset (or one restriction of it). None when unscoreable (<2 plans)."""
    if len(rows) < 2:
        return None
    sums = [row["rtt"] for row in rows]
    maxes = [makespan(row) for row in rows]

    min_s = min(sums)
    min_m = min(maxes)
    tol_s = rel_tol * max(abs(min_s), 1e-12)
    tol_m = rel_tol * max(abs(min_m), 1e-12)

    s_opt_idx = [i for i, s in enumerate(sums) if s <= min_s + tol_s]
    # Conservative: the sum-argmin "achieves" the makespan optimum if ANY
    # sum-optimal plan is makespan-optimal.
    best_m_among_s_opt = min(maxes[i] for i in s_opt_idx)
    disagree = best_m_among_s_opt > min_m + tol_m
    m_regret_of_s_opt = (best_m_among_s_opt - min_m) / min_m if min_m > 0 else 0.0

    # Reverse direction, for context only.
    m_opt_idx = [i for i, m in enumerate(maxes) if m <= min_m + tol_m]
    best_s_among_m_opt = min(sums[i] for i in m_opt_idx)
    s_regret_of_m_opt = (best_s_among_m_opt - min_s) / min_s if min_s > 0 else 0.0

    return {
        "n_plans": len(rows),
        "disagree": bool(disagree),
        "makespan_regret_of_sum_argmin": m_regret_of_s_opt,
        "sum_regret_of_makespan_argmin": s_regret_of_m_opt,
        "n_sum_optimal": len(s_opt_idx),
        "n_makespan_optimal": len(m_opt_idx),
    }


def check_sum_consistency(rows: List[dict], jsonl_path: Path) -> float:
    """Recorded rtt must equal the sum of retained per-task elapsed. Max rel err."""
    worst = 0.0
    for row in rows:
        s = sum_elapsed(row)
        rel = abs(s - row["rtt"]) / max(abs(row["rtt"]), 1e-12)
        worst = max(worst, rel)
    if worst > 1e-6:
        raise RuntimeError(
            f"{jsonl_path}: recorded rtt disagrees with sum of retained task times "
            f"(max rel err {worst:.3e}) - task_times are not the objective's own "
            "per-task decomposition; refusing to score"
        )
    return worst


def aggregate(per_ds: List[dict]) -> dict:
    n = len(per_ds)
    if n == 0:
        return {"n_datasets": 0}
    dis = [d for d in per_ds if d["disagree"]]
    regrets = [d["makespan_regret_of_sum_argmin"] for d in per_ds]
    return {
        "n_datasets": n,
        "disagree_fraction": len(dis) / n,
        "disagree_gt_1pct_fraction": sum(
            1 for d in per_ds if d["makespan_regret_of_sum_argmin"] > 0.01
        ) / n,
        "makespan_regret_mean": sum(regrets) / n,
        "makespan_regret_max": max(regrets),
        "sum_regret_of_makespan_argmin_mean": sum(
            d["sum_regret_of_makespan_argmin"] for d in per_ds
        ) / n,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-dir", required=True, nargs="+")
    ap.add_argument("--report", default=None)
    ap.add_argument("--rel-tol", type=float, default=1e-9)
    args = ap.parse_args()

    per_ds_full: List[dict] = []
    per_ds_spread: List[dict] = []
    ds_records: List[dict] = []
    skipped_spread = 0

    for base in args.base_dir:
        base_path = Path(base)
        ds_dirs = sorted(base_path.glob("ds_*"))
        if not ds_dirs:
            raise RuntimeError(f"No ds_* under {base_path}")
        for ds in ds_dirs:
            jsonl = ds / "placements" / "placements.jsonl"
            if not jsonl.exists():
                raise RuntimeError(f"{ds} has no placements/placements.jsonl")
            rows = load_rows(jsonl)
            check_sum_consistency(rows, jsonl)
            full = score_rows(rows, args.rel_tol)
            if full is None:
                continue
            spread_rows = [r for r in rows if is_spread(r)]
            spread = score_rows(spread_rows, args.rel_tol)
            if spread is None:
                skipped_spread += 1
            rec = {
                "dataset": str(ds),
                "full": full,
                "spread": spread,
                "n_spread_plans": len(spread_rows),
            }
            ds_records.append(rec)
            per_ds_full.append(full)
            if spread is not None:
                per_ds_spread.append(spread)

    report = {
        "registered_rule": {
            "full_sweep_disagree_threshold": 0.10,
            "spread_plans_disagree_threshold": 0.05,
            "disagreement": "no sum-optimal plan is makespan-optimal (ties favor agreement)",
        },
        "full_sweep": aggregate(per_ds_full),
        "spread_plans_only": aggregate(per_ds_spread),
        "spread_unscoreable_datasets": skipped_spread,
        "datasets": ds_records,
    }

    full_agg = report["full_sweep"]
    spread_agg = report["spread_plans_only"]
    print(f"datasets scored: {full_agg.get('n_datasets', 0)} "
          f"(spread-scoreable: {spread_agg.get('n_datasets', 0)}, "
          f"spread-unscoreable: {skipped_spread})")
    for label, agg, thr in (
        ("FULL SWEEP  ", full_agg, 0.10),
        ("SPREAD-ONLY ", spread_agg, 0.05),
    ):
        if agg.get("n_datasets", 0) == 0:
            print(f"{label}: no scoreable datasets")
            continue
        verdict = "FIRES" if agg["disagree_fraction"] >= thr else "below threshold"
        print(
            f"{label}: disagree {agg['disagree_fraction']:.1%} "
            f"(>1% regret: {agg['disagree_gt_1pct_fraction']:.1%})  "
            f"makespan-regret mean {agg['makespan_regret_mean']:.4%} "
            f"max {agg['makespan_regret_max']:.2%}  "
            f"[threshold {thr:.0%}: {verdict}]"
        )

    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=1)
        print(f"report written: {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
