#!/usr/bin/env python3
"""Per-task duration decomposition over an exhaustive co-sim sweep (2026-09-08).

WHY. `measure_route_b_additivity.py` says WHETHER a corpus's rtt is additive over (task,
platform) one-hots. When it is not, this says WHICH task carries the residual and WHAT it
depends on, using the per-task `task_times` retained by `HEROSIM_RETAIN_TASK_TIMES=1`
(duration = end - start, so the DAG's dispatch `max` never enters -- rtt is the plain sum of
these durations, checked on the route_b sweeps).

For every task t it fits duration_t on nested one-hot designs and reports R^2:
  own                 -- t's own placement only (the pointwise class)
  own+othernodes      -- + the node of every OTHER task (pairwise, node-level)
  own+coresident_set  -- + the set of tasks sharing t's node (any-order co-residency)
  own_x_coresident    -- own placement x co-resident set (the full node-multiset class)
and prints, for the first task whose `own` R^2 < 0.99, the mean duration by
(own node, co-resident set) so the mechanism's shape is visible without a model.

Measured 2026-09-08 (route_b_env_pivot_v1 H2 separable control, 3 datasets): task 0 (root)
R^2_own = 1.000; children R^2_own 0.75-0.79; the whole residual is whether the PARENT is on
the child's node (dnn2 on node0: ~10.7-12.7 s with the parent remote vs ~2.6-6.2 s with it
local; sibling co-residency changes nothing to 3 decimals), and for the fan-in task the
3-way AND `local_dependencies = all(parents local)` (infrastructure.py) -- which is why
pairwise indicators reached R^2 0.92 but not 1.0 in the 2026-08-28 additivity entry.
Storage-read neutrality WAS applied in that control (both tiers 100 MB/s, 0.00012 s
latency; input 150 KB), so the read tier is not the carrier; the carrier inside the
local-parent branch is unidentified here. On route_b arm_s, task 0's duration is exactly
pointwise (R^2_own = 1.000): the "39 s node-write-contention term" is warmup-write
contention on the node, not co-placement.

Usage:
  task_duration_decomposition.py --corpus <dir> [--max-datasets N] --out report.json
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

TASKS = [0, 1, 2, 3]


def r2(X, y):
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    p = X @ coef
    ss = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - p) ** 2).sum() / ss) if ss > 1e-12 else 1.0


def onehot(rows, key_fn):
    keys = sorted({key_fn(r) for r in rows})
    idx = {k: i for i, k in enumerate(keys)}
    X = np.zeros((len(rows), len(keys)))
    for i, r in enumerate(rows):
        X[i, idx[key_fn(r)]] = 1.0
    return X


def decompose(ds_dir: Path):
    infra = json.load(open(ds_dir / "infrastructure.json"))
    pmap = {r["platform_id"]: r["node_name"] for v in infra["replica_placements"].values() for r in v}
    rows = [json.loads(l) for l in open(ds_dir / "placements/placements.jsonl")]
    if not rows or not rows[0].get("task_times"):
        raise RuntimeError(f"{ds_dir}: no task_times -- sweep was not run with HEROSIM_RETAIN_TASK_TIMES=1")
    plans = [{int(k): int(v[1]) for k, v in r["placement_plan"].items()} for r in rows]
    tasks = sorted(plans[0])
    dur = {t: np.array([next(tt[2] - tt[1] for tt in r["task_times"] if tt[0] == t) for r in rows]) for t in tasks}
    rtt = np.array([r["rtt"] for r in rows])
    rep = {"ds": ds_dir.name, "n_rows": len(rows),
           "rtt_equals_sum_durations_max_abs_err": float(np.abs(rtt - sum(dur.values())).max()),
           "tasks": {}, "example": None}
    for t in tasks:
        cores = [tuple(sorted(u for u in tasks if u != t and pmap[p[u]] == pmap[p[t]])) for p in plans]
        X_own = onehot(plans, lambda p, t=t: p[t])
        X_other = np.hstack([onehot(plans, lambda p, u=u: pmap[p[u]]) for u in tasks if u != t])
        X_set = onehot(list(zip(plans, cores)), lambda pc: pc[1])
        X_own_x_set = onehot(list(zip(plans, cores)), lambda pc, t=t: (pc[0][t], pc[1]))
        y = dur[t]
        rep["tasks"][str(t)] = {
            "mean_duration": float(y.mean()), "sd": float(y.std()),
            "r2_own": r2(X_own, y),
            "r2_own+othernodes": r2(np.hstack([X_own, X_other]), y),
            "r2_own+coresident_set": r2(np.hstack([X_own, X_set]), y),
            "r2_own_x_coresident_set": r2(X_own_x_set, y),
        }
        if rep["example"] is None and rep["tasks"][str(t)]["r2_own"] < 0.99:
            tab = defaultdict(list)
            for p, s, d in zip(plans, cores, y):
                tab[(pmap[p[t]], s)].append(d)
            rep["example"] = {"task": t, "by_node_and_coresidents": {
                f"{k[0]} co={list(k[1])}": {"mean": float(np.mean(v)), "sd": float(np.std(v)), "n": len(v)}
                for k, v in sorted(tab.items())}}
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--max-datasets", type=int, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    dirs = sorted(p for p in args.corpus.glob("ds_*") if (p / "placements/placements.jsonl").exists())
    if args.max_datasets:
        dirs = dirs[: args.max_datasets]
    reps = [decompose(d) for d in dirs]
    tasks = sorted({t for r in reps for t in r["tasks"]})
    summary = {"corpus": str(args.corpus), "n_datasets": len(reps),
               "rtt_equals_sum_durations_max_abs_err": max(r["rtt_equals_sum_durations_max_abs_err"] for r in reps),
               "median_r2_by_task": {t: {k: float(np.median([r["tasks"][t][k] for r in reps]))
                                         for k in ["r2_own", "r2_own+othernodes", "r2_own+coresident_set", "r2_own_x_coresident_set"]}
                                     for t in tasks}}
    args.out.write_text(json.dumps({"summary": summary, "per_dataset": reps}, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
