#!/usr/bin/env python3
"""Independent cross-check of score_route_b_contention's R_greedy and R_exact.

A from-scratch recomputation of the two registered route_b_v1 statistics straight from
each dataset's files — deliberately importing NOTHING from the scorer, using no numpy,
and structured differently (flat dict passes), so a shared implementation bug has to be
made twice to survive. Disagreement beyond 1e-9 percentage points on any dataset is a
VOID condition for the lineage (see route_b_v1 in LINEAGES.md).

Definitions verified (must match the registration exactly):
  cap_node(alpha)  = alpha * max single (task,placement) demand on that node
  feasible(plan)   = every node's summed demand <= cap + 1e-12
  m_t(p)           = min total cost over sweep plans containing (t, p)   [full sweep]
  R_exact          = regret of the feasible plan minimizing sum_t m_t(p_t), ties broken
                     by sorted plan key
  R_greedy         = regret of the sequential greedy: tasks in ascending
                     (min marginal, task_id) order, each taking its cheapest placement
                     (ties by placement id) that neither reuses a replica nor overflows
                     a node's remaining capacity
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

EPS = 1e-12


def fail(msg: str) -> None:
    print(f"!! DISAGREEMENT: {msg}")
    sys.exit(1)


def load(ds, task_types_path):
    task_db = json.load(open(task_types_path))
    infra = json.load(open(ds / "infrastructure.json"))
    pid_map = {}
    for reps in infra["replica_placements"].values():
        for r in reps:
            pid_map[int(r["platform_id"])] = (r["node_name"], r["platform_type"])
    workload = json.load(open(ds / "workload.json"))
    from graphlib import TopologicalSorter
    ttypes = []
    for ev in workload["events"]:
        dag = ev["application"]["dag"]
        ttypes += list(dag) if isinstance(dag, list) else list(
            TopologicalSorter(dag).static_order())
    rows = []
    for line in open(ds / "placements" / "placements.jsonl"):
        row = json.loads(line)
        plan = {int(k): (int(v[0]), int(v[1]))
                for k, v in row["placement_plan"].items()}
        rows.append((plan, float(row["rtt"])))
    return rows, ttypes, pid_map, task_db


def demand_of(task_id, placement, ttypes, pid_map, task_db):
    node, ptype = pid_map[placement[1]]
    return node, float(task_db[ttypes[task_id]]["memoryRequirements"][ptype])


def recompute(rows, ttypes, pid_map, task_db, alpha):
    # capacities
    caps = None
    if alpha is not None:
        peak = {}
        for plan, _v in rows:
            for t, p in plan.items():
                node, d = demand_of(t, p, ttypes, pid_map, task_db)
                peak[node] = max(peak.get(node, 0.0), d)
        caps = {n: alpha * m for n, m in peak.items()}

    def feasible(plan):
        if caps is None:
            return True
        load_ = {}
        for t, p in plan.items():
            node, d = demand_of(t, p, ttypes, pid_map, task_db)
            load_[node] = load_.get(node, 0.0) + d
        return all(v <= caps.get(n, math.inf) + EPS for n, v in load_.items())

    feas = [(plan, v) for plan, v in rows if feasible(plan)]
    if not feas:
        return None, None
    best = min(v for _p, v in feas)

    marg = {}
    for plan, v in rows:
        for t, p in plan.items():
            cur = marg.setdefault(t, {})
            if p not in cur or v < cur[p]:
                cur[p] = v

    # R_exact: feasible argmin of sum of marginals, ties by sorted plan key
    keyed = sorted(
        ((sum(marg[t][p] for t, p in plan.items()),
          tuple(sorted(plan.items())), v) for plan, v in feas))
    r_exact = 100.0 * (keyed[0][2] - best) / best

    # R_greedy
    order = sorted(marg, key=lambda t: (min(marg[t].values()), t))
    used, load_, chosen_total = set(), {}, {}
    for t in order:
        pick = None
        for p, _mv in sorted(marg[t].items(), key=lambda kv: (kv[1], kv[0])):
            if p in used:
                continue
            node, d = demand_of(t, p, ttypes, pid_map, task_db)
            cap = math.inf if caps is None else caps.get(node, math.inf)
            if load_.get(node, 0.0) + d > cap + EPS:
                continue
            pick = p
            break
        if pick is None:
            return r_exact, "stuck"
        chosen_total[t] = pick
        used.add(pick)
        node, d = demand_of(t, pick, ttypes, pid_map, task_db)
        load_[node] = load_.get(node, 0.0) + d
    lookup = {tuple(sorted(plan.items())): v for plan, v in rows}
    key = tuple(sorted(chosen_total.items()))
    if key not in lookup:
        fail(f"greedy plan {key} not in sweep")
    r_greedy = 100.0 * (lookup[key] - best) / best
    return r_exact, r_greedy


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", action="append", required=True)
    ap.add_argument("--report", required=True,
                    help="scorer report JSON written with --include-per-dataset")
    ap.add_argument("--task-types", default="data/nofs-ids/task-types.json")
    args = ap.parse_args()

    reports = json.load(open(args.report))
    by_corpus = {r["corpus"]: r for r in reports}
    checked = 0
    for corpus in args.corpus:
        rep = by_corpus.get(corpus)
        if rep is None:
            fail(f"{corpus} not present in {args.report}")
        if "per_dataset" not in rep:
            fail(f"{args.report} lacks per_dataset rows — rerun scorer with "
                 "--include-per-dataset")
        ds_dirs = sorted(d for d in Path(corpus).glob("ds_*") if d.is_dir())
        for alpha_key, results in rep["per_dataset"].items():
            alpha = None if alpha_key == "None" else float(alpha_key)
            if len(results) != len(ds_dirs):
                fail(f"{corpus} alpha={alpha_key}: {len(results)} scored rows vs "
                     f"{len(ds_dirs)} datasets")
            for ds, scored in zip(ds_dirs, results):
                rows, ttypes, pid_map, task_db = load(ds, args.task_types)
                r_exact, r_greedy = recompute(rows, ttypes, pid_map, task_db, alpha)
                if r_exact is None:
                    if not scored.get("no_feasible_rows"):
                        fail(f"{ds} alpha={alpha_key}: verifier finds no feasible rows, "
                             "scorer scored it")
                    continue
                if scored.get("no_feasible_rows"):
                    fail(f"{ds} alpha={alpha_key}: scorer says no feasible rows, "
                         "verifier disagrees")
                if abs(scored["r_exact_pct"] - r_exact) > 1e-9:
                    fail(f"{ds} alpha={alpha_key}: R_exact scorer="
                         f"{scored['r_exact_pct']!r} verifier={r_exact!r}")
                if r_greedy == "stuck":
                    if not scored.get("greedy_stuck"):
                        fail(f"{ds} alpha={alpha_key}: verifier greedy stuck, scorer "
                             "not")
                elif scored.get("greedy_stuck"):
                    fail(f"{ds} alpha={alpha_key}: scorer greedy stuck, verifier not")
                elif abs(scored["r_greedy_pct"] - r_greedy) > 1e-9:
                    fail(f"{ds} alpha={alpha_key}: R_greedy scorer="
                         f"{scored['r_greedy_pct']!r} verifier={r_greedy!r}")
                checked += 1
    print(f"OK: {checked} (dataset, alpha) cells agree to 1e-9")
    return 0


if __name__ == "__main__":
    sys.exit(main())
