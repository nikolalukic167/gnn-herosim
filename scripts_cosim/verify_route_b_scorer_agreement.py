#!/usr/bin/env python3
"""Independent cross-check of score_route_b_contention's R_greedy, R_exact, and repairs.

A from-scratch recomputation of the registered route_b_v1 statistics straight from each
dataset's files — deliberately importing NOTHING from the scorer, using no numpy (a
hand-rolled Gaussian-elimination least-squares solver instead), and structured
differently (flat dict passes), so a shared implementation bug has to be made twice to
survive. Disagreement beyond 1e-9 percentage points on any dataset is a VOID condition
for the lineage (see route_b_v1 in LINEAGES.md).

The repair check exists because gate condition 2 (count repairs close < 0.5 median) is
the single statistic distinguishing "genuine non-count structure" from "sixth
count-shaped confirmation," and the repair-fit code path had already produced one
saturation bug caught only by the rig, never independently reverified against the real
corpus. --check-repairs turns this on (slower: one small linear solve per dataset).

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
  repair(1int)     = LS fit of y ~ a + b*marginal_sum + max_node_occupancy_excess,
                     over the FULL sweep; repaired R_exact = regret of the feasible
                     argmin of the fitted surrogate, reported as min(base, repaired)
  repair(kint)     = same, with one column per (node, task_type) occupancy count
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


def node_of(task_id, placement, ttypes, pid_map, task_db):
    return demand_of(task_id, placement, ttypes, pid_map, task_db)[0]


def solve_normal_equations(X, y):
    """Pure-Python (no numpy) least squares via normal equations + Gaussian
    elimination with partial pivoting. X is a list of row-lists, y a flat list.

    Normal equations square the design matrix's condition number, and this problem's
    columns span wildly different scales (an intercept of 1.0 next to a marginal-sum
    column in the hundreds next to 0/1 count columns) — squaring that spread lost
    precision badly enough to disagree with the scorer's numpy/SVD fit on a real
    dataset (10.6% vs 42.0%, cross-checked independently). Fix: standardize every
    non-intercept column to zero mean / unit norm before solving, un-scale after.
    """
    n_rows = len(X)
    n_params = len(X[0])
    means = [0.0] * n_params
    scales = [1.0] * n_params
    Xs = [row[:] for row in X]
    for j in range(1, n_params):  # column 0 is the intercept: leave it alone
        col = [row[j] for row in X]
        mean = sum(col) / n_rows
        centered = [v - mean for v in col]
        norm = math.sqrt(sum(v * v for v in centered)) or 1.0
        means[j] = mean
        scales[j] = norm
        for i in range(n_rows):
            Xs[i][j] = centered[i] / norm

    XtX = [[0.0] * n_params for _ in range(n_params)]
    Xty = [0.0] * n_params
    for row, target in zip(Xs, y):
        for i in range(n_params):
            Xty[i] += row[i] * target
            for j in range(n_params):
                XtX[i][j] += row[i] * row[j]
    aug = [XtX[i] + [Xty[i]] for i in range(n_params)]
    for col in range(n_params):
        pivot_row = max(range(col, n_params), key=lambda r: abs(aug[r][col]))
        aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
        pivot = aug[col][col]
        if abs(pivot) < 1e-12:
            continue  # singular direction: leave coefficient at 0
        for r in range(n_params):
            if r == col:
                continue
            factor = aug[r][col] / pivot
            for c in range(col, n_params + 1):
                aug[r][c] -= factor * aug[col][c]
    beta_s = [0.0] * n_params
    for i in range(n_params):
        pivot = aug[i][i]
        beta_s[i] = aug[i][n_params] / pivot if abs(pivot) > 1e-12 else 0.0

    # Un-scale: y = b0 + sum_j bj_s * (x_j - mean_j)/scale_j
    #            = (b0 - sum_j bj_s*mean_j/scale_j) + sum_j (bj_s/scale_j) * x_j
    beta = [0.0] * n_params
    beta[0] = beta_s[0]
    for j in range(1, n_params):
        beta[j] = beta_s[j] / scales[j]
        beta[0] -= beta_s[j] * means[j] / scales[j]
    return beta


def repair_columns(kind, plan, ttypes, pid_map, task_db, kint_keys=None):
    if kind == "1int":
        counts = {}
        for t, p in plan.items():
            node = node_of(t, p, ttypes, pid_map, task_db)
            counts[node] = counts.get(node, 0) + 1
        return [float(max((c - 1 for c in counts.values() if c > 1), default=0))]
    cols = [0.0] * len(kint_keys)
    idx = {k: i for i, k in enumerate(kint_keys)}
    for t, p in plan.items():
        node = node_of(t, p, ttypes, pid_map, task_db)
        cols[idx[(node, ttypes[t])]] += 1.0
    return cols


def repaired_r_exact(rows, feas, marg, kind, ttypes, pid_map, task_db, best):
    kint_keys = None
    if kind == "kint":
        kint_keys = sorted({(node_of(t, p, ttypes, pid_map, task_db), ttypes[t])
                            for plan, _v in rows for t, p in plan.items()})
    n_params = 2 + (len(kint_keys) if kint_keys is not None else 1)
    if len(rows) < 2 * n_params:
        return None  # saturation guard, mirrors the scorer
    X, y = [], []
    for plan, v in rows:
        msum = sum(marg[t][p] for t, p in plan.items())
        cols = repair_columns(kind, plan, ttypes, pid_map, task_db, kint_keys)
        X.append([1.0, msum] + cols)
        y.append(v)
    beta = solve_normal_equations(X, y)
    scored = []
    for plan, v in feas:
        msum = sum(marg[t][p] for t, p in plan.items())
        cols = repair_columns(kind, plan, ttypes, pid_map, task_db, kint_keys)
        pred = sum(b * x for b, x in zip(beta, [1.0, msum] + cols))
        scored.append((pred, tuple(sorted(plan.items())), v))
    scored.sort()
    return 100.0 * (scored[0][2] - best) / best


def recompute(rows, ttypes, pid_map, task_db, alpha, check_repairs=False):
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
        return None, None, None
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

    repairs = None
    if check_repairs:
        repairs = {}
        for kind in ("1int", "kint"):
            r = repaired_r_exact(rows, feas, marg, kind, ttypes, pid_map, task_db, best)
            repairs[kind] = None if r is None else min(r_exact, r)

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
            return r_exact, "stuck", repairs
        chosen_total[t] = pick
        used.add(pick)
        node, d = demand_of(t, pick, ttypes, pid_map, task_db)
        load_[node] = load_.get(node, 0.0) + d
    lookup = {tuple(sorted(plan.items())): v for plan, v in rows}
    key = tuple(sorted(chosen_total.items()))
    if key not in lookup:
        fail(f"greedy plan {key} not in sweep")
    r_greedy = 100.0 * (lookup[key] - best) / best
    return r_exact, r_greedy, repairs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", action="append", required=True)
    ap.add_argument("--report", required=True,
                    help="scorer report JSON written with --include-per-dataset")
    ap.add_argument("--task-types", default="data/nofs-ids/task-types.json")
    ap.add_argument("--check-repairs", action="store_true",
                    help="also independently recompute the 1int/kint repair fits "
                         "(slower: one small linear solve per dataset)")
    ap.add_argument("--alpha", action="append",
                    help="restrict to these alpha keys (e.g. --alpha 2.0), default all "
                         "in the report")
    args = ap.parse_args()

    reports = json.load(open(args.report))
    by_corpus = {r["corpus"]: r for r in reports}
    checked = 0
    repairs_checked = 0
    for corpus in args.corpus:
        rep = by_corpus.get(corpus)
        if rep is None:
            fail(f"{corpus} not present in {args.report}")
        if "per_dataset" not in rep:
            fail(f"{args.report} lacks per_dataset rows — rerun scorer with "
                 "--include-per-dataset")
        ds_dirs = sorted(d for d in Path(corpus).glob("ds_*") if d.is_dir())
        for alpha_key, results in rep["per_dataset"].items():
            if args.alpha and alpha_key not in args.alpha:
                continue
            alpha = None if alpha_key == "None" else float(alpha_key)
            if len(results) != len(ds_dirs):
                fail(f"{corpus} alpha={alpha_key}: {len(results)} scored rows vs "
                     f"{len(ds_dirs)} datasets")
            for ds, scored in zip(ds_dirs, results):
                rows, ttypes, pid_map, task_db = load(ds, args.task_types)
                r_exact, r_greedy, repairs = recompute(
                    rows, ttypes, pid_map, task_db, alpha,
                    check_repairs=args.check_repairs)
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
                if args.check_repairs and repairs is not None:
                    for kind in ("1int", "kint"):
                        s_key = f"r_exact_repaired_{kind}_pct"
                        sat_key = f"repair_{kind}_saturated"
                        v = repairs[kind]
                        if v is None:
                            if not scored.get(sat_key):
                                fail(f"{ds} alpha={alpha_key}: verifier repair "
                                     f"{kind} saturated, scorer not")
                        else:
                            if scored.get(sat_key):
                                fail(f"{ds} alpha={alpha_key}: scorer repair {kind} "
                                     "saturated, verifier not")
                            elif abs(scored[s_key] - v) > 1e-6:
                                fail(f"{ds} alpha={alpha_key}: repair {kind} scorer="
                                     f"{scored[s_key]!r} verifier={v!r}")
                            repairs_checked += 1
    extra = f", {repairs_checked} repair values" if args.check_repairs else ""
    print(f"OK: {checked} (dataset, alpha) cells agree to 1e-9{extra}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
