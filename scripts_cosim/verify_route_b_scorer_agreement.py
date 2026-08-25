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
  repair(1int)     = LS fit of y ~ a + b*marginal_sum + sum_node_occupancy_excess,
                     over the FULL sweep; repaired R_exact = regret of the feasible
                     argmin of the fitted surrogate, reported as min(base, repaired)
  repair(kint)     = same, with one column per (node, task_type) occupancy count
  repair(t1)       = same, with the stage-2 T1 (partial-state) column set of
                     ROUTE_B_STAGE2_PREREGISTRATION.md §9a: kint + per-type quadratic
                     co-residency + load/cap + over-cap count + min/max parent-hop sums
                     + hops/bottleneck + latency + same-node-parent count, from the
                     dataset's own link_topology routes. Constrained alphas only.
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
    # DAG edges as (parent_task_id, child_task_id), ids in static_order assignment;
    # and the network pieces the t1 repair columns need.
    dag_edges = []
    offset = 0
    for ev in workload["events"]:
        dag = ev["application"]["dag"]
        if isinstance(dag, list):
            offset += len(dag)
            continue
        order = list(TopologicalSorter(dag).static_order())
        local = {name: offset + i for i, name in enumerate(order)}
        for child, parents in dag.items():
            for parent in parents:
                dag_edges.append((local[parent], local[child]))
        offset += len(order)
    lt = infra.get("link_topology") or {}
    net = {"routes": lt.get("routes") or {}, "links": lt.get("links") or {},
           "maps": infra.get("network_maps") or {}}
    return rows, ttypes, pid_map, task_db, dag_edges, net


def demand_of(task_id, placement, ttypes, pid_map, task_db):
    node, ptype = pid_map[placement[1]]
    return node, float(task_db[ttypes[task_id]]["memoryRequirements"][ptype])


def node_of(task_id, placement, ttypes, pid_map, task_db):
    return demand_of(task_id, placement, ttypes, pid_map, task_db)[0]


def solve_least_squares(X, y):
    """Pure-Python (no numpy) least squares. X is a list of row-lists, y a flat list.

    History, because this function has now been wrong twice in instructive ways:
    (1) the first version used raw normal equations + Gaussian elimination — squaring
    the condition number of columns spanning wildly different scales lost enough
    precision to disagree with the scorer's numpy/SVD fit on a real dataset (10.6% vs
    42.0%); fixed by standardizing every non-intercept column first. (2) standardized
    normal equations were still not enough for the wider t1 column set: on ds_00008
    the solver returned coefficients whose FITTED VALUES differed from the true LS
    projection (fitted values on the fit rows are solver-independent — any exact LS
    solution projects onto the same column space — so a fitted-value mismatch is a
    numerical failure of the solver, not a tie). Fix: thin QR via modified
    Gram-Schmidt with one re-orthogonalization pass and dependent-column dropping —
    no normal equations, condition number never squared. Dropped (dependent) columns
    get coefficient 0; the projection, and therefore every prediction on the fit
    rows, is unchanged by which basis was kept.
    """
    n_rows = len(X)
    n_params = len(X[0])
    means = [0.0] * n_params
    scales = [1.0] * n_params
    cols = []
    for j in range(n_params):
        col = [row[j] for row in X]
        if j > 0:  # column 0 is the intercept: leave it alone
            mean = sum(col) / n_rows
            centered = [v - mean for v in col]
            norm = math.sqrt(sum(v * v for v in centered)) or 1.0
            means[j] = mean
            scales[j] = norm
            col = [v / norm for v in centered]
        cols.append(col)

    kept = []      # param indices whose columns are independent
    q_list = []    # orthonormal basis, one vector per kept column
    r_proj = {}    # param index -> projections onto q_list existing at its time + diag
    for j in range(n_params):
        v = cols[j][:]
        col_norm = math.sqrt(sum(t * t for t in v)) or 1.0
        proj = [0.0] * len(q_list)
        for _pass in range(2):  # MGS + one re-orthogonalization pass
            for k, q in enumerate(q_list):
                c = sum(a * b for a, b in zip(q, v))
                proj[k] += c
                v = [a - c * b for a, b in zip(v, q)]
        norm = math.sqrt(sum(t * t for t in v))
        if norm <= 1e-10 * col_norm:
            continue  # dependent column: coefficient stays 0
        q_list.append([t / norm for t in v])
        proj.append(norm)
        r_proj[j] = proj
        kept.append(j)

    z = [sum(a * b for a, b in zip(q, y)) for q in q_list]
    beta_s = [0.0] * n_params
    for pos in range(len(kept) - 1, -1, -1):
        j = kept[pos]
        acc = z[pos]
        for pos2 in range(pos + 1, len(kept)):
            acc -= r_proj[kept[pos2]][pos] * beta_s[kept[pos2]]
        beta_s[j] = acc / r_proj[j][pos]

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
        # SUM of excess occupants, matching the registration and
        # separability_diagnostic._excess_sharing. This was `max` until 2026-08-25 —
        # a genuine verifier bug that the old normal-equations solver masked (its
        # imprecise fits happened to argmin onto the same plans); the QR solver
        # exposed it on ds_00019 (scorer 1.036 vs verifier-with-max 9.633).
        counts = {}
        for t, p in plan.items():
            node = node_of(t, p, ttypes, pid_map, task_db)
            counts[node] = counts.get(node, 0) + 1
        return [float(sum(c - 1 for c in counts.values() if c > 1))]
    cols = [0.0] * len(kint_keys)
    idx = {k: i for i, k in enumerate(kint_keys)}
    for t, p in plan.items():
        node = node_of(t, p, ttypes, pid_map, task_db)
        cols[idx[(node, ttypes[t])]] += 1.0
    return cols


def route_hops_bneck_latency(net, parent_node, child_node):
    """Independent walk of the dataset's own route data. Same node: (0, inf, 0.0)."""
    if parent_node == child_node:
        return 0, math.inf, 0.0
    path = net["routes"].get(parent_node, {}).get(child_node)
    if not path:
        fail(f"t1 columns: no route {parent_node}->{child_node}")
    bneck = math.inf
    for i in range(len(path) - 1):
        pair = sorted((path[i], path[i + 1]))
        link = net["links"].get(pair[0] + "|" + pair[1])
        if link is None:
            fail(f"t1 columns: link {pair} missing from link_topology.links")
        bneck = min(bneck, float(link["bandwidth_mbps"]))
    entry = net["maps"].get(child_node, {}).get(parent_node)
    if entry is None:
        fail(f"t1 columns: no network_maps[{child_node}][{parent_node}]")
    lat = float(entry["latency"]) if isinstance(entry, dict) else float(entry)
    return len(path) - 1, bneck, lat


def t1_columns(plan, ttypes, pid_map, task_db, kint_keys, caps, dag_edges, net):
    """The §9a T1 column set, recomputed from scratch (see the scorer's t1_cols for the
    registered definition this must reproduce). Uncapped nodes (no caps entry) are
    treated as infinite-capacity, the same convention feasibility uses."""
    cols = repair_columns("kint", plan, ttypes, pid_map, task_db, kint_keys)
    per_node = {}
    for t, p in plan.items():
        node, d = demand_of(t, p, ttypes, pid_map, task_db)
        slot = per_node.setdefault(node, {"types": {}, "tot": 0, "load": 0.0})
        slot["types"][ttypes[t]] = slot["types"].get(ttypes[t], 0) + 1
        slot["tot"] += 1
        slot["load"] += d
    for ttype in sorted(set(ttypes)):
        cols.append(float(sum(s["tot"] * s["types"].get(ttype, 0)
                              for s in per_node.values())))
    # The verifier's caps dict carries 0.0 entries for nodes whose max single demand
    # is 0; the scorer's node_caps omits them (uncapped). Feasibility-identical, but
    # these columns divide by cap — normalize to the scorer's uncapped convention.
    def cap_of(n):
        cap = caps.get(n, math.inf)
        return math.inf if cap <= 0 else cap
    cols.append(sum(s["tot"] * s["load"] / cap_of(n) for n, s in per_node.items()))
    cols.append(float(sum(s["tot"] for n, s in per_node.items()
                          if s["load"] > cap_of(n) + EPS)))
    min_sum = max_sum = transfer = lat_sum = same = 0.0
    children = {}
    for parent, child in dag_edges:
        children.setdefault(child, []).append(parent)
    for child, parents in children.items():
        child_node = node_of(child, plan[child], ttypes, pid_map, task_db)
        hop_list = []
        for parent in parents:
            parent_node = node_of(parent, plan[parent], ttypes, pid_map, task_db)
            hops, bneck, lat = route_hops_bneck_latency(net, parent_node, child_node)
            hop_list.append(hops)
            if hops == 0:
                same += 1.0
            else:
                transfer += hops / bneck
                lat_sum += lat
        min_sum += min(hop_list)
        max_sum += max(hop_list)
    return cols + [min_sum, max_sum, transfer, lat_sum, same]


def repaired_r_exact(rows, feas, marg, kind, ttypes, pid_map, task_db, best,
                     caps=None, dag_edges=None, net=None):
    kint_keys = None
    if kind in ("kint", "t1"):
        kint_keys = sorted({(node_of(t, p, ttypes, pid_map, task_db), ttypes[t])
                            for plan, _v in rows for t, p in plan.items()})

    def columns(plan):
        if kind == "t1":
            return t1_columns(plan, ttypes, pid_map, task_db, kint_keys, caps,
                              dag_edges, net)
        return repair_columns(kind, plan, ttypes, pid_map, task_db, kint_keys)

    n_params = 2 + len(columns(rows[0][0]))
    if len(rows) < 2 * n_params:
        return None  # saturation guard, mirrors the scorer
    X, y = [], []
    for plan, v in rows:
        msum = sum(marg[t][p] for t, p in plan.items())
        X.append([1.0, msum] + columns(plan))
        y.append(v)
    beta = solve_least_squares(X, y)
    scored = []
    for plan, v in feas:
        msum = sum(marg[t][p] for t, p in plan.items())
        pred = sum(b * x for b, x in zip(beta, [1.0, msum] + columns(plan)))
        scored.append((pred, tuple(sorted(plan.items())), v))
    scored.sort()
    regret = 100.0 * (scored[0][2] - best) / best
    # Plans whose predicted surrogate ties the argmin at machine precision (the
    # ds_00008 case from the stage-1 scrutiny: predictions equal to ~13 sig figs,
    # true costs very different). Two independent LS implementations can then pick
    # different plans without either being wrong — the caller accepts a scorer value
    # matching any tied plan, LOUDLY, and counts it.
    tol = 1e-9 * max(1.0, abs(scored[0][0]))
    tied_regrets = {100.0 * (v - best) / best
                    for pred, _key, v in scored if pred - scored[0][0] <= tol}
    return regret, tied_regrets


def recompute(rows, ttypes, pid_map, task_db, alpha, check_repairs=False,
              dag_edges=None, net=None):
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
        kinds = ["1int", "kint"] + (["t1"] if caps is not None else [])
        for kind in kinds:
            r = repaired_r_exact(rows, feas, marg, kind, ttypes, pid_map, task_db,
                                 best, caps=caps, dag_edges=dag_edges, net=net)
            if r is None:
                repairs[kind] = None
            else:
                regret, tied = r
                repairs[kind] = (min(r_exact, regret),
                                 {min(r_exact, t) for t in tied})

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
    tie_accepted = 0
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
                rows, ttypes, pid_map, task_db, dag_edges, net = load(
                    ds, args.task_types)
                r_exact, r_greedy, repairs = recompute(
                    rows, ttypes, pid_map, task_db, alpha,
                    check_repairs=args.check_repairs,
                    dag_edges=dag_edges, net=net)
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
                    for kind in repairs:
                        s_key = f"r_exact_repaired_{kind}_pct"
                        sat_key = f"repair_{kind}_saturated"
                        v = repairs[kind]
                        if v is None:
                            if not scored.get(sat_key):
                                fail(f"{ds} alpha={alpha_key}: verifier repair "
                                     f"{kind} saturated, scorer not")
                        else:
                            regret, tied = v
                            if scored.get(sat_key):
                                fail(f"{ds} alpha={alpha_key}: scorer repair {kind} "
                                     "saturated, verifier not")
                            elif abs(scored[s_key] - regret) > 1e-6:
                                if any(abs(scored[s_key] - t) <= 1e-6 for t in tied):
                                    print(f"TIE (accepted): {ds.name} "
                                          f"alpha={alpha_key} repair {kind}: "
                                          f"scorer={scored[s_key]:.6f} "
                                          f"verifier={regret:.6f} — predicted "
                                          "surrogate values tie at machine "
                                          "precision, both argmins valid")
                                    tie_accepted += 1
                                else:
                                    fail(f"{ds} alpha={alpha_key}: repair {kind} "
                                         f"scorer={scored[s_key]!r} "
                                         f"verifier={regret!r}")
                            repairs_checked += 1
    extra = (f", {repairs_checked} repair values ({tie_accepted} machine-precision "
             "ties accepted)" if args.check_repairs else "")
    print(f"OK: {checked} (dataset, alpha) cells agree to 1e-9{extra}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
