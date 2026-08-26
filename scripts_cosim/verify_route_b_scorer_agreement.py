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
  krank arms       = --check-krank (PP0' of the corrected stage-2 registration §10):
                     the per-dataset and POOLED repair fractions of the krank block —
                     occupancy indexed by node rank under the identity-free canonical
                     ordering ascending (cap at alpha, mean hop, node name), padded to
                     a common width for pooling — plus the dim36crk-expressible block
                     set, and the linkrank ingress-route co-use block when the report
                     was produced with --add-linkrank. Tie bands included. The pooled
                     fit is recomputed by a DIFFERENT algorithm on purpose: per-dataset
                     intercepts are projected out by within-dataset demeaning
                     (Frisch-Waugh) and the reduced system solved with this file's own
                     QR, instead of numpy lstsq on the full indicator design. Any exact
                     LS solution's shared-part predictions differ from any other's by a
                     per-dataset CONSTANT only (the difference lies in the design's
                     null space, so it is absorbed by the intercepts), and the decode
                     fraction and tie band are invariant to a constant shift.
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
    # Submitting client node per task_id (the linkrank ingress endpoint). May be None
    # on pre-fabric corpora; the linkrank columns fail loudly if they ever need one.
    sources = []
    for ev in workload["events"]:
        sources.extend([ev.get("node_name")] * len(ev["application"]["dag"]))
    return rows, ttypes, pid_map, task_db, dag_edges, net, sources


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


# The scorer's T1_BLOCKS, deliberately RE-TYPED rather than imported: this file exists to
# reproduce the scorer's definitions without sharing a line of code with it, so an ordering
# or membership drift in one has to be made twice to survive.
T1_BLOCKS = ("kint", "quad", "cap", "hop", "coupling")


def t1_columns(plan, ttypes, pid_map, task_db, kint_keys, caps, dag_edges, net,
               blocks=T1_BLOCKS):
    """The §9a T1 column set, recomputed from scratch (see the scorer's t1_cols for the
    registered definition this must reproduce). Uncapped nodes (no caps entry) are
    treated as infinite-capacity, the same convention feasibility uses.

    `blocks` selects a subset, always emitted in T1_BLOCKS order — the §9b block
    attribution and the pooled (kint-free) column set both need subsets."""
    unknown = [b for b in blocks if b not in T1_BLOCKS]
    if unknown:
        fail(f"unknown T1 block(s) {unknown}")
    cols = (repair_columns("kint", plan, ttypes, pid_map, task_db, kint_keys)
            if "kint" in blocks else [])
    per_node = {}
    for t, p in plan.items():
        node, d = demand_of(t, p, ttypes, pid_map, task_db)
        slot = per_node.setdefault(node, {"types": {}, "tot": 0, "load": 0.0})
        slot["types"][ttypes[t]] = slot["types"].get(ttypes[t], 0) + 1
        slot["tot"] += 1
        slot["load"] += d
    if "quad" in blocks:
        for ttype in sorted(set(ttypes)):
            cols.append(float(sum(s["tot"] * s["types"].get(ttype, 0)
                                  for s in per_node.values())))

    # The verifier's caps dict carries 0.0 entries for nodes whose max single demand
    # is 0; the scorer's node_caps omits them (uncapped). Feasibility-identical, but
    # these columns divide by cap — normalize to the scorer's uncapped convention.
    def cap_of(n):
        cap = caps.get(n, math.inf)
        return math.inf if cap <= 0 else cap
    if "cap" in blocks:
        cols.append(sum(s["tot"] * s["load"] / cap_of(n) for n, s in per_node.items()))
        cols.append(float(sum(s["tot"] for n, s in per_node.items()
                              if s["load"] > cap_of(n) + EPS)))
    if not ({"hop", "coupling"} & set(blocks)):
        return cols
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
    if "hop" in blocks:
        cols += [min_sum, max_sum]
    if "coupling" in blocks:
        cols += [transfer, lat_sum, same]
    return cols


def repaired_r_exact(rows, feas, marg, kind, ttypes, pid_map, task_db, best,
                     caps=None, dag_edges=None, net=None, blocks=T1_BLOCKS):
    kint_keys = None
    if kind in ("kint", "t1"):
        kint_keys = sorted({(node_of(t, p, ttypes, pid_map, task_db), ttypes[t])
                            for plan, _v in rows for t, p in plan.items()})

    def columns(plan):
        if kind == "t1":
            return t1_columns(plan, ttypes, pid_map, task_db, kint_keys, caps,
                              dag_edges, net, blocks=blocks)
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


def check_blocks(corpus, transfer_report, task_types_path):
    """Independent recomputation of the §9b per-dataset arms (cells A and B, and every
    block-attribution arm) straight from each dataset's files.

    Same discipline as --check-repairs and for the same reason: §9b's registered VOID
    condition turns on cell B's median, an arm no committed code computed before today,
    and this file's own solver has been wrong twice. The pooled cell C is deliberately NOT
    checked here — it is a single global fit whose value is decoded per dataset, and the
    VOID it produced is driven by cell B, which is checked.
    """
    report = json.load(open(transfer_report))
    rows_by_ds = {r["ds"]: r for r in report["per_dataset"]}
    ds_dirs = [d for d in sorted(Path(corpus).glob("ds_*")) if d.name in rows_by_ds]
    if len(ds_dirs) != len(rows_by_ds):
        fail(f"{corpus}: {len(ds_dirs)} of {len(rows_by_ds)} §9b datasets found")
    alpha = float(report["alpha"])
    pooled = tuple(report["pooled_blocks"])
    arms = {"cell_a": tuple(T1_BLOCKS), "cell_b": pooled}
    for name, arm in report.get("ablation", {}).items():
        arms[name] = tuple(arm["blocks"])
    order = [r["ds"] for r in report["per_dataset"]]

    checked = 0
    for ds in ds_dirs:
        rows, ttypes, pid_map, task_db, dag_edges, net, _src = load(
            ds, task_types_path)
        peak = {}
        for plan, _v in rows:
            for t, p in plan.items():
                node, d = demand_of(t, p, ttypes, pid_map, task_db)
                peak[node] = max(peak.get(node, 0.0), d)
        caps = {n: alpha * m for n, m in peak.items()}

        def feasible(plan):
            load_ = {}
            for t, p in plan.items():
                node, d = demand_of(t, p, ttypes, pid_map, task_db)
                load_[node] = load_.get(node, 0.0) + d
            return all(v <= caps.get(n, math.inf) + EPS for n, v in load_.items())

        feas = [(plan, v) for plan, v in rows if feasible(plan)]
        best = min(v for _p, v in feas)
        marg = {}
        for plan, v in rows:
            for t, p in plan.items():
                cur = marg.setdefault(t, {})
                if p not in cur or v < cur[p]:
                    cur[p] = v
        keyed = sorted(((sum(marg[t][p] for t, p in plan.items()),
                         tuple(sorted(plan.items())), v) for plan, v in feas))
        r_exact = 100.0 * (keyed[0][2] - best) / best
        row = rows_by_ds[ds.name]
        if abs(row["r_exact_pct"] - r_exact) > 1e-9:
            fail(f"{ds}: R_exact §9b={row['r_exact_pct']!r} verifier={r_exact!r}")

        for name, blocks in arms.items():
            got = repaired_r_exact(rows, feas, marg, "t1", ttypes, pid_map, task_db,
                                   best, caps=caps, dag_edges=dag_edges, net=net,
                                   blocks=blocks)
            if got is None:
                fail(f"{ds}: verifier says arm {name} saturated, §9b reported a value")
            regret, tied = got
            fraction = 1.0 - min(r_exact, regret) / r_exact
            if name in ("cell_a", "cell_b"):
                expected = row[f"{name}_fraction"]
            else:
                expected = report["ablation"][name]["fractions"][order.index(ds.name)]
            if abs(expected - fraction) > 1e-9:
                # the same machine-precision tie escape --check-repairs grants
                tied_fracs = [1.0 - min(r_exact, t) / r_exact for t in tied]
                if any(abs(expected - t) <= 1e-6 for t in tied_fracs):
                    print(f"TIE (accepted): {ds.name} arm {name}: §9b={expected:.9f} "
                          f"verifier={fraction:.9f}")
                else:
                    fail(f"{ds} arm {name}: §9b fraction={expected!r} "
                         f"verifier={fraction!r}")
            checked += 1
    print(f"OK: {checked} §9b (dataset, arm) repair fractions agree to 1e-9 "
          f"across {len(ds_dirs)} datasets and {len(arms)} arms")
    return 0


# ---------------------------------------------------------------------------
# PP0' (corrected stage-2 registration §10): the krank arms + linkrank block
# ---------------------------------------------------------------------------

def krank_rank_map(rows, ttypes, pid_map, task_db, net, alpha):
    """node -> rank under the canonical identity-free ordering ascending
    (cap at alpha, mean hop from the other candidate-hosting nodes, node name),
    recomputed straight from the raw files. Mirrors the definition pinned at
    route_b_coefficient_transfer.krank_cols/node_features while sharing no code:
    the node set is every node hosting a candidate placement, cap is
    alpha * max single demand (0.0 for an all-zero-demand node, matching the
    scorer's node_caps omission read back as .get(node, 0.0)), and hops come from
    the dataset's own routes."""
    peak = {}
    for plan, _v in rows:
        for t, p in plan.items():
            node, d = demand_of(t, p, ttypes, pid_map, task_db)
            if node not in peak or d > peak[node]:
                peak[node] = d
    nodes = sorted(peak)

    def mean_hop(node):
        hops = [float(route_hops_bneck_latency(net, other, node)[0])
                for other in nodes if other != node]
        return sum(hops) / len(hops) if hops else 0.0

    order = sorted(nodes, key=lambda n: (alpha * peak[n] if peak[n] > 0 else 0.0,
                                         mean_hop(n), n))
    return {n: i for i, n in enumerate(order)}


def krank_columns_fn(ttypes, pid_map, task_db, rank, width):
    """Per-plan krank block: occupancy count at (node rank, task type), rank-major,
    types in sorted order, padded to `width` ranks (top slots stay zero)."""
    types = sorted(set(ttypes))
    if len(rank) > width:
        fail(f"krank: {len(rank)} nodes exceed pad width {width}")

    def fn(plan):
        cols = [0.0] * (width * len(types))
        for t, p in plan.items():
            r = rank[node_of(t, p, ttypes, pid_map, task_db)]
            cols[r * len(types) + types.index(ttypes[t])] += 1.0
        return cols
    return fn


def ingress_links_indep(net, src, dst):
    """Undirected link keys on the client->destination route. Re-types
    network_fabric.route_links: forward lookup, reversed-route fallback, sorted
    'a|b' keys; same node or no fabric at all -> the true empty set."""
    if src == dst or not net["routes"]:
        return []
    path = net["routes"].get(src, {}).get(dst)
    if not path:
        rev = net["routes"].get(dst, {}).get(src)
        if not rev:
            fail(f"linkrank: no route between {src} and {dst}")
        path = list(reversed(rev))
    return ["|".join(sorted((path[i], path[i + 1]))) for i in range(len(path) - 1)]


def linkrank_columns(plan, ttypes, pid_map, task_db, net, sources):
    """The 8 linkrank order-statistic columns (the scorer's t1_cols 'linkrank'
    branch, re-typed): per-link co-use over each task's ingress route, emitted as
    top-4 counts, excess sums, and >=2-co-use link counts, core-restricted twins
    included. Never a link identity."""
    couse = {}
    for t, p in plan.items():
        src = sources[t]
        if src is None:
            fail("linkrank: workload event without node_name — cannot resolve "
                 "ingress routes")
        for lk in ingress_links_indep(
                net, src, node_of(t, p, ttypes, pid_map, task_db)):
            couse[lk] = couse.get(lk, 0) + 1

    def core(lk):
        a, _, b = lk.partition("|")
        return a.startswith("core") and b.startswith("core")

    top = sorted(couse.values(), reverse=True)[:4]
    top += [0] * (4 - len(top))
    return ([float(v) for v in top]
            + [float(sum(c - 1 for c in couse.values() if c > 1)),
               float(sum(c - 1 for lk, c in couse.items() if c > 1 and core(lk))),
               float(sum(1 for c in couse.values() if c >= 2)),
               float(sum(1 for lk, c in couse.items() if c >= 2 and core(lk)))])


def decode_band(feas, predicted, best, r_base):
    """(fraction, band, tied_fractions): the registered decode (ties by sorted plan
    key), the [pessimistic, mean_tied, optimistic] repair-fraction band over the
    argmin tie group, and every tied plan's fraction (the machine-precision tie
    escape the other checks already grant)."""
    scored = sorted((predicted[i], tuple(sorted(feas[i][0].items())), feas[i][1])
                    for i in range(len(feas)))
    lo = scored[0][0]
    tol = 1e-9 * max(1.0, abs(lo))
    tied = [v for i, (_p, v) in enumerate(feas) if predicted[i] - lo <= tol]

    def frac(rtt):
        return 1.0 - min(r_base, 100.0 * (rtt - best) / best) / r_base

    band = {"registered": frac(scored[0][2]),
            "optimistic": frac(min(tied)),
            "pessimistic": frac(max(tied)),
            "mean_tied": frac(sum(tied) / len(tied)),
            "n_tied": len(tied)}
    return band["registered"], band, [frac(v) for v in tied]


def compare_krank_row(ds_name, arm, got_frac, got_band, tied_fracs,
                      exp_frac, exp_band, counters):
    for key in ("optimistic", "pessimistic", "mean_tied"):
        if abs(got_band[key] - exp_band[key]) > 1e-9:
            fail(f"{ds_name} {arm}: band {key} transfer={exp_band[key]!r} "
                 f"verifier={got_band[key]!r}")
    if got_band["n_tied"] != exp_band["n_tied"]:
        fail(f"{ds_name} {arm}: n_tied transfer={exp_band['n_tied']} "
             f"verifier={got_band['n_tied']}")
    if exp_band["registered"] != exp_frac:
        fail(f"{ds_name} {arm}: report-internal mismatch — band registered "
             f"{exp_band['registered']!r} vs fraction {exp_frac!r}")
    if abs(got_frac - exp_frac) > 1e-9:
        if any(abs(exp_frac - t) <= 1e-6 for t in tied_fracs):
            print(f"TIE (accepted): {ds_name} {arm}: transfer={exp_frac:.9f} "
                  f"verifier={got_frac:.9f} — both argmins of one machine-"
                  "precision tie group")
            counters["ties"] += 1
        else:
            fail(f"{ds_name} {arm}: fraction transfer={exp_frac!r} "
                 f"verifier={got_frac!r}")
    counters["checked"] += 1


def check_krank(corpus, transfer_report, task_types_path):
    """PP0' verification gate: independently recompute the per-dataset and pooled
    krank repair fractions (and the linkrank block when the report used
    --add-linkrank) from raw files, tie bands included. 1e-9 agreement with
    route_b_coefficient_transfer.py or the lineage is VOID until resolved. This is
    a verification gate, NOT a kill test — no threshold reading is taken here."""
    report = json.load(open(transfer_report))
    kre = report.get("krank_exploratory")
    kpe = report.get("krank_pooled_exploratory")
    for label, block in (("krank_exploratory", kre),
                         ("krank_pooled_exploratory", kpe)):
        if not block:
            fail(f"{transfer_report}: no {label} block")
        if "fractions" not in block:
            fail(f"{transfer_report}: {label} lacks per-dataset fractions/bands — "
                 "re-run route_b_coefficient_transfer.py (PP0' report extension)")
    alpha = float(report["alpha"])
    blocks = tuple(kre["blocks"])
    if tuple(kpe["blocks"]) != blocks:
        fail(f"krank arms disagree on blocks: {kre['blocks']} vs {kpe['blocks']}")
    base_blocks = tuple(b for b in blocks if b != "linkrank")
    want_linkrank = "linkrank" in blocks
    unknown = [b for b in base_blocks if b not in T1_BLOCKS]
    if unknown:
        fail(f"unknown krank pool block(s) {unknown}")
    names = kre["ds"]
    if kpe["ds"] != names:
        fail("krank arms list different datasets")
    rows_by_ds = {r["ds"]: r for r in report["per_dataset"]}

    # The report's own aggregates must be the aggregates of its own per-dataset
    # fractions (upper-middle median, the registered gate convention).
    for label, block in (("krank_exploratory", kre),
                         ("krank_pooled_exploratory", kpe)):
        fr = block["fractions"]
        if abs(sorted(fr)[len(fr) // 2] - block["median_fraction"]) > 1e-12:
            fail(f"{label}: median_fraction inconsistent with its own fractions")
        if abs(sum(fr) / len(fr) - block["mean_fraction"]) > 1e-9:
            fail(f"{label}: mean_fraction inconsistent with its own fractions")
        if sum(1 for f in fr if f >= 0.5) != block["n_closed_ge_half"]:
            fail(f"{label}: n_closed_ge_half inconsistent with its own fractions")
    mt = [b["mean_tied"] for b in kpe["bands"]]
    if abs(sorted(mt)[len(mt) // 2] - kpe["median_mean_tied"]) > 1e-12:
        fail("krank_pooled_exploratory: median_mean_tied inconsistent with bands")

    ctx = []
    for name in names:
        ds = Path(corpus) / name
        if not ds.is_dir():
            fail(f"{ds}: firing dataset missing from corpus")
        rows, ttypes, pid_map, task_db, dag_edges, net, sources = load(
            ds, task_types_path)
        peak = {}
        for plan, _v in rows:
            for t, p in plan.items():
                node, d = demand_of(t, p, ttypes, pid_map, task_db)
                if node not in peak or d > peak[node]:
                    peak[node] = d
        caps = {n: alpha * m for n, m in peak.items()}

        def feasible(plan):
            load_ = {}
            for t, p in plan.items():
                node, d = demand_of(t, p, ttypes, pid_map, task_db)
                load_[node] = load_.get(node, 0.0) + d
            return all(v <= caps.get(n, math.inf) + EPS for n, v in load_.items())

        feas = [(plan, v) for plan, v in rows if feasible(plan)]
        if not feas:
            fail(f"{ds}: no feasible rows at alpha={alpha}")
        best = min(v for _p, v in feas)
        marg = {}
        for plan, v in rows:
            for t, p in plan.items():
                cur = marg.setdefault(t, {})
                if p not in cur or v < cur[p]:
                    cur[p] = v
        keyed = sorted(((sum(marg[t][p] for t, p in plan.items()),
                         tuple(sorted(plan.items())), v) for plan, v in feas))
        r_exact = 100.0 * (keyed[0][2] - best) / best
        row = rows_by_ds.get(name)
        if row is None:
            fail(f"{name}: not in the report's per_dataset rows")
        if abs(row["r_exact_pct"] - r_exact) > 1e-9:
            fail(f"{ds}: R_exact report={row['r_exact_pct']!r} "
                 f"verifier={r_exact!r}")
        ctx.append({"name": name, "rows": rows, "feas": feas, "best": best,
                    "marg": marg, "r_exact": r_exact, "caps": caps,
                    "ttypes": ttypes, "pid_map": pid_map, "task_db": task_db,
                    "dag_edges": dag_edges, "net": net, "sources": sources,
                    "rank": krank_rank_map(rows, ttypes, pid_map, task_db, net,
                                           alpha)})
    n_ranks = max(len(c["rank"]) for c in ctx)
    if n_ranks != int(kpe["n_ranks"]):
        fail(f"pad width: verifier {n_ranks} vs report n_ranks {kpe['n_ranks']}")

    def merged_fn(c, width):
        kfn = krank_columns_fn(c["ttypes"], c["pid_map"], c["task_db"],
                               c["rank"], width)

        def fn(plan):
            cols = kfn(plan) + t1_columns(
                plan, c["ttypes"], c["pid_map"], c["task_db"], None, c["caps"],
                c["dag_edges"], c["net"], blocks=base_blocks)
            if want_linkrank:
                cols += linkrank_columns(plan, c["ttypes"], c["pid_map"],
                                         c["task_db"], c["net"], c["sources"])
            return cols
        return fn

    def msum(c, plan):
        return sum(c["marg"][t][p] for t, p in plan.items())

    counters = {"checked": 0, "ties": 0}

    # arm 1: per-dataset fits (own width, own QR)
    for i, c in enumerate(ctx):
        fn = merged_fn(c, len(c["rank"]))
        n_params = 2 + len(fn(c["rows"][0][0]))
        if len(c["rows"]) < 2 * n_params:
            fail(f"{c['name']}: verifier says the per-dataset krank fit is "
                 "saturated, the report carries a value")
        X = [[1.0, msum(c, plan)] + fn(plan) for plan, _v in c["rows"]]
        y = [v for _p, v in c["rows"]]
        beta = solve_least_squares(X, y)
        pred = [sum(b * x for b, x in zip(beta, [1.0, msum(c, plan)] + fn(plan)))
                for plan, _v in c["feas"]]
        got_frac, got_band, tied = decode_band(c["feas"], pred, c["best"],
                                               c["r_exact"])
        compare_krank_row(c["name"], "krank_per_dataset", got_frac, got_band,
                          tied, kre["fractions"][i], kre["bands"][i], counters)

    # arm 2: the pooled fit. The transfer solves one numpy lstsq over the full
    # design with explicit per-dataset intercept indicator columns; here the
    # intercepts are projected out by within-dataset demeaning (Frisch-Waugh) and
    # the reduced system goes through this file's own QR — a deliberately
    # different algorithm. Any exact LS solution's shared-part predictions differ
    # from any other's by a per-dataset constant only (the difference lies in the
    # design's null space and is absorbed by the intercepts), and the decode
    # fraction and tie band are invariant to a constant shift.
    X_all, y_all, shared_fns = [], [], []
    for c in ctx:
        fn = merged_fn(c, n_ranks)
        shared_fns.append(fn)
        rows_X = [[msum(c, plan)] + fn(plan) for plan, _v in c["rows"]]
        rows_y = [v for _p, v in c["rows"]]
        m = len(rows_X)
        col_mean = [sum(r[j] for r in rows_X) / m for j in range(len(rows_X[0]))]
        y_mean = sum(rows_y) / m
        for r, v in zip(rows_X, rows_y):
            X_all.append([1.0] + [a - mu for a, mu in zip(r, col_mean)])
            y_all.append(v - y_mean)
    sb = solve_least_squares(X_all, y_all)[1:]
    for i, c in enumerate(ctx):
        fn = shared_fns[i]
        pred = [sum(b * x for b, x in zip(sb, [msum(c, plan)] + fn(plan)))
                for plan, _v in c["feas"]]
        got_frac, got_band, tied = decode_band(c["feas"], pred, c["best"],
                                               c["r_exact"])
        compare_krank_row(c["name"], "krank_pooled", got_frac, got_band, tied,
                          kpe["fractions"][i], kpe["bands"][i], counters)

    print(f"OK: {counters['checked']} krank (dataset, arm) fractions + tie bands "
          f"agree to 1e-9 across {len(ctx)} datasets and 2 arms "
          f"({counters['ties']} machine-precision ties accepted; "
          f"blocks={list(blocks)}, pad width {n_ranks})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", action="append", required=True)
    ap.add_argument("--report",
                    help="scorer report JSON written with --include-per-dataset; "
                         "required unless --check-blocks is used")
    ap.add_argument("--task-types", default="data/nofs-ids/task-types.json")
    ap.add_argument("--check-blocks", metavar="TRANSFER_REPORT",
                    help="instead of the scorer report, independently recompute the §9b "
                         "block arms in route_b_coefficient_transfer.py's output")
    ap.add_argument("--check-krank", metavar="TRANSFER_REPORT",
                    help="PP0': independently recompute the per-dataset and pooled "
                         "krank repair fractions (and the linkrank block when the "
                         "report used --add-linkrank), tie bands included, in "
                         "route_b_coefficient_transfer.py's output")
    ap.add_argument("--check-repairs", action="store_true",
                    help="also independently recompute the 1int/kint repair fits "
                         "(slower: one small linear solve per dataset)")
    ap.add_argument("--alpha", action="append",
                    help="restrict to these alpha keys (e.g. --alpha 2.0), default all "
                         "in the report")
    args = ap.parse_args()

    if args.check_blocks and args.check_krank:
        fail("--check-blocks and --check-krank are separate passes; run one at "
             "a time")
    if args.check_krank:
        if len(args.corpus) != 1:
            fail("--check-krank takes exactly one --corpus")
        return check_krank(args.corpus[0], args.check_krank, args.task_types)
    if args.check_blocks:
        if len(args.corpus) != 1:
            fail("--check-blocks takes exactly one --corpus")
        return check_blocks(args.corpus[0], args.check_blocks, args.task_types)
    if not args.report:
        fail("--report is required unless --check-blocks is used")

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
                rows, ttypes, pid_map, task_db, dag_edges, net, _src = load(
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
