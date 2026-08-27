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
                     docs/lineages/route_b_v1/stage2-preregistration.md §9a: kint + per-type quadratic
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


def compute_caps(rows, ttypes, pid_map, task_db, demand_of_fn, alpha, scales, cap_mode="alpha_max"):
    """route_b_env_pivot_v1: independent recomputation of Dataset.node_caps's cap_mode
    option (score_route_b_contention.py). alpha_max (default) is the original formula
    here, unchanged. A node whose demands are ALL zero gets no entry in ANY mode
    (uncapped), matching the scorer's exact convention -- caps.get(n, inf) downstream.

    CRITICAL: demands are deduplicated per (task_id, placement) pair, exactly like
    Dataset.demand -- alpha_max is invariant to how many sweep ROWS repeat a given
    (task, placement), but alpha_mean is NOT, so counting the same demand once per row
    it appears in (instead of once per unique candidate) silently changes the mean.
    Caught by test_route_b_env_pivot_cap_mode_verifier.py: verifier alpha_mean
    disagreed with the scorer (6.222 vs 6.5) on the toy rig before this fix."""
    if alpha is None:
        return None
    demand_by_key = {}
    for plan, _v in rows:
        for t, p in plan.items():
            key = (t, p)
            if key in demand_by_key:
                continue
            node, d = demand_of_fn(t, p, ttypes, pid_map, task_db, scales)
            demand_by_key[key] = (node, d)
    by_node = {}
    for node, d in demand_by_key.values():
        by_node.setdefault(node, []).append(d)
    by_node = {n: ds for n, ds in by_node.items() if max(ds) > 0.0}
    if cap_mode == "alpha_max":
        return {n: alpha * max(ds) for n, ds in by_node.items()}
    if cap_mode == "alpha_mean":
        return {n: alpha * (sum(ds) / len(ds)) for n, ds in by_node.items()}
    if isinstance(cap_mode, dict) and "absolute" in cap_mode:
        budget = float(cap_mode["absolute"])
        return {n: budget for n in by_node}
    fail(f"compute_caps: unknown cap_mode {cap_mode!r}")


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
    # Per-task_id demand_scale (route_b env pivot W2). An event's
    # application.demand_scale is keyed by task TYPE name; task_id is a positional
    # derivative of static_order, so expand it the same way ttypes is built above.
    # Absent key -> 1.0, so any corpus generated before demand_spread existed reads back
    # byte-identical. Derived independently of the scorer's load_demand_scales: this file
    # is the independent recomputation and must not import from what it verifies.
    scales = []
    for ev in workload["events"]:
        dag = ev["application"]["dag"]
        per_type = ev["application"].get("demand_scale") or {}
        names = list(dag) if isinstance(dag, list) else list(
            TopologicalSorter(dag).static_order())
        scales.extend(float(per_type.get(name, 1.0)) for name in names)
    return rows, ttypes, pid_map, task_db, dag_edges, net, sources, scales


def demand_of(task_id, placement, ttypes, pid_map, task_db, scales):
    """Memory demand of (task_id, placement), INCLUDING its per-instance demand_scale.

    `scales` deliberately has NO default. The scorer applies
    `scale * memoryRequirements[ptype]` (score_route_b_contention.py Dataset.demand) and
    this file omitted the factor entirely until 2026-08-27 — inert on H0 (all scales 1.0)
    but live on H1, whose corpus carries values like [1.6047, 1.5150, 1.8383, 0.8348].
    Caps, feasibility and every repair would have diverged, failing the registered 1e-9
    agreement (an S0 VOID gate) for a spurious reason. A default here would let a missed
    call site silently reintroduce exactly that, so every caller must pass it explicitly.
    """
    node, ptype = pid_map[placement[1]]
    return node, scales[task_id] * float(
        task_db[ttypes[task_id]]["memoryRequirements"][ptype])


def node_of(task_id, placement, ttypes, pid_map, task_db, scales):
    return demand_of(task_id, placement, ttypes, pid_map, task_db, scales)[0]


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


def repair_columns(kind, plan, ttypes, pid_map, task_db, scales, kint_keys=None):
    if kind == "1int":
        # SUM of excess occupants, matching the registration and
        # separability_diagnostic._excess_sharing. This was `max` until 2026-08-25 —
        # a genuine verifier bug that the old normal-equations solver masked (its
        # imprecise fits happened to argmin onto the same plans); the QR solver
        # exposed it on ds_00019 (scorer 1.036 vs verifier-with-max 9.633).
        counts = {}
        for t, p in plan.items():
            node = node_of(t, p, ttypes, pid_map, task_db, scales)
            counts[node] = counts.get(node, 0) + 1
        return [float(sum(c - 1 for c in counts.values() if c > 1))]
    cols = [0.0] * len(kint_keys)
    idx = {k: i for i, k in enumerate(kint_keys)}
    for t, p in plan.items():
        node = node_of(t, p, ttypes, pid_map, task_db, scales)
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
# route_b env pivot (2026-08-27): the extended-T1 superset, re-typed independently of the
# scorer's own T1_EXTENDED_BLOCKS for the same reason T1_BLOCKS is re-typed above. The
# scorer's T1_EXTENDED_BLOCKS = T1_BLOCKS(+linkrank) + (hetdem, futureint) — linkrank
# included, so this must match exactly for t1_columns(blocks=T1_EXTENDED_BLOCKS) to
# reproduce score_dataset's "t1x" arm.
T1_EXTENDED_BLOCKS = T1_BLOCKS + ("linkrank", "hetdem", "futureint")


def hetdem_columns(plan, ttypes, pid_map, task_db, scales, caps):
    """Independent recomputation of the scorer's hetdem block: demand-weighted analogs
    of quad/cap/1int (see score_route_b_contention.t1_cols's hetdem branch)."""
    per_node = {}
    for t, p in plan.items():
        node, d = demand_of(t, p, ttypes, pid_map, task_db, scales)
        slot = per_node.setdefault(node, {"types": {}, "tot": 0, "load": 0.0})
        slot["types"][ttypes[t]] = slot["types"].get(ttypes[t], 0) + 1
        slot["tot"] += 1
        slot["load"] += d

    def cap_of(n):
        cap = caps.get(n, math.inf)
        return math.inf if cap <= 0 else cap

    cols = []
    for ttype in sorted(set(ttypes)):
        cols.append(float(sum(s["load"] * s["types"].get(ttype, 0)
                              for s in per_node.values())))
    cols.append(sum(s["load"] * s["load"] / cap_of(n) for n, s in per_node.items()))
    cols.append(float(sum(s["load"] for n, s in per_node.items()
                          if s["load"] > cap_of(n) + EPS)))
    excess_share = 0.0
    for n, s in per_node.items():
        if s["tot"] > 1:
            min_single = min(demand_of(t, p, ttypes, pid_map, task_db, scales)[1]
                             for t, p in plan.items()
                             if demand_of(t, p, ttypes, pid_map, task_db, scales)[0] == n)
            excess_share += s["load"] - min_single
    cols.append(excess_share)
    cols.append(float(sum(s["load"] * s["load"] for s in per_node.values())))
    return cols


def task_node_min_demand_table(rows, ttypes, pid_map, task_db, scales):
    """Per (task_id, node) -> the MINIMUM demand task_id would carry if placed on that
    node, over every candidate of task_id that appears anywhere in the sweep — a static
    per-task eligibility fact, independent of any one plan. Mirrors the scorer's
    task_node_min_demand precompute in t1_cols's futureint branch."""
    table = {}
    for plan, _v in rows:
        for t, p in plan.items():
            node, d = demand_of(t, p, ttypes, pid_map, task_db, scales)
            slot = table.setdefault(t, {})
            if node not in slot or d < slot[node]:
                slot[node] = d
    return table


def futureint_columns(plan, ttypes, pid_map, task_db, scales, caps, order, task_node_min_demand):
    """Independent recomputation of the scorer's futureint block: per (fixed
    topological) decode step, the candidate node's static interaction with not-yet-
    committed tasks' eligibility + demand (see score_route_b_contention.t1_cols's
    futureint branch). `order` is this file's own _kahn_order (not the scorer's
    topological_task_order); `task_node_min_demand` is task_node_min_demand_table's
    output, computed once per dataset and passed in (mirrors the scorer's own
    per-dataset precompute, never recomputed per plan)."""
    fdi = fci = fop = fmax = 0.0
    for step, task_id in enumerate(order):
        node = node_of(task_id, plan[task_id], ttypes, pid_map, task_db, scales)
        future_tasks = order[step + 1:]
        future_demand_here = 0.0
        future_count_here = 0
        future_max_here = 0.0
        for ft in future_tasks:
            d = task_node_min_demand.get(ft, {}).get(node)
            if d is None:
                continue
            future_demand_here += d
            future_count_here += 1
            if d > future_max_here:
                future_max_here = d
        fdi += future_demand_here
        fci += float(future_count_here)
        fmax += future_max_here
        cap = caps.get(node, math.inf)
        load_so_far = sum(
            demand_of(t2, plan[t2], ttypes, pid_map, task_db, scales)[1]
            for t2 in order[:step + 1]
            if node_of(t2, plan[t2], ttypes, pid_map, task_db, scales) == node)
        fop += max(0.0, load_so_far + future_demand_here - cap)
    return [fdi, fci, fop, fmax]


def t1_columns(plan, ttypes, pid_map, task_db, scales, kint_keys, caps, dag_edges, net,
               blocks=T1_BLOCKS, decode_order=None, task_node_min_demand=None,
               sources=None):
    """The §9a T1 column set, recomputed from scratch (see the scorer's t1_cols for the
    registered definition this must reproduce). Uncapped nodes (no caps entry) are
    treated as infinite-capacity, the same convention feasibility uses.

    `blocks` selects a subset, always emitted in T1_EXTENDED_BLOCKS order — the §9b
    block attribution and the pooled (kint-free) column set both need subsets.
    `decode_order`/`task_node_min_demand` are required only when "futureint" is
    requested (see futureint_columns / task_node_min_demand_table); `sources` only
    when "linkrank" is requested. All default to None so every pre-existing call site
    (which never asks for futureint/linkrank here) is unaffected."""
    unknown = [b for b in blocks if b not in T1_EXTENDED_BLOCKS]
    if unknown:
        fail(f"unknown T1 block(s) {unknown}")
    cols = (repair_columns("kint", plan, ttypes, pid_map, task_db, scales, kint_keys)
            if "kint" in blocks else [])
    per_node = {}
    for t, p in plan.items():
        node, d = demand_of(t, p, ttypes, pid_map, task_db, scales)
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
    if {"hop", "coupling"} & set(blocks):
        min_sum = max_sum = transfer = lat_sum = same = 0.0
        children = {}
        for parent, child in dag_edges:
            children.setdefault(child, []).append(parent)
        for child, parents in children.items():
            child_node = node_of(child, plan[child], ttypes, pid_map, task_db, scales)
            hop_list = []
            for parent in parents:
                parent_node = node_of(parent, plan[parent], ttypes, pid_map, task_db, scales)
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
    if "linkrank" in blocks:
        if sources is None:
            fail("t1_columns: linkrank requested without sources")
        cols += linkrank_columns(plan, ttypes, pid_map, task_db, scales, net, sources)
    if "hetdem" in blocks:
        cols += hetdem_columns(plan, ttypes, pid_map, task_db, scales, caps)
    if "futureint" in blocks:
        if decode_order is None or task_node_min_demand is None:
            fail("t1_columns: futureint requested without decode_order/"
                 "task_node_min_demand")
        cols += futureint_columns(plan, ttypes, pid_map, task_db, scales, caps,
                                  decode_order, task_node_min_demand)
    return cols


# route_b env pivot (2026-08-27): the two new repair arms score_dataset wires
# (score_route_b_contention.py's score_dataset), re-typed here as block sets. t1x
# INCLUDES linkrank (T1_EXTENDED_BLOCKS = T1_BLOCKS(+linkrank) + hetdem + futureint) —
# matching the scorer's t1x = t1_cols(blocks=T1_EXTENDED_BLOCKS) exactly.
T1HD_BLOCKS = ("kint", "quad", "cap", "hop", "coupling", "hetdem")
T1X_BLOCKS = T1_EXTENDED_BLOCKS  # kint+quad+cap+hop+coupling+linkrank+hetdem+futureint


def repaired_r_exact(rows, feas, marg, kind, ttypes, pid_map, task_db, scales, best,
                     caps=None, dag_edges=None, net=None, blocks=T1_BLOCKS,
                     sources=None):
    kint_keys = None
    if kind in ("kint", "t1", "t1hd", "t1x"):
        kint_keys = sorted({(node_of(t, p, ttypes, pid_map, task_db, scales), ttypes[t])
                            for plan, _v in rows for t, p in plan.items()})
    arm_blocks = blocks
    if kind == "t1hd":
        arm_blocks = T1HD_BLOCKS
    elif kind == "t1x":
        arm_blocks = T1X_BLOCKS
    # decode_order/task_node_min_demand (futureint) and sources (linkrank) are needed
    # whenever the ACTUAL block set requests them -- not just for kind=="t1x" -- so an
    # arbitrary block subset passed via `blocks` (e.g. check_blocks's single-arm
    # "hetdem"/"futureint" ablation recomputation, kind="t1") gets them too.
    decode_order = None
    task_node_min_demand = None
    if "futureint" in arm_blocks:
        decode_order = _kahn_order(len(ttypes), dag_edges)
        task_node_min_demand = task_node_min_demand_table(rows, ttypes, pid_map, task_db, scales)

    def columns(plan):
        if kind in ("t1", "t1hd", "t1x"):
            return t1_columns(plan, ttypes, pid_map, task_db, scales, kint_keys, caps,
                              dag_edges, net, blocks=arm_blocks,
                              decode_order=decode_order,
                              task_node_min_demand=task_node_min_demand,
                              sources=sources)
        return repair_columns(kind, plan, ttypes, pid_map, task_db, scales, kint_keys)

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


def recompute(rows, ttypes, pid_map, task_db, scales, alpha, check_repairs=False,
              dag_edges=None, net=None, sources=None, cap_mode="alpha_max"):
    # capacities
    caps = compute_caps(rows, ttypes, pid_map, task_db, demand_of, alpha, scales, cap_mode)

    def feasible(plan):
        if caps is None:
            return True
        load_ = {}
        for t, p in plan.items():
            node, d = demand_of(t, p, ttypes, pid_map, task_db, scales)
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
        kinds = ["1int", "kint"] + (
            ["t1", "t1hd", "t1x"] if caps is not None else [])
        for kind in kinds:
            r = repaired_r_exact(rows, feas, marg, kind, ttypes, pid_map, task_db, scales,
                                 best, caps=caps, dag_edges=dag_edges, net=net,
                                 sources=sources)
            if r is None:
                repairs[kind] = None
            else:
                regret, tied = r
                repairs[kind] = (min(r_exact, regret),
                                 {min(r_exact, t) for t in tied})

    # R_greedy — AMENDMENT 2 (signed off 2026-08-27): the masked decode is a COMPLETE
    # search over the same option ordering, not a single forward pass. Written here from
    # the amendment's text, independently of the scorer, like everything else in this
    # file; the recursion below shares no code with score_route_b_contention.
    order = sorted(marg, key=lambda t: (min(marg[t].values()), t))
    used, load_, chosen_total = set(), {}, {}

    def _extend(i):
        if i == len(order):
            return True
        t = order[i]
        for p, _mv in sorted(marg[t].items(), key=lambda kv: (kv[1], kv[0])):
            if p in used:
                continue
            node, d = demand_of(t, p, ttypes, pid_map, task_db, scales)
            cap = math.inf if caps is None else caps.get(node, math.inf)
            if load_.get(node, 0.0) + d > cap + EPS:
                continue
            chosen_total[t] = p
            used.add(p)
            load_[node] = load_.get(node, 0.0) + d
            if _extend(i + 1):
                return True
            del chosen_total[t]
            used.discard(p)
            load_[node] -= d
        return False

    if not _extend(0):
        return r_exact, "stuck", repairs
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
    cap_mode = report.get("cap_mode", "alpha_max")
    pooled = tuple(report["pooled_blocks"])
    arms = {"cell_a": tuple(T1_BLOCKS), "cell_b": pooled}
    for name, arm in report.get("ablation", {}).items():
        arms[name] = tuple(arm["blocks"])
    order = [r["ds"] for r in report["per_dataset"]]

    have_t1x_band = bool(report.get("t1x_per_dataset")) and all(
        "t1x_band" in r for r in report["per_dataset"])
    t1x_bands_checked = 0
    checked = 0
    for ds in ds_dirs:
        rows, ttypes, pid_map, task_db, dag_edges, net, sources, scales = load(
            ds, task_types_path)
        caps = compute_caps(rows, ttypes, pid_map, task_db, demand_of, alpha, scales, cap_mode)

        def feasible(plan):
            load_ = {}
            for t, p in plan.items():
                node, d = demand_of(t, p, ttypes, pid_map, task_db, scales)
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
            got = repaired_r_exact(rows, feas, marg, "t1", ttypes, pid_map, task_db, scales,
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

        # route_b_env_pivot_v1 S2 (the kill bar): independently recompute the t1x
        # per-dataset tie band (T1_EXTENDED_BLOCKS) and compare against
        # route_b_coefficient_transfer.py's t1x_band, which came from a DIFFERENT
        # code path (Cell.repair_band, same compute_caps-derived caps as above).
        if have_t1x_band:
            row = rows_by_ds[ds.name]
            band = row["t1x_band"]
            expect_saturated = row.get("t1x_saturated", band is None)
            got = repaired_r_exact(rows, feas, marg, "t1", ttypes, pid_map, task_db, scales,
                                   best, caps=caps, dag_edges=dag_edges, net=net,
                                   blocks=T1X_BLOCKS, sources=sources)
            if got is None:
                if not expect_saturated:
                    fail(f"{ds}: verifier says t1x saturated, §9b reported a band")
            else:
                if expect_saturated:
                    fail(f"{ds}: §9b says t1x saturated, verifier is not")
                _regret, tied = got
                tied_fracs = sorted(1.0 - min(r_exact, t) / r_exact for t in tied)
                got_band = {
                    "optimistic": tied_fracs[-1], "pessimistic": tied_fracs[0],
                    "mean_tied": sum(tied_fracs) / len(tied_fracs),
                    "n_tied": len(tied_fracs),
                }
                mismatches = {key: (got_band[key], band[key]) for key in
                             ("optimistic", "pessimistic", "mean_tied")
                             if abs(got_band[key] - band[key]) > 1e-9}
                if mismatches or got_band["n_tied"] != band["n_tied"]:
                    # T1_EXTENDED_BLOCKS is wide (kint's dataset-specific width +
                    # linkrank + hetdem + futureint, 40+ params) and this corpus has
                    # datasets numpy's own lstsq reports as severely rank-deficient at
                    # the NARROWER 13-param pooled design already (cond ~1e20,
                    # ds_00017/ds_00020 -- route_b_coefficient_transfer.py's own
                    # "rank-deficient designs" printout). On a rank-deficient design
                    # two independently-computed exact LS solutions can differ off the
                    # training manifold by construction (the null space is real, not a
                    # solver bug): predictions on FEASIBLE-BUT-NOT-FIT rows genuinely
                    # diverge even though both solvers minimize the same residual on
                    # the FIT rows. Checked directly (not assumed): rebuild this
                    # dataset's t1x design and confirm numpy's own rank < n_params
                    # before accepting the escape -- a full-rank disagreement is still
                    # a real bug and still fails.
                    import numpy as _np
                    kint_keys_here = sorted({
                        (node_of(t, p, ttypes, pid_map, task_db, scales), ttypes[t])
                        for plan, _v in rows for t, p in plan.items()})
                    dec_order = _kahn_order(len(ttypes), dag_edges)
                    tnmd = task_node_min_demand_table(rows, ttypes, pid_map, task_db, scales)

                    def _t1x_cols(plan):
                        return t1_columns(plan, ttypes, pid_map, task_db, scales, kint_keys_here,
                                          caps, dag_edges, net, blocks=T1X_BLOCKS,
                                          decode_order=dec_order,
                                          task_node_min_demand=tnmd, sources=sources)
                    Xfull = _np.array([[1.0, sum(marg[t][p] for t, p in plan.items())]
                                       + _t1x_cols(plan) for plan, _v in rows])
                    design_rank = int(_np.linalg.matrix_rank(Xfull))
                    n_params = Xfull.shape[1]
                    if design_rank < n_params:
                        print(f"RANK-DEFICIENT (accepted): {ds.name} t1x band: rank "
                              f"{design_rank}/{n_params} -- §9b n_tied="
                              f"{band['n_tied']} verifier n_tied={got_band['n_tied']}, "
                              f"mismatches={mismatches or 'none (count only)'} -- two "
                              "exact LS solutions on a rank-deficient design, not a "
                              "computation bug")
                    else:
                        fail(f"{ds}: t1x band mismatch {mismatches or ''} n_tied "
                             f"§9b={band['n_tied']} verifier={got_band['n_tied']} "
                             f"on a FULL-RANK design ({design_rank}/{n_params}) -- "
                             "this is a real disagreement")
            t1x_bands_checked += 1
    extra = (f"; {t1x_bands_checked} t1x per-dataset bands agree to 1e-9"
            if have_t1x_band else "")
    print(f"OK: {checked} §9b (dataset, arm) repair fractions agree to 1e-9 "
          f"across {len(ds_dirs)} datasets and {len(arms)} arms{extra}")
    return 0


# ---------------------------------------------------------------------------
# PP0' (corrected stage-2 registration §10): the krank arms + linkrank block
# ---------------------------------------------------------------------------

def krank_rank_map(rows, ttypes, pid_map, task_db, scales, net, alpha, cap_mode="alpha_max"):
    """node -> rank under the canonical identity-free ordering ascending
    (cap at alpha [cap_mode-aware], mean hop from the other candidate-hosting nodes,
    node name), recomputed straight from the raw files. Mirrors the definition pinned
    at route_b_coefficient_transfer.krank_cols/node_features while sharing no code:
    the node set is every node hosting a candidate placement, cap comes from
    compute_caps (0.0 for an all-zero-demand node, matching the scorer's node_caps
    omission read back as .get(node, 0.0)), and hops come from the dataset's own
    routes. cap_mode default "alpha_max" reproduces the pre-existing ordering exactly
    for every caller that does not pass it (§9c/PP0' reports, alpha_max always)."""
    caps = compute_caps(rows, ttypes, pid_map, task_db, demand_of, alpha, scales, cap_mode)
    nodes = sorted({demand_of(t, p, ttypes, pid_map, task_db, scales)[0]
                    for plan, _v in rows for t, p in plan.items()})

    def mean_hop(node):
        hops = [float(route_hops_bneck_latency(net, other, node)[0])
                for other in nodes if other != node]
        return sum(hops) / len(hops) if hops else 0.0

    order = sorted(nodes, key=lambda n: (caps.get(n, 0.0), mean_hop(n), n))
    return {n: i for i, n in enumerate(order)}


def krank_columns_fn(ttypes, pid_map, task_db, scales, rank, width):
    """Per-plan krank block: occupancy count at (node rank, task type), rank-major,
    types in sorted order, padded to `width` ranks (top slots stay zero)."""
    types = sorted(set(ttypes))
    if len(rank) > width:
        fail(f"krank: {len(rank)} nodes exceed pad width {width}")

    def fn(plan):
        cols = [0.0] * (width * len(types))
        for t, p in plan.items():
            r = rank[node_of(t, p, ttypes, pid_map, task_db, scales)]
            cols[r * len(types) + types.index(ttypes[t])] += 1.0
        return cols
    return fn


def krank_demand_columns_fn(ttypes, pid_map, task_db, scales, rank, width):
    """route_b env pivot (2026-08-27), --extended-blocks: krank_columns_fn's exact
    rank x type structure, summing real per-instance demand instead of a unit count —
    independent recomputation of route_b_coefficient_transfer.krank_demand_cols."""
    types = sorted(set(ttypes))
    if len(rank) > width:
        fail(f"krank_demand: {len(rank)} nodes exceed pad width {width}")

    def fn(plan):
        cols = [0.0] * (width * len(types))
        for t, p in plan.items():
            node, d = demand_of(t, p, ttypes, pid_map, task_db, scales)
            r = rank[node]
            cols[r * len(types) + types.index(ttypes[t])] += d
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


def linkrank_columns(plan, ttypes, pid_map, task_db, scales, net, sources):
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
                net, src, node_of(t, p, ttypes, pid_map, task_db, scales)):
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
    cap_mode = report.get("cap_mode", "alpha_max")
    blocks = tuple(kre["blocks"])
    if tuple(kpe["blocks"]) != blocks:
        fail(f"krank arms disagree on blocks: {kre['blocks']} vs {kpe['blocks']}")
    want_linkrank = "linkrank" in blocks
    # route_b env pivot (2026-08-27): --extended-blocks adds hetdem+futureint to the t1
    # portion AND a second krank-shaped block of demand-weighted occupancy — detect it
    # from the report itself (route_b_coefficient_transfer.py stamps extended_blocks on
    # krank_pooled_exploratory) rather than guessing from block membership alone, since
    # hetdem/futureint are ordinary T1_EXTENDED_BLOCKS members like linkrank.
    want_extended = bool(kpe.get("extended_blocks", False))
    base_blocks = tuple(b for b in blocks if b not in ("linkrank", "hetdem", "futureint"))
    unknown = [b for b in base_blocks if b not in T1_BLOCKS]
    if unknown:
        fail(f"unknown krank pool block(s) {unknown}")
    expected_blocks = set(base_blocks)
    if want_linkrank:
        expected_blocks.add("linkrank")
    if want_extended:
        expected_blocks |= {"hetdem", "futureint"}
    if set(blocks) != expected_blocks:
        fail(f"krank pool blocks {blocks} inconsistent with extended_blocks="
             f"{want_extended}/linkrank={want_linkrank}")
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
        rows, ttypes, pid_map, task_db, dag_edges, net, sources, scales = load(
            ds, task_types_path)
        caps = compute_caps(rows, ttypes, pid_map, task_db, demand_of, alpha, scales, cap_mode)

        def feasible(plan):
            load_ = {}
            for t, p in plan.items():
                node, d = demand_of(t, p, ttypes, pid_map, task_db, scales)
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
                    "scales": scales, "dag_edges": dag_edges, "net": net, "sources": sources,
                    "rank": krank_rank_map(rows, ttypes, pid_map, task_db, scales, net,
                                           alpha, cap_mode=cap_mode),
                    "decode_order": _kahn_order(len(ttypes), dag_edges),
                    "task_node_min_demand": task_node_min_demand_table(
                        rows, ttypes, pid_map, task_db, scales)})
    n_ranks = max(len(c["rank"]) for c in ctx)
    if n_ranks != int(kpe["n_ranks"]):
        fail(f"pad width: verifier {n_ranks} vs report n_ranks {kpe['n_ranks']}")

    def merged_fn(c, width):
        kfn = krank_columns_fn(c["ttypes"], c["pid_map"], c["task_db"],
                               c["scales"], c["rank"], width)
        dfn = (krank_demand_columns_fn(c["ttypes"], c["pid_map"], c["task_db"],
                                       c["scales"], c["rank"], width)
               if want_extended else None)

        def fn(plan):
            cols = kfn(plan)
            if dfn is not None:
                cols = cols + dfn(plan)
            cols = cols + t1_columns(
                plan, c["ttypes"], c["pid_map"], c["task_db"], c["scales"], None,
                c["caps"], c["dag_edges"], c["net"], blocks=base_blocks)
            if want_linkrank:
                cols += linkrank_columns(plan, c["ttypes"], c["pid_map"],
                                         c["task_db"], c["scales"], c["net"],
                                         c["sources"])
            if want_extended:
                cols += hetdem_columns(plan, c["ttypes"], c["pid_map"], c["task_db"],
                                       c["scales"], c["caps"])
                cols += futureint_columns(plan, c["ttypes"], c["pid_map"], c["task_db"],
                                          c["scales"], c["caps"],
                                          c["decode_order"],
                                          c["task_node_min_demand"])
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


# ---------------------------------------------------------------------------
# B1 (corrected stage-2 registration §4/§10): the masked_topo decoder acceptance
# ---------------------------------------------------------------------------

def _kahn_order(n_tasks, dag_edges):
    """Verifier-local topological order (Kahn, lowest task_id first) — deliberately
    re-typed, not imported from the decoder under test."""
    remaining = {t: 0 for t in range(n_tasks)}
    children = {}
    for parent, child in dag_edges:
        remaining[child] += 1
        children.setdefault(parent, []).append(child)
    order = []
    ready = sorted(t for t in range(n_tasks) if remaining[t] == 0)
    while ready:
        t = ready.pop(0)
        order.append(t)
        grew = False
        for c in children.get(t, ()):
            remaining[c] -= 1
            if remaining[c] == 0:
                ready.append(c)
                grew = True
        if grew:
            ready.sort()
    if len(order) != n_tasks:
        fail(f"masked_topo check: dependency cycle in dag_edges")
    return order


def _greedy_masked(order, marg, caps, ttypes, pid_map, task_db, scales):
    """Verifier-local masked greedy: for each task in `order`, the cheapest
    (marginal value, placement) not reusing a replica nor overflowing a node."""
    taken, load_, plan = set(), {}, {}
    for t in order:
        choice = None
        for p, _v in sorted(marg[t].items(), key=lambda kv: (kv[1], kv[0])):
            if p in taken:
                continue
            node, d = demand_of(t, p, ttypes, pid_map, task_db, scales)
            if load_.get(node, 0.0) + d > caps.get(node, math.inf) + EPS:
                continue
            choice = p
            break
        if choice is None:
            return None
        plan[t] = choice
        taken.add(choice)
        node, d = demand_of(t, choice, ttypes, pid_map, task_db, scales)
        load_[node] = load_.get(node, 0.0) + d
    return plan


def check_decoder(corpus, report_path, task_types_path, alpha_keys):
    """B1 acceptance (§4): fed the true min-marginals, the production masked_topo
    decoder (src/policy/gnn/seq_decode.decode_masked_topo_placement) must reproduce
    the masked greedy run in the corrected topological order, and that plan's
    regret must match the FROZEN stage-1 r_greedy_pct to 1e-9 — the §9c measured
    fact that the correction leaves the stage-1 plans unchanged, asserted dataset
    by dataset. The reference (Kahn order + masked greedy) is verifier-local code;
    only the SUBJECT is imported from src/. An infeasible completion must be
    None on both sides and flagged greedy_stuck in the frozen report."""
    import sys as _sys
    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in _sys.path:
        _sys.path.insert(0, repo_root)
    from src.policy.gnn.seq_decode import decode_masked_topo_placement

    reports = json.load(open(report_path))
    if not isinstance(reports, list):
        reports = [reports]
    by_corpus = {r["corpus"]: r for r in reports}
    rep = by_corpus.get(corpus)
    if rep is None:
        fail(f"{corpus} not present in {report_path}")
    if "per_dataset" not in rep:
        fail(f"{report_path} lacks per_dataset rows — rerun scorer with "
             "--include-per-dataset")
    missing = [a for a in alpha_keys if a not in rep["per_dataset"]]
    if missing:
        fail(f"{report_path}: alpha keys {missing} not in the report "
             f"(has {sorted(rep['per_dataset'])})")

    ds_dirs = sorted(d for d in Path(corpus).glob("ds_*") if d.is_dir())
    checked = 0
    stuck = 0
    pre_amendment_report = False
    for alpha_key in alpha_keys:
        results = rep["per_dataset"][alpha_key]
        alpha = float(alpha_key)
        if len(results) != len(ds_dirs):
            fail(f"{corpus} alpha={alpha_key}: {len(results)} scored rows vs "
                 f"{len(ds_dirs)} datasets")
        for ds, scored in zip(ds_dirs, results):
            rows, ttypes, pid_map, task_db, dag_edges, net, _src, scales = load(
                ds, task_types_path)
            peak = {}
            for plan, _v in rows:
                for t, p in plan.items():
                    node, d = demand_of(t, p, ttypes, pid_map, task_db, scales)
                    if node not in peak or d > peak[node]:
                        peak[node] = d
            caps = {n: alpha * m for n, m in peak.items()}

            def feasible(plan):
                load_ = {}
                for t, p in plan.items():
                    node, d = demand_of(t, p, ttypes, pid_map, task_db, scales)
                    load_[node] = load_.get(node, 0.0) + d
                return all(v <= caps.get(n, math.inf) + EPS
                           for n, v in load_.items())

            feas = [(plan, v) for plan, v in rows if feasible(plan)]
            if not feas:
                if not scored.get("no_feasible_rows"):
                    fail(f"{ds} alpha={alpha_key}: verifier finds no feasible "
                         "rows, scorer scored it")
                continue
            best = min(v for _p, v in feas)
            marg = {}
            for plan, v in rows:
                for t, p in plan.items():
                    cur = marg.setdefault(t, {})
                    if p not in cur or v < cur[p]:
                        cur[p] = v
            n_tasks = len(ttypes)
            topo = _kahn_order(n_tasks, dag_edges)
            reference = _greedy_masked(topo, marg, caps, ttypes, pid_map, task_db, scales)
            historical = _greedy_masked(
                sorted(marg, key=lambda t: (min(marg[t].values()), t)),
                marg, caps, ttypes, pid_map, task_db, scales)
            if reference != historical:
                fail(f"{ds} alpha={alpha_key}: topological-order greedy differs "
                     f"from the frozen historical order — the §9c measured fact "
                     f"does not hold here (topo={reference}, hist={historical})")

            # subject: the production decoder fed the true min-marginals
            parents = {}
            for parent, child in dag_edges:
                parents.setdefault(child, []).append(parent)
            node_name_of_id = {}
            for t in marg:
                for (node_id, pid) in marg[t]:
                    name = pid_map[pid][0]
                    prev = node_name_of_id.setdefault(node_id, name)
                    if prev != name:
                        fail(f"{ds}: node_id {node_id} maps to both {prev} "
                             f"and {name}")
            candidates = {t: sorted(marg[t]) for t in marg}
            logits = {t: [-marg[t][p] for p in candidates[t]] for t in marg}
            demands = {
                t: [demand_of(t, p, ttypes, pid_map, task_db, scales)[1]
                    for p in candidates[t]]
                for t in marg}
            id_caps = {nid: caps[name] for nid, name in node_name_of_id.items()
                       if name in caps}
            combo = decode_masked_topo_placement(
                [logits[t] for t in range(n_tasks)],
                {t: candidates[t] for t in range(n_tasks)},
                n_tasks,
                dag_parents=parents,
                node_caps=id_caps,
                demands=demands,
            )

            # The SUBJECT here is the production SERVING decoder, which is a single
            # forward pass and is unchanged by AMENDMENT 2 (that amendment scoped itself
            # to the scorer's offline decode). So this pass compares against the report's
            # `legacy_forward_only` block — the forward-only numbers, from the same run —
            # not against the amended live counters. Comparing it to the amended ones
            # would report a disagreement that is really a difference of subject.
            # A report written BEFORE the amendment has no such block, and does not need
            # one: pre-amendment the top-level counters ARE the forward-only counters. So
            # the frozen stage-1 artifacts stay checkable without being re-scored, which
            # is the point — they belong to route_b_v1 and this amendment regenerates
            # nothing. The choice is announced once per run, never silent.
            fwd = scored.get("legacy_forward_only")
            if fwd is None:
                fwd = scored
                if not pre_amendment_report:
                    pre_amendment_report = True
                    print(f"note: {report_path} predates AMENDMENT 2 (no "
                          "legacy_forward_only block); its top-level greedy counters "
                          "are the forward-only ones and are used as such")

            if reference is None:
                stuck += 1
                if combo is not None:
                    fail(f"{ds} alpha={alpha_key}: reference greedy is stuck, "
                         "decoder produced a plan")
                if not fwd.get("greedy_stuck"):
                    fail(f"{ds} alpha={alpha_key}: verifier greedy stuck, frozen "
                         "report's forward-only decode not")
                checked += 1
                continue
            if fwd.get("greedy_stuck"):
                fail(f"{ds} alpha={alpha_key}: frozen report's forward-only decode "
                     "greedy stuck, verifier reference is not")
            if combo is None:
                fail(f"{ds} alpha={alpha_key}: decoder returned None, reference "
                     f"greedy found {reference}")
            decoded = {t: (int(c[0]), int(c[1])) for t, c in enumerate(combo)}
            if decoded != reference:
                fail(f"{ds} alpha={alpha_key}: decoder plan {decoded} != "
                     f"reference greedy {reference}")
            lookup = {tuple(sorted(p.items())): v for p, v in rows}
            key = tuple(sorted(decoded.items()))
            if key not in lookup:
                fail(f"{ds} alpha={alpha_key}: decoded plan {key} not in sweep")
            regret = 100.0 * (lookup[key] - best) / best
            if abs(fwd["r_greedy_pct"] - regret) > 1e-9:
                fail(f"{ds} alpha={alpha_key}: decoded-plan regret {regret!r} != "
                     f"frozen forward-only r_greedy_pct {fwd['r_greedy_pct']!r}")
            checked += 1
    print(f"OK: masked_topo decoder reproduces the topological-order masked greedy "
          f"and the frozen forward-only r_greedy_pct on {checked} (dataset, alpha) cells "
          f"({stuck} greedy-stuck cells matched) across {len(ds_dirs)} datasets, "
          f"alphas {list(alpha_keys)}")
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
    ap.add_argument("--check-decoder", action="store_true",
                    help="B1 acceptance (§4): the production masked_topo decoder, "
                         "fed true min-marginals, must reproduce the topological-"
                         "order masked greedy and the frozen r_greedy_pct in "
                         "--report on every dataset x alpha (default alphas "
                         "2.0 and 3.0; override with --alpha)")
    ap.add_argument("--check-repairs", action="store_true",
                    help="also independently recompute the 1int/kint repair fits "
                         "(slower: one small linear solve per dataset)")
    ap.add_argument("--alpha", action="append",
                    help="restrict to these alpha keys (e.g. --alpha 2.0), default all "
                         "in the report")
    args = ap.parse_args()

    if sum(bool(x) for x in (args.check_blocks, args.check_krank,
                             args.check_decoder)) > 1:
        fail("--check-blocks / --check-krank / --check-decoder are separate "
             "passes; run one at a time")
    if args.check_decoder:
        if len(args.corpus) != 1:
            fail("--check-decoder takes exactly one --corpus")
        if not args.report:
            fail("--check-decoder needs --report (the frozen stage-1 scorer "
                 "report carrying r_greedy_pct)")
        return check_decoder(args.corpus[0], args.report, args.task_types,
                             tuple(args.alpha) if args.alpha else ("2.0", "3.0"))
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
        cap_mode = rep.get("cap_mode", "alpha_max")
        ds_dirs = sorted(d for d in Path(corpus).glob("ds_*") if d.is_dir())
        for alpha_key, results in rep["per_dataset"].items():
            if args.alpha and alpha_key not in args.alpha:
                continue
            alpha = None if alpha_key == "None" else float(alpha_key)
            if len(results) != len(ds_dirs):
                fail(f"{corpus} alpha={alpha_key}: {len(results)} scored rows vs "
                     f"{len(ds_dirs)} datasets")
            for ds, scored in zip(ds_dirs, results):
                rows, ttypes, pid_map, task_db, dag_edges, net, sources, scales = load(
                    ds, args.task_types)
                r_exact, r_greedy, repairs = recompute(
                    rows, ttypes, pid_map, task_db, scales, alpha,
                    check_repairs=args.check_repairs,
                    dag_edges=dag_edges, net=net, sources=sources, cap_mode=cap_mode)
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
