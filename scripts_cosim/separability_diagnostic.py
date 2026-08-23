#!/usr/bin/env python3
"""Quantify how much *irreducible joint coupling* exists in the co-sim oracle labels.

Central question for "is a GNN necessary":
  A pointwise edge scorer f(task, platform) -> logit picks argmax per task,
  INDEPENDENTLY. A GNN can (in principle) make the score of one edge depend on
  the rest of the batch. The GNN can only beat a pointwise model when the OPTIMAL
  placement is NOT recoverable by independent per-task decisions.

We measure three label-grounded quantities per dataset, from placements/placements.jsonl
(the full (placement_plan, rtt) brute-force sweep):

  (M1) marginal-greedy regret:
       pi(t) = argmin_p [ min RTT over all combos with task t on platform p ]
       regret = RTT(joint combo of pi) - RTT(optimal).
       If ~0 everywhere, each task's best platform is independent of the others
       => separable => pointwise MLP suffices.

  (M2) identical-task symmetry:
       tasks with identical (type, source_node) get IDENTICAL features, hence a
       pointwise model (and a frozen-decode GNN) MUST assign them the same platform.
       - frac of datasets with >=2 identical tasks
       - among those, does the optimum SPREAD them?
       - regret of forcing identical tasks to co-locate (pointwise lower bound):
         min RTT over combos where every identical group shares one platform,
         minus optimal RTT.

  (M3) collision in optimum:
       fraction of optimal combos that place 2+ tasks on the SAME platform
       (i.e., the optimum itself "double books") vs. spreads.

  (M4) variance decomposition of the RTT target itself:
       M1-M3 ask whether the *argmin* is recoverable pointwise. M4 asks the stronger
       question of whether the whole RTT surface is, by fitting

           rtt(plan) ~ mu + sum_t f_t(plan[t])

       -- a fully pointwise model, exactly what PointwiseEdgeMLP can express -- by least
       squares over every plan in the sweep. additive_r2 near 1.0 means a pointwise
       scorer is the correctly specified model class and NO amount of data or
       architecture work can give a GNN an edge.

       interaction_r2_gain adds a single collision-count column. If that recovers the
       remaining variance, the entire coupling is one integer you can hand an MLP,
       not graph structure.

       node_only_r2 / platform_only_r2 describe WHERE the separable signal sits (which
       node vs which platform on it). Platform ids are globally unique, so
       platform_only_r2 == additive_r2 by construction -- it is a sanity check, not a
       signal. Neither is evidence about whether topology matters; use the network
       share of RTT for that.

Run:
  pipenv run python3 scripts_cosim/separability_diagnostic.py <corpus_dir> [--limit N]
  pipenv run python3 scripts_cosim/separability_diagnostic.py <corpus_dir> --gate-additive-r2 0.95
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# link_contention_v1: the same route resolution the simulator uses, so the offline
# analysis charges a plan to exactly the links the run would.
from src.placement.network_fabric import route_links  # noqa: E402


def load_workload_task_sigs(ds_dir: Path) -> Optional[List[Tuple[str, str]]]:
    """Return per-task (type_name, source_node) in event order = task index order."""
    wl = ds_dir / "workload.json"
    if not wl.exists():
        return None
    try:
        data = json.loads(wl.read_text())
    except Exception:
        return None
    sigs: List[Tuple[str, str]] = []
    for ev in data.get("events", []):
        app = ev.get("application", {})
        # task type is the dag key, e.g. {"dnn2": []}
        dag = app.get("dag", {})
        ttype = next(iter(dag.keys()), app.get("name", "?"))
        src = str(ev.get("node_name", "?"))
        sigs.append((str(ttype), src))
    return sigs


def load_combos(ds_dir: Path) -> Optional[List[Tuple[Dict[int, Tuple[int, int]], float]]]:
    jp = ds_dir / "placements" / "placements.jsonl"
    if not jp.exists():
        return None
    combos: List[Tuple[Dict[int, Tuple[int, int]], float]] = []
    with jp.open() as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception as exc:
                raise RuntimeError(
                    f"{jp}:{line_number}: invalid JSON"
                ) from exc
            plan = rec.get("placement_plan")
            rtt = rec.get("rtt")
            if plan is None or rtt is None:
                raise RuntimeError(
                    f"{jp}:{line_number}: missing placement_plan or rtt"
                )
            pp: Dict[int, Tuple[int, int]] = {}
            for k, v in plan.items():
                try:
                    if isinstance(v, (list, tuple)) and len(v) >= 2:
                        pp[int(k)] = (int(v[0]), int(v[1]))
                    else:
                        raise ValueError("placement must contain node and platform")
                except Exception as exc:
                    raise RuntimeError(
                        f"{jp}:{line_number}: invalid placement plan"
                    ) from exc
            if not pp:
                raise RuntimeError(f"{jp}:{line_number}: empty placement plan")
            combos.append((pp, float(rtt)))
    return combos or None


def _excess_sharing(row: Sequence[Any]) -> int:
    """How many assignments in a plan are 'extra' occupants of an already-used slot."""
    return sum(count - 1 for count in Counter(row).values() if count > 1)


def load_link_context(ds_dir: Path) -> Optional[dict]:
    """Routes + task sources, so a plan can be mapped onto the links it loads.

    Returns None for any corpus without a ``link_topology`` -- i.e. everything generated
    before link_contention_v1 -- and the link repair columns are then simply absent.
    """
    infra_path = ds_dir / "infrastructure.json"
    space_path = ds_dir / "space_with_network.json"
    workload_path = ds_dir / "workload.json"
    if not (infra_path.exists() and space_path.exists() and workload_path.exists()):
        return None
    infra = json.loads(infra_path.read_text())
    link_topology = infra.get("link_topology")
    if not link_topology:
        return None
    space = json.loads(space_path.read_text())
    workload = json.loads(workload_path.read_text())
    return {
        "routes": link_topology["routes"],
        "sources": [event["node_name"] for event in workload.get("events", [])],
        "n_clients": int(space["nodes"]["client_nodes"]["count"]),
    }


def _plan_link_load(
    plan_nodes: Sequence[int],
    task_ids: Sequence[int],
    context: dict,
) -> Counter:
    """Link keys crossed by a plan, counted with multiplicity across tasks."""
    routes = context["routes"]
    sources = context["sources"]
    n_clients = context["n_clients"]
    load: Counter = Counter()
    for slot, task_id in enumerate(task_ids):
        node_index = int(plan_nodes[slot])
        dest = (
            f"client_node{node_index}"
            if node_index < n_clients
            else f"node{node_index - n_clients}"
        )
        src = sources[int(task_id)]
        if dest == src:
            # Local execution never touches the network, matching the simulator's
            # `task.node_name != self.node.node_name` guard.
            continue
        for key in route_links(routes, src, dest):
            load[key] += 1
    return load


def _link_repair_columns(load: Counter) -> List[float]:
    """The scalar summaries a pointwise model could be handed for free.

    k1 = load on the busiest link, k2 = the top two, excess = total link-sharing excess.
    These are the link-level analogue of node-occupancy excess. If any of them repairs
    the additive fit's regret, the "coupling" is again a count and this mechanism has
    merely moved the previous four failures up one level.
    """
    counts = sorted(load.values(), reverse=True)
    busiest = float(counts[0]) if counts else 0.0
    second = float(counts[1]) if len(counts) > 1 else 0.0
    excess = float(sum(count - 1 for count in counts if count > 1))
    return [busiest, second, excess]


def _indicator_matrix(
    assignments: List[Tuple[Any, ...]],
    extra: Optional[List[List[float]]] = None,
) -> np.ndarray:
    """Intercept + per-(task slot, value) indicators, optionally plus numeric columns.

    The one-hot block per slot is collinear with the intercept; lstsq takes the
    minimum-norm solution and R^2 is unaffected by that rank deficiency.
    """
    n_slots = len(assignments[0])
    vocab = [sorted({row[slot] for row in assignments}) for slot in range(n_slots)]
    index = [{value: i for i, value in enumerate(values)} for values in vocab]
    offsets: List[int] = []
    width = 1
    for values in vocab:
        offsets.append(width)
        width += len(values)
    n_extra = 0 if extra is None else len(extra[0])
    matrix = np.zeros((len(assignments), width + n_extra), dtype=float)
    matrix[:, 0] = 1.0
    for row_idx, row in enumerate(assignments):
        for slot, value in enumerate(row):
            matrix[row_idx, offsets[slot] + index[slot][value]] = 1.0
    if extra is not None:
        matrix[:, width:] = np.asarray(extra, dtype=float)
    return matrix


def _fit_r2(matrix: np.ndarray, y: np.ndarray, ss_tot: float) -> float:
    beta, *_ = np.linalg.lstsq(matrix, y, rcond=None)
    residual = y - matrix @ beta
    return 1.0 - float(residual @ residual) / ss_tot


def variance_decomposition(
    combos: List[Tuple[Dict[int, Tuple[int, int]], float]],
    task_ids: List[int],
    link_context: Optional[dict] = None,
) -> dict:
    """M4: how much of the RTT surface a pointwise model can express.

    Returns ``{"degenerate": True}`` when every plan has the same RTT -- the placement
    decision is then inconsequential and R^2 is undefined, which is a real property of
    the dataset rather than a failure.
    """
    y = np.array([rtt for _, rtt in combos], dtype=float)
    centered = y - y.mean()
    ss_tot = float(centered @ centered)
    if ss_tot <= 0.0:
        return {"degenerate": True}

    candidate_rows = [tuple(plan[t] for t in task_ids) for plan, _ in combos]
    node_rows = [tuple(node for node, _ in row) for row in candidate_rows]
    platform_rows = [tuple(platform for _, platform in row) for row in candidate_rows]

    if len(candidate_rows) < sum(len({row[s] for row in candidate_rows}) for s in range(len(task_ids))) + 2:
        return {"degenerate": True, "reason": "underdetermined"}

    platform_collisions = [[float(_excess_sharing(row))] for row in candidate_rows]
    node_collisions = [[float(_excess_sharing(row))] for row in node_rows]

    additive_r2 = _fit_r2(_indicator_matrix(candidate_rows), y, ss_tot)
    with_platform_collision = _fit_r2(
        _indicator_matrix(candidate_rows, platform_collisions), y, ss_tot
    )
    with_node_collision = _fit_r2(
        _indicator_matrix(candidate_rows, node_collisions), y, ss_tot
    )
    node_only_r2 = _fit_r2(_indicator_matrix(node_rows), y, ss_tot)
    platform_only_r2 = _fit_r2(_indicator_matrix(platform_rows), y, ss_tot)

    # Regret of following the additive fit's argmin instead of the true optimum,
    # restricted to plans actually enumerated in the sweep.
    matrix = _indicator_matrix(candidate_rows)
    beta, *_ = np.linalg.lstsq(matrix, y, rcond=None)
    additive_choice = int(np.argmin(matrix @ beta))
    best = float(y.min())
    additive_regret_rel = (float(y[additive_choice]) - best) / best if best > 0 else None

    # THE ONE-INTEGER CONTROL. Re-take the argmin after handing the additive fit a single
    # scalar column: the plan's node-occupancy excess. If that repairs the regret, the
    # "coupling" is a count-like feature a pointwise MLP learns from one extra input, not
    # graph structure -- and a GNN win on such a corpus would only mean the baseline was
    # under-featurised. Every mechanism tried so far (added_in_batch, deep queues, node
    # ingress) has collapsed this way at whatever setting produced decision-level regret,
    # so this must be reported next to additive_choice_regret_rel, never on its own.
    aug_matrix = _indicator_matrix(candidate_rows, node_collisions)
    aug_beta, *_ = np.linalg.lstsq(aug_matrix, y, rcond=None)
    aug_choice = int(np.argmin(aug_matrix @ aug_beta))
    aug_regret_rel = (float(y[aug_choice]) - best) / best if best > 0 else None
    if additive_regret_rel is None or aug_regret_rel is None:
        one_integer_repair = None
    elif additive_regret_rel <= 0.0:
        one_integer_repair = None  # nothing to repair; excluded from the mean
    else:
        one_integer_repair = max(
            0.0, (additive_regret_rel - aug_regret_rel) / additive_regret_rel
        )

    # THE LINK REPAIR CONTROL (link_contention_v1). The node-occupancy column above is
    # structurally BLIND to link contention: two tasks on different destination nodes can
    # queue on a shared core segment while node-occupancy excess reads exactly zero. That
    # blindness is the whole reason to expect this mechanism to survive -- and it also
    # means the node column is the wrong control for it. These are the right ones: the
    # scalar summaries of the plan's link load a pointwise model could be handed free.
    #
    # k1 alone repairing the regret is the specific failure to watch for: it would mean
    # the topology has one bottleneck segment and the degeneracy has simply moved up a
    # level from "the busiest node" to "the busiest link".
    link_repairs: Dict[str, Optional[float]] = {}
    if link_context is not None and additive_regret_rel is not None and additive_regret_rel > 0:
        link_columns = [
            _link_repair_columns(_plan_link_load(row, task_ids, link_context))
            for row in node_rows
        ]
        for name, width in (("k1", 1), ("k2", 2), ("excess", 3)):
            cols = [values[:width] if width < 3 else [values[2]] for values in link_columns]
            fit = _indicator_matrix(candidate_rows, cols)
            fit_beta, *_ = np.linalg.lstsq(fit, y, rcond=None)
            choice = int(np.argmin(fit @ fit_beta))
            regret = (float(y[choice]) - best) / best if best > 0 else None
            link_repairs[f"link_repair_frac_{name}"] = (
                None
                if regret is None
                else max(0.0, (additive_regret_rel - regret) / additive_regret_rel)
            )
            link_repairs[f"link_choice_regret_rel_{name}"] = regret
    else:
        for name in ("k1", "k2", "excess"):
            link_repairs[f"link_repair_frac_{name}"] = None
            link_repairs[f"link_choice_regret_rel_{name}"] = None

    return {
        "degenerate": False,
        "additive_r2": additive_r2,
        **link_repairs,
        "additive_plus_collision_choice_regret_rel": aug_regret_rel,
        "additive_plus_collision_choice_is_optimal": aug_regret_rel == 0.0,
        "one_integer_repair_frac": one_integer_repair,
        "interaction_r2_gain_platform_collision": with_platform_collision - additive_r2,
        "interaction_r2_gain_node_collision": with_node_collision - additive_r2,
        "node_only_r2": node_only_r2,
        "platform_only_r2": platform_only_r2,
        "node_share_of_additive": (
            node_only_r2 / additive_r2 if additive_r2 > 0 else None
        ),
        "additive_choice_regret_rel": additive_regret_rel,
        "additive_choice_is_optimal": additive_regret_rel == 0.0,
        "frac_plans_with_platform_collision": float(
            np.mean([row[0] > 0 for row in platform_collisions])
        ),
    }


def _spread_plans_only(
    combos: List[Tuple[Dict[int, Tuple[int, int]], float]],
) -> List[Tuple[Dict[int, Tuple[int, int]], float]]:
    """Keep only plans that place every task on a DISTINCT node.

    The isolation control for link_contention_v1. On this subset node-occupancy excess is
    identically zero, so the repair column that killed the four previous mechanisms is a
    constant and cannot explain anything; `added_in_batch` is zero too, since distinct
    nodes imply distinct platforms. Link contention, by contrast, survives untouched --
    it acts *between* tasks on different destinations.

    So any regret remaining here is coupling that is neither platform-collision nor
    node-occupancy shaped. That is the question the mixed n=48 result could not answer,
    because the node term dominated and confounded the link term.
    """
    spread = []
    for plan, rtt in combos:
        nodes = [node for node, _platform in plan.values()]
        if len(set(nodes)) == len(nodes):
            spread.append((plan, rtt))
    return spread


def analyze_dataset(ds_dir: Path, spread_only: bool = False) -> Optional[dict]:
    combos = load_combos(ds_dir)
    if not combos:
        return None
    if spread_only:
        combos = _spread_plans_only(combos)
        if len(combos) < 4:
            return None
    sigs = load_workload_task_sigs(ds_dir)

    n_tasks = max(len(pp) for pp, _ in combos)
    # task indices present
    task_ids = sorted({t for pp, _ in combos for t in pp.keys()})
    if len(task_ids) < 1:
        return None

    # optimal
    opt_plan, opt_rtt = min(combos, key=lambda x: x[1])
    if opt_rtt <= 0:
        return None

    # --- M1: marginal greedy ---
    # marginal_min[t][p] = min rtt over combos with task t -> p
    marginal_min: Dict[int, Dict[Tuple[int, int], float]] = defaultdict(lambda: defaultdict(lambda: float("inf")))
    combo_lookup: Dict[Tuple[Tuple[int, int], ...], float] = {}
    for pp, rtt in combos:
        key = tuple(pp[t] for t in task_ids)
        # keep min rtt if duplicate combo keys
        if key not in combo_lookup or rtt < combo_lookup[key]:
            combo_lookup[key] = rtt
        for t in task_ids:
            p = pp[t]
            if rtt < marginal_min[t][p]:
                marginal_min[t][p] = rtt

    greedy_choice: Dict[int, Tuple[int, int]] = {}
    for t in task_ids:
        greedy_choice[t] = min(marginal_min[t].items(), key=lambda kv: kv[1])[0]
    greedy_key = tuple(greedy_choice[t] for t in task_ids)
    greedy_rtt = combo_lookup.get(greedy_key)  # may be None if combo not enumerated
    m1_regret = None
    m1_regret_rel = None
    greedy_in_sweep = greedy_rtt is not None
    # A greedy plan that re-uses a platform is structurally absent from a
    # unique-replicas sweep; only a collision-FREE absence indicates truncation.
    greedy_has_collision = len(set(greedy_key)) < len(greedy_key)
    if greedy_rtt is not None:
        m1_regret = greedy_rtt - opt_rtt
        m1_regret_rel = m1_regret / opt_rtt

    # --- M2: identical-task symmetry (needs sigs) ---
    m2 = {
        "has_identical": False,
        "n_identical_groups": 0,
        "max_group_size": 0,
        "opt_spreads_identical": None,
        "colocate_regret_rel": None,
    }
    if sigs is not None and len(sigs) >= len(task_ids):
        groups: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        for t in task_ids:
            if t < len(sigs):
                groups[sigs[t]].append(t)
        ident_groups = [g for g in groups.values() if len(g) >= 2]
        m2["n_identical_groups"] = len(ident_groups)
        m2["has_identical"] = len(ident_groups) > 0
        m2["max_group_size"] = max((len(g) for g in ident_groups), default=0)
        if ident_groups:
            # does optimum place each identical group on a single platform?
            opt_spreads = False
            for g in ident_groups:
                plats = {opt_plan[t] for t in g}
                if len(plats) > 1:
                    opt_spreads = True
                    break
            m2["opt_spreads_identical"] = opt_spreads
            # pointwise lower bound: best combo where each identical group co-locates
            best_colo = float("inf")
            for key, rtt in combo_lookup.items():
                ok = True
                for g in ident_groups:
                    plats = {key[task_ids.index(t)] for t in g}
                    if len(plats) > 1:
                        ok = False
                        break
                if ok and rtt < best_colo:
                    best_colo = rtt
            if best_colo < float("inf"):
                m2["colocate_regret_rel"] = (best_colo - opt_rtt) / opt_rtt

    # --- M3: collision in optimum ---
    opt_plats = [opt_plan[t] for t in task_ids]
    opt_has_collision = len(set(opt_plats)) < len(opt_plats)
    opt_unique_plats = len(set(opt_plats))

    return {
        "n_tasks": len(task_ids),
        "n_combos": len(combos),
        "opt_rtt": opt_rtt,
        "greedy_in_sweep": greedy_in_sweep,
        "greedy_has_collision": greedy_has_collision,
        "m1_regret_rel": m1_regret_rel,
        "m1_greedy_eq_opt": (greedy_key == tuple(opt_plan[t] for t in task_ids)),
        "m2": m2,
        "opt_has_collision": opt_has_collision,
        "opt_unique_plats": opt_unique_plats,
        "m4": variance_decomposition(combos, task_ids, load_link_context(ds_dir)),
    }


def pctl(vals: List[float], q: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    return s[min(len(s) - 1, int(q * len(s)))]


def summarize_results(results: List[dict]) -> dict[str, Any]:
    if not results:
        raise ValueError("Cannot summarize empty separability results")
    n = len(results)
    m1_rel = [
        r["m1_regret_rel"]
        for r in results
        if r["m1_regret_rel"] is not None
    ]
    greedy_eq = sum(1 for r in results if r["m1_greedy_eq_opt"])
    greedy_in = sum(1 for r in results if r["greedy_in_sweep"])
    has_ident = sum(1 for r in results if r["m2"]["has_identical"])
    spreads = sum(
        1 for r in results if r["m2"]["opt_spreads_identical"] is True
    )
    colo_rel = [
        r["m2"]["colocate_regret_rel"]
        for r in results
        if r["m2"]["colocate_regret_rel"] is not None
    ]
    opt_coll = sum(1 for r in results if r["opt_has_collision"])
    multitask = [r for r in results if r["n_tasks"] >= 2]
    fitted = [r["m4"] for r in results if not r["m4"]["degenerate"]]
    degenerate = n - len(fitted)

    def distribution(values: List[float]) -> dict[str, float | int | None]:
        if not values:
            return {
                "count": 0,
                "mean": None,
                "median": None,
                "p90": None,
                "p99": None,
                "max": None,
            }
        return {
            "count": len(values),
            "mean": sum(values) / len(values),
            "median": pctl(values, 0.5),
            "p90": pctl(values, 0.9),
            "p99": pctl(values, 0.99),
            "max": max(values),
        }

    return {
        "datasets_analyzed": n,
        "multitask_datasets": len(multitask),
        "mean_n_combos": sum(r["n_combos"] for r in results) / n,
        "m1_marginal_greedy": {
            "greedy_in_sweep_count": greedy_in,
            "greedy_in_sweep_fraction": greedy_in / n,
            "greedy_exact_optimum_count": greedy_eq,
            "greedy_exact_optimum_fraction": greedy_eq / n,
            "regret_relative": distribution(m1_rel),
            "coupled_gt_1pct_count": sum(value > 0.01 for value in m1_rel),
            "coupled_gt_1pct_fraction": (
                sum(value > 0.01 for value in m1_rel) / len(m1_rel)
                if m1_rel
                else None
            ),
            "coupled_gt_5pct_count": sum(value > 0.05 for value in m1_rel),
            "coupled_gt_5pct_fraction": (
                sum(value > 0.05 for value in m1_rel) / len(m1_rel)
                if m1_rel
                else None
            ),
            "coupled_gt_10pct_count": sum(value > 0.10 for value in m1_rel),
            "coupled_gt_10pct_fraction": (
                sum(value > 0.10 for value in m1_rel) / len(m1_rel)
                if m1_rel
                else None
            ),
        },
        "m2_identical_tasks": {
            "has_identical_count": has_ident,
            "has_identical_fraction": has_ident / n,
            "optimum_spreads_identical_count": spreads,
            "optimum_spreads_identical_fraction": (
                spreads / has_ident if has_ident else None
            ),
            "forced_colocation_regret_relative": distribution(colo_rel),
        },
        "m3_optimum_collision": {
            "collision_count": opt_coll,
            "collision_fraction": opt_coll / n,
            "avg_unique_platforms_multitask": (
                sum(r["opt_unique_plats"] for r in multitask) / len(multitask)
                if multitask
                else None
            ),
            "avg_tasks_multitask": (
                sum(r["n_tasks"] for r in multitask) / len(multitask)
                if multitask
                else None
            ),
        },
        "m4_variance_decomposition": {
            "fitted_datasets": len(fitted),
            "degenerate_datasets": degenerate,
            "additive_r2": distribution([f["additive_r2"] for f in fitted]),
            "interaction_r2_gain_platform_collision": distribution(
                [f["interaction_r2_gain_platform_collision"] for f in fitted]
            ),
            "interaction_r2_gain_node_collision": distribution(
                [f["interaction_r2_gain_node_collision"] for f in fitted]
            ),
            "node_share_of_additive": distribution(
                [f["node_share_of_additive"] for f in fitted if f["node_share_of_additive"] is not None]
            ),
            "additive_choice_regret_rel": distribution(
                [f["additive_choice_regret_rel"] for f in fitted if f["additive_choice_regret_rel"] is not None]
            ),
            "additive_choice_optimal_fraction": (
                sum(1 for f in fitted if f["additive_choice_is_optimal"]) / len(fitted)
                if fitted
                else None
            ),
            # The one-integer control -- see variance_decomposition().
            "additive_plus_collision_choice_regret_rel": distribution(
                [
                    f["additive_plus_collision_choice_regret_rel"]
                    for f in fitted
                    if f.get("additive_plus_collision_choice_regret_rel") is not None
                ]
            ),
            "additive_plus_collision_choice_optimal_fraction": (
                sum(1 for f in fitted if f.get("additive_plus_collision_choice_is_optimal"))
                / len(fitted)
                if fitted
                else None
            ),
            "one_integer_repair_frac": distribution(
                [
                    f["one_integer_repair_frac"]
                    for f in fitted
                    if f.get("one_integer_repair_frac") is not None
                ]
            ),
            # link_contention_v1 controls -- the node column above cannot see link
            # sharing between different destinations, so these are the ones that matter
            # for this mechanism. See variance_decomposition().
            **{
                f"link_repair_frac_{name}": distribution(
                    [
                        f[f"link_repair_frac_{name}"]
                        for f in fitted
                        if f.get(f"link_repair_frac_{name}") is not None
                    ]
                )
                for name in ("k1", "k2", "excess")
            },
            "mean_frac_plans_with_platform_collision": (
                sum(f["frac_plans_with_platform_collision"] for f in fitted) / len(fitted)
                if fitted
                else None
            ),
        },
    }


def print_summary(base_name: str, summary: dict[str, Any]) -> None:
    n = summary["datasets_analyzed"]
    m1 = summary["m1_marginal_greedy"]
    m2 = summary["m2_identical_tasks"]
    m3 = summary["m3_optimum_collision"]
    regret = m1["regret_relative"]
    colo = m2["forced_colocation_regret_relative"]

    print(f"\n===== Separability diagnostic: {base_name} =====")
    print(
        f"Datasets analyzed: {n} "
        f"(multi-task >=2: {summary['multitask_datasets']})"
    )
    print(f"Mean n_combos: {summary['mean_n_combos']:.0f}")
    print("\n--- M1: marginal-greedy (independent per-task best) vs joint optimum ---")
    print(
        f"  greedy combo present in sweep: {m1['greedy_in_sweep_count']}/{n} "
        f"({100 * m1['greedy_in_sweep_fraction']:.1f}%)"
    )
    print(
        f"  greedy == optimum (exact):     {m1['greedy_exact_optimum_count']}/{n} "
        f"({100 * m1['greedy_exact_optimum_fraction']:.1f}%)"
    )
    if regret["count"]:
        print(
            "  regret_rel (greedy vs opt): "
            f"mean={100 * regret['mean']:.2f}%  "
            f"median={100 * regret['median']:.2f}%  "
            f"p90={100 * regret['p90']:.2f}%  "
            f"p99={100 * regret['p99']:.2f}%  "
            f"max={100 * regret['max']:.1f}%"
        )
        print(
            "  coupled datasets: "
            f">1%={100 * m1['coupled_gt_1pct_fraction']:.1f}%  "
            f">5%={100 * m1['coupled_gt_5pct_fraction']:.1f}%  "
            f">10%={100 * m1['coupled_gt_10pct_fraction']:.1f}%"
        )
    print("\n--- M2: identical (type,src) tasks => pointwise MUST co-assign ---")
    print(
        f"  datasets with >=2 identical tasks: {m2['has_identical_count']}/{n} "
        f"({100 * m2['has_identical_fraction']:.1f}%)"
    )
    if m2["has_identical_count"]:
        print(
            "  among identical: optimum SPREADS them: "
            f"{m2['optimum_spreads_identical_count']}/{m2['has_identical_count']} "
            f"({100 * m2['optimum_spreads_identical_fraction']:.1f}%)"
        )
    if colo["count"]:
        print(
            "  forced-colocation regret_rel (pointwise floor): "
            f"mean={100 * colo['mean']:.2f}%  "
            f"median={100 * colo['median']:.2f}%  "
            f"p90={100 * colo['p90']:.2f}%  "
            f"max={100 * colo['max']:.1f}%"
        )
    print("\n--- M3: does the OPTIMUM itself collide (2+ tasks same platform)? ---")
    print(
        f"  optimal combo has collision: {m3['collision_count']}/{n} "
        f"({100 * m3['collision_fraction']:.1f}%)"
    )
    print(
        "  avg unique platforms in optimum: "
        f"{m3['avg_unique_platforms_multitask']:.2f} / "
        f"avg n_tasks {m3['avg_tasks_multitask']:.2f} (multi-task)"
    )

    m4 = summary["m4_variance_decomposition"]
    print("\n--- M4: can a pointwise model express the RTT surface at all? ---")
    print(
        f"  fitted: {m4['fitted_datasets']}/{n}"
        f"  (degenerate/underdetermined: {m4['degenerate_datasets']})"
    )
    if m4["fitted_datasets"]:
        r2d = m4["additive_r2"]
        print(
            "  additive R^2 (pointwise ceiling): "
            f"mean={r2d['mean']:.5f}  median={r2d['median']:.5f}  max={r2d['max']:.5f}"
        )
        print(
            "  R^2 gain from ONE collision-count column: "
            f"platform={100 * m4['interaction_r2_gain_platform_collision']['mean']:.3f} pp  "
            f"node={100 * m4['interaction_r2_gain_node_collision']['mean']:.3f} pp"
        )
        areg = m4["additive_choice_regret_rel"]
        if areg["count"]:
            print(
                "  additive-fit argmin regret: "
                f"mean={100 * areg['mean']:.2f}%  max={100 * areg['max']:.1f}%  "
                f"exactly optimal in {100 * m4['additive_choice_optimal_fraction']:.0f}% of datasets"
            )
        aug = m4.get("additive_plus_collision_choice_regret_rel", {"count": 0})
        rep = m4.get("one_integer_repair_frac", {"count": 0})
        if aug.get("count"):
            print(
                "  ONE-INTEGER CONTROL (+1 node-collision-count column): "
                f"argmin regret mean={100 * aug['mean']:.2f}%  "
                f"optimal in {100 * m4['additive_plus_collision_choice_optimal_fraction']:.0f}%"
            )
        if rep.get("count"):
            print(
                f"    -> repairs {100 * rep['mean']:.0f}% of the additive fit's regret "
                f"(n={rep['count']} datasets where it had any)."
            )
            if rep["mean"] >= 0.5:
                print(
                    "    !! DEGENERATE: most of the 'coupling' is a single count-like "
                    "feature. A pointwise MLP given that column closes the gap, so a GNN "
                    "win here would only mean the baseline was under-featurised."
                )
        link_reported = False
        for name, label in (
            ("k1", "busiest link load"),
            ("k2", "top-2 link loads"),
            ("excess", "total link-sharing excess"),
        ):
            dist = m4.get(f"link_repair_frac_{name}", {"count": 0})
            if not dist.get("count"):
                continue
            if not link_reported:
                print("  LINK REPAIR CONTROL (link_contention_v1) — the node column above")
                print("  cannot see contention between different destinations, so these are")
                print("  the controls that actually bind for this mechanism:")
                link_reported = True
            print(
                f"    +{label:<26} repairs {100 * dist['mean']:3.0f}% of the regret "
                f"(n={dist['count']})"
            )
        if link_reported:
            worst = max(
                m4[f"link_repair_frac_{name}"]["mean"]
                for name in ("k1", "k2", "excess")
                if m4.get(f"link_repair_frac_{name}", {}).get("count")
            )
            if worst >= 0.5:
                print(
                    "    !! DEGENERATE: a handful of scalar link summaries close the gap. "
                    "The coupling is still count-shaped, just counted over links instead "
                    "of nodes."
                )
        print(
            "  node share of additive signal: "
            f"{m4['node_share_of_additive']['mean']:.3f}   "
            f"plans containing a platform collision: "
            f"{100 * m4['mean_frac_plans_with_platform_collision']:.0f}%"
        )


def _load_integrity_manifest(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("status") != "clean" or manifest.get("clean") is not True:
        raise RuntimeError(f"Integrity manifest is not clean: {path}")
    return manifest, hashlib.sha256(raw).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus_dir", nargs="+", type=Path)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--integrity-manifest", type=Path)
    ap.add_argument("--output", type=Path)
    ap.add_argument(
        "--skip-corrupt",
        action="store_true",
        help=(
            "Skip datasets whose placements.jsonl fails to parse instead of aborting the "
            "corpus, and list them in the report as excluded. Diagnose-only alternative "
            "to archiving them via audit_placements_jsonl_integrity.py --exclude."
        ),
    )
    ap.add_argument(
        "--gate-coupled-fraction",
        type=float,
        default=None,
        help=(
            "DEPRECATED as a primary gate -- see --gate-additive-argmin-regret. M1's "
            "marginal greedy scores each option as 'min RTT over all joint plans with task "
            "t there', i.e. it has ORACLE access to the joint sweep, so it is not a "
            "pointwise model and cannot fail once the candidate set is small. Measured "
            "2026-08-18: a corpus where the additive fit picked a suboptimal plan 75%% of "
            "the time at 21.5%% mean regret still reported M1 coupled = 0%%. Fails (exit 1) "
            "if the fraction of datasets with M1 greedy regret >1%% is BELOW this."
        ),
    )
    ap.add_argument(
        "--gate-additive-r2",
        type=float,
        default=None,
        help=(
            "Fail (exit 1) if mean additive R^2 is at or above this threshold. A corpus "
            "above ~0.95 is pointwise-separable, so a GNN cannot beat a pointwise MLP on "
            "it no matter how it is trained. Run this on a ~30-dataset pilot BEFORE "
            "generating a full corpus."
        ),
    )
    ap.add_argument(
        "--gate-additive-argmin-regret",
        type=float,
        default=None,
        help=(
            "PREFERRED PRIMARY GATE. Fail (exit 1) if mean additive-fit argmin regret is "
            "BELOW this. The additive fit is literally what PointwiseEdgeMLP can express, "
            "so its argmin regret is the headroom a graph model could actually capture -- "
            "unlike M1, whose greedy sees the joint sweep. Reference points: contention_v2 "
            "~0.003, shallow_v1 ~0.030."
        ),
    )
    ap.add_argument(
        "--gate-one-integer-repair",
        type=float,
        default=None,
        help=(
            "MANDATORY COMPANION to --gate-additive-argmin-regret. Fail (exit 1) if a "
            "single node-collision-count column repairs AT OR ABOVE this fraction of the "
            "additive fit's regret -- that means the 'coupling' is one count-like feature "
            "a pointwise MLP learns from one extra input, not graph structure, and a GNN "
            "win would only show the baseline was under-featurised. Suggested 0.5. Every "
            "mechanism tried so far (added_in_batch, deep queues, node ingress) collapsed "
            "this way at whatever setting produced decision-level regret."
        ),
    )
    ap.add_argument(
        "--spread-plans-only",
        action="store_true",
        help=(
            "ISOLATION CONTROL. Restrict every metric to plans that place each task on a "
            "DISTINCT node. Node-occupancy excess is then identically zero across the "
            "retained plans, so the repair column that collapsed the four mechanisms "
            "before link_contention_v1 is a constant and cannot explain anything; "
            "`added_in_batch` is zero too. Link contention survives, because it acts "
            "between tasks on different destinations. Any regret left on this subset is "
            "coupling that is neither platform-collision nor node-occupancy shaped. Use it "
            "to separate a new mechanism from the collision term that otherwise dominates "
            "-- NOT as a corpus gate, since it discards most of the sweep."
        ),
    )
    ap.add_argument(
        "--gate-link-repair",
        type=float,
        default=None,
        help=(
            "link_contention_v1 companion control. Fail (exit 1) if the strongest scalar "
            "LINK summary -- busiest link load (k1), top-2 link loads (k2), or total "
            "link-sharing excess (excess) -- repairs AT OR ABOVE this fraction of the additive "
            "fit's regret. The node-collision control is structurally blind to contention "
            "between tasks on different destination nodes, which is exactly what a per-link "
            "model creates, so it cannot bind here; these can. k1 alone repairing the gap "
            "means the topology has one bottleneck segment and the degeneracy has merely "
            "moved from the busiest node to the busiest link. Suggested 0.5."
        ),
    )
    args = ap.parse_args()
    if args.gate_additive_argmin_regret is not None and args.gate_one_integer_repair is None:
        ap.error(
            "--gate-additive-argmin-regret requires --gate-one-integer-repair (suggested "
            "0.5). Headroom that a single collision-count column repairs is not GNN "
            "headroom; gating on regret alone is how a degenerate corpus passes."
        )

    integrity_manifest = None
    integrity_sha256 = None
    if args.integrity_manifest:
        integrity_manifest, integrity_sha256 = _load_integrity_manifest(
            args.integrity_manifest
        )

    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "integrity_manifest": (
            str(args.integrity_manifest) if args.integrity_manifest else None
        ),
        "integrity_manifest_sha256": integrity_sha256,
        "corpora": {},
    }
    for base in args.corpus_dir:
        if not base.is_dir():
            raise FileNotFoundError(f"Corpus directory not found: {base}")
        if integrity_manifest is not None:
            corpus_inventory = integrity_manifest["corpora"].get(base.name)
            if corpus_inventory is None:
                raise RuntimeError(
                    f"Corpus absent from integrity manifest: {base.name}"
                )
            retained_names = sorted(corpus_inventory["datasets"])
            ds_dirs = [base / name for name in retained_names]
            excluded = corpus_inventory["excluded_datasets"]
        else:
            ds_dirs = sorted(d for d in base.glob("ds_*") if d.is_dir())
            excluded = []
        if args.limit:
            ds_dirs = ds_dirs[: args.limit]
        if not ds_dirs:
            raise RuntimeError(f"No retained datasets under {base}")

        results: list[dict] = []
        skipped_corrupt: list[str] = []
        for dataset_dir in ds_dirs:
            try:
                result = analyze_dataset(dataset_dir, spread_only=args.spread_plans_only)
            except RuntimeError as exc:
                if args.skip_corrupt and "invalid JSON" in str(exc):
                    skipped_corrupt.append(dataset_dir.name)
                    print(f"  SKIPPED (corrupt placements.jsonl): {dataset_dir.name}")
                    continue
                raise
            if result is None:
                if args.spread_plans_only:
                    # Too few all-distinct-node plans to fit anything. A real property of
                    # the dataset under this restriction, not a corpus defect.
                    continue
                raise RuntimeError(f"Dataset is not analyzable: {dataset_dir}")
            result["dataset_id"] = dataset_dir.name
            results.append(result)
        summary = summarize_results(results)
        # Under --spread-plans-only the sweep is deliberately incomplete, so the marginal
        # greedy plan may legitimately have been filtered out. Everywhere else its absence
        # means a truncated placements.jsonl and must still fail loudly.
        # A greedy plan containing a platform collision is structurally absent from a
        # unique-replicas sweep (highq_safe hits this on complete 25/25 sweeps), so only
        # a collision-free absence is evidence of truncation.
        suspicious_absent = [
            r["dataset_id"]
            for r in results
            if not r["greedy_in_sweep"] and not r.get("greedy_has_collision")
        ]
        if not args.spread_plans_only and suspicious_absent:
            raise RuntimeError(
                f"{base.name}: collision-free marginal greedy combo absent from "
                f"{len(suspicious_absent)} retained full sweeps "
                f"(e.g. {suspicious_absent[:5]}) — truncated placements.jsonl"
            )
        print_summary(base.name, summary)
        if skipped_corrupt:
            print(f"  corrupt datasets skipped ({len(skipped_corrupt)}): {skipped_corrupt}")
        report["corpora"][base.name] = {
            "path": str(base),
            "excluded_datasets": excluded,
            "skipped_corrupt_datasets": skipped_corrupt,
            "summary": summary,
            "datasets": results,
        }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"\nFrozen report: {args.output}")

    if args.gate_additive_argmin_regret is not None:
        regret_failures: list[str] = []
        degenerate: list[str] = []
        for name, entry in report["corpora"].items():
            m4 = entry["summary"]["m4_variance_decomposition"]
            areg = m4["additive_choice_regret_rel"]
            if not areg["count"]:
                raise RuntimeError(
                    f"{name}: no additive-fit argmin regrets, cannot evaluate the gate"
                )
            mean_regret = areg["mean"]
            rep = m4.get("one_integer_repair_frac", {"count": 0})
            repair = rep["mean"] if rep.get("count") else 0.0
            ok_regret = mean_regret >= args.gate_additive_argmin_regret
            ok_repair = repair < args.gate_one_integer_repair

            # link_contention_v1: the strongest scalar link summary is the binding
            # control for a per-link mechanism, since the node column cannot see
            # cross-destination contention at all.
            link_repair = 0.0
            worst_link_control = None
            for control in ("k1", "k2", "excess"):
                dist = m4.get(f"link_repair_frac_{control}", {"count": 0})
                if dist.get("count") and dist["mean"] > link_repair:
                    link_repair = dist["mean"]
                    worst_link_control = control
            ok_link = (
                args.gate_link_repair is None or link_repair < args.gate_link_repair
            )

            verdict = "PASS" if (ok_regret and ok_repair and ok_link) else "FAIL"
            link_note = (
                ""
                if args.gate_link_repair is None
                else (
                    f", link repair={link_repair:.3f}"
                    f"{f' ({worst_link_control})' if worst_link_control else ''} "
                    f"(must stay below {args.gate_link_repair:.3f})"
                )
            )
            print(
                f"\n[gate] {name}: additive-argmin regret={mean_regret:.4f} "
                f"(threshold {args.gate_additive_argmin_regret:.4f}), "
                f"one-integer repair={repair:.3f} "
                f"(must stay below {args.gate_one_integer_repair:.3f})"
                f"{link_note} -> {verdict}"
            )
            if not ok_regret:
                regret_failures.append(f"{name} ({mean_regret:.4f})")
            if not ok_repair:
                degenerate.append(f"{name} (node repair={repair:.3f})")
            if not ok_link:
                degenerate.append(
                    f"{name} (link repair={link_repair:.3f} via {worst_link_control})"
                )
        if regret_failures:
            print(
                "\n[gate] FAILED - too little pointwise headroom: "
                + ", ".join(regret_failures)
                + "\n[gate] The additive fit already picks a near-optimal plan, so there is "
                "little for a graph model to capture."
            )
        if degenerate:
            print(
                "\n[gate] FAILED - DEGENERATE coupling: "
                + ", ".join(degenerate)
                + "\n[gate] A single node-collision-count column repairs most of the "
                "regret. Hand that column to the MLP and the gap closes; this is not "
                "graph structure."
            )
        if regret_failures or degenerate:
            return 1

    if args.gate_coupled_fraction is not None:
        coupled_failures: list[str] = []
        for name, entry in report["corpora"].items():
            m1 = entry["summary"]["m1_marginal_greedy"]
            fraction = m1["coupled_gt_1pct_fraction"]
            if fraction is None:
                raise RuntimeError(
                    f"{name}: no marginal-greedy regrets, cannot evaluate the gate"
                )
            verdict = "PASS" if fraction >= args.gate_coupled_fraction else "FAIL"
            print(
                f"\n[gate] {name}: coupled(>1%) fraction={fraction:.3f} "
                f"threshold={args.gate_coupled_fraction:.3f} -> {verdict}"
            )
            if verdict == "FAIL":
                coupled_failures.append(f"{name} ({fraction:.3f})")
        if coupled_failures:
            print(
                "\n[gate] FAILED - too few coupled datasets: "
                + ", ".join(coupled_failures)
                + "\n[gate] A pointwise MLP recovers the optimum on nearly every dataset "
                "here, so there is little for a GNN to learn."
            )
            return 1

    if args.gate_additive_r2 is not None:
        failures: list[str] = []
        for name, entry in report["corpora"].items():
            m4 = entry["summary"]["m4_variance_decomposition"]
            if not m4["fitted_datasets"]:
                raise RuntimeError(
                    f"{name}: no dataset could be fitted, cannot evaluate the gate"
                )
            mean_r2 = m4["additive_r2"]["mean"]
            verdict = "FAIL" if mean_r2 >= args.gate_additive_r2 else "PASS"
            print(
                f"\n[gate] {name}: additive R^2 mean={mean_r2:.5f} "
                f"threshold={args.gate_additive_r2:.5f} -> {verdict}"
            )
            if verdict == "FAIL":
                failures.append(f"{name} (R^2 {mean_r2:.5f})")
        if failures:
            print(
                "\n[gate] FAILED - pointwise-separable corpora: "
                + ", ".join(failures)
                + "\n[gate] A GNN cannot beat a pointwise MLP on these. Change the "
                "physics (node contention, congestible links, fan-out DAGs) before "
                "generating a full corpus."
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
