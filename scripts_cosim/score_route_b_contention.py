#!/usr/bin/env python3
"""route_b_v1 — constrained-argmin regret under node-memory contention.

Route A closed with: breaking separability is necessary but NOT sufficient — coupling
without competition leaves every task free to take its individual favourite. This scorer
measures the other hypothesis of the composition theorem, FREE CHOICE: a plan is feasible
only if every node's co-resident memory demand fits under a hard per-node capacity

    sum_{tasks on node} memReq[task_type][platform_type]  <=  cap_node(alpha)

with cap_node(alpha) = alpha * (max single demand among the sweep's candidate replicas on
that node). Memory occupancy does not change episode physics, so the constraint is a pure
function of (plan, capacities) and is applied to the FULL enumerated sweep at scoring
time — one corpus serves the whole tightness ladder, exactly like --spread-plans-only.

Statistics per dataset, per alpha, per objective:

  R_greedy  true cost of the plan chosen by feasibility-masked sequential greedy over
            additive min-marginals (the deployable pointwise scheduler), relative to the
            constrained sweep optimum.
  R_exact   (PRIMARY) true cost of the feasible plan minimizing the min-marginal-sum
            surrogate sum_t m_t(p_t), decoded by exhaustive search over the feasible
            sweep — the pointwise-scores-plus-PERFECT-decoder bound. On separable
            physics m_t(p) = c_t(p) + const, so sum_t m_t is the true cost up to a
            constant and `r_exact_band["optimistic"] == 0` EXACTLY, under ANY feasibility
            restriction — the theorem-predicted zero this scorer must reproduce on Arm B0.
            Unconstrained, it reduces to route A's componentwise-argmin statistic
            (measured 0.000%).

            UP TO TIES, and the qualifier is load-bearing. When the surrogate scores
            several feasible plans EQUAL, `decode_regret` resolves them by sorted plan key
            — platform id, unrelated to cost — so the registered `r_exact_pct` can be
            nonzero on provably separable physics. Measured 2026-08-27 on the H0 separable
            control: 12 of 16 firing datasets had the true optimum inside the argmin tie
            set. This file previously claimed "nonzero constrained R_exact can be neither a
            decoder artifact nor an LS fitting artifact"; that was FALSE and is why a
            decoder tie-break was read as physics. Read `r_exact_band` (§4: mean_tied is
            the fair reading, optimistic an upper bound only) — never `r_exact_pct` alone.
  repairs   R_exact recomputed with the surrogate y ~ a + b*sum_t m_t(p_t) + counts,
            where counts are (a) one-integer: node-occupancy excess sharing
            sum(count-1), the program's established collision column; (b) k-integer:
            per-node x per-type co-residency counts (the constraint's own sufficient
            statistic). Coefficients fit by LS on the FULL sweep — 3 to ~k+2 params, so
            saturation is impossible at these sweep sizes.

A full per-(task,placement) indicator LS surrogate is also reported (`r_exact_ls`) as a
sensitivity row only: measured on the m3 pilot it fires ~12% even unconstrained (the
known count-shaped collision channel plus LS argmin noise near ties), where the
min-marginal statistic measures the established 0.000% — it is NOT the gate statistic.

Fail-loud contract: missing placements.jsonl raises; makespan scoring on rows without
task_times raises; a plan referencing an unknown platform_id raises; a greedy decode that
cannot complete is counted and printed (`greedy_stuck`), never silently dropped; an alpha
that leaves zero feasible rows is counted and printed (`no_feasible_rows`), never
silently dropped.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.placement.network_fabric import is_core_link, route_links  # noqa: E402

Plan = Dict[int, Tuple[int, int]]

EPS = 1e-12


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_task_types(path: Path) -> Dict[str, dict]:
    with open(path) as fh:
        return json.load(fh)


def load_rows(ds_dir: Path, objective: str) -> List[Tuple[Plan, float]]:
    jsonl = ds_dir / "placements" / "placements.jsonl"
    if not jsonl.exists():
        raise RuntimeError(f"{ds_dir}: placements/placements.jsonl missing — the full "
                           "sweep is mandatory (CO_SIMULATION_GUIDE), refusing to score")
    # A truncated sweep biases every statistic here (marginals, optima, feasible sets)
    # toward whatever subset survived, and route B's smoke lost 66-72/240 rows to
    # mid-episode replica scale-down before this check existed. Unscoreable — refuse.
    meta_path = ds_dir / "placement_metadata.json"
    if not meta_path.exists():
        raise RuntimeError(f"{ds_dir}: placement_metadata.json missing — cannot verify "
                           "the sweep is complete, refusing to score")
    with open(meta_path) as fh:
        meta = json.load(fh)
    if not meta.get("sweep_complete", False):
        raise RuntimeError(
            f"{ds_dir}: sweep is TRUNCATED ({meta.get('rows_written')}/"
            f"{meta.get('num_placements')} rows, worker_failed="
            f"{meta.get('worker_failed')}, timed_out={meta.get('timed_out')}) — see "
            "placement_errors.log in the dataset dir; refusing to score")
    rows: List[Tuple[Plan, float]] = []
    with open(jsonl) as fh:
        for i, line in enumerate(fh):
            row = json.loads(line)
            plan = {int(k): (int(v[0]), int(v[1]))
                    for k, v in row["placement_plan"].items()}
            if objective == "rtt":
                value = float(row["rtt"])
            elif objective == "makespan":
                tt = row.get("task_times")
                if not tt:
                    raise RuntimeError(
                        f"{jsonl}:{i + 1} has no task_times — this sweep was not run "
                        "with HEROSIM_RETAIN_TASK_TIMES=1 and cannot be scored under a "
                        "makespan")
                value = max(t[2] for t in tt) - min(t[1] for t in tt)
            else:
                raise ValueError(f"unknown objective {objective!r}")
            rows.append((plan, value))
    if not rows:
        raise RuntimeError(f"{jsonl}: zero rows")
    return rows


def load_task_type_names(ds_dir: Path) -> List[str]:
    """Task type per task_id, in the id-assignment order (static_order per dag)."""
    from graphlib import TopologicalSorter
    with open(ds_dir / "workload.json") as fh:
        workload = json.load(fh)
    names: List[str] = []
    for event in workload["events"]:
        dag = event["application"]["dag"]
        if isinstance(dag, list):
            names.extend(dag)
        elif isinstance(dag, dict):
            names.extend(TopologicalSorter(dag).static_order())
        else:
            raise RuntimeError(f"{ds_dir}: unrecognised dag shape {type(dag)}")
    return names


def load_demand_scales(ds_dir: Path) -> List[float]:
    """Per-task_id demand_scale (route_b env pivot W2), same id-assignment order as
    load_task_type_names. An event's application.demand_scale is keyed by task TYPE
    name (generate_gnn_datasets_fast.py's generate_workload_templates writes one key
    per DAG task type per event, not per task_id, since task_id is a positional
    derivative of static_order). Absent key -> scale 1.0, so a dataset generated
    before this option existed (or without the demand_spread grid key) reads back
    byte-identical demand = 1.0 * type-table value everywhere."""
    from graphlib import TopologicalSorter
    with open(ds_dir / "workload.json") as fh:
        workload = json.load(fh)
    scales: List[float] = []
    for event in workload["events"]:
        dag = event["application"]["dag"]
        per_type = event["application"].get("demand_scale") or {}
        if isinstance(dag, list):
            names = dag
        elif isinstance(dag, dict):
            names = list(TopologicalSorter(dag).static_order())
        else:
            raise RuntimeError(f"{ds_dir}: unrecognised dag shape {type(dag)}")
        scales.extend(float(per_type.get(name, 1.0)) for name in names)
    return scales


def load_task_sources(ds_dir: Path) -> List[str]:
    """Submitting client node per task_id, same id-assignment order as
    load_task_type_names. Every task of an event ingresses from that event's
    node_name — the source endpoint the simulator's store-and-forward hop loop
    charges (infrastructure.py, link_contention_v1 block)."""
    with open(ds_dir / "workload.json") as fh:
        workload = json.load(fh)
    sources: List[str] = []
    for event in workload["events"]:
        node_name = event.get("node_name")
        if not node_name:
            raise RuntimeError(
                f"{ds_dir}: workload event without node_name — cannot resolve "
                "ingress routes for the linkrank block")
        # len(dag) is the event's task count for both dag shapes (list of names,
        # or dict task -> parents).
        sources.extend([node_name] * len(event["application"]["dag"]))
    return sources


def load_dag_edges(ds_dir: Path) -> List[Tuple[int, int]]:
    """(parent_task_id, child_task_id) edges, task ids in the same static_order id
    assignment load_task_type_names uses."""
    from graphlib import TopologicalSorter
    with open(ds_dir / "workload.json") as fh:
        workload = json.load(fh)
    edges: List[Tuple[int, int]] = []
    offset = 0
    for event in workload["events"]:
        dag = event["application"]["dag"]
        if isinstance(dag, list):
            offset += len(dag)  # linear-list DAGs carry no explicit edges here
            continue
        order = list(TopologicalSorter(dag).static_order())
        local = {name: offset + i for i, name in enumerate(order)}
        for child, parents in dag.items():
            for parent in parents:
                edges.append((local[parent], local[child]))
        offset += len(order)
    return edges


def load_network(ds_dir: Path) -> Tuple[dict, dict, dict]:
    """(routes, links, network_maps) straight from infrastructure.json."""
    with open(ds_dir / "infrastructure.json") as fh:
        infra = json.load(fh)
    lt = infra.get("link_topology") or {}
    return lt.get("routes") or {}, lt.get("links") or {}, infra.get("network_maps") or {}


def load_platform_map(ds_dir: Path) -> Dict[int, Tuple[str, str]]:
    """platform_id -> (node_name, platform_type), merged over all task types."""
    with open(ds_dir / "infrastructure.json") as fh:
        infra = json.load(fh)
    mapping: Dict[int, Tuple[str, str]] = {}
    for _task_type, replicas in infra["replica_placements"].items():
        for rep in replicas:
            key = int(rep["platform_id"])
            value = (rep["node_name"], rep["platform_type"])
            if key in mapping and mapping[key] != value:
                raise RuntimeError(
                    f"{ds_dir}: platform_id {key} maps to both {mapping[key]} and "
                    f"{value}")
            mapping[key] = value
    if not mapping:
        raise RuntimeError(f"{ds_dir}: empty replica_placements")
    return mapping


class Dataset:
    def __init__(self, ds_dir: Path, task_types_db: Dict[str, dict], objective: str):
        self.ds_dir = ds_dir
        self.rows = load_rows(ds_dir, objective)
        self.task_type_names = load_task_type_names(ds_dir)
        self.platform_map = load_platform_map(ds_dir)
        self.task_types_db = task_types_db
        self.dag_edges = load_dag_edges(ds_dir)
        self.task_sources = load_task_sources(ds_dir)
        self.demand_scales = load_demand_scales(ds_dir)
        self.routes, self.links, self.network_maps = load_network(ds_dir)
        self._route_cache: Dict[Tuple[str, str], Tuple[int, float, float]] = {}
        self._ingress_cache: Dict[Tuple[str, str], List[str]] = {}
        # Per-task demand for every placement that appears in the sweep. demand =
        # demand_scale[task_id] * type-table value; demand_scale defaults to 1.0
        # everywhere a dataset carries no application.demand_scale (route_b env pivot
        # W2), so this is byte-identical to the pre-W2 formula on every existing
        # corpus.
        self.demand: Dict[Tuple[int, Tuple[int, int]], float] = {}
        for plan, _v in self.rows:
            for task_id, placement in plan.items():
                if (task_id, placement) in self.demand:
                    continue
                node_name, ptype = self._resolve(placement)
                ttype = self.task_type_names[task_id]
                mem = task_types_db[ttype].get("memoryRequirements", {})
                if ptype not in mem:
                    raise RuntimeError(
                        f"{ds_dir}: no memoryRequirements[{ttype}][{ptype}] — refusing "
                        "to invent a demand")
                scale = self.demand_scales[task_id]
                self.demand[(task_id, placement)] = scale * float(mem[ptype])

    def _resolve(self, placement: Tuple[int, int]) -> Tuple[str, str]:
        pid = placement[1]
        if pid not in self.platform_map:
            raise RuntimeError(
                f"{self.ds_dir}: plan references platform_id {pid} absent from "
                "replica_placements")
        return self.platform_map[pid]

    def node_of(self, placement: Tuple[int, int]) -> str:
        return self._resolve(placement)[0]

    def node_caps(self, alpha: Optional[float],
                  cap_mode: Union[str, Dict[str, float]] = "alpha_max"
                  ) -> Optional[Dict[str, float]]:
        """cap_node(alpha_max) [DEFAULT, unchanged] = alpha * MAX single demand among
        candidates on that node — the sweep's own max auto-scales the cap with
        whatever demands happen to appear, which is exactly what lets per-instance
        demand heterogeneity (W2) silently widen every cap and erase the scarcity a
        rung was built to create (score_route_b_contention.py:248-257's own note).

        Two independent-tightness alternatives (route_b env pivot W2), both requiring
        `alpha` to be non-None like the default:
          cap_mode="alpha_mean": alpha * MEAN single demand among candidates on that
            node — a fixed reference that does not track the sweep's own extreme.
          cap_mode={"absolute": x}: a flat per-node budget x, alpha ignored (still
            required non-None so the caller's None=unconstrained convention is
            unambiguous — an absolute cap under alpha=None would be a silent behavior
            change for every existing None-alpha caller).

        A node with zero candidates observed carries no entry in ANY mode, preserving
        the existing "absent = uncapped" convention plan_feasible/greedy/t1_cols share.
        """
        if alpha is None:
            return None
        by_node: Dict[str, List[float]] = {}
        for (task_id, placement), demand in self.demand.items():
            node = self.node_of(placement)
            by_node.setdefault(node, []).append(demand)
        # A node whose demands are ALL zero gets no entry at all (uncapped), matching
        # the original alpha_max convention exactly: `if demand > max_demand.get(node,
        # 0.0)` never registers a 0.0-only node, so caps.get(n, inf) treats it as
        # unbounded downstream (t1_cols, plan_feasible, the greedy). Preserved across
        # every cap_mode so switching modes never flips which nodes are capped.
        by_node = {node: ds for node, ds in by_node.items() if max(ds) > 0.0}
        if cap_mode == "alpha_max":
            return {node: alpha * max(ds) for node, ds in by_node.items()}
        if cap_mode == "alpha_mean":
            return {node: alpha * (sum(ds) / len(ds)) for node, ds in by_node.items()}
        if isinstance(cap_mode, dict) and "absolute" in cap_mode:
            budget = float(cap_mode["absolute"])
            return {node: budget for node in by_node}
        raise ValueError(f"{self.ds_dir}: unknown cap_mode {cap_mode!r}")

    def plan_feasible(self, plan: Plan, caps: Optional[Dict[str, float]]) -> bool:
        if caps is None:
            return True
        load: Dict[str, float] = {}
        for task_id, placement in plan.items():
            node = self.node_of(placement)
            load[node] = load.get(node, 0.0) + self.demand[(task_id, placement)]
        return all(total <= caps.get(node, math.inf) + EPS
                   for node, total in load.items())

    def is_spread(self, plan: Plan) -> bool:
        nodes = [self.node_of(p) for p in plan.values()]
        return len(set(nodes)) == len(nodes)

    def route_metrics(self, parent_node: str, child_node: str
                      ) -> Tuple[int, float, float]:
        """(n_hops, bottleneck_bandwidth_mbps, latency) for a parent->child transfer,
        from the dataset's own precomputed routes — the same quantities
        `Platform._dependency_transfer_time` charges (payload uniform, so scale is a
        surrogate coefficient). Same node: (0, inf, 0.0). Missing route/link: fail loud,
        mirroring the simulator's unreachable-parent RuntimeError."""
        if parent_node == child_node:
            return 0, math.inf, 0.0
        key = (parent_node, child_node)
        if key in self._route_cache:
            return self._route_cache[key]
        path = (self.routes.get(parent_node) or {}).get(child_node)
        if not path:
            raise RuntimeError(
                f"{self.ds_dir}: no route {parent_node}->{child_node} in link_topology "
                "— server mesh reachability missing, refusing to invent a distance")
        bandwidths = []
        for a, b in zip(path, path[1:]):
            link = self.links.get("|".join(sorted((a, b))))
            if link is None:
                raise RuntimeError(
                    f"{self.ds_dir}: route {parent_node}->{child_node} uses link "
                    f"{a}|{b} absent from link_topology.links")
            bandwidths.append(float(link["bandwidth_mbps"]))
        entry = (self.network_maps.get(child_node) or {}).get(parent_node)
        if entry is None:
            raise RuntimeError(
                f"{self.ds_dir}: no network_maps[{child_node}][{parent_node}] — the "
                "simulator would raise here too (unreachable parent)")
        latency = (float(entry.get("latency", 0.0)) if isinstance(entry, dict)
                   else float(entry))
        result = (len(path) - 1, min(bandwidths), latency)
        self._route_cache[key] = result
        return result

    def ingress_links(self, src: str, dst: str) -> List[str]:
        """Link keys on the client->destination route — exactly the links the
        simulator's store-and-forward hop loop holds for this task's input
        transmission. Same node: no traversal. No link_topology at all: there are
        structurally no links to contend on, so the answer is the true empty set
        (this is a value, not a silent skip). A missing route on a dataset that
        HAS a fabric fails loud inside route_links, mirroring route_metrics."""
        if src == dst or not self.routes:
            return []
        key = (src, dst)
        cached = self._ingress_cache.get(key)
        if cached is None:
            cached = route_links(self.routes, src, dst)
            self._ingress_cache[key] = cached
        return cached


# ---------------------------------------------------------------------------
# Pointwise baselines
# ---------------------------------------------------------------------------

def min_marginals(rows: Sequence[Tuple[Plan, float]]
                  ) -> Dict[int, Dict[Tuple[int, int], float]]:
    marginal: Dict[int, Dict[Tuple[int, int], float]] = {}
    for plan, value in rows:
        for task_id, placement in plan.items():
            slot = marginal.setdefault(task_id, {})
            if placement not in slot or value < slot[placement]:
                slot[placement] = value
    return marginal


def topological_task_order(ds: Dataset) -> List[int]:
    """DAG topological order over the dataset's tasks, lowest task_id first among
    the ready set (Kahn) — the §4 corrected decode order of masked_topo. On the
    stage-1 corpora this equals ascending task_id (the ids are assigned by
    static_order), and §9c measured the historical (min marginal, task_id) order
    collapsing to the same thing; the helper exists so the order is model- and
    score-independent by construction rather than by measurement."""
    n = len(ds.task_type_names)
    remaining = {t: 0 for t in range(n)}
    children: Dict[int, List[int]] = {}
    for parent, child in ds.dag_edges:
        remaining[child] += 1
        children.setdefault(parent, []).append(child)
    import heapq
    ready = [t for t in range(n) if remaining[t] == 0]
    heapq.heapify(ready)
    order: List[int] = []
    while ready:
        t = heapq.heappop(ready)
        order.append(t)
        for c in children.get(t, ()):
            remaining[c] -= 1
            if remaining[c] == 0:
                heapq.heappush(ready, c)
    if len(order) != n:
        raise RuntimeError(f"{ds.ds_dir}: dependency cycle in dag_edges")
    return order


def greedy_masked_plan(ds: Dataset,
                       marginal: Dict[int, Dict[Tuple[int, int], float]],
                       caps: Optional[Dict[str, float]],
                       order: Optional[List[int]] = None) -> Optional[Plan]:
    """Sequential greedy: ascending best-marginal order, mask replica reuse and any
    placement that would push a node over its remaining capacity. Returns None when no
    feasible completion exists (caller counts it loudly).

    `order` overrides the task loop order (default: the frozen stage-1 ascending
    (min marginal, task_id) order, unchanged). The §4 acceptance check runs it with
    topological_task_order(ds) — measured fact (§9c): on the stage-1 corpora the two
    orders produce identical plans, which is exactly what the check asserts."""
    taken: set = set()
    load: Dict[str, float] = {}
    plan: Plan = {}
    if order is None:
        order = sorted(marginal, key=lambda t: (min(marginal[t].values()), t))
    for task_id in order:
        options = sorted(marginal[task_id].items(), key=lambda kv: (kv[1], kv[0]))
        choice = None
        for placement, _v in options:
            if placement in taken:
                continue
            node = ds.node_of(placement)
            demand = ds.demand[(task_id, placement)]
            cap = math.inf if caps is None else caps.get(node, math.inf)
            if load.get(node, 0.0) + demand > cap + EPS:
                continue
            choice = placement
            break
        if choice is None:
            return None
        plan[task_id] = choice
        taken.add(choice)
        node = ds.node_of(choice)
        load[node] = load.get(node, 0.0) + ds.demand[(task_id, choice)]
    return plan


def additive_argmin_plan(marginal: Dict[int, Dict[Tuple[int, int], float]]) -> Plan:
    """The unconstrained componentwise argmin (free-choice plan)."""
    return {task_id: min(options.items(), key=lambda kv: (kv[1], kv[0]))[0]
            for task_id, options in marginal.items()}


# --- LS surrogate ----------------------------------------------------------

def _design_matrix(plans: List[Plan], task_ids: List[int],
                   vocab: Dict[int, List[Tuple[int, int]]],
                   extra_fn=None) -> np.ndarray:
    index = {t: {p: i for i, p in enumerate(vocab[t])} for t in task_ids}
    offsets: Dict[int, int] = {}
    width = 1
    for t in task_ids:
        offsets[t] = width
        width += len(vocab[t])
    n_extra = 0 if extra_fn is None else len(extra_fn(plans[0]))
    matrix = np.zeros((len(plans), width + n_extra), dtype=float)
    matrix[:, 0] = 1.0
    for row_idx, plan in enumerate(plans):
        for t in task_ids:
            matrix[row_idx, offsets[t] + index[t][plan[t]]] = 1.0
        if extra_fn is not None:
            matrix[row_idx, width:] = extra_fn(plan)
    return matrix


def surrogate_regret(ds: Dataset,
                     fit_rows: Sequence[Tuple[Plan, float]],
                     feasible_rows: Sequence[Tuple[Plan, float]],
                     extra_fn=None) -> Tuple[float, dict]:
    """Regret of the feasible plan minimizing the LS-additive surrogate.

    Fit on fit_rows; exhaustive decode over feasible_rows. Returns (regret_pct, info).
    """
    plans = [p for p, _v in fit_rows]
    y = np.array([v for _p, v in fit_rows], dtype=float)
    task_ids = sorted(plans[0].keys())
    vocab = {t: sorted({p[t] for p in plans}) for t in task_ids}
    matrix = _design_matrix(plans, task_ids, vocab, extra_fn)
    beta, *_ = np.linalg.lstsq(matrix, y, rcond=None)

    feas_plans = [p for p, _v in feasible_rows]
    # A feasible plan can use a placement absent from the fit rows only if fit_rows is a
    # subset — with full-sweep fitting this cannot happen; guard anyway (fail loud).
    for p in feas_plans:
        for t in task_ids:
            if p[t] not in set(vocab[t]):
                raise RuntimeError(
                    f"{ds.ds_dir}: feasible plan uses placement {p[t]} for task {t} "
                    "that the surrogate never saw — fit/decode row mismatch")
    feas_matrix = _design_matrix(feas_plans, task_ids, vocab, extra_fn)
    predicted = feas_matrix @ beta
    pick = int(np.argmin(predicted))
    truths = np.array([v for _p, v in feasible_rows], dtype=float)
    best = float(truths.min())
    if best <= 0:
        raise RuntimeError(f"{ds.ds_dir}: non-positive optimum {best}")
    chosen = float(truths[pick])
    n_params = matrix.shape[1]
    return 100.0 * (chosen - best) / best, {
        "n_fit_rows": len(fit_rows),
        "n_params": n_params,
        "saturated": len(fit_rows) < 2 * n_params,
    }


# --- marginal-sum surrogate (primary) --------------------------------------

def marginal_sum(marginal: Dict[int, Dict[Tuple[int, int], float]],
                 plan: Plan) -> float:
    return sum(marginal[t][p] for t, p in plan.items())


def decode_regret(feasible_rows: Sequence[Tuple[Plan, float]],
                  predicted: Sequence[float],
                  best: float) -> float:
    """Regret (%) of the feasible plan a surrogate's scores rank first.

    The tie-break is part of the registered statistic: equal scores are resolved by the
    plan's sorted (task_id, placement) items, so the decode is deterministic and
    independent of sweep row order. Shared, so that every consumer — the per-dataset
    repairs, the block ablation, the pooled cross-dataset fit — decodes identically
    rather than through a re-typed copy of these four lines.
    """
    order = sorted(range(len(feasible_rows)),
                   key=lambda i: (predicted[i],
                                  tuple(sorted(feasible_rows[i][0].items()))))
    pick = order[0]
    return 100.0 * (float(feasible_rows[pick][1]) - best) / best


def tie_set_indices(predicted: Sequence[float]) -> List[int]:
    """Indices whose scores a surrogate cannot separate from its own minimum.

    ONE definition of "tied" for the whole program. route_b_coefficient_transfer's
    Cell.tie_band uses this, and so does decode_regret_band below — a second tolerance
    rule here would make the scorer's band and the transfer's band incomparable while
    both looked correct.
    """
    lo = float(min(predicted))
    tol = 1e-9 * max(1.0, abs(lo))
    return [i for i, p in enumerate(predicted) if float(p) - lo <= tol]


def decode_regret_band(feasible_rows: Sequence[Tuple[Plan, float]],
                       predicted: Sequence[float],
                       best: float) -> dict:
    """The registered decode plus the band its score ties leave open.

    The registered statistic (`decode_regret`) breaks ties by sorted plan key — platform
    id, a quantity with no relation to cost. On a degenerate feasible set the min-marginal
    surrogate ties the true optimum against strictly worse plans, and that arbitrary rule
    then decides the number. Measured 2026-08-27 on the H0 separable control: of 16 firing
    datasets, 12 had the true optimum INSIDE the argmin tie set, i.e. `optimistic == 0`
    exactly on physics that is provably separable.

    So the docstring claim this module shipped with — "Nonzero constrained R_exact can be
    neither a decoder artifact nor an LS fitting artifact" — was false, and is now
    corrected at the top of this file. Reporting the band is what makes the artifact
    visible instead of letting a tie-break decide a verdict.

    Members follow docs/lineages/route_b_env_pivot_v1/screen-preregistration.md §4 and Cell.tie_band: `mean_tied` is the
    fair reading (a real decoder must pick one tied plan and cannot pick the best by
    oracle, so what it achieves under a fixed uninformative rule is the group MEAN);
    `optimistic` credits the surrogate with plans it cannot distinguish and is an upper
    bound only, never a verdict.
    """
    tied_idx = tie_set_indices(predicted)
    tied = [float(feasible_rows[i][1]) for i in tied_idx]
    pct = lambda v: 100.0 * (v - best) / best  # noqa: E731
    return {
        "registered": decode_regret(feasible_rows, predicted, best),
        "optimistic": pct(min(tied)),
        "pessimistic": pct(max(tied)),
        "mean_tied": pct(sum(tied) / len(tied)),
        "n_tied": len(tied),
    }


def marginal_surrogate_regret(ds: Dataset,
                              marginal: Dict[int, Dict[Tuple[int, int], float]],
                              feasible_rows: Sequence[Tuple[Plan, float]],
                              count_fn=None,
                              fit_rows: Optional[Sequence[Tuple[Plan, float]]] = None,
                              return_beta: bool = False,
                              ):
    """Regret of the feasible plan minimizing sum_t m_t(p_t) (+ fitted count columns).

    Without count_fn no fitting happens at all: the surrogate is the raw marginal sum
    (monotone-equivalent to any a + b*sum with b > 0). With count_fn, coefficients for
    y ~ a + b*marginal_sum + counts are fit by LS on fit_rows (default: the full sweep).

    Saturation guard (the P7 / audit_spread_fit_saturation lesson, and what route B's
    own Control 2 caught at rig scale): a repair fit with fewer than 2x as many rows as
    parameters can interpolate the sweep — coupling included — and mechanically zero
    the regret it exists to measure. Such a repair is refused: returns None, and the
    caller records it as saturated rather than repaired.

    With return_beta the fitted coefficient vector is returned alongside the regret
    ((regret, beta)); beta is None on the no-fit path and on a saturated refusal. The
    default single-value return is unchanged.
    """
    truths = np.array([v for _p, v in feasible_rows], dtype=float)
    best = float(truths.min())
    if best <= 0:
        raise RuntimeError(f"{ds.ds_dir}: non-positive constrained optimum {best}")
    if count_fn is None:
        predicted = [marginal_sum(marginal, p) for p, _v in feasible_rows]
        regret = decode_regret(feasible_rows, predicted, best)
        return (regret, None) if return_beta else regret

    rows = list(fit_rows) if fit_rows is not None else list(ds.rows)
    n_params = 2 + len(count_fn(rows[0][0]))
    if len(rows) < 2 * n_params:
        return (None, None) if return_beta else None
    X = np.array([[1.0, marginal_sum(marginal, p)] + count_fn(p) for p, _v in rows])
    y = np.array([v for _p, v in rows], dtype=float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    Xf = np.array([[1.0, marginal_sum(marginal, p)] + count_fn(p)
                   for p, _v in feasible_rows])
    predicted = Xf @ beta
    regret = decode_regret(feasible_rows, predicted, best)
    return (regret, beta) if return_beta else regret


# --- repair columns --------------------------------------------------------

def one_integer_cols(ds: Dataset):
    """Node-occupancy excess sharing sum(count-1) — the program's established
    collision column (separability_diagnostic._excess_sharing)."""
    def fn(plan: Plan) -> List[float]:
        counts: Dict[str, int] = {}
        for placement in plan.values():
            node = ds.node_of(placement)
            counts[node] = counts.get(node, 0) + 1
        return [float(sum(c - 1 for c in counts.values() if c > 1))]
    return fn


def k_integer_keys(ds: Dataset) -> List[Tuple[str, str]]:
    """The (node, task_type) column vocabulary of the k-integer block, sorted.

    Dataset-dependent by construction — K ranges 8..13 over the route_b_v1 corpus — which
    is why this block cannot carry a coefficient shared across datasets (see
    route_b_coefficient_transfer.py).
    """
    return sorted({(ds.node_of(placement), ds.task_type_names[task_id])
                   for (task_id, placement) in ds.demand})


def k_integer_cols(ds: Dataset):
    """Per-node x per-type co-residency counts — the constraint's own sufficient
    statistic. Column space fixed from the sweep's node/type vocabulary."""
    keys = k_integer_keys(ds)
    key_index = {k: i for i, k in enumerate(keys)}

    def fn(plan: Plan) -> List[float]:
        cols = [0.0] * len(keys)
        for task_id, placement in plan.items():
            cols[key_index[(ds.node_of(placement), ds.task_type_names[task_id])]] += 1.0
        return cols
    return fn


# The T1 column set, split into named blocks. The ORDER of these tuples is the column
# order of t1_cols and must not change: the frozen §9a pre-probe report
# (simulation_data/route_b_stage2_preprobe_t1_rtt.json) was produced by the flat
# concatenation kint + quad + [load_over_cap, overcap_tasks, min_hop_sum, max_hop_sum,
# transfer, latency_sum, same_node_edges], and t1_cols(ds, caps) with the default blocks
# reproduces it exactly (proven byte-identical, see LINEAGES route_b_v1).
#
# T1_REGISTERED_BLOCKS is that frozen §9a layout and stays the DEFAULT of t1_cols /
# t1_column_names, so every registered statistic is unchanged by later block additions.
# `linkrank` (2026-08-26, route-C screen) is a known-but-opt-in extension: fixed-width
# order statistics of per-link co-use over the plan's ingress routes (client -> executing
# node), i.e. the honest pointwise competitor for a link-contention environment. It is
# identity-free by construction (counts, not link names), so it pools across datasets.
#
# `hetdem` and `futureint` (route_b env pivot, 2026-08-27) are opt-in extensions for the
# per-instance-demand-heterogeneous environment (W1 of the pivot plan): `hetdem` renders
# the existing count blocks' demand-weighted sufficient statistics (real per-instance
# demand instead of unit counts) and `futureint` renders the (dataset, prefix)-expressible
# future-interaction quantities under the fixed topological decode order. Neither is in
# T1_REGISTERED_BLOCKS, T1_BLOCKS (unchanged — the frozen §9b/§9c "every known block"
# superset stays exactly kint+quad+cap+hop+coupling+linkrank so `t1lnk` and the block-
# partition fixture stay byte-identical) or the default of t1_cols/t1_column_names;
# T1_EXTENDED_BLOCKS is the new pivot superset that adds them.
T1_REGISTERED_BLOCKS: Tuple[str, ...] = ("kint", "quad", "cap", "hop", "coupling")
T1_BLOCKS: Tuple[str, ...] = T1_REGISTERED_BLOCKS + ("linkrank",)
T1_EXTENDED_BLOCKS: Tuple[str, ...] = T1_BLOCKS + ("hetdem", "futureint")


def _check_blocks(blocks: Sequence[str]) -> None:
    unknown = [b for b in blocks if b not in T1_EXTENDED_BLOCKS]
    if unknown:
        raise ValueError(
            f"unknown T1 block(s) {unknown}; known blocks {list(T1_EXTENDED_BLOCKS)}")
    if not blocks:
        raise ValueError("empty T1 block set — a repair with no columns is just the "
                         "unrepaired marginal surrogate; ask for that explicitly")


def t1_column_names(ds: Dataset, blocks: Sequence[str] = T1_REGISTERED_BLOCKS) -> List[str]:
    """Column labels for t1_cols(ds, caps, blocks), in the same order as the values.

    Nothing in the pipeline persisted a fitted coefficient before 2026-08-25, so the
    question "which column carried the closure" was unanswerable from the frozen
    artifacts. These names are what make a reported coefficient vector readable.
    """
    _check_blocks(blocks)
    names: List[str] = []
    for block in T1_EXTENDED_BLOCKS:
        if block not in blocks:
            continue
        if block == "kint":
            names.extend(f"kint[{node}|{ttype}]" for node, ttype in k_integer_keys(ds))
        elif block == "quad":
            names.extend(f"quad[{k}]" for k in sorted(set(ds.task_type_names)))
        elif block == "cap":
            names.extend(["load_over_cap", "overcap_tasks"])
        elif block == "hop":
            names.extend(["min_hop_sum", "max_hop_sum"])
        elif block == "coupling":
            names.extend(["transfer", "latency_sum", "same_node_edges"])
        elif block == "linkrank":
            names.extend(["link_couse_top1", "link_couse_top2", "link_couse_top3",
                          "link_couse_top4", "link_excess", "link_excess_core",
                          "link_shared_links", "link_shared_core_links"])
        elif block == "hetdem":
            names.extend(f"hd_quad[{k}]" for k in sorted(set(ds.task_type_names)))
            names.extend(["hd_load_over_cap", "hd_overcap_load",
                          "hd_excess_share", "hd_node_load_l2"])
        elif block == "futureint":
            names.extend(["future_demand_interaction", "future_count_interaction",
                          "future_overcap_pressure", "future_max_single_interaction"])
    return names


def t1_cols(ds: Dataset, caps: Dict[str, float], blocks: Sequence[str] = T1_REGISTERED_BLOCKS):
    """Plan-level sums of the stage-2 T1 (partial-state) per-edge features — the §9a
    pre-probe-zero column set of docs/lineages/route_b_v1/stage2-preregistration.md. T1 ⊇ kint, plus:
    per-type quadratic co-residency Σ_t occ_{node(t)}[k]; Σ_t load/cap; over-cap task
    count; per-task min/max parent-hop sums; and the coupling-term columns
    Σ_edges hops/bottleneck, Σ_edges latency, same-node-parent count — computed from the
    dataset's own routes, i.e. exactly what `_dependency_transfer_time` charges (uniform
    payload absorbed by the LS coefficient). Requires a constrained rung (caps), because
    the capacity-normalized columns are undefined at alpha=inf.

    `blocks` selects a SUBSET of T1_BLOCKS, always emitted in T1_BLOCKS order. The
    default is T1_REGISTERED_BLOCKS, the registered §9a column set; subsets exist for the
    block-attribution ablation (which columns actually close the effect) and for the
    pooled cross-dataset fit, which cannot carry the dataset-specific kint vocabulary.
    The opt-in `linkrank` block (ingress-route per-link co-use order statistics) is
    NOT in the default — asking for it is a deliberate act, so no registered statistic
    silently changes meaning. Also opt-in, from the route_b env pivot (2026-08-27):
    `hetdem` (demand-weighted analogs of quad/cap/1int, real per-instance demand
    instead of unit counts — see the fixture proving it is redundant with quad+cap
    on a uniform-demand rig) and `futureint` (per-step candidate-node x not-yet-
    committed-task-demand interaction under the fixed topological decode order — the
    honest pointwise-expressible lookahead, never the future tasks' actual placement).
    """
    _check_blocks(blocks)
    want = {block: (block in blocks) for block in T1_EXTENDED_BLOCKS}
    kint = k_integer_cols(ds) if want["kint"] else None
    type_order = sorted(set(ds.task_type_names))
    parents_of: Dict[int, List[int]] = {}
    for parent, child in ds.dag_edges:
        parents_of.setdefault(child, []).append(parent)
    need_parents = want["hop"] or want["coupling"]

    # futureint precompute: per (task, node) the MINIMUM demand that task would carry
    # if placed on that node, over its own candidate set only — a static per-task
    # eligibility fact, independent of any plan. Node set is every node hosting ANY
    # candidate in the sweep (ds.demand's own node vocabulary), so a task ineligible
    # on a node simply has no entry (treated as demand 0 / not-eligible below).
    if want["futureint"]:
        decode_order = topological_task_order(ds)
        step_of = {t: i for i, t in enumerate(decode_order)}
        task_node_min_demand: Dict[int, Dict[str, float]] = {}
        for (task_id, placement), d in ds.demand.items():
            node = ds.node_of(placement)
            slot = task_node_min_demand.setdefault(task_id, {})
            if node not in slot or d < slot[node]:
                slot[node] = d

    def fn(plan: Plan) -> List[float]:
        occ: Dict[str, Dict[str, int]] = {}
        tot: Dict[str, int] = {}
        load: Dict[str, float] = {}
        for task_id, placement in plan.items():
            node = ds.node_of(placement)
            ttype = ds.task_type_names[task_id]
            occ.setdefault(node, {})[ttype] = occ.setdefault(node, {}).get(ttype, 0) + 1
            tot[node] = tot.get(node, 0) + 1
            load[node] = load.get(node, 0.0) + ds.demand[(task_id, placement)]
        cols: List[float] = kint(plan) if want["kint"] else []
        if want["quad"]:
            cols += [float(sum(tot[n] * occ[n].get(k, 0) for n in occ))
                     for k in type_order]
        if want["cap"]:
            # A node whose max single demand is 0 has no cap entry and is uncapped —
            # caps.get(n, inf), the same convention plan_feasible and the greedy use;
            # load/inf contributes 0.0 and such a node can never be over cap.
            cols += [sum(tot[n] * load[n] / caps.get(n, math.inf) for n in occ),
                     float(sum(tot[n] for n in occ
                               if load[n] > caps.get(n, math.inf) + EPS))]
        if need_parents:
            min_hop_sum = max_hop_sum = transfer = latency_sum = 0.0
            same_node_edges = 0.0
            for task_id, placement in plan.items():
                parents = parents_of.get(task_id)
                if not parents:
                    continue
                child_node = ds.node_of(placement)
                hops_here = []
                for parent in parents:
                    n_hops, bottleneck, latency = ds.route_metrics(
                        ds.node_of(plan[parent]), child_node)
                    hops_here.append(n_hops)
                    if n_hops == 0:
                        same_node_edges += 1.0
                    else:
                        transfer += n_hops / bottleneck
                        latency_sum += latency
                min_hop_sum += min(hops_here)
                max_hop_sum += max(hops_here)
            if want["hop"]:
                cols += [min_hop_sum, max_hop_sum]
            if want["coupling"]:
                cols += [transfer, latency_sum, same_node_edges]
        if want["linkrank"]:
            # Per-link co-use over the plan's ingress routes (client -> executing
            # node) — the exact links the store-and-forward hop loop serializes on.
            # Emitted as fixed-width order statistics, never link identities: the
            # block must pool across datasets and stay far from the saturation
            # guard (38 links vs 576 rows would trip 2*n_params).
            couse: Dict[str, int] = {}
            for task_id, placement in plan.items():
                for lk in ds.ingress_links(
                        ds.task_sources[task_id], ds.node_of(placement)):
                    couse[lk] = couse.get(lk, 0) + 1
            top = sorted(couse.values(), reverse=True)[:4]
            top += [0] * (4 - len(top))
            cols += [float(v) for v in top]
            cols += [
                float(sum(c - 1 for c in couse.values() if c > 1)),
                float(sum(c - 1 for lk, c in couse.items()
                          if c > 1 and is_core_link(lk))),
                float(sum(1 for c in couse.values() if c >= 2)),
                float(sum(1 for lk, c in couse.items()
                          if c >= 2 and is_core_link(lk))),
            ]
        if want["hetdem"]:
            # Demand-weighted analogs of the quad/cap count columns, real per-instance
            # demand in place of unit counts. Under UNIFORM demand d (every candidate
            # placement demand == d, load[n] = d*tot[n]), every column here is a fixed
            # multiple of an existing count-block column, hence gains NO new closure
            # over quad+cap on a uniform-demand rig:
            #   hd_quad[k]        = d * quad[k]
            #   hd_load_over_cap  = d * load_over_cap      (load[n]^2/cap = d*tot[n]*load[n]/cap)
            #   hd_overcap_load   = d * overcap_tasks       (indicator load[n]>cap[n] is
            #                                                 identical to tot[n]-weighted; the
            #                                                 indicator itself doesn't depend on
            #                                                 which of load/tot triggered it)
            #   hd_excess_share   = d * one_integer_cols     (sum(tot[n]-1) for co-resident nodes)
            #   hd_node_load_l2   = d^2 * sum_k quad[k]      (sum_n tot[n]^2 = sum_k quad[k]
            #                                                 identically, any plan)
            # Verified exactly: test_route_b_env_pivot_fixtures.py.
            cols += [float(sum(load[n] * occ[n].get(k, 0) for n in occ))
                     for k in type_order]
            cap_of = lambda n: caps.get(n, math.inf)  # noqa: E731
            cols.append(sum(load[n] * load[n] / cap_of(n) for n in occ))
            cols.append(float(sum(load[n] for n in occ if load[n] > cap_of(n) + EPS)))
            excess_share = 0.0
            for n in occ:
                if tot[n] > 1:
                    min_single = min(ds.demand[(t, p)] for t, p in plan.items()
                                     if ds.node_of(p) == n)
                    excess_share += load[n] - min_single
            cols.append(excess_share)
            cols.append(float(sum(load[n] * load[n] for n in occ)))
        if want["futureint"]:
            # Future-interaction columns: per decode step, the CHOSEN node's static
            # interaction with the not-yet-committed tasks' eligibility+demand — never
            # where those future tasks actually land (that would be leaking the label).
            # Plain aggregate future demand alone is CONSTANT across the sweep (every
            # plan commits the same task set in the same order) and is absorbed by the
            # intercept; only the candidate-node x future-demand INTERACTION varies
            # with the plan, which is what makes these columns non-vacuous.
            fdi = fci = fop = fmax = 0.0
            for step, task_id in enumerate(decode_order):
                node = ds.node_of(plan[task_id])
                future_tasks = decode_order[step + 1:]
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
                    ds.demand[(t2, plan[t2])]
                    for t2 in decode_order[:step + 1] if ds.node_of(plan[t2]) == node)
                fop += max(0.0, load_so_far + future_demand_here - cap)
            cols += [fdi, fci, fop, fmax]
        return cols
    return fn


# ---------------------------------------------------------------------------
# Per-dataset scoring
# ---------------------------------------------------------------------------

def score_dataset(ds: Dataset, alpha: Optional[float],
                  cap_mode: Union[str, Dict[str, float]] = "alpha_max") -> dict:
    caps = ds.node_caps(alpha, cap_mode=cap_mode)
    feasible_rows = [(p, v) for p, v in ds.rows if ds.plan_feasible(p, caps)]
    out: Dict[str, Any] = {
        "alpha": alpha,
        "n_rows": len(ds.rows),
        "n_feasible": len(feasible_rows),
    }
    if not feasible_rows:
        out["no_feasible_rows"] = True
        return out

    truths = [v for _p, v in feasible_rows]
    best = min(truths)
    if best <= 0:
        raise RuntimeError(f"{ds.ds_dir}: non-positive constrained optimum {best}")
    lookup = {tuple(sorted(p.items())): v for p, v in ds.rows}

    marginal = min_marginals(ds.rows)  # full-sweep marginals (registered choice)

    # Is the free-choice plan still feasible? (the constraint's bite)
    free_plan = additive_argmin_plan(marginal)
    free_key = tuple(sorted(free_plan.items()))
    out["componentwise_plan_enumerated"] = free_key in lookup
    out["componentwise_plan_feasible"] = (
        free_key in lookup and ds.plan_feasible(free_plan, caps))

    # R_greedy
    gplan = greedy_masked_plan(ds, marginal, caps)
    if gplan is None:
        out["greedy_stuck"] = True
    else:
        gkey = tuple(sorted(gplan.items()))
        if gkey not in lookup:
            raise RuntimeError(
                f"{ds.ds_dir}: greedy plan {gkey} absent from the enumerated sweep — "
                "the sweep is not the full unique-replica enumeration or the mask "
                "disagrees with the enumerator")
        out["r_greedy_pct"] = 100.0 * (lookup[gkey] - best) / best

    # R_exact (primary: min-marginal-sum surrogate, perfect decode) + count repairs.
    # A repaired surrogate that scores WORSE than the base would simply not be used by
    # the pointwise side, so the repaired regret is min(base, repaired).
    r_base = marginal_surrogate_regret(ds, marginal, feasible_rows)
    out["r_exact_pct"] = r_base
    # The band the surrogate's own score ties leave open (AMENDMENT-era addition,
    # 2026-08-27). Purely additive: `registered` is the same number `r_exact_pct` already
    # carries — asserted below, and pinned by test_control3_* — so no consumer that keys
    # off r_exact_pct changes. §4 already mandates a band for S2/S3 and eb6e93a added one
    # for t1x; R_exact simply never got one, which is how a decoder tie-break came to be
    # read as physics on the H0 control.
    r_exact_predicted = [marginal_sum(marginal, p) for p, _v in feasible_rows]
    out["r_exact_band"] = decode_regret_band(feasible_rows, r_exact_predicted, best)
    if out["r_exact_band"]["registered"] != r_base:
        raise RuntimeError(
            f"{ds.ds_dir}: r_exact band 'registered' "
            f"{out['r_exact_band']['registered']!r} != r_exact_pct {r_base!r}; the band "
            "must reproduce the registered statistic exactly, not approximate it")
    repair_sets = [("1int", one_integer_cols(ds)), ("kint", k_integer_cols(ds))]
    # route-C screen arm (2026-08-26): ingress-route link co-use alone — the honest
    # pointwise competitor for a link-contention environment. Needs no caps (link
    # co-use is cap-free), so it runs on the unconstrained row too, where any nonzero
    # R_exact is attributable to the fabric alone (memory constraint absent).
    # Diagnostic, not a registered §9a statistic.
    repair_sets.append(("lnk", t1_cols(ds, caps or {}, blocks=("linkrank",))))
    if caps is not None:
        # Stage-2 §9a pre-probe zero: the T1 (partial-state) column set. Constrained
        # rungs only — the capacity-normalized columns are undefined at alpha=inf.
        repair_sets.append(("t1", t1_cols(ds, caps)))
        repair_sets.append(("t1lnk", t1_cols(ds, caps, blocks=T1_BLOCKS)))
        # route_b env pivot (2026-08-27, W1): the extended honest pointwise
        # competitor. t1hd = registered T1 + demand-weighted sufficient statistics;
        # t1x = t1hd + futureint + linkrank, the full extended-T1 arm the pivot
        # screen reads. Both opt-in additions to the existing arm set — no existing
        # key's value changes.
        repair_sets.append(("t1hd", t1_cols(
            ds, caps, blocks=T1_REGISTERED_BLOCKS + ("hetdem",))))
        repair_sets.append(("t1x", t1_cols(ds, caps, blocks=T1_EXTENDED_BLOCKS)))
    for name, cols in repair_sets:
        repaired = marginal_surrogate_regret(ds, marginal, feasible_rows, cols)
        if repaired is None:
            out[f"repair_{name}_saturated"] = True
            out[f"r_exact_repaired_{name}_pct"] = None
        else:
            out[f"r_exact_repaired_{name}_pct"] = min(r_base, repaired)

    # sensitivity row: full per-(task,placement) indicator LS surrogate
    r_ls, fit_info = surrogate_regret(ds, ds.rows, feasible_rows)
    out["r_exact_ls_pct"] = r_ls
    out["fit"] = fit_info

    # spread view (memory-feasible AND all-distinct-node)
    spread_rows = [(p, v) for p, v in feasible_rows if ds.is_spread(p)]
    out["n_spread_feasible"] = len(spread_rows)
    if spread_rows:
        out["r_exact_spread_pct"] = marginal_surrogate_regret(
            ds, marginal, spread_rows)
    return out


# ---------------------------------------------------------------------------
# Corpus aggregation
# ---------------------------------------------------------------------------

def summarize(values: List[float]) -> dict:
    if not values:
        return {"n": 0}
    arr = sorted(values)
    return {
        "n": len(arr),
        "mean": sum(arr) / len(arr),
        "median": arr[len(arr) // 2],
        "max": arr[-1],
        "nonzero_frac": sum(1 for v in arr if v > 1e-9) / len(arr),
        "frac_gt_1pct": sum(1 for v in arr if v > 1.0) / len(arr),
        "frac_gt_5pct": sum(1 for v in arr if v > 5.0) / len(arr),
    }


def score_corpus(corpus: Path, task_types_db: Dict[str, dict], objective: str,
                 alphas: List[Optional[float]], limit: Optional[int] = None,
                 cap_mode: Union[str, Dict[str, float]] = "alpha_max") -> dict:
    ds_dirs = sorted(d for d in corpus.glob("ds_*") if d.is_dir())
    if limit:
        ds_dirs = ds_dirs[:limit]
    if not ds_dirs:
        raise RuntimeError(f"{corpus}: no ds_* directories")

    per_alpha: Dict[str, dict] = {}
    all_results: Dict[str, List[dict]] = {str(a): [] for a in alphas}
    failures: List[str] = []
    for ds_dir in ds_dirs:
        try:
            ds = Dataset(ds_dir, task_types_db, objective)
        except RuntimeError as exc:
            failures.append(f"{ds_dir.name}: {exc}")
            raise
        for alpha in alphas:
            all_results[str(alpha)].append(score_dataset(ds, alpha, cap_mode=cap_mode))

    for alpha in alphas:
        results = all_results[str(alpha)]
        no_feasible = sum(1 for r in results if r.get("no_feasible_rows"))
        stuck = sum(1 for r in results if r.get("greedy_stuck"))

        # Three denominators, each named for what legitimately censors it (2026-08-27).
        # `no_feasible_rows` censors everything: with no feasible plan there is no
        # constrained optimum to be regretful against. `greedy_stuck` censors ONLY the
        # greedy statistic — R_exact is a perfect-decoder quantity and the greedy decoder
        # has nothing to do with it, so a greedy dead-end must not delete a valid R_exact.
        #
        # This was not a neutral censoring. On the H0 control the (n_feasible, stuck)
        # histogram is exactly {(9, False): 102, (16, True): 101, (16, False): 1} — stuck
        # is PERFECTLY confounded with the replica-config arm, so one whole cell of the
        # 2x2x3x17 design was silently dropped from every r_exact statistic and the
        # reported frac_gt_1pct was really "over the 9-feasible-row arm only".
        feasible_scored = [r for r in results if not r.get("no_feasible_rows")]
        greedy_scored = [r for r in feasible_scored if not r.get("greedy_stuck")]
        exact_scored = feasible_scored
        # Retained so the pre-fix number is recoverable from the SAME artifact rather than
        # asserted from a commit message — see legacy_greedy_censored below.
        scored = greedy_scored

        stuck_by_n_feasible: Dict[str, int] = {}
        for r in results:
            if r.get("greedy_stuck"):
                key = str(r.get("n_feasible"))
                stuck_by_n_feasible[key] = stuck_by_n_feasible.get(key, 0) + 1

        # Both censors, keyed on the arm rather than on a post-cap quantity (2026-08-27).
        # `stuck_by_n_feasible` above cannot be reused for `no_feasible_rows`: n_feasible
        # is 0 by construction on a censored dataset, so it collapses every arm into one
        # bucket. `n_rows` — the UNCONSTRAINED sweep size — is the replica-config arm's
        # signature (16 vs 64 on the 2x2x3x17 grid) and survives censoring.
        #
        # This exists because `no_feasible_rows` is the STRICTER censor: it removes a
        # dataset from r_exact and every LS/repair statistic, not just from r_greedy, and
        # it had no arm breakdown at all while greedy_stuck did. Measured on H1 at its
        # registered primary alpha=2.0: all 70 censored datasets sit in the 64-row arm and
        # none in the 16-row arm, so r_exact there is 102 of 102 on one arm and 32 of 102
        # on the other — the same "over one arm only" defect as the greedy denominator,
        # one counter over.
        def _by_arm(pred) -> Dict[str, int]:
            hist: Dict[str, int] = {}
            for r in results:
                if pred(r):
                    key = str(r.get("n_rows"))
                    hist[key] = hist.get(key, 0) + 1
            return hist

        per_alpha[str(alpha)] = {
            "alpha": alpha,
            "n_datasets": len(results),
            "no_feasible_rows": no_feasible,
            "greedy_stuck": stuck,
            "componentwise_infeasible_frac": (
                sum(1 for r in results
                    if not r.get("no_feasible_rows")
                    and not r.get("componentwise_plan_feasible", True))
                / max(1, len(results) - no_feasible)),
            "mean_feasible_rows": (
                sum(r["n_feasible"] for r in results) / len(results)),
            "saturated_fit_frac": (
                sum(1 for r in exact_scored if r.get("fit", {}).get("saturated"))
                / max(1, len(exact_scored))),
            "r_greedy": summarize([r["r_greedy_pct"] for r in greedy_scored
                                   if "r_greedy_pct" in r]),
            "r_exact": summarize([r["r_exact_pct"] for r in exact_scored]),
            "repair_1int_saturated": sum(
                1 for r in exact_scored if r.get("repair_1int_saturated")),
            "repair_kint_saturated": sum(
                1 for r in exact_scored if r.get("repair_kint_saturated")),
            "r_exact_repaired_1int": summarize(
                [r["r_exact_repaired_1int_pct"] for r in exact_scored
                 if r.get("r_exact_repaired_1int_pct") is not None]),
            "r_exact_repaired_kint": summarize(
                [r["r_exact_repaired_kint_pct"] for r in exact_scored
                 if r.get("r_exact_repaired_kint_pct") is not None]),
            "repair_t1_saturated": sum(
                1 for r in exact_scored if r.get("repair_t1_saturated")),
            "r_exact_repaired_t1": summarize(
                [r["r_exact_repaired_t1_pct"] for r in exact_scored
                 if r.get("r_exact_repaired_t1_pct") is not None]),
            "repair_lnk_saturated": sum(
                1 for r in exact_scored if r.get("repair_lnk_saturated")),
            "r_exact_repaired_lnk": summarize(
                [r["r_exact_repaired_lnk_pct"] for r in exact_scored
                 if r.get("r_exact_repaired_lnk_pct") is not None]),
            "repair_t1lnk_saturated": sum(
                1 for r in exact_scored if r.get("repair_t1lnk_saturated")),
            "r_exact_repaired_t1lnk": summarize(
                [r["r_exact_repaired_t1lnk_pct"] for r in exact_scored
                 if r.get("r_exact_repaired_t1lnk_pct") is not None]),
            "r_exact_ls": summarize(
                [r["r_exact_ls_pct"] for r in exact_scored]),
            "r_exact_spread": summarize(
                [r["r_exact_spread_pct"] for r in exact_scored
                 if "r_exact_spread_pct" in r]),
            # Denominators, stated rather than inferred.
            "n_exact_scored": len(exact_scored),
            "n_greedy_scored": len(greedy_scored),
            # The band R_exact's score ties leave open, per band member.
            # docs/lineages/route_b_env_pivot_v1/screen-preregistration.md §4: mean_tied is the fair reading; optimistic is
            # an upper bound only, never a verdict.
            "r_exact_band": {
                member: summarize([r["r_exact_band"][member] for r in exact_scored
                                   if "r_exact_band" in r])
                for member in ("registered", "optimistic", "pessimistic", "mean_tied")
            },
            "r_exact_n_tied": {
                "datasets_with_ties": sum(
                    1 for r in exact_scored
                    if r.get("r_exact_band", {}).get("n_tied", 1) > 1),
                "max_tie_group": max(
                    [r.get("r_exact_band", {}).get("n_tied", 0) for r in exact_scored],
                    default=0),
            },
            # greedy_stuck's confound with the design, made self-evident in every rung's
            # artifact instead of requiring a bespoke investigation to rediscover.
            "greedy_stuck_by_n_feasible": stuck_by_n_feasible,
            # Both censors and the surviving denominator, per ARM (unconstrained sweep
            # size). Read these before quoting any statistic: a denominator concentrated
            # in one arm makes the statistic "over that arm only", whatever its name.
            "censoring_by_arm": {
                "key": "n_rows (unconstrained sweep size = replica-config arm)",
                "n_datasets": _by_arm(lambda r: True),
                "no_feasible_rows": _by_arm(lambda r: bool(r.get("no_feasible_rows"))),
                "greedy_stuck": _by_arm(lambda r: bool(r.get("greedy_stuck"))),
                "n_exact_scored": _by_arm(lambda r: not r.get("no_feasible_rows")),
                "n_greedy_scored": _by_arm(
                    lambda r: not r.get("no_feasible_rows") and not r.get("greedy_stuck")),
            },
            # The PRE-FIX numbers, reproduced from this same run so a deviation can be
            # audited against one artifact rather than a commit message. This is the
            # greedy-censored denominator R_exact used to be summarized over.
            "legacy_greedy_censored": {
                "note": "pre-2026-08-27 denominator: r_exact censored by greedy_stuck. "
                        "Retained for audit; not a statistic to read.",
                "n": len(greedy_scored),
                "r_exact": summarize([r["r_exact_pct"] for r in greedy_scored]),
                "r_exact_band": {
                    member: summarize([r["r_exact_band"][member] for r in greedy_scored
                                       if "r_exact_band" in r])
                    for member in ("registered", "optimistic", "pessimistic", "mean_tied")
                },
            },
        }
    return {
        "corpus": str(corpus),
        "objective": objective,
        "cap_mode": cap_mode,
        "n_datasets": len(ds_dirs),
        "failures": failures,
        "per_alpha": per_alpha,
        "per_dataset": all_results,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", action="append", required=True)
    ap.add_argument("--objective", choices=["rtt", "makespan"], default="rtt")
    ap.add_argument("--task-types", default="data/nofs-ids/task-types.json")
    ap.add_argument("--alphas", default="1.0,1.5,2.0,2.5,3.0",
                    help="comma-separated capacity multipliers; 'inf' allowed")
    ap.add_argument("--limit", type=int, default=None,
                    help="score only the first N datasets (smoke runs)")
    ap.add_argument("--out", default=None, help="write the frozen report JSON here")
    ap.add_argument("--include-per-dataset", action="store_true",
                    help="keep per-dataset rows in the report (large)")
    ap.add_argument("--cap-mode", default="alpha_max",
                    help="route_b env pivot W2: 'alpha_max' (default, unchanged) | "
                         "'alpha_mean' | an absolute per-node budget as a bare number "
                         "(interpreted as {'absolute': x}). See Dataset.node_caps.")
    args = ap.parse_args()

    task_types_db = load_task_types(Path(args.task_types))
    alphas: List[Optional[float]] = []
    for tok in args.alphas.split(","):
        tok = tok.strip()
        alphas.append(None if tok in ("inf", "none") else float(tok))
    if None not in alphas:
        alphas.append(None)  # the unconstrained arm is always scored (sanity anchor)

    cap_mode: Union[str, Dict[str, float]]
    if args.cap_mode in ("alpha_max", "alpha_mean"):
        cap_mode = args.cap_mode
    else:
        try:
            cap_mode = {"absolute": float(args.cap_mode)}
        except ValueError:
            raise SystemExit(
                f"--cap-mode: unrecognised value {args.cap_mode!r}; expected "
                "'alpha_max', 'alpha_mean', or a number")

    reports = []
    for corpus in args.corpus:
        rep = score_corpus(Path(corpus), task_types_db, args.objective, alphas,
                           args.limit, cap_mode=cap_mode)
        reports.append(rep)
        print(f"\n=== {corpus}  ({rep['n_datasets']} datasets, {args.objective}) ===")
        header = (f"{'alpha':>6} {'feas_rows':>9} {'cw_infeas':>9} {'stuck':>5} "
                  f"{'nofeas':>6} {'sat':>5} | {'Rg>1%':>6} {'Rg max':>8} | "
                  f"{'Rx>1%':>6} {'Rx max':>8} {'Rx1i>1%':>7} {'Rxki>1%':>7} "
                  f"{'Rxt1>1%':>7} {'Rxt1>5%':>7} {'Rxlk>1%':>7} {'Rxtl>1%':>7}")
        print(header)
        for key, s in rep["per_alpha"].items():
            rg, rx = s["r_greedy"], s["r_exact"]
            r1, rk = s["r_exact_repaired_1int"], s["r_exact_repaired_kint"]
            rt = s["r_exact_repaired_t1"]
            rl = s["r_exact_repaired_lnk"]
            rtl = s["r_exact_repaired_t1lnk"]
            print(f"{key:>6} {s['mean_feasible_rows']:>9.1f} "
                  f"{s['componentwise_infeasible_frac']:>9.2f} "
                  f"{s['greedy_stuck']:>5d} {s['no_feasible_rows']:>6d} "
                  f"{s['saturated_fit_frac']:>5.2f} | "
                  f"{rg.get('frac_gt_1pct', float('nan')):>6.2f} "
                  f"{rg.get('max', float('nan')):>8.2f} | "
                  f"{rx.get('frac_gt_1pct', float('nan')):>6.2f} "
                  f"{rx.get('max', float('nan')):>8.2f} "
                  f"{r1.get('frac_gt_1pct', float('nan')):>7.2f} "
                  f"{rk.get('frac_gt_1pct', float('nan')):>7.2f} "
                  f"{rt.get('frac_gt_1pct', float('nan')):>7.2f} "
                  f"{rt.get('frac_gt_5pct', float('nan')):>7.2f} "
                  f"{rl.get('frac_gt_1pct', float('nan')):>7.2f} "
                  f"{rtl.get('frac_gt_1pct', float('nan')):>7.2f}")

    if args.out:
        payload = reports
        if not args.include_per_dataset:
            payload = [{k: v for k, v in rep.items() if k != "per_dataset"}
                       for rep in reports]
        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=1, sort_keys=True)
        print(f"\nreport written: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
