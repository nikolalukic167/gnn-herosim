"""plan_rerank_v1 Phase 1, serving: the learned plan selector as a live arm (GNN_PLAN_SELECTOR=<selector .pt>).

Held by the GNN scheduler. After the no-split masked_topo decode of a declared-slate batch, the batch's pool is built on this very
state exactly as the training pool was built offline:
  * the step-0 branch completions of the base GNN's decode (src/policy/plan_rerank/pool.py), each with its summed step logits;
  * every argmin-S plan over the declared slate's product (S = the CD rule's whole-plan cost, `_pg_plan_cost`, ties within 1e-12);
  * per candidate: S and its components (src/policy/plan_rerank/s_components.py, checked against `_pg_plan_cost`), the base GNN's
    plan score.
The row is scored by the selector (src/policy/plan_rerank/selector.py) and the selected plan is served. A batch whose slate holds more
than PRSEL_MAX_PLANS plans has no argmin-S pool: it is served cd_exactS's own plan (`GnnCdRefiner._cd_native_plan`: greedy, passes,
exact search with the expansion as its over-cap fallback, passes) and counted. That plan is also computed for every pooled batch,
for counting only (how often the selector, and the slate's argmin-S, differ from cd_exactS).

PRSEL_DIAG_DIR (not a HEROSIM_ name: the gate scrubs those) appends every pooled batch's row, costs and choice to
<dir>/prsel_<pid>.jsonl. Changes no decision.
"""
from __future__ import annotations

import json
import os
from itertools import product
from pathlib import Path
from timeit import default_timer
from typing import Any, Dict, List, Tuple

import torch

PLAN_SELECTOR_ENV = "GNN_PLAN_SELECTOR"
ALLOW_SMOKE_ENV = "GNN_PLAN_SELECTOR_SMOKE"
MAX_PLANS_ENV = "PRSEL_MAX_PLANS"
DIAG_DIR_ENV = "PRSEL_DIAG_DIR"
S_TIE_EPS = 1e-12  # fidelity_cd_plans --argmin-s's tie tolerance, the training pool's
COUNTERS = ("prsel_batches", "prsel_tasks", "prsel_pool_plans", "prsel_branches", "prsel_argmin_s_ties", "prsel_changed_from_decode",
            "prsel_chose_argmin_s", "prsel_chose_branch_only", "prsel_differs_native", "prsel_argmin_s_differs_native",
            "prsel_fallback_batches", "prsel_fallback_tasks", "prsel_order_not_id", "prsel_seconds")


def selector_path() -> str:
    return os.environ.get(PLAN_SELECTOR_ENV, "").strip()


class LivePlanSelector:
    def __init__(self, host) -> None:
        from src.policy.peer_greedy_network.scheduler import GnnCdRefiner
        from src.policy.plan_rerank.selector import load_selector

        for name, want in (("HEROSIM_PG_CD_EXACT", "1"), ("HEROSIM_PG_CD_EXPANSION", "1"), ("GNN_SLATE_NO_SPLIT", "1")):
            if os.environ.get(name, "").strip() != want:
                raise RuntimeError(f"FAIL LOUD: {PLAN_SELECTOR_ENV} needs {name}={want} (the fallback is cd_exactS's plan; the pool is the "
                                   "no-split decode's)")
        for name in ("GNN_CD_REFINE", "GNN_SELF_SEARCH", "GNN_PREFIX_SELF_REFINE", "GNN_PREFIX_SIBLING_SPREAD", "GNN_REPLICA_KEEPWARM",
                     "HEROSIM_PG_CD_EXACT_GNN", "HEROSIM_PG_CD_EXACT_PRUNE", "HEROSIM_PG_ORACLE"):
            if os.environ.get(name, "").strip() not in ("", "0"):
                raise RuntimeError(f"FAIL LOUD: {PLAN_SELECTOR_ENV} serves the selected plan as is; {name} would change it")
        from src.placement import declared_slate

        if declared_slate.serving_slate() is None:
            raise RuntimeError(f"FAIL LOUD: {PLAN_SELECTOR_ENV} needs the declared slate (GNN_SERVE_CANDIDATE_SLATE=declared_pruning_v1)")
        base = os.environ.get("GNN_MODEL_PATH", "").strip()
        if not base:
            raise RuntimeError(f"FAIL LOUD: {PLAN_SELECTOR_ENV} needs GNN_MODEL_PATH (the base whose pool the selector was trained on)")
        self.model, self.contract = load_selector(Path(selector_path()), base_checkpoint=Path(base),
                                                  allow_smoke=os.environ.get(ALLOW_SMOKE_ENV, "").strip() == "1")
        self.host = host
        self.refiner = GnnCdRefiner(host)
        self.max_plans = int(os.environ.get(MAX_PLANS_ENV, "100000"))
        self.diag_dir = os.environ.get(DIAG_DIR_ENV, "").strip()
        self._diag = None
        for name in COUNTERS:
            setattr(host, name, 0.0 if name.endswith("_seconds") else 0)
        host.prsel_armed = 1
        host.prsel_variant = str(self.contract["variant"])
        host.prsel_mp_off = int(bool(self.contract["disable_message_passing"]))

    def _write_diag(self, rec: Dict[str, Any]) -> None:
        if self._diag is None:
            os.makedirs(self.diag_dir, exist_ok=True)
            self._diag = open(os.path.join(self.diag_dir, f"prsel_{os.getpid()}.jsonl"), "a", buffering=1 << 16)
        self._diag.write(json.dumps(rec) + "\n")
        self._diag.flush()

    def _native(self, batch_tasks, system_state) -> Tuple[Tuple[int, int], ...]:
        self.refiner._pg_batch_ids = frozenset(int(t.id) for t in batch_tasks)
        nat = self.refiner._cd_native_plan(batch_tasks, system_state, passes=3)
        return tuple((int(nat[i][0]), int(nat[i][1])) for i in range(len(batch_tasks)))

    def build_row(self, batch_tasks, system_state, sl) -> Tuple[Dict[str, Any], Any, Dict[str, Any]]:
        """(pool row in the training schema, the graph the selector reads, info). Raises nothing silently."""
        from src.policy.gnn.seq_decode import topological_task_order
        from src.policy.plan_rerank import s_components
        from src.policy.plan_rerank.pool import plan_score, step0_branches, teacher_forced_sum

        host = self.host
        n = len(batch_tasks)
        cands = [[(int(c["node_id"]), int(c["platform_id"])) for c in sl.kept[i]] for i in range(n)]
        score_fn, tl, graph = host._exact_score_fn_graph(batch_tasks, system_state, {i: set(cands[i]) for i in range(n)})
        tl = [[(int(c[0]), int(c[1])) for c in tl[t]] for t in range(n)]
        order = topological_task_order(n, graph.dag_parents)
        with torch.no_grad():
            branches = step0_branches(score_fn, tl, order)
        orch = host._orchestrator()
        memo: Dict[str, float] = {}
        ref = self.refiner
        ref._pg_batch_ids = frozenset(int(t.id) for t in batch_tasks)
        couple: Dict[Tuple[int, int], Any] = {}
        for i, task in enumerate(batch_tasks):
            for node, platform in host._get_valid_replicas(system_state.replicas.get(task.type["name"], set()), task):
                couple.setdefault((int(node.id), int(platform.id)), (node, platform))
        nodes = host.nodes.items
        slates = [sorted(c) for c in cands]
        best, ties = None, []
        for combo in product(*slates):
            sv, _svc, _sc = ref._pg_plan_cost(batch_tasks, [couple[c] for c in combo], orch, memo, nodes)
            if best is None or sv < best - S_TIE_EPS:
                best, ties = sv, [tuple(combo)]
            elif abs(sv - best) <= S_TIE_EPS:
                ties.append(tuple(combo))
        rows: Dict[Tuple[Tuple[int, int], ...], Dict[str, Any]] = {}
        srcs = [(f"branch:{i}", plan, acc) for i, (plan, acc) in enumerate(branches)] + [("argmin_s", p, None) for p in ties]
        with torch.no_grad():
            for src, plan, acc in srcs:
                if plan in rows:
                    rows[plan]["sources"].append(src)
                    continue
                comps = s_components.components(ref, batch_tasks, [couple[c] for c in plan], orch, memo, nodes)
                sv = s_components.s_of(comps)
                full, _svc, _sc = ref._pg_plan_cost(batch_tasks, [couple[c] for c in plan], orch, memo, nodes)
                if abs(full - sv) > 1e-9 * max(1.0, abs(full)):
                    raise RuntimeError(f"FAIL LOUD: S components sum {sv} != S {full}")
                rows[plan] = {"plan": [list(c) for c in plan], "sources": [src],
                              "plan_score": plan_score(score_fn, tl, plan),
                              "step_logit_sum": acc if acc is not None else teacher_forced_sum(score_fn, tl, order, plan),
                              "s": sv, "s_comps": comps}
        row = {"n_tasks": n, "candidates": list(rows.values())}
        return row, graph, {"branches": len(branches), "ties": len(ties), "order": order}

    def choose(self, batch_tasks, system_state, sl, decoded: Dict[int, Tuple[int, int]]) -> Dict[int, Tuple[int, int]]:
        from src.policy.plan_rerank.selector import select

        host = self.host
        t0 = default_timer()
        n = len(batch_tasks)
        dec = tuple((int(decoded[i][0]), int(decoded[i][1])) for i in range(n))
        native = self._native(batch_tasks, system_state)
        if sl.plans > self.max_plans:
            host.prsel_fallback_batches += 1
            host.prsel_fallback_tasks += n
            host.prsel_seconds += default_timer() - t0
            return {i: native[i] for i in range(n)}
        row, graph, info = self.build_row(batch_tasks, system_state, sl)
        branch_plans = [tuple(map(tuple, c["plan"])) for c in row["candidates"] if any(s.startswith("branch") for s in c["sources"])]
        if dec not in branch_plans:
            raise RuntimeError("FAIL LOUD: no step-0 branch reproduces the served decode; the pool's decode is not the served one")
        with torch.no_grad():
            cost, plans, cands = self.model.costs(graph, row)
        k = select([float(c) for c in cost], plans, cands)
        chosen = plans[k]
        argmin_s = {tuple(map(tuple, c["plan"])) for c in row["candidates"] if "argmin_s" in c["sources"]}
        host.prsel_batches += 1
        host.prsel_tasks += n
        host.prsel_pool_plans += len(plans)
        host.prsel_branches += info["branches"]
        host.prsel_argmin_s_ties += info["ties"]
        host.prsel_changed_from_decode += int(chosen != dec)
        host.prsel_chose_argmin_s += int(chosen in argmin_s)
        host.prsel_chose_branch_only += int(chosen not in argmin_s)
        host.prsel_differs_native += int(chosen != native)
        host.prsel_argmin_s_differs_native += int(native not in argmin_s)
        host.prsel_order_not_id += int([int(batch_tasks[t].id) for t in info["order"]] != sorted(int(t.id) for t in batch_tasks))
        host.prsel_seconds += default_timer() - t0
        if self.diag_dir:
            self._write_diag({"t": float(host.env.now), "row": row, "cost": [float(c) for c in cost], "plans": [list(map(list, p)) for p in plans],
                              "chosen": list(map(list, chosen)), "decoded": list(map(list, dec)), "native": list(map(list, native))})
        return {i: chosen[i] for i in range(n)}
