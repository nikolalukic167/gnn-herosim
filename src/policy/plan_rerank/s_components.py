"""`_pg_plan_cost`'s S split into summed terms, the components the plan_rerank_v1 pool carries (fidelity_cd_plans --s-comps,
rp/plan-rerank-data 10c9081a, which computed the training pool's columns with this arithmetic on the replayed decision state):

  drain_platform  the platform's own drain (+ in-flight)
  drain_batch     the service of the other batch tasks the plan stacks on the same platform
  cold            cold start
  cold_dup        the part of cold charged to a task whose platform already holds an earlier batch task (index order) of the same
                  type, i.e. a cold start co-location avoids; NOT a term of S
  exec_lat        exec + network latency
  exch            exchange_scale * exchange

S = drain_platform + drain_batch + cold + exec_lat + exch. Defined only where `_pg_plan_cost` has no further term (no pull hold, no
externality, no input-link wait, no shared-link pricing, both alphas 1), which `components` asserts.
"""
from __future__ import annotations

from typing import Dict, Sequence

KEYS = ("drain_platform", "drain_batch", "cold", "cold_dup", "exec_lat", "exch")


def components(sched, batch_tasks, plan: Sequence, orch, memo: Dict[str, float], nodes) -> Dict[str, float]:
    from src.policy.peer_greedy_network import scheduler as PG

    if getattr(sched, "pg_pull_hold", False) or sched.pg_ext_rate is not None or sched.pg_input_link or sched.pg_shared_link \
            or sched.pg_alpha_on:
        raise RuntimeError("FAIL LOUD: S components assume no pull hold, no externality, no input-link or shared-link term and alpha 1")
    n = len(batch_tasks)
    planned = {int(batch_tasks[i].id): plan[i][0].node_name for i in range(n)}
    books = (sched.pg_partners_known, sched.pg_partners_unknown, sched.pg_partners_blinded)
    peer_nodes_of = []
    for i in range(n):
        others = dict(planned)
        others.pop(int(batch_tasks[i].id), None)
        peer_nodes_of.append(sched._pg_peer_nodes(batch_tasks[i], orch, others) if sched.exchange_on else [])
    sched.pg_partners_known, sched.pg_partners_unknown, sched.pg_partners_blinded = books
    exch = [sched._pg_exchange_seconds(plan[i][0], plan[i][1], peer_nodes_of[i]) for i in range(n)]
    xf = [sched._pg_xf(plan[i][1]) for i in range(n)]
    exec_s = [float(batch_tasks[i].type["executionTime"].get(plan[i][1].type["shortName"], 0.0) or 0.0) * xf[i] for i in range(n)]
    service = [exec_s[i] + PG._approx_comm(batch_tasks[i].type) + exch[i] for i in range(n)]
    committed: Dict[str, float] = {}
    for i in range(n):
        k = f"{plan[i][0].node_name}:{plan[i][1].id}"
        committed[k] = committed.get(k, 0.0) + service[i]
    out = dict.fromkeys(KEYS, 0.0)
    seen_type = set()
    for i in range(n):
        node, platform = plan[i]
        k = f"{node.node_name}:{platform.id}"
        dp = PG.platform_queue_drain_seconds(platform, orch, memo, exec_scale=xf[i])
        if sched.pg_inflight:
            dp += PG.inflight_remaining_seconds(platform)
        cold = PG.incoming_cold_start_time(batch_tasks[i], platform)
        tk = (k, batch_tasks[i].type["name"])
        if tk in seen_type:
            out["cold_dup"] += cold
        seen_type.add(tk)
        out["drain_platform"] += dp
        out["drain_batch"] += committed[k] - service[i]
        out["cold"] += cold
        out["exec_lat"] += exec_s[i] + PG.network_latency_between(batch_tasks[i].node_name, node, nodes)
        out["exch"] += sched.pg_exchange_scale * exch[i]
    return out


def s_of(c: Dict[str, float]) -> float:
    return c["drain_platform"] + c["drain_batch"] + c["cold"] + c["exec_lat"] + c["exch"]
