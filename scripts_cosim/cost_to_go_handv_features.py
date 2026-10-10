"""cost_to_go_v1 hand V: the post-commit features of one plan, read off the scheduler at the batch decision.

UNVERIFIED against live objects: written from rollout_replay.py / peer_greedy_network/scheduler.py by a code read, exercised only
with fakes in tests/test_cost_to_go_handv.py. Run it on ONE real state and compare `load_after` with S's own backlog terms before
trusting any row. Integration point: inside rollout_replay's patched `_prefix_inference`, after the batch decision is identified
(the call whose tasks include the dataset batch), once per plan (policy plan and each given plan), before `res` is used.
"""
from __future__ import annotations

import math

from src.placement.live_audit import inflight_remaining_seconds, platform_queue_drain_seconds


def _key(node, platform):
    return f"{int(node.id)}:{int(platform.id)}"


def post_commit_features(sched, tasks, plan, system_state, now, type_defs, scale_in_after_s, *, ds, slot, t0, workload):
    """plan: [(Node, Platform)] aligned with `tasks`. type_defs: {type name: the task.type dict S reads} -- a replica's cold seconds are
    type_defs[type]["coldStartDuration"][platform.type["shortName"]], as scheduling_cost.incoming_cold_start_time charges them
    (a missing type fails loudly)."""
    orch = sched._pg_orchestrator()
    memo = {}
    nodes = sched.nodes.items
    _total, service, _scores = sched._pg_plan_cost(tasks, plan, orch, memo, nodes)
    own = {}
    for (node, platform), svc in zip(plan, service):
        own[_key(node, platform)] = own.get(_key(node, platform), 0.0) + float(svc)
    used = set(own)
    type_platforms, load_after, load_after_inflight, drain_of, inflight_of, replicas = {}, {}, {}, {}, {}, []
    for typ, reps in system_state.replicas.items():
        if typ not in type_defs:
            raise KeyError(f"FAIL LOUD: type_defs has no entry for task type {typ!r}")
        type_platforms[typ] = []
        for node, platform in sorted(reps, key=lambda c: (int(c[0].id), int(c[1].id))):
            k = _key(node, platform)
            type_platforms[typ].append(k)
            if k not in load_after:
                xf = sched._pg_xf(platform)
                drain = float(platform_queue_drain_seconds(platform, orch, memo, exec_scale=xf))
                fly = float(inflight_remaining_seconds(platform) or 0.0)      # logged whatever the pg_inflight switch says
                drain_of[k], inflight_of[k] = drain, fly
                load_after[k] = drain + (fly if sched.pg_inflight else 0.0) + own.get(k, 0.0)    # as S sees it
                load_after_inflight[k] = drain + fly + own.get(k, 0.0)                            # plus in-flight remaining
            ref = platform.idle_since if math.isfinite(platform.idle_since) else platform.last_allocated
            cold_s = float(type_defs[typ]["coldStartDuration"].get(platform.type["shortName"], 0.0) or 0.0)
            replicas.append({"key": k, "type": typ, "idle_s": float(now - ref), "used": k in used, "cold_s": cold_s})
    return {"ds": ds, "slot": slot, "t0": float(t0), "workload": workload, "load_after": load_after,
            "load_after_inflight": load_after_inflight, "drain": drain_of, "inflight": inflight_of,
            "type_platforms": type_platforms, "replicas": replicas,
            "scale_in_after_s": float(scale_in_after_s)}


def decision_hook(sched, batch_tasks, plan, system_state, ctx):
    """rollout_replay `decision_hook` entry point ("scripts_cosim.cost_to_go_handv_features:decision_hook"): called once at the batch
    decision, before anything is enqueued. plan is [(Node, Platform)] aligned with batch_tasks, which may include queued tasks forced
    at the same decision; all of them are committed to their platforms here, so their service is in load_after.
    Scale-in time is sched.autoscaler.kpa.config.stable_window, the KPA's own horizon (it has no fixed scale-in time)."""
    job = ctx["job"]
    feat = post_commit_features(
        sched, batch_tasks, plan, system_state, float(sched.env.now),
        sched._pg_orchestrator().data.task_types, float(sched.autoscaler.kpa.config.stable_window),
        ds=job["ds"], slot=job["tag"].split("|", 1)[1], t0=float(ctx["t0"]), workload=job["workload"])
    # Plumbing check material: S's own totals for this plan, to compare with the features offline.
    orch = sched._pg_orchestrator()
    total, service, scores = sched._pg_plan_cost(batch_tasks, plan, orch, {}, sched.nodes.items)
    feat["s_total"], feat["s_service"], feat["s_scores"] = float(total), [float(x) for x in service], [float(x) for x in scores]
    feat["pg_inflight"] = bool(sched.pg_inflight)
    feat["local_batch"] = sorted(int(ctx["local_of"][int(t.id)]) for t in batch_tasks if int(t.id) in ctx["local_of"])
    return feat
