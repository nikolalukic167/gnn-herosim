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


def post_commit_features(sched, tasks, plan, system_state, now, cold_cost_s, scale_in_after_s, *, ds, slot, t0, workload):
    """plan: [(Node, Platform)] aligned with `tasks`. cold_cost_s: {type: seconds}, supplied by the caller (a missing type fails)."""
    orch = sched._pg_orchestrator()
    memo = {}
    nodes = sched.nodes.items
    _total, service, _scores = sched._pg_plan_cost(tasks, plan, orch, memo, nodes)
    own = {}
    for (node, platform), svc in zip(plan, service):
        own[_key(node, platform)] = own.get(_key(node, platform), 0.0) + float(svc)
    used = set(own)
    type_platforms, load_after, replicas = {}, {}, []
    for typ, reps in system_state.replicas.items():
        if typ not in cold_cost_s:
            raise KeyError(f"FAIL LOUD: cold_cost_s has no entry for task type {typ!r}")
        type_platforms[typ] = []
        for node, platform in sorted(reps, key=lambda c: (int(c[0].id), int(c[1].id))):
            k = _key(node, platform)
            type_platforms[typ].append(k)
            if k not in load_after:
                xf = sched._pg_xf(platform)
                drain = float(platform_queue_drain_seconds(platform, orch, memo, exec_scale=xf))
                if sched.pg_inflight:
                    drain += float(inflight_remaining_seconds(platform) or 0.0)
                load_after[k] = drain + own.get(k, 0.0)
            ref = platform.idle_since if math.isfinite(platform.idle_since) else platform.last_allocated
            replicas.append({"key": k, "type": typ, "idle_s": float(now - ref), "used": k in used})
    return {"ds": ds, "slot": slot, "t0": float(t0), "workload": workload, "load_after": load_after,
            "type_platforms": type_platforms, "replicas": replicas, "cold_cost_s": dict(cold_cost_s),
            "scale_in_after_s": float(scale_in_after_s)}
