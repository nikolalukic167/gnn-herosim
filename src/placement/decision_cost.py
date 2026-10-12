"""decision_cost_s0_v1: the scheduler's decision time charged as simulated time (docs/lineages/decision_cost_s0_v1.md).

``HEROSIM_DECISION_COST_MS`` = c, milliseconds per plan evaluated. Unset or 0 is the old code path: no resource, no timeout, no SimPy event.
With c > 0 the batch decision path requests a capacity-1 FIFO server before the state mutex, computes the placements, then holds the mutex for
``t = c * plans_evaluated`` seconds of simulated time before applying them. ``plans_evaluated`` is the growth of ``sched_plan_evals``, a monotone
counter every plan or candidate score increments through ``meter`` (one candidate score in the greedy, one whole-plan score in the exact search and
the expansion, one plan scored by the GNN self-search, one forward per task of a GNN decode).
"""
from __future__ import annotations

import os
from typing import Any

ENV = "HEROSIM_DECISION_COST_MS"
COUNTERS = ("sched_plan_evals", "sched_busy_seconds", "sched_batches_charged", "sched_queue_wait_seconds", "sched_queue_wait_max",
            "sched_mutex_blocked_lb_s", "sched_mutex_waiters_max")


def cost_ms() -> float:
    raw = os.environ.get(ENV, "").strip()
    if not raw:
        return 0.0
    try:
        c = float(raw)
    except ValueError:
        raise ValueError(f"FAIL LOUD: {ENV} must be a number of milliseconds per plan, got {raw!r}") from None
    if c < 0:
        raise ValueError(f"FAIL LOUD: {ENV} must be >= 0, got {c}")
    return c


def meter(obj: Any, k: int) -> None:
    """Count k plan evaluations on the scheduler the orchestrator reads counters from (a refiner forwards to its host)."""
    tgt = getattr(obj, "_host", obj)
    tgt.sched_plan_evals = getattr(tgt, "sched_plan_evals", 0) + k


def server_of(scheduler):
    srv = scheduler.__dict__.get("_decision_server")
    if srv is None:
        from simpy.resources.resource import Resource

        srv = scheduler.__dict__["_decision_server"] = Resource(scheduler.env, capacity=1)
    return srv


def book(scheduler, name: str, v: float, mx: bool = False) -> None:
    cur = getattr(scheduler, name, 0)
    setattr(scheduler, name, max(cur, v) if mx else cur + v)
