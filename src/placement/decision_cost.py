"""decision_cost_s0_v1: the scheduler's decision time charged as simulated time (docs/lineages/decision_cost_s0_v1.md).

``HEROSIM_DECISION_COST_MS`` = c, milliseconds per plan evaluated. Unset or 0 is the old code path: no resource, no timeout, no SimPy event.
With c > 0 the batch decision path requests a capacity-1 FIFO server before the state mutex, computes the placements, then holds the mutex for
``t = c * plans_evaluated`` seconds of simulated time before applying them. ``plans_evaluated`` is the growth of ``sched_plan_evals``, a monotone
counter every plan or candidate score increments through ``meter`` (one candidate score in the greedy, one whole-plan score in the exact search and
the expansion, one plan scored by the GNN self-search, one forward per task of a GNN decode).

uncut_groups_s0_v1: ``HEROSIM_DECISION_COST_MS_BY_BUCKET`` = a JSON object {"le10": c, "11_50": c, "gt50": c} charges the rate of the batch's
size bucket (tasks decided, the buckets of ``decision_timing.SIZE_BUCKETS``) instead of one c. It and ``HEROSIM_DECISION_COST_MS`` never both serve.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from src.placement.decision_timing import SIZE_BUCKETS

ENV = "HEROSIM_DECISION_COST_MS"
BUCKET_ENV = "HEROSIM_DECISION_COST_MS_BY_BUCKET"
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


def bucket_of(n_tasks: int) -> str:
    return next(tag for hi, tag in SIZE_BUCKETS if hi is None or n_tasks <= hi)


def bucket_rates() -> Optional[Dict[str, float]]:
    raw = os.environ.get(BUCKET_ENV, "").strip()
    if not raw:
        return None
    if os.environ.get(ENV, "").strip():
        raise ValueError(f"FAIL LOUD: {ENV} and {BUCKET_ENV} are both set; a cell serves one rate rule")
    tags = [tag for _hi, tag in SIZE_BUCKETS]
    try:
        rates = json.loads(raw)
        ok = isinstance(rates, dict) and sorted(rates) == sorted(tags)
        out = {tag: float(rates[tag]) for tag in tags} if ok else None
    except (ValueError, TypeError):
        out = None
    if out is None or any(type(rates[t]) is bool or v < 0 for t, v in out.items()):
        raise ValueError(f"FAIL LOUD: {BUCKET_ENV} must be a JSON object of ms per plan >= 0 keyed exactly {tags}, got {raw!r}")
    return out


def enabled() -> bool:
    rates = bucket_rates()
    return any(v > 0 for v in rates.values()) if rates is not None else cost_ms() > 0.0


def rate_ms(n_tasks: int) -> float:
    """ms per plan for a batch of n_tasks decided: the bucket's rate when the bucket rule serves, else the one c."""
    rates = bucket_rates()
    return rates[bucket_of(n_tasks)] if rates is not None else cost_ms()


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
