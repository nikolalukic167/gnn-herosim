"""sparse_obs_s0_v1: what the peer-greedy scheduler OBSERVES, degraded one channel at a time. The physics is untouched.

Every flag is default off, and with none set `Observation.from_env()` returns None, so the scheduler runs its registered arithmetic unchanged.

  HEROSIM_OBS_ESTIMATOR   naive | ec   (required when any channel is set)
  HEROSIM_OBS_QUEUE=count     a platform's drain = queued count x a service estimate (naive: the global mean service; ec: the per-type mean service of each
                              queued task's type). No per-task exec, I/O, latency or partner transfer.
  HEROSIM_OBS_EXEC_SIGMA=s    every exec estimate x a log-normal factor, one per (task type, platform type), seeded by HEROSIM_OBS_SEED, fixed for the run.
  HEROSIM_OBS_INFLIGHT=running  the running task is known only as running: naive charges nothing, ec charges half its type's mean service.
  HEROSIM_OBS_PARTNERS=arrival  a partner's node is known only once the partner is placed: the batch's own plan is invisible. ec predicts the node of a
                              partner with no known node as the node of its bytes-heaviest known partner (the lookahead rule's two-step estimate).
"""
from __future__ import annotations

import math
import os
import random
from typing import Dict, List, Optional, Tuple

from src.placement.live_snapshot_seed import _approx_comm

QUEUE_ENV = "HEROSIM_OBS_QUEUE"
EXEC_SIGMA_ENV = "HEROSIM_OBS_EXEC_SIGMA"
INFLIGHT_ENV = "HEROSIM_OBS_INFLIGHT"
PARTNERS_ENV = "HEROSIM_OBS_PARTNERS"
ESTIMATOR_ENV = "HEROSIM_OBS_ESTIMATOR"
SEED_ENV = "HEROSIM_OBS_SEED"
OBS_ENVS = (QUEUE_ENV, EXEC_SIGMA_ENV, INFLIGHT_ENV, PARTNERS_ENV, ESTIMATOR_ENV, SEED_ENV)
OBS_COUNTERS = ("obs_drain_degraded", "obs_drain_exact", "obs_exec_degraded", "obs_inflight_degraded", "obs_partner_hidden", "obs_partner_predicted")


def _choice(env: str, allowed: Tuple[str, ...]) -> str:
    raw = os.environ.get(env, "").strip()
    if raw not in ("",) + allowed:
        raise ValueError(f"FAIL LOUD: {env}={raw!r}; expected one of {allowed}")
    return raw


class Observation:
    def __init__(self, queue: str, sigma: float, inflight: str, partners: str, estimator: str, seed: str) -> None:
        self.queue_count = queue == "count"
        self.sigma = sigma
        self.inflight_running = inflight == "running"
        self.partners_arrival = partners == "arrival"
        self.ec = estimator == "ec"
        self.seed = seed
        self.counts: Dict[str, int] = {n: 0 for n in OBS_COUNTERS}
        self._factor: Dict[Tuple[str, str], float] = {}
        self._type_service: Dict[str, float] = {}
        self._global_service: Optional[float] = None

    @classmethod
    def from_env(cls) -> Optional["Observation"]:
        queue = _choice(QUEUE_ENV, ("count",))
        inflight = _choice(INFLIGHT_ENV, ("running",))
        partners = _choice(PARTNERS_ENV, ("arrival",))
        raw_sigma = os.environ.get(EXEC_SIGMA_ENV, "").strip()
        try:
            sigma = float(raw_sigma) if raw_sigma else 0.0
        except ValueError as exc:
            raise ValueError(f"FAIL LOUD: {EXEC_SIGMA_ENV}={raw_sigma!r} is not a float") from exc
        if sigma < 0.0:
            raise ValueError(f"FAIL LOUD: {EXEC_SIGMA_ENV} must be >= 0, got {sigma}")
        estimator = _choice(ESTIMATOR_ENV, ("naive", "ec"))
        if not (queue or inflight or partners or sigma):
            if estimator:
                raise ValueError(f"FAIL LOUD: {ESTIMATOR_ENV}={estimator!r} with no degradation set")
            return None
        if not estimator:
            raise ValueError(f"FAIL LOUD: a degradation is set; {ESTIMATOR_ENV} must be naive or ec")
        if inflight and os.environ.get("HEROSIM_PG_INFLIGHT", "0").strip() not in ("", "0"):
            raise ValueError(f"FAIL LOUD: {INFLIGHT_ENV} replaces HEROSIM_PG_INFLIGHT's exact remaining seconds; set it to 0")
        return cls(queue, sigma, inflight, partners, estimator, os.environ.get(SEED_ENV, "0").strip() or "0")

    def exec_est(self, task_type, plat_type: str, table_value: float) -> float:
        if not self.sigma:
            return table_value
        key = (str(task_type["name"]), plat_type)
        f = self._factor.get(key)
        if f is None:
            f = math.exp(self.sigma * random.Random(f"{self.seed}|{key[0]}|{key[1]}").gauss(0.0, 1.0))
            self._factor[key] = f
        self.counts["obs_exec_degraded"] += 1
        return table_value * f

    def exec_of(self):
        """The exec estimator for the exact queue walk (None when the exec channel is exact)."""
        if not self.sigma:
            return None
        return lambda task_type, plat_type: self.exec_est(task_type, plat_type, float(task_type.get("executionTime", {}).get(plat_type, 0.0) or 0.0))

    def type_service(self, task_type) -> float:
        """Mean over platform types of the (estimated) exec, plus the storage I/O the scheduler is told."""
        name = str(task_type["name"])
        v = self._type_service.get(name)
        if v is None:
            ex = task_type.get("executionTime", {}) or {}
            vals = [self.exec_est(task_type, pt, float(x or 0.0)) for pt, x in sorted(ex.items())]
            v = (sum(vals) / len(vals) if vals else 0.0) + _approx_comm(task_type)
            self._type_service[name] = v
        return v

    def global_service(self, orch) -> float:
        if self._global_service is None:
            types = [orch.data.task_types[k] for k in sorted(orch.data.task_types)]
            self._global_service = sum(self.type_service(t) for t in types) / len(types)
        return self._global_service

    def drain(self, platform, orch, memo: Dict[str, float], exec_scale: float, exact) -> float:
        if not self.queue_count:
            self.counts["obs_drain_exact"] += 1
            return exact(platform, orch, memo, exec_scale=exec_scale, exec_of=self.exec_of())
        key = f"obs:{platform.node.node_name}:{platform.id}"
        if key in memo:
            return memo[key]
        items = list(platform.queue.items)
        if self.ec:
            total = sum(self.type_service(t.type) for t in items)
        else:
            total = len(items) * self.global_service(orch)
        self.counts["obs_drain_degraded"] += 1
        memo[key] = float(total) * exec_scale
        return memo[key]

    def running_seconds(self, platform) -> float:
        task = platform.current_task
        if not self.inflight_running or task is None or not self.ec:
            return 0.0
        self.counts["obs_inflight_degraded"] += 1
        return 0.5 * self.type_service(task.type)

    @staticmethod
    def _placed_node(orch, peer_id: int) -> Optional[str]:
        peer = orch.task_by_id.get(int(peer_id))
        if peer is None:
            return None
        plat = getattr(peer, "platform", None)
        return plat.node.node_name if plat is not None else getattr(peer, "planned_node_name", None)

    def predict_node(self, task, orch, peer_id: int) -> Optional[str]:
        table = getattr(orch, "peer_exchange", None) or {}
        weight: Dict[str, float] = {}
        for q, b in (table.get(int(peer_id)) or {}).items():
            if int(q) == int(task.id):
                continue
            node_name = self._placed_node(orch, int(q))
            if node_name is not None:
                weight[node_name] = weight.get(node_name, 0.0) + float(b)
        if not weight:
            return None
        return max(sorted(weight), key=lambda n: weight[n])
