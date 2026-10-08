"""Scale-out physics: which rule decides how many replicas a function gets (kpa_scaleout_v1).

`HEROSIM_SCALEOUT=legacy` (default) is every run before 2026-10-07: target concurrency 100 (int) per
replica over queued tasks only, one 1 s sample, no panic mode, idle replicas removed after keep-alive,
replicas otherwise created when a task's source reaches none (reachability).

`HEROSIM_SCALEOUT=kpa` is the Knative Pod Autoscaler at its defaults, with `containerConcurrency: 1`:
target 0.7 per replica (1 x 70 % utilisation), concurrency = tasks in flight (queued + admitted +
in service, including tasks mid-transfer when replicas are released), averaged over a 60 s stable
window and a 6 s panic window; panic when the panic-window demand is >= 200 % of the ready capacity,
no scale-down while panicking; the last replica is removed only after a full stable window without
traffic. Reachability-triggered creation is kept and logged as its own cause. Both windows scale with
HEROSIM_POLICY_TIME_SCALE exactly as keep-alive does.
"""
from __future__ import annotations

import math
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Optional, Tuple

SCALEOUT_ENV = "HEROSIM_SCALEOUT"
LEGACY = "legacy"
KPA = "kpa"
SCALEOUT_MODES = (LEGACY, KPA)

# Knative defaults (config-autoscaler): stable-window 60s, panic-window-percentage 10,
# panic-threshold-percentage 200, container-concurrency-target-percentage 70.
KPA_TARGET = 0.7
KPA_STABLE_WINDOW_S = 60.0
KPA_PANIC_WINDOW_S = 6.0
KPA_PANIC_THRESHOLD = 2.0
# amendment A2 (2026-10-07): max-scale-up-rate 1000, max-scale-down-rate 2.0 (at most halve per decision)
KPA_MAX_SCALE_UP_RATE = 1000.0
KPA_MAX_SCALE_DOWN_RATE = 2.0


def scaleout_mode() -> str:
    raw = (os.environ.get(SCALEOUT_ENV) or "").strip() or LEGACY
    if raw not in SCALEOUT_MODES:
        raise ValueError(f"{SCALEOUT_ENV}={raw!r}; expected one of {SCALEOUT_MODES}")
    return raw


SHARED_AUTOSCALER_ENV = "HEROSIM_SHARED_AUTOSCALER"


def shared_autoscaler() -> bool:
    """kpa_scaleout_v1 A4: every arm on the GNN-family autoscaler and the shared starved-task defer. Always on under
    kpa; HEROSIM_SHARED_AUTOSCALER=1 turns it on under legacy (the control that separates the autoscaler swap from
    KPA). Default off under legacy, so every earlier run replays."""
    raw = (os.environ.get(SHARED_AUTOSCALER_ENV) or "0").strip()
    if raw not in ("0", "1"):
        raise ValueError(f"{SHARED_AUTOSCALER_ENV}={raw!r}; expected 0 or 1")
    return scaleout_mode() == KPA or raw == "1"


def policy_time_scale() -> float:
    """HEROSIM_POLICY_TIME_SCALE multiplies keep_alive and the reconcile interval, so a workload whose
    timestamps were stretched by a factor keeps every policy time constant in proportion (the
    `drainable_regime_v1` rule; cd_gap_v1 B stretched arrivals and left these two unscaled)."""
    raw = (os.environ.get("HEROSIM_POLICY_TIME_SCALE") or "").strip()
    if not raw:
        return 1.0
    try:
        scale = float(raw)
    except ValueError:
        raise ValueError(f"HEROSIM_POLICY_TIME_SCALE={raw!r} is not a number") from None
    if not scale > 0:
        raise ValueError(f"HEROSIM_POLICY_TIME_SCALE={raw!r} must be > 0")
    return scale


@dataclass(frozen=True)
class KpaConfig:
    target: float
    stable_window: float
    panic_window: float
    panic_threshold: float = KPA_PANIC_THRESHOLD
    max_scale_up_rate: float = KPA_MAX_SCALE_UP_RATE
    max_scale_down_rate: float = KPA_MAX_SCALE_DOWN_RATE

    @classmethod
    def from_env(cls, target: float) -> "KpaConfig":
        scale = policy_time_scale()
        if not (isinstance(target, (int, float)) and target > 0 and math.isfinite(target)):
            raise ValueError(f"kpa target concurrency must be a finite number > 0, got {target!r}")
        return cls(float(target), KPA_STABLE_WINDOW_S * scale, KPA_PANIC_WINDOW_S * scale)

    def describe(self) -> Dict[str, Any]:
        return {"mode": KPA, "target": self.target, "stable_window_s": self.stable_window,
                "panic_window_s": self.panic_window, "panic_threshold": self.panic_threshold,
                "max_scale_up_rate": self.max_scale_up_rate, "max_scale_down_rate": self.max_scale_down_rate,
                "policy_time_scale": policy_time_scale()}


def pending_in_flight(pending: Any) -> int:
    """Amendment A1 (2026-10-07): tasks that arrived and are not yet on any replica -- held in a batching
    scheduler's peer-group buffer or postponed for want of a reachable replica -- count as demand for their
    type, as Knative's activator reports the requests it buffers. Prunes placed tasks in place."""
    if not pending:
        return 0
    pending[:] = [t for t in pending if not t.scheduled.triggered]
    return len(pending)


def platform_in_flight(platform: Any) -> int:
    """Tasks a replica holds: queued (plus seeded backlog), popped but not yet started (network/ingress
    transfer), and in service. Under HEROSIM_REPLICA_RELEASE=1 `inflight` holds every task past cold start
    still in rendezvous, exchange, compute or output, and `current_task` is one of them."""
    n = platform.queue_length()
    if getattr(platform, "admitted", None) is not None:
        n += 1
    inflight = getattr(platform, "inflight", None) or ()
    n += len(inflight)
    current = getattr(platform, "current_task", None)
    if current is not None and current not in inflight:
        n += 1
    return n


@dataclass
class KpaFunctionState:
    samples: Deque[Tuple[float, float]] = field(default_factory=deque)
    first_sample: Optional[float] = None
    last_traffic: float = -math.inf
    panic_since: Optional[float] = None
    panic_last_over: Optional[float] = None
    max_panic_pods: int = 0


@dataclass
class KpaDecision:
    desired: int
    stable_avg: float
    panic_avg: float
    panicking: bool
    entered_panic: bool


class KpaScaler:
    """The per-function KPA decision, kept free of SimPy so it can be tested directly."""

    def __init__(self, config: KpaConfig):
        self.config = config
        self.functions: Dict[str, KpaFunctionState] = {}

    def observe(self, function_name: str, now: float, in_flight: float) -> None:
        st = self.functions.setdefault(function_name, KpaFunctionState())
        if st.first_sample is None:
            st.first_sample = now
        st.samples.append((now, float(in_flight)))
        if in_flight > 0:
            st.last_traffic = now
        horizon = now - self.config.stable_window
        while st.samples and st.samples[0][0] <= horizon:
            st.samples.popleft()

    def _average(self, st: KpaFunctionState, now: float, window: float) -> float:
        vals = [v for t, v in st.samples if t > now - window]
        return sum(vals) / len(vals) if vals else 0.0

    def decide(self, function_name: str, now: float, current: int, ready: int) -> KpaDecision:
        cfg = self.config
        st = self.functions.setdefault(function_name, KpaFunctionState())
        stable_avg = self._average(st, now, cfg.stable_window)
        panic_avg = self._average(st, now, cfg.panic_window)
        raw_stable = math.ceil(stable_avg / cfg.target - 1e-9)
        raw_panic = math.ceil(panic_avg / cfg.target - 1e-9)
        # rate limits (A2): the panic test reads the unclamped demand; both counts are then held in
        # [floor(ready / max_scale_down_rate), ceil(max_scale_up_rate * max(1, ready))]
        lo = math.floor(ready / cfg.max_scale_down_rate)
        hi = math.ceil(cfg.max_scale_up_rate * max(1, ready))
        desired_stable = min(max(raw_stable, lo), hi)
        desired_panic = min(max(raw_panic, lo), hi)

        over = (panic_avg / cfg.target) / max(1, ready) >= cfg.panic_threshold
        entered = False
        if over:
            if st.panic_since is None:
                st.panic_since = now
                entered = True
            st.panic_last_over = now
        elif st.panic_since is not None and now - st.panic_last_over >= cfg.stable_window:
            st.panic_since = None
            st.panic_last_over = None
        panicking = st.panic_since is not None

        if panicking:
            desired = max(desired_panic, st.max_panic_pods, current)
            st.max_panic_pods = desired
        else:
            st.max_panic_pods = 0
            desired = desired_stable
            if desired == 0 and current > 0:
                window_covered = st.first_sample is not None and now - st.first_sample >= cfg.stable_window
                quiet = now - st.last_traffic >= cfg.stable_window
                if not (window_covered and quiet):
                    desired = 1
        return KpaDecision(desired, stable_avg, panic_avg, panicking, entered)


def new_scaleout_stats() -> Dict[str, Any]:
    return {
        "scale_ups_by_cause": {"load": 0, "reachability": 0},
        "scale_up_failures_by_cause": {"load": 0, "reachability": 0},
        "scale_downs": 0,
        "scale_to_zero": 0,
        "panic_entries": 0,
        "panic_ticks": 0,
        # candidate (node, platform) couples skipped because the node's free memory cannot hold the
        # replica, and scale-up attempts that found no couple only because of that
        "memory_cap_refusals": 0,
        "memory_cap_blocked": 0,
        "replica_lifetimes_closed": 0,
        "replica_lifetimes_within_stable_window": 0,
    }
