"""decima_rule_v1: Decima's strongest hand baseline, the tuned weighted-fair scheduler, in the burst seat.

Decima (Mao et al., SIGCOMM 2019, §7) compares against FIFO, SJF-CP, fair, naive weighted fair,
Tetris and Graphene*. The strongest in its Spark batched- and continuous-arrival experiments is
the TUNED weighted-fair scheduler: at every scheduling event the free executors are split across
the jobs in the system in proportion to w_j = W_j ** alpha, W_j = job j's remaining total work,
alpha grid-searched (reported best alpha = -1, favouring short jobs). alpha = 0 is the plain fair
scheduler, alpha = 1 the naive weighted-fair one: one knob spans Decima's fair family.

Mapping onto HeROsim's burst seat (the learned arms' and CD's serving stack, peer-group batching):
  job      = a peer group (connected component of the trace's peer_exchange table)
  executor = a platform; tasks are committed into its FIFO queue at placement
  share    = the batch's job gets k_t = round(s_J * |P_t|) platforms of its candidate pool P_t for
             each task type t, s_J = w_J / (sum of w over the batch's jobs and every other job with
             work left); clamp to [1, |P_t|]
  which k  = the k platforms of P_t that can finish a task earliest (drain + cold + exec): the
             "free executors" of a heterogeneous cluster
  within   = each task, in id order, goes to the earliest-finishing platform of its allowed set,
             batch-mates' service charged (the drain-greedy score, exchange OFF)

What does not transfer: Decima's lever is ORDER -- which job's stage gets the next free executor.
HeROsim platform queues are FIFO and placement is committed at arrival, so the only part of the
rule a placement policy can express is the per-job PARALLELISM (how many platforms a group
spreads over). The rule is locality-blind by construction (Decima's baselines ignore data
locality), so it prices no peer exchange; co-location happens only as a side effect of a small k.
"""
from __future__ import annotations

import math
import os
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, TYPE_CHECKING

from src.placement.live_snapshot_seed import _approx_comm
from src.placement.live_audit import platform_queue_drain_seconds
from src.placement.model import SystemState
from src.placement.scheduling_cost import incoming_cold_start_time
from src.policy.peer_greedy_network.scheduler import PeerGreedyNetworkBatchScheduler

if TYPE_CHECKING:
    from src.placement.infrastructure import Node, Platform, Task

DECIMA_ALPHA_ENV = "HEROSIM_DECIMA_ALPHA"
DECIMA_DEFAULT_ALPHA = -1.0
DECIMA_COUNTERS = ("decima_batches", "decima_jobs", "decima_active_jobs", "decima_pool_platforms",
                   "decima_allowed_platforms", "decima_fallbacks")


def decima_alpha() -> float:
    raw = os.environ.get(DECIMA_ALPHA_ENV, "").strip()
    if not raw:
        return DECIMA_DEFAULT_ALPHA
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"FAIL LOUD: {DECIMA_ALPHA_ENV}={raw!r} is not a float") from exc
    if not math.isfinite(value):
        raise ValueError(f"FAIL LOUD: {DECIMA_ALPHA_ENV} must be finite, got {raw!r}")
    return value


def peer_groups(table: Mapping[int, Mapping[int, float]]) -> Dict[int, int]:
    """task id -> job id (the smallest task id of its connected component in the peer table)."""
    parent: Dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, peers in table.items():
        for j in peers:
            a, b = find(int(i)), find(int(j))
            if a != b:
                parent[max(a, b)] = min(a, b)
    return {t: find(t) for t in list(parent)}


def job_weight(work: float, alpha: float) -> float:
    if work <= 0.0:
        raise ValueError(f"job_weight: remaining work must be > 0, got {work}")
    return work ** alpha


def fair_share_count(pool: int, w_self: float, w_total: float) -> int:
    """Decima's weighted-fair executor share for one job, as a platform count in [1, pool]."""
    if pool < 1:
        raise ValueError(f"fair_share_count: empty pool ({pool})")
    if not (0.0 < w_self <= w_total):
        raise ValueError(f"fair_share_count: need 0 < w_self <= w_total, got {w_self}, {w_total}")
    return max(1, min(pool, int(math.floor(pool * w_self / w_total + 0.5))))


def earliest_platforms(ranked: Iterable[Tuple[float, int, int]], k: int) -> List[Tuple[int, int]]:
    """The k (node_id, platform_id) with the smallest finish estimate; ties by node id, platform id."""
    return [(n, p) for _f, n, p in sorted(ranked)[:k]]


class DecimaWeightedFairBatchScheduler(PeerGreedyNetworkBatchScheduler):
    """Tuned weighted fair (Decima's best hand baseline) on the learned arms' serving stack."""

    _policy_label = "decima_wfair_network"
    _live_audit_policy_name = "decima_wfair_network"
    exchange_on = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decima_alpha = decima_alpha()
        for name in DECIMA_COUNTERS:
            setattr(self, name, 0)
        self._decima_job_of: Optional[Dict[int, int]] = None
        # tasks this rule placed and has not seen finish: job -> {task id: (task, work seconds)}
        self._decima_live: Dict[int, Dict[int, Tuple["Task", float]]] = {}

    def _job_of(self, orch, task_id: int) -> int:
        if self._decima_job_of is None:
            self._decima_job_of = peer_groups(getattr(orch, "peer_exchange", None) or {})
        return self._decima_job_of.get(int(task_id), int(task_id))

    @staticmethod
    def _remaining(task: "Task", work: float) -> float:
        started = getattr(task, "started_time", None)
        if started is None:
            return work
        return max(0.0, work - (float(task.env.now) - float(started)))

    def _active_weights(self) -> Dict[int, float]:
        """Remaining work per job still in the system (tasks placed by this rule, not finished)."""
        out: Dict[int, float] = {}
        for job in list(self._decima_live):
            live = self._decima_live[job]
            for tid in [t for t, (task, _w) in live.items() if getattr(task, "finished", False)]:
                del live[tid]
            if not live:
                del self._decima_live[job]
                continue
            rem = sum(self._remaining(task, w) for task, w in live.values())
            if rem > 0.0:
                out[job] = rem
        return out

    def _prefix_inference(
        self,
        batch_tasks: List["Task"],
        system_state: SystemState,
        queue_snapshot: Dict[str, int],
        temporal_state: Optional[Dict[str, Dict[str, float]]],
    ) -> Dict[int, Tuple[int, int]]:
        orch = self._pg_orchestrator()
        memo: Dict[str, float] = {}
        committed_service: Dict[str, float] = {}
        placements: Dict[int, Tuple[int, int]] = {}
        self._pg_batch_ids = frozenset(int(t.id) for t in batch_tasks)

        cands: Dict[int, List[Tuple["Node", "Platform"]]] = {}
        for idx, task in enumerate(batch_tasks):
            valid = self._get_valid_replicas(system_state.replicas.get(task.type["name"], set()), task)
            if not valid:
                raise RuntimeError(
                    f"{self._policy_label}: task {task.id} reached the decoder without a "
                    "network-accessible replica; the batch path defers those before decoding"
                )
            initialized = [r for r in valid if r[1].initialized.triggered]
            cands[idx] = initialized if initialized else valid

        def work_of(task: "Task", options: Sequence[Tuple["Node", "Platform"]]) -> float:
            execs = [float(task.type["executionTime"].get(p.type["shortName"], 0.0) or 0.0) for _n, p in options]
            return min(execs) + _approx_comm(task.type)

        jobs: Dict[int, List[int]] = {}
        for idx, task in enumerate(batch_tasks):
            jobs.setdefault(self._job_of(orch, int(task.id)), []).append(idx)
        others = {j: w for j, w in self._active_weights().items() if j not in jobs}
        batch_work = {j: sum(work_of(batch_tasks[i], cands[i]) for i in idxs) for j, idxs in jobs.items()}
        for j in jobs:
            batch_work[j] += sum(self._remaining(t, w) for t, w in self._decima_live.get(j, {}).values())
        weights = {j: job_weight(w, self.decima_alpha) for j, w in batch_work.items()}
        w_total = sum(weights.values()) + sum(job_weight(w, self.decima_alpha) for w in others.values())
        self.decima_batches += 1
        self.decima_jobs += len(jobs)
        self.decima_active_jobs += len(others)

        for job in sorted(jobs):
            idxs = sorted(jobs[job], key=lambda i: int(batch_tasks[i].id))
            by_type: Dict[str, List[int]] = {}
            for i in idxs:
                by_type.setdefault(batch_tasks[i].type["name"], []).append(i)
            for ttype in sorted(by_type):
                members = by_type[ttype]
                probe = batch_tasks[members[0]]
                pool: Dict[Tuple[int, int], Tuple["Node", "Platform"]] = {}
                for i in members:
                    for node, platform in cands[i]:
                        pool[(node.id, platform.id)] = (node, platform)
                k = fair_share_count(len(pool), weights[job], w_total)
                ranked = []
                for (nid, pid), (node, platform) in pool.items():
                    finish = (platform_queue_drain_seconds(platform, orch, memo)
                              + incoming_cold_start_time(probe, platform)
                              + float(probe.type["executionTime"].get(platform.type["shortName"], 0.0) or 0.0))
                    ranked.append((finish, nid, pid))
                allowed: Set[Tuple[int, int]] = set(earliest_platforms(ranked, k))
                self.decima_pool_platforms += len(pool)
                self.decima_allowed_platforms += k
                for i in members:
                    task = batch_tasks[i]
                    options = [c for c in cands[i] if (c[0].id, c[1].id) in allowed]
                    if not options:
                        self.decima_fallbacks += 1
                        options = cands[i]
                    node, platform, service = self._pg_choose(
                        task, options, orch, memo=memo, committed_service=committed_service,
                        planned={}, nodes=self.nodes.items,
                    )
                    key = f"{node.node_name}:{platform.id}"
                    committed_service[key] = committed_service.get(key, 0.0) + service
                    placements[i] = (node.id, platform.id)
                    exec_s = float(task.type["executionTime"].get(platform.type["shortName"], 0.0) or 0.0)
                    self._decima_live.setdefault(job, {})[int(task.id)] = (task, exec_s + _approx_comm(task.type))
        self.pg_batches += 1
        return placements
