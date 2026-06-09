"""Sequential GNN decode with optional seqblend min-queue override."""

from __future__ import annotations

import itertools
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:
    from torch import Tensor
except ImportError:  # pragma: no cover
    Tensor = object  # type: ignore

PlacementCombo = Tuple[Tuple[int, int], ...]


@dataclass
class GnnDecodeRunStats:
    """Per-run GNN decode instrumentation (all decode modes)."""

    decode_mode: str = ""
    top_k: int = 0
    gnn_batches: int = 0
    decode_time_ms: List[float] = field(default_factory=list)
    combo_search_size: List[int] = field(default_factory=list)
    intra_batch_platform_collisions: List[int] = field(default_factory=list)
    chosen_queue_minus_min: List[int] = field(default_factory=list)

    # Seqblend-specific (argmax + seqblend mode only)
    total_tasks: int = 0
    p1_override_count: int = 0
    classic_would_override_count: int = 0
    classic_only_count: int = 0  # classic would override, p1 kept GNN (gnn_q == min+1)
    gnn_queue_when_p1_override: List[int] = field(default_factory=list)
    final_queue_when_p1_override: List[int] = field(default_factory=list)
    min_queue_when_p1_override: List[int] = field(default_factory=list)
    gnn_queue_when_classic_only: List[int] = field(default_factory=list)
    gnn_queue_all: List[int] = field(default_factory=list)
    final_queue_all: List[int] = field(default_factory=list)

    def record_task(
        self,
        gnn_queue: int,
        final_queue: int,
        min_queue: int,
        *,
        p1_margin: int,
    ) -> None:
        self.total_tasks += 1
        self.gnn_queue_all.append(int(gnn_queue))
        self.final_queue_all.append(int(final_queue))

        classic = gnn_queue > min_queue
        p1_override = gnn_queue > min_queue + p1_margin

        if classic:
            self.classic_would_override_count += 1
        if p1_override:
            self.p1_override_count += 1
            self.gnn_queue_when_p1_override.append(int(gnn_queue))
            self.final_queue_when_p1_override.append(int(final_queue))
            self.min_queue_when_p1_override.append(int(min_queue))
        elif classic:
            self.classic_only_count += 1
            self.gnn_queue_when_classic_only.append(int(gnn_queue))

    def merge(self, other: GnnDecodeRunStats) -> None:
        self.gnn_batches += other.gnn_batches
        self.decode_time_ms.extend(other.decode_time_ms)
        self.combo_search_size.extend(other.combo_search_size)
        self.intra_batch_platform_collisions.extend(other.intra_batch_platform_collisions)
        self.chosen_queue_minus_min.extend(other.chosen_queue_minus_min)
        self.total_tasks += other.total_tasks
        self.p1_override_count += other.p1_override_count
        self.classic_would_override_count += other.classic_would_override_count
        self.classic_only_count += other.classic_only_count
        self.gnn_queue_when_p1_override.extend(other.gnn_queue_when_p1_override)
        self.final_queue_when_p1_override.extend(other.final_queue_when_p1_override)
        self.min_queue_when_p1_override.extend(other.min_queue_when_p1_override)
        self.gnn_queue_when_classic_only.extend(other.gnn_queue_when_classic_only)
        self.gnn_queue_all.extend(other.gnn_queue_all)
        self.final_queue_all.extend(other.final_queue_all)

    @staticmethod
    def _mean(values: Sequence[int]) -> float:
        return sum(values) / len(values) if values else 0.0

    @staticmethod
    def _median(values: Sequence[int]) -> float:
        if not values:
            return 0.0
        s = sorted(values)
        mid = len(s) // 2
        if len(s) % 2:
            return float(s[mid])
        return (s[mid - 1] + s[mid]) / 2.0

    @staticmethod
    def _mean_float(values: Sequence[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    @staticmethod
    def _p95(values: Sequence[float]) -> float:
        if not values:
            return 0.0
        s = sorted(values)
        return float(s[min(len(s) - 1, int(0.95 * len(s)))])

    def summary(self, *, p1_margin: int = 1) -> Dict[str, Any]:
        total = max(1, self.total_tasks)
        p1_rate = self.p1_override_count / total
        classic_rate = self.classic_would_override_count / total
        classic_only_rate = self.classic_only_count / total

        gnn_on_p1 = self.gnn_queue_when_p1_override
        final_on_p1 = self.final_queue_when_p1_override
        queue_saved = [g - f for g, f in zip(gnn_on_p1, final_on_p1)] if gnn_on_p1 else []

        collision_batches = sum(1 for c in self.intra_batch_platform_collisions if c > 0)
        batch_n = max(1, self.gnn_batches)

        return {
            "decode_mode": self.decode_mode,
            "top_k": int(self.top_k),
            "gnn_batches": self.gnn_batches,
            "decode_time_ms": {
                "mean": round(self._mean_float(self.decode_time_ms), 4),
                "p95": round(self._p95(self.decode_time_ms), 4),
                "total": round(sum(self.decode_time_ms), 2),
            },
            "combo_search_size": {
                "mean": round(self._mean(self.combo_search_size), 2),
                "max": max(self.combo_search_size) if self.combo_search_size else 0,
            },
            "intra_batch_platform_collisions": {
                "total": sum(self.intra_batch_platform_collisions),
                "batches_with_collision": collision_batches,
                "collision_batch_rate": round(collision_batches / batch_n, 6),
            },
            "chosen_queue_vs_min": {
                "mean": round(self._mean(self.chosen_queue_minus_min), 3),
                "median": round(self._median(self.chosen_queue_minus_min), 3),
                "p95": round(self._p95([float(v) for v in self.chosen_queue_minus_min]), 3),
            },
            "p1_margin": int(p1_margin),
            "total_decode_tasks": self.total_tasks,
            "p1_override_count": self.p1_override_count,
            "p1_override_rate": round(p1_rate, 6),
            "classic_would_override_count": self.classic_would_override_count,
            "classic_would_override_rate": round(classic_rate, 6),
            "classic_only_count": self.classic_only_count,
            "classic_only_rate": round(classic_only_rate, 6),
            "gnn_kept_count": self.total_tasks - self.p1_override_count,
            "queue_on_p1_override": {
                "gnn_mean": round(self._mean(gnn_on_p1), 3),
                "gnn_median": round(self._median(gnn_on_p1), 3),
                "final_mean": round(self._mean(final_on_p1), 3),
                "final_median": round(self._median(final_on_p1), 3),
                "min_mean": round(self._mean(self.min_queue_when_p1_override), 3),
                "saved_mean": round(self._mean(queue_saved), 3),
            },
            "queue_classic_only": {
                "gnn_mean": round(self._mean(self.gnn_queue_when_classic_only), 3),
                "gnn_median": round(self._median(self.gnn_queue_when_classic_only), 3),
                "count": self.classic_only_count,
            },
            "queue_all_tasks": {
                "gnn_mean": round(self._mean(self.gnn_queue_all), 3),
                "final_mean": round(self._mean(self.final_queue_all), 3),
            },
        }

    def to_dict(self, *, p1_margin: int = 1) -> Dict[str, Any]:
        return self.summary(p1_margin=p1_margin)


# Backward-compatible alias
SeqblendDecodeStats = GnnDecodeRunStats

# Accumulator merged across batches in one simulation run.
_RUN_STATS: Optional[GnnDecodeRunStats] = None


def reset_run_decode_stats() -> GnnDecodeRunStats:
    global _RUN_STATS
    _RUN_STATS = GnnDecodeRunStats()
    return _RUN_STATS


def get_run_decode_stats() -> Optional[GnnDecodeRunStats]:
    return _RUN_STATS


def write_run_decode_stats(path: Path, *, p1_margin: int = 1) -> Optional[Dict[str, Any]]:
    if _RUN_STATS is None or _RUN_STATS.gnn_batches == 0:
        return None
    payload = _RUN_STATS.to_dict(p1_margin=p1_margin)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
    return payload


def combo_search_size_for_mode(
    decode_mode: str,
    n_tasks: int,
    *,
    top_k: int = 10,
    per_task_branching: Optional[Sequence[int]] = None,
) -> int:
    """Effective combo search space size for this decode mode and batch."""
    if decode_mode in ("frozen_topk", "topk", "topk_joint"):
        if per_task_branching:
            size = 1
            for b in per_task_branching:
                size *= max(1, int(b))
            return size
        return max(1, top_k) ** max(1, n_tasks)
    return 1


def intra_batch_collision_count(combo: PlacementCombo) -> int:
    """Number of tasks beyond unique platforms (0 = all distinct)."""
    if not combo:
        return 0
    return max(0, len(combo) - len(set(combo)))


def queue_regret_for_combo(
    combo: PlacementCombo,
    task_logit_to_placement: Mapping[int, Sequence[Tuple[int, int]]],
    queue_snapshot: Optional[Mapping[str, int]],
    task_logit_to_queue_key: Optional[Mapping[int, Sequence[str]]],
    *,
    roll_forward: bool = False,
) -> List[int]:
    """Per-task chosen_queue - min_queue among candidates at decode time."""
    if not combo or not queue_snapshot:
        return []

    live_queues: Dict[str, int] = {str(k): int(v) for k, v in queue_snapshot.items()}
    keys_map = task_logit_to_queue_key or {}
    regrets: List[int] = []

    for t_idx, placement in enumerate(combo):
        if t_idx not in task_logit_to_placement:
            continue
        candidates = task_logit_to_placement[t_idx]
        keys = _queue_keys_for_task(t_idx, candidates, keys_map)
        queues = _candidate_queues(keys, live_queues)
        if not queues:
            continue
        min_q = min(queues)
        try:
            chosen_idx = candidates.index(tuple(placement))
        except ValueError:
            regrets.append(0)
            continue
        regrets.append(int(queues[chosen_idx]) - min_q)
        if roll_forward:
            live_queues[keys[chosen_idx]] = live_queues.get(keys[chosen_idx], 0) + 1

    return regrets


def record_decode_batch(
    stats: GnnDecodeRunStats,
    *,
    combo: PlacementCombo,
    decode_mode: str,
    decode_time_ms: float,
    combo_search_size: int,
    task_logit_to_placement: Mapping[int, Sequence[Tuple[int, int]]],
    queue_snapshot: Optional[Mapping[str, int]] = None,
    task_logit_to_queue_key: Optional[Mapping[int, Sequence[str]]] = None,
    roll_forward: bool = False,
    top_k: int = 0,
    count_tasks: bool = True,
) -> None:
    stats.gnn_batches += 1
    stats.decode_time_ms.append(float(decode_time_ms))
    stats.combo_search_size.append(int(combo_search_size))
    stats.intra_batch_platform_collisions.append(intra_batch_collision_count(combo))
    stats.chosen_queue_minus_min.extend(
        queue_regret_for_combo(
            combo,
            task_logit_to_placement,
            queue_snapshot,
            task_logit_to_queue_key,
            roll_forward=roll_forward,
        )
    )
    if count_tasks:
        stats.total_tasks += len(combo)
    if not stats.decode_mode:
        stats.decode_mode = decode_mode
    if top_k > 0:
        stats.top_k = top_k


def _queue_keys_for_task(
    t_idx: int,
    candidates: Sequence[Tuple[int, int]],
    keys_map: Mapping[int, Sequence[str]],
) -> List[str]:
    keys = keys_map.get(t_idx)
    if keys and len(keys) == len(candidates):
        return [str(k) for k in keys]
    return [f"unknown:{plat_id}" for _, plat_id in candidates]


def _candidate_queues(keys: Sequence[str], live_queues: Mapping[str, int]) -> List[int]:
    return [int(live_queues.get(str(k), 0)) for k in keys]


def seqblend_chosen_idx(
    gnn_idx: int,
    keys: Sequence[str],
    live_queues: Mapping[str, int],
    queue_margin: int,
) -> int:
    """Override GNN pick with min-queue only if live queue exceeds min by more than margin."""
    queues = _candidate_queues(keys, live_queues)
    min_q = min(queues)
    if queues[gnn_idx] > min_q + queue_margin:
        return min(range(len(keys)), key=lambda i: (queues[i], str(keys[i])))
    return gnn_idx


def logit_queue_blend_idx(
    logits_t: Tensor,
    keys: Sequence[str],
    live_queues: Mapping[str, int],
    lam: float,
) -> int:
    """Continuous log-scale queue penalty: Score = Logit - lam * log1p(queue).

    Bridges the training/live queue regime gap: at training range (queue 0–6),
    penalty = lam * log1p(6) ≈ 1.9*lam. At live range (queue 2857), penalty =
    lam * log1p(2857) ≈ 7.96*lam. With lam=1.5 this puts the max live penalty
    (~11.9) in the same ballpark as the ranking margin cap (8.0), letting the
    queue signal override sharpened logits without destroying network/type signal.
    """
    queues = _candidate_queues(keys, live_queues)
    scores = [
        float(logits_t[i].item()) - lam * math.log1p(queues[i])
        for i in range(len(keys))
    ]
    return max(range(len(keys)), key=lambda i: scores[i])


def queue_filter_chosen_idx(
    logits_t: Tensor,
    keys: Sequence[str],
    live_queues: Mapping[str, int],
    max_delta: int,
) -> Optional[int]:
    """Choose best-logit platform among queues within min_queue + max_delta."""
    if max_delta < 0:
        return None
    queues = _candidate_queues(keys, live_queues)
    if not queues:
        return None
    min_q = min(queues)
    allowed = [i for i, q in enumerate(queues) if q <= min_q + max_delta]
    if not allowed:
        return None
    return max(allowed, key=lambda i: float(logits_t[i].item()))


def decode_sequential_placement(
    logits_per_task: Sequence[Tensor],
    task_logit_to_placement: Mapping[int, Sequence[Tuple[int, int]]],
    n_tasks: int,
    queue_snapshot: Optional[Mapping[str, int]] = None,
    task_logit_to_queue_key: Optional[Mapping[int, Sequence[str]]] = None,
    *,
    seqblend: bool = False,
    queue_margin: int = 1,
    queue_filter_max_delta: Optional[int] = None,
    lqb_lambda: Optional[float] = None,
    stats: Optional[GnnDecodeRunStats] = None,
) -> Optional[PlacementCombo]:
    """Sequential decode with live queue roll-forward; optional seqblend override."""
    if len(logits_per_task) != n_tasks:
        return None

    live_queues: Dict[str, int] = {
        str(k): int(v) for k, v in (queue_snapshot or {}).items()
    }
    keys_map = task_logit_to_queue_key or {}
    combo_list: List[Tuple[int, int]] = []
    batch_stats = GnnDecodeRunStats() if stats is not None else None

    for t_idx in range(n_tasks):
        if t_idx not in task_logit_to_placement:
            return None
        logits_t = logits_per_task[t_idx]
        if logits_t.numel() == 0:
            return None

        candidates = task_logit_to_placement[t_idx]
        gnn_idx = int(logits_t.argmax().item())
        if gnn_idx >= len(candidates):
            return None

        keys = _queue_keys_for_task(t_idx, candidates, keys_map)
        queues = _candidate_queues(keys, live_queues)
        min_q = min(queues)
        gnn_q = queues[gnn_idx]

        chosen_idx = gnn_idx
        if lqb_lambda is not None:
            chosen_idx = logit_queue_blend_idx(logits_t, keys, live_queues, lqb_lambda)
        elif queue_filter_max_delta is not None:
            filtered_idx = queue_filter_chosen_idx(
                logits_t, keys, live_queues, int(queue_filter_max_delta)
            )
            if filtered_idx is not None:
                chosen_idx = filtered_idx
        elif seqblend:
            chosen_idx = seqblend_chosen_idx(gnn_idx, keys, live_queues, queue_margin)

        final_q = queues[chosen_idx]
        if batch_stats is not None and seqblend:
            batch_stats.record_task(gnn_q, final_q, min_q, p1_margin=queue_margin)

        chosen_key = keys[chosen_idx]
        node_id, plat_id = candidates[chosen_idx]
        combo_list.append((int(node_id), int(plat_id)))
        live_queues[chosen_key] = live_queues.get(chosen_key, 0) + 1

    if stats is not None and batch_stats is not None:
        stats.merge(batch_stats)

    return tuple(combo_list)


def _per_task_topk_branching(
    logits_per_task: Sequence[Tensor],
    task_logit_to_placement: Mapping[int, Sequence[Tuple[int, int]]],
    n_tasks: int,
    top_k: int,
) -> List[int]:
    branching: List[int] = []
    for t_idx in range(n_tasks):
        if t_idx not in task_logit_to_placement:
            continue
        logits_t = logits_per_task[t_idx]
        candidates = task_logit_to_placement[t_idx]
        branching.append(min(top_k, logits_t.numel(), len(candidates)))
    return branching


def decode_frozen_argmax_placement(
    logits_per_task: Sequence[Tensor],
    task_logit_to_placement: Mapping[int, Sequence[Tuple[int, int]]],
    n_tasks: int,
) -> Optional[PlacementCombo]:
    """Per-task argmax from one inference pass; no queue roll-forward between tasks."""
    if len(logits_per_task) != n_tasks:
        return None

    combo_list: List[Tuple[int, int]] = []
    for t_idx in range(n_tasks):
        if t_idx not in task_logit_to_placement:
            return None
        logits_t = logits_per_task[t_idx]
        if logits_t.numel() == 0:
            return None
        candidates = task_logit_to_placement[t_idx]
        chosen_idx = int(logits_t.argmax().item())
        if chosen_idx >= len(candidates):
            return None
        node_id, plat_id = candidates[chosen_idx]
        combo_list.append((int(node_id), int(plat_id)))
    return tuple(combo_list)


def decode_frozen_topk_joint_placement(
    logits_per_task: Sequence[Tensor],
    task_logit_to_placement: Mapping[int, Sequence[Tuple[int, int]]],
    n_tasks: int,
    top_k: int = 10,
) -> Optional[PlacementCombo]:
    """Joint top-k decode from one snapshot: pick the combo with highest summed logits."""
    if len(logits_per_task) != n_tasks or top_k <= 0:
        return None

    choices: List[List[Tuple[float, Tuple[int, int]]]] = []
    for t_idx in range(n_tasks):
        if t_idx not in task_logit_to_placement:
            return None
        logits_t = logits_per_task[t_idx]
        if logits_t.numel() == 0:
            return None
        candidates = task_logit_to_placement[t_idx]
        k = min(top_k, logits_t.numel(), len(candidates))
        values, indices = logits_t.float().topk(k)
        choices.append([
            (float(val.item()), tuple(candidates[int(idx.item())]))
            for val, idx in zip(values, indices)
        ])

    best_combo: Optional[PlacementCombo] = None
    best_score = float("-inf")
    for product in itertools.product(*choices):
        score = sum(item[0] for item in product)
        if score > best_score:
            best_score = score
            best_combo = tuple(item[1] for item in product)
    return best_combo


def run_decode_with_timing(
    decode_mode: str,
    logits_per_task: Sequence[Tensor],
    task_logit_to_placement: Mapping[int, Sequence[Tuple[int, int]]],
    n_tasks: int,
    *,
    queue_snapshot: Optional[Mapping[str, int]] = None,
    task_logit_to_queue_key: Optional[Mapping[int, Sequence[str]]] = None,
    seqblend: bool = False,
    queue_margin: int = 1,
    queue_filter_max_delta: Optional[int] = None,
    lqb_lambda: Optional[float] = None,
    top_k: int = 10,
    stats: Optional[GnnDecodeRunStats] = None,
) -> Optional[PlacementCombo]:
    """Run decode for the requested mode and record batch instrumentation."""
    t0 = time.perf_counter()
    combo: Optional[PlacementCombo] = None
    branching: Optional[List[int]] = None

    if decode_mode in ("frozen", "frozen_argmax"):
        combo = decode_frozen_argmax_placement(
            logits_per_task, task_logit_to_placement, n_tasks
        )
    elif decode_mode in ("frozen_topk", "topk", "topk_joint"):
        branching = _per_task_topk_branching(
            logits_per_task, task_logit_to_placement, n_tasks, top_k
        )
        combo = decode_frozen_topk_joint_placement(
            logits_per_task, task_logit_to_placement, n_tasks, top_k=top_k
        )
    else:
        if lqb_lambda is None:
            lqb_env = os.environ.get("GNN_LQB_LAMBDA", "").strip()
            lqb_lambda = float(lqb_env) if lqb_env else None
        if queue_filter_max_delta is None:
            qf_env = int(os.environ.get("GNN_QUEUE_FILTER_MAX_DELTA", "-1"))
            queue_filter_max_delta = qf_env if qf_env >= 0 else None
        combo = decode_sequential_placement(
            logits_per_task,
            task_logit_to_placement,
            n_tasks,
            queue_snapshot,
            task_logit_to_queue_key,
            seqblend=seqblend,
            queue_margin=queue_margin,
            queue_filter_max_delta=queue_filter_max_delta,
            lqb_lambda=lqb_lambda,
            stats=stats,
        )

    decode_time_ms = (time.perf_counter() - t0) * 1000.0
    if combo is None or stats is None:
        return combo

    search_size = combo_search_size_for_mode(
        decode_mode,
        n_tasks,
        top_k=top_k,
        per_task_branching=branching,
    )
    roll_forward = decode_mode not in ("frozen", "frozen_argmax", "frozen_topk", "topk", "topk_joint")
    count_tasks = decode_mode in ("frozen", "frozen_argmax", "frozen_topk", "topk", "topk_joint")
    record_decode_batch(
        stats,
        combo=combo,
        decode_mode=decode_mode,
        decode_time_ms=decode_time_ms,
        combo_search_size=search_size,
        task_logit_to_placement=task_logit_to_placement,
        queue_snapshot=queue_snapshot,
        task_logit_to_queue_key=task_logit_to_queue_key,
        roll_forward=roll_forward,
        top_k=top_k if decode_mode in ("frozen_topk", "topk", "topk_joint") else 0,
        count_tasks=count_tasks,
    )
    return combo
