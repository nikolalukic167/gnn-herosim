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


def uniq_platform_chosen_idx(
    logits_t: Tensor,
    candidates: Sequence[Tuple[int, int]],
    used_placements: set[Tuple[int, int]],
) -> int:
    """Per-task argmax excluding platforms already picked earlier in the batch."""
    import torch

    if logits_t.numel() == 0:
        raise RuntimeError("uniq_platform decode: empty logits for task.")
    masked = logits_t.clone()
    for i, placement in enumerate(candidates):
        if tuple(placement) in used_placements:
            masked[i] = float("-inf")
    if not torch.isfinite(masked).any():
        raise RuntimeError(
            "uniq_platform decode: no unused platform among candidates "
            f"(candidates={len(candidates)}, used_in_batch={len(used_placements)})."
        )
    return int(masked.argmax().item())


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
    uniq_platform: bool = False,
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
    used_placements: set[Tuple[int, int]] = set()

    for t_idx in range(n_tasks):
        if t_idx not in task_logit_to_placement:
            return None
        logits_t = logits_per_task[t_idx]
        if logits_t.numel() == 0:
            return None

        candidates = task_logit_to_placement[t_idx]
        if uniq_platform:
            chosen_idx = uniq_platform_chosen_idx(logits_t, candidates, used_placements)
        else:
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

        if chosen_idx >= len(candidates):
            return None

        keys = _queue_keys_for_task(t_idx, candidates, keys_map)
        chosen_key = keys[chosen_idx]
        node_id, plat_id = candidates[chosen_idx]
        placement = (int(node_id), int(plat_id))
        combo_list.append(placement)
        used_placements.add(placement)
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


def _task_logit_mapping(graph: Any) -> Optional[Mapping[int, Sequence[Tuple[int, int]]]]:
    mapping = getattr(graph, "task_logit_to_placement", None)
    if mapping is None:
        mapping = getattr(graph, "_task_logit_to_placement", None)
    return mapping


def _queue_norm_for_reforward(live_queues: Mapping[str, int], meta: Mapping[str, Mapping[str, Any]]) -> float:
    from src.policy.tabular.feature_builder import _scheduler_adaptive_queue_norm

    queue_values = [int(live_queues.get(str(key), 0)) for key in meta.keys()]
    norm_mode = os.environ.get("GNN_QUEUE_NORM_MODE", "scheduler_adaptive").strip().lower()
    if norm_mode == "adaptive":
        norm_mode = "scheduler_adaptive"
    return _scheduler_adaptive_queue_norm(queue_values, norm_mode)


def _refresh_queue_dependent_platform_features(
    graph: Any,
    live_queues: Mapping[str, int],
    meta: Mapping[str, Mapping[str, Any]],
) -> None:
    import torch

    platform_features = graph.platform_features
    feat_dim = int(platform_features.size(-1))
    layout = os.environ.get("INFERENCE_FEATURE_LAYOUT", "dim22").strip().lower()
    ce_reduced = layout in ("ce_reduced", "reduced_ce", "reduced1060")
    atomic21 = layout in ("atomic21", "21")
    if feat_dim < 6:
        raise RuntimeError(
            f"seq_reforward expects >=6-dim platform features; got {feat_dim}."
        )
    if not ce_reduced and feat_dim < 14:
        raise RuntimeError(
            f"seq_reforward expects >=14-dim platform features; got {feat_dim}."
        )
    queue_norm = 1.0 if atomic21 else _queue_norm_for_reforward(live_queues, meta)
    n_platforms = int(getattr(graph, "n_platforms", platform_features.size(0)))

    for queue_key, info in meta.items():
        if "platform_pos" not in info:
            raise RuntimeError(f"seq_reforward metadata missing platform_pos for {queue_key}.")
        pos = int(info["platform_pos"])
        if pos < 0 or pos >= n_platforms:
            raise RuntimeError(f"seq_reforward platform_pos out of range for {queue_key}: {pos}.")
        raw_q = float(live_queues.get(str(queue_key), 0))
        if ce_reduced:
            platform_features[pos, 5] = raw_q / float(queue_norm)
        elif atomic21:
            platform_features[pos, 7] = raw_q
        else:
            target_concurrency = max(float(info.get("target_concurrency", 1.0)), 1e-9)
            platform_features[pos, 7] = raw_q / float(queue_norm)
            platform_features[pos, 13] = (raw_q / target_concurrency) / 5.0


def decode_sequential_reforward_placement(
    model: Any,
    graph: Any,
    n_tasks: int,
    queue_snapshot: Optional[Mapping[str, int]] = None,
    *,
    stats: Optional[GnnDecodeRunStats] = None,
) -> Optional[PlacementCombo]:
    """Per-task argmax with queue-feature refresh + full GNN re-forward between tasks."""
    import torch

    mapping = _task_logit_mapping(graph)
    keys_map = getattr(graph, "task_logit_to_queue_key", None) or getattr(
        graph, "_task_logit_to_queue_key", None
    )
    meta = getattr(graph, "queue_key_to_platform_meta", None)
    snapshot = queue_snapshot if queue_snapshot is not None else getattr(graph, "queue_snapshot", None)
    if not mapping or not keys_map or not meta or snapshot is None:
        raise RuntimeError(
            "seq_reforward requires task_logit_to_placement, task_logit_to_queue_key, "
            "queue_key_to_platform_meta, and queue_snapshot on the inference graph."
        )

    live_queues: Dict[str, int] = {str(k): int(v) for k, v in dict(snapshot).items()}
    combo_list: List[Tuple[int, int]] = []
    original_platform_features = graph.platform_features
    graph.platform_features = original_platform_features.clone()

    t0 = time.perf_counter()
    try:
        _refresh_queue_dependent_platform_features(graph, live_queues, meta)
        with torch.no_grad():
            for task_idx in range(n_tasks):
                if task_idx not in mapping or task_idx not in keys_map:
                    return None
                logits_per_task = model(graph)
                if task_idx >= len(logits_per_task):
                    return None
                logits_t = logits_per_task[task_idx]
                if logits_t.numel() == 0:
                    return None
                chosen_idx = int(logits_t.argmax().item())
                candidates = mapping[task_idx]
                task_keys = keys_map[task_idx]
                if chosen_idx >= len(candidates) or chosen_idx >= len(task_keys):
                    return None
                queue_key = str(task_keys[chosen_idx])
                combo_list.append(tuple(candidates[chosen_idx]))
                live_queues[queue_key] = live_queues.get(queue_key, 0) + 1
                _refresh_queue_dependent_platform_features(graph, live_queues, meta)
    finally:
        graph.platform_features = original_platform_features

    combo = tuple(combo_list)
    if stats is not None:
        decode_time_ms = (time.perf_counter() - t0) * 1000.0
        record_decode_batch(
            stats,
            combo=combo,
            decode_mode="seq_reforward",
            decode_time_ms=decode_time_ms,
            combo_search_size=n_tasks,
            task_logit_to_placement=mapping,
            queue_snapshot=snapshot,
            task_logit_to_queue_key=keys_map,
            roll_forward=True,
            top_k=0,
            count_tasks=False,
        )
    return combo


def _unit_pull_from_platform_row(row: Any, cold_count: float) -> float:
    """Recover T_pull from dim15/dim14, else default."""
    from src.placement.warmth import DEFAULT_T_PULL_S, ESTIMATED_PULL_REMAINING_NORM_S

    if cold_count > 1e-9 and row.numel() > 15:
        # dim15 = (cold_count * T_pull) / NORM
        return float(row[15].item()) * float(ESTIMATED_PULL_REMAINING_NORM_S) / float(
            cold_count
        )
    return float(DEFAULT_T_PULL_S)


def _refresh_pull_dependent_platform_features(
    graph: Any,
    live_queues: Mapping[str, int],
    meta: Mapping[str, Mapping[str, Any]],
    *,
    base_cold_count_by_node: Mapping[str, float],
    pulls_committed: Mapping[str, int],
    unit_pull_by_node: Mapping[str, float],
    n_platforms_by_node: Mapping[str, int],
) -> None:
    """Queue refresh + dim24 pull-ledger roll-forward (node_cold_count / pull_remaining)."""
    from src.placement.warmth import (
        estimated_pull_remaining_sec,
        normalize_estimated_pull_remaining_sec,
    )

    _refresh_queue_dependent_platform_features(graph, live_queues, meta)

    platform_features = graph.platform_features
    feat_dim = int(platform_features.size(-1))
    if feat_dim < 16:
        raise RuntimeError(
            f"seq_reforward_pull requires dim24 platform features (>=16 dims); got {feat_dim}."
        )

    for queue_key, info in meta.items():
        pos = int(info["platform_pos"])
        node_name = str(info["node_name"])
        base_cold = float(base_cold_count_by_node.get(node_name, 0.0))
        committed = int(pulls_committed.get(node_name, 0))
        effective_cold = base_cold + float(committed)
        unit = float(unit_pull_by_node.get(node_name, 0.0))
        pull_rem = estimated_pull_remaining_sec(effective_cold, unit)
        platform_features[pos, 14] = effective_cold
        platform_features[pos, 15] = normalize_estimated_pull_remaining_sec(pull_rem)
        n_col = max(int(n_platforms_by_node.get(node_name, 1)), 1)
        # shared_fate density tracks effective FilterStore depth / co-located plats
        platform_features[pos, 8] = min(1.0, effective_cold / float(n_col))


def decode_sequential_reforward_pull_placement(
    model: Any,
    graph: Any,
    n_tasks: int,
    queue_snapshot: Optional[Mapping[str, int]] = None,
    *,
    platform_needs_pull: Optional[Mapping[str, bool]] = None,
    stats: Optional[GnnDecodeRunStats] = None,
) -> Optional[PlacementCombo]:
    """Phase 1 ablation: CE argmax + queue refresh + pulls_committed ledger + re-forward.

    After each placement, if the chosen platform still needs an image pull
    (``not initialized`` at batch start), increment ``pulls_committed[node]`` and
    rewrite dim24 ``node_cold_count`` / ``estimated_pull_remaining`` (and shared_fate)
    for every platform on that node before the next GNN forward. Matches ect_pull's
    decision-time FilterStore bookkeeping without changing model weights.
    """
    import torch

    mapping = _task_logit_mapping(graph)
    keys_map = getattr(graph, "task_logit_to_queue_key", None) or getattr(
        graph, "_task_logit_to_queue_key", None
    )
    meta = getattr(graph, "queue_key_to_platform_meta", None)
    snapshot = (
        queue_snapshot
        if queue_snapshot is not None
        else getattr(graph, "queue_snapshot", None)
    )
    if not mapping or not keys_map or not meta or snapshot is None:
        raise RuntimeError(
            "seq_reforward_pull requires task_logit_to_placement, task_logit_to_queue_key, "
            "queue_key_to_platform_meta, and queue_snapshot on the inference graph."
        )

    platform_features = graph.platform_features
    if int(platform_features.size(-1)) < 16:
        raise RuntimeError(
            f"seq_reforward_pull requires dim24 (>=16 platform dims); "
            f"got {int(platform_features.size(-1))}. Set INFERENCE_FEATURE_LAYOUT=dim24."
        )

    # Freeze per-platform pull need at batch start (matches ect_pull: no mid-batch init).
    needs_pull: Dict[str, bool] = {}
    for queue_key, info in meta.items():
        qk = str(queue_key)
        if platform_needs_pull is not None and qk in platform_needs_pull:
            needs_pull[qk] = bool(platform_needs_pull[qk])
        elif "initialized" in info:
            needs_pull[qk] = not bool(info["initialized"])
        else:
            raise RuntimeError(
                f"seq_reforward_pull: missing initialized/needs_pull for {qk}. "
                "Rebuild graph via feature_builder (initialized in meta) or pass "
                "platform_needs_pull from the live scheduler."
            )

    base_cold_count_by_node: Dict[str, float] = {}
    unit_pull_by_node: Dict[str, float] = {}
    n_platforms_by_node: Dict[str, int] = {}
    for queue_key, info in meta.items():
        node_name = str(info["node_name"])
        pos = int(info["platform_pos"])
        n_platforms_by_node[node_name] = n_platforms_by_node.get(node_name, 0) + 1
        if node_name not in base_cold_count_by_node:
            cold = float(platform_features[pos, 14].item())
            base_cold_count_by_node[node_name] = cold
            unit_pull_by_node[node_name] = _unit_pull_from_platform_row(
                platform_features[pos], cold
            )

    live_queues: Dict[str, int] = {str(k): int(v) for k, v in dict(snapshot).items()}
    pulls_committed: Dict[str, int] = {}
    combo_list: List[Tuple[int, int]] = []
    original_platform_features = graph.platform_features
    graph.platform_features = original_platform_features.clone()

    t0 = time.perf_counter()
    try:
        _refresh_pull_dependent_platform_features(
            graph,
            live_queues,
            meta,
            base_cold_count_by_node=base_cold_count_by_node,
            pulls_committed=pulls_committed,
            unit_pull_by_node=unit_pull_by_node,
            n_platforms_by_node=n_platforms_by_node,
        )
        with torch.no_grad():
            for task_idx in range(n_tasks):
                if task_idx not in mapping or task_idx not in keys_map:
                    return None
                logits_per_task = model(graph)
                if task_idx >= len(logits_per_task):
                    return None
                logits_t = logits_per_task[task_idx]
                if logits_t.numel() == 0:
                    return None
                chosen_idx = int(logits_t.argmax().item())
                candidates = mapping[task_idx]
                task_keys = keys_map[task_idx]
                if chosen_idx >= len(candidates) or chosen_idx >= len(task_keys):
                    return None
                queue_key = str(task_keys[chosen_idx])
                node_id, plat_id = candidates[chosen_idx]
                combo_list.append((int(node_id), int(plat_id)))
                live_queues[queue_key] = live_queues.get(queue_key, 0) + 1

                info = meta.get(queue_key)
                if info is None:
                    raise RuntimeError(
                        f"seq_reforward_pull: queue_key {queue_key} missing from meta"
                    )
                node_name = str(info["node_name"])
                if needs_pull.get(queue_key, False):
                    pulls_committed[node_name] = int(pulls_committed.get(node_name, 0)) + 1

                _refresh_pull_dependent_platform_features(
                    graph,
                    live_queues,
                    meta,
                    base_cold_count_by_node=base_cold_count_by_node,
                    pulls_committed=pulls_committed,
                    unit_pull_by_node=unit_pull_by_node,
                    n_platforms_by_node=n_platforms_by_node,
                )
    finally:
        graph.platform_features = original_platform_features

    combo = tuple(combo_list)
    if stats is not None:
        decode_time_ms = (time.perf_counter() - t0) * 1000.0
        record_decode_batch(
            stats,
            combo=combo,
            decode_mode="seq_reforward_pull",
            decode_time_ms=decode_time_ms,
            combo_search_size=n_tasks,
            task_logit_to_placement=mapping,
            queue_snapshot=snapshot,
            task_logit_to_queue_key=keys_map,
            roll_forward=True,
            top_k=0,
            count_tasks=False,
        )
    return combo


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
    uniq_platform: bool = False,
    top_k: int = 10,
    stats: Optional[GnnDecodeRunStats] = None,
) -> Optional[PlacementCombo]:
    """Run decode for the requested mode and record batch instrumentation."""
    t0 = time.perf_counter()
    combo: Optional[PlacementCombo] = None
    branching: Optional[List[int]] = None
    effective_mode = decode_mode

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
        if uniq_platform or decode_mode in ("argmax_uniq", "uniq_platform", "uniq"):
            effective_mode = "argmax_uniq"
            uniq_platform = True
        if lqb_lambda is None:
            lqb_env = os.environ.get("GNN_LQB_LAMBDA", "").strip()
            lqb_lambda = float(lqb_env) if lqb_env else None
        if queue_filter_max_delta is None:
            qf_env = int(os.environ.get("GNN_QUEUE_FILTER_MAX_DELTA", "-1"))
            queue_filter_max_delta = qf_env if qf_env >= 0 else None
        if uniq_platform:
            lqb_lambda = None
            queue_filter_max_delta = None
            seqblend = False
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
            uniq_platform=uniq_platform,
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
        decode_mode=effective_mode,
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
