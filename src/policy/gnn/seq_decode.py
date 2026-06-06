"""Sequential GNN decode with optional seqblend min-queue override."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:
    from torch import Tensor
except ImportError:  # pragma: no cover
    Tensor = object  # type: ignore

PlacementCombo = Tuple[Tuple[int, int], ...]


@dataclass
class SeqblendDecodeStats:
    """Per-task decode counters for seqblend+1 vs classic seqblend comparison."""

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

    def merge(self, other: SeqblendDecodeStats) -> None:
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

    def summary(self, *, p1_margin: int = 1) -> Dict[str, Any]:
        total = max(1, self.total_tasks)
        p1_rate = self.p1_override_count / total
        classic_rate = self.classic_would_override_count / total
        classic_only_rate = self.classic_only_count / total

        gnn_on_p1 = self.gnn_queue_when_p1_override
        final_on_p1 = self.final_queue_when_p1_override
        queue_saved = [g - f for g, f in zip(gnn_on_p1, final_on_p1)] if gnn_on_p1 else []

        return {
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


# Accumulator merged across batches in one simulation run.
_RUN_STATS: Optional[SeqblendDecodeStats] = None


def reset_run_decode_stats() -> SeqblendDecodeStats:
    global _RUN_STATS
    _RUN_STATS = SeqblendDecodeStats()
    return _RUN_STATS


def get_run_decode_stats() -> Optional[SeqblendDecodeStats]:
    return _RUN_STATS


def write_run_decode_stats(path: Path, *, p1_margin: int = 1) -> Optional[Dict[str, Any]]:
    if _RUN_STATS is None or _RUN_STATS.total_tasks == 0:
        return None
    payload = _RUN_STATS.to_dict(p1_margin=p1_margin)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
    return payload


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


def decode_sequential_placement(
    logits_per_task: Sequence[Tensor],
    task_logit_to_placement: Mapping[int, Sequence[Tuple[int, int]]],
    n_tasks: int,
    queue_snapshot: Optional[Mapping[str, int]] = None,
    task_logit_to_queue_key: Optional[Mapping[int, Sequence[str]]] = None,
    *,
    seqblend: bool = False,
    queue_margin: int = 1,
    stats: Optional[SeqblendDecodeStats] = None,
) -> Optional[PlacementCombo]:
    """Sequential decode with live queue roll-forward; optional seqblend override."""
    if len(logits_per_task) != n_tasks:
        return None

    live_queues: Dict[str, int] = {
        str(k): int(v) for k, v in (queue_snapshot or {}).items()
    }
    keys_map = task_logit_to_queue_key or {}
    combo_list: List[Tuple[int, int]] = []
    batch_stats = SeqblendDecodeStats() if stats is not None else None

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
        if seqblend:
            chosen_idx = seqblend_chosen_idx(gnn_idx, keys, live_queues, queue_margin)

        final_q = queues[chosen_idx]
        if batch_stats is not None:
            batch_stats.record_task(gnn_q, final_q, min_q, p1_margin=queue_margin)

        chosen_key = keys[chosen_idx]
        node_id, plat_id = candidates[chosen_idx]
        combo_list.append((int(node_id), int(plat_id)))
        live_queues[chosen_key] = live_queues.get(chosen_key, 0) + 1

    if stats is not None and batch_stats is not None:
        stats.merge(batch_stats)

    return tuple(combo_list)
