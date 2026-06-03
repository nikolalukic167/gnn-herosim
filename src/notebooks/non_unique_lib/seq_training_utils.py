"""Shared sequential training: prefix labels, sequential argmax decode, queue roll-forward."""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor

PlacementCombo = Tuple[Tuple[int, int], ...]
ComboRTT = Tuple[PlacementCombo, float]


def optimal_combo_from_tasks(
    task_rows: Sequence[Tuple[int, Optional[int], Optional[int]]],
) -> PlacementCombo:
    """task_rows: (task_id, optimal_node_id, optimal_platform_id) sorted by task_id."""
    out: List[Tuple[int, int]] = []
    for _tid, node_id, plat_id in task_rows:
        if node_id is None or plat_id is None:
            continue
        out.append((int(node_id), int(plat_id)))
    return tuple(out)


def prefix_optimal_placement_at_step(
    combos: Sequence[ComboRTT],
    prefix: PlacementCombo,
    step: int,
) -> Optional[Tuple[int, int]]:
    """Min-RTT choice for task `step` among combos agreeing on `prefix` (tasks 0..step-1)."""
    if step < 0:
        return None
    best_rtt = float("inf")
    best: Optional[Tuple[int, int]] = None
    for combo, rtt in combos:
        if len(combo) <= step:
            continue
        if combo[:step] != prefix:
            continue
        if float(rtt) < best_rtt:
            best_rtt = float(rtt)
            best = combo[step]
    return best


def logit_index_for_placement(
    task_logit_to_placement: Mapping[int, Sequence[Tuple[int, int]]],
    task_idx: int,
    node_id: int,
    plat_id: int,
) -> Optional[int]:
    placements = task_logit_to_placement.get(task_idx)
    if not placements:
        return None
    target = (int(node_id), int(plat_id))
    for idx, placement in enumerate(placements):
        if tuple(placement) == target:
            return idx
    return None


def apply_prefix_optimal_labels(
    y: Tensor,
    ce_label_mask: Tensor,
    task_logit_to_placement: Mapping[int, Sequence[Tuple[int, int]]],
    optimal_combo: PlacementCombo,
    combos: Sequence[ComboRTT],
) -> int:
    """Overwrite active CE label slots with prefix-optimal class indices. Returns count updated."""
    updated = 0
    n_tasks = int(y.numel())
    for task_idx in range(n_tasks):
        if not bool(ce_label_mask[task_idx].item()):
            continue
        prefix = optimal_combo[:task_idx]
        choice = prefix_optimal_placement_at_step(combos, prefix, task_idx)
        if choice is None:
            continue
        logit_idx = logit_index_for_placement(
            task_logit_to_placement, task_idx, choice[0], choice[1]
        )
        if logit_idx is None:
            continue
        y[task_idx] = int(logit_idx)
        updated += 1
    return updated


def decode_sequential_argmax_placement(
    logits_per_task: Sequence[Tensor],
    task_logit_to_placement: Mapping[int, Sequence[Tuple[int, int]]],
    n_tasks: int,
    queue_snapshot: Optional[Mapping[str, int]] = None,
    task_logit_to_queue_key: Optional[Mapping[int, Sequence[str]]] = None,
) -> Optional[PlacementCombo]:
    """Sequential argmax with live queue updates (no shortest-queue override)."""
    if len(logits_per_task) != n_tasks:
        return None

    live_queues: Dict[str, int] = {
        str(k): int(v) for k, v in (queue_snapshot or {}).items()
    }
    keys_map = task_logit_to_queue_key or {}
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

        keys = keys_map.get(t_idx)
        if keys and len(keys) == len(candidates):
            chosen_key = keys[chosen_idx]
        else:
            chosen_key = f"unknown:{candidates[chosen_idx][1]}"

        node_id, plat_id = candidates[chosen_idx]
        combo_list.append((int(node_id), int(plat_id)))
        live_queues[chosen_key] = live_queues.get(chosen_key, 0) + 1

    return tuple(combo_list)


def is_final_sequential_graph(data) -> bool:
    step = getattr(data, "seq_step", None)
    n_tasks = getattr(data, "seq_n_tasks", None)
    if step is None or n_tasks is None:
        return True
    return int(step) == int(n_tasks) - 1


def initial_queue_snapshot_for_graph(data) -> Dict[str, int]:
    snap = getattr(data, "initial_queue_snapshot", None)
    if isinstance(snap, dict) and snap:
        return {str(k): int(v) for k, v in snap.items()}
    snap = getattr(data, "queue_snapshot", None)
    if isinstance(snap, dict):
        return {str(k): int(v) for k, v in snap.items()}
    return {}
