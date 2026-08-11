from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data


ExactRttEntry = Tuple[List[int], float]
ExactRttLookupMap = Dict[str, List[ExactRttEntry]]


def _dataset_id(data: Data) -> str:
    parent_id = getattr(data, "parent_dataset_id", None)
    dataset_id = parent_id if parent_id else getattr(data, "dataset_id", "")
    s = str(dataset_id or "")
    for sep in ("@os", "@seq"):
        idx = s.find(sep)
        if idx >= 0:
            s = s[:idx]
    return s


def soft_combo_ce_loss(
    logits_per_task: Sequence[Tensor],
    data: Data,
    exact_rtt_map: ExactRttLookupMap,
    *,
    tau: float,
    max_combos: int = 4096,
) -> Tuple[Tensor, int, Dict[str, float]]:
    """Soft CE over exact-RTT placement combos.

    Each combo score is the additive sum of per-task placement logits. The target
    distribution is a softmax over negative combo regret from the sidecar.
    """
    if not logits_per_task:
        device = torch.device("cpu")
    else:
        device = logits_per_task[0].device
    zero = torch.zeros((), device=device)
    dataset_id = _dataset_id(data)
    entries = exact_rtt_map.get(dataset_id, [])
    if not entries:
        return zero, 0, {}

    usable = entries[: max(1, int(max_combos))]
    scores: List[Tensor] = []
    rtts: List[float] = []
    for indices, rtt in usable:
        if len(indices) != len(logits_per_task):
            continue
        score = zero
        ok = True
        for task_idx, logit_idx in enumerate(indices):
            logits_t = logits_per_task[task_idx]
            if logit_idx < 0 or logit_idx >= logits_t.numel():
                ok = False
                break
            score = score + logits_t[int(logit_idx)]
        if ok:
            scores.append(score)
            rtts.append(float(rtt))

    if not scores:
        return zero, 0, {}

    scores_t = torch.stack(scores)
    rtt_t = torch.tensor(rtts, dtype=torch.float32, device=device)
    regret_t = rtt_t - torch.min(rtt_t)
    tau_safe = max(float(tau), 1e-6)
    target = torch.softmax(-regret_t / tau_safe, dim=0).detach()
    log_probs = F.log_softmax(scores_t.float(), dim=0)
    loss = -(target * log_probs).sum()

    with torch.no_grad():
        model_probs = torch.softmax(scores_t.float(), dim=0)
        target_entropy = -(target * torch.log(target.clamp_min(1e-12))).sum()
        model_entropy = -(model_probs * torch.log(model_probs.clamp_min(1e-12))).sum()
        best_idx = int(torch.argmax(model_probs).item())
        stats = {
            "combos": float(len(scores)),
            "target_entropy": float(target_entropy.item()),
            "model_entropy": float(model_entropy.item()),
            "model_regret": float(regret_t[best_idx].item()),
        }
    return loss, 1, stats


def _task_queue_keys(data: Data, task_idx: int, num_logits: int) -> List[str]:
    keys_map = getattr(data, "task_logit_to_queue_key", None) or {}
    keys = keys_map.get(task_idx) if isinstance(keys_map, Mapping) else None
    if keys is not None and len(keys) == num_logits:
        return [str(k) for k in keys]

    placement_map = getattr(
        data,
        "task_logit_to_placement",
        getattr(data, "_task_logit_to_placement", {}),
    )
    placements = placement_map.get(task_idx, []) if isinstance(placement_map, Mapping) else []
    if len(placements) == num_logits:
        return [f"{node_id}:{plat_id}" for node_id, plat_id in placements]
    return [f"task{task_idx}:candidate{i}" for i in range(num_logits)]


def concentration_penalty(
    logits_per_task: Sequence[Tensor],
    data: Data,
    *,
    cap: float,
    baseline_concurrency: float = 5.0,
    require_metadata: bool = True,
) -> Tuple[Tensor, int, Dict[str, float]]:
    """Penalize expected within-batch load on the same physical queue key."""
    if not logits_per_task:
        return torch.zeros(()), 0, {}
    device = logits_per_task[0].device
    queue_meta = getattr(data, "queue_key_to_platform_meta", None)
    if require_metadata and not isinstance(queue_meta, Mapping):
        raise ValueError(
            "Missing queue_key_to_platform_meta on graph; regenerate cache before "
            "using soft_combo_conc."
        )
    expected_load: Dict[str, Tensor] = {}

    for task_idx, logits_t in enumerate(logits_per_task):
        if logits_t.numel() == 0:
            continue
        probs = torch.softmax(logits_t.float(), dim=0)
        keys = _task_queue_keys(data, task_idx, int(logits_t.numel()))
        for key, prob in zip(keys, probs):
            if require_metadata and str(key) not in queue_meta:
                raise ValueError(
                    f"Missing queue metadata for candidate queue key {key!r}; "
                    "cache metadata is incomplete."
                )
            expected_load[key] = expected_load.get(key, torch.zeros((), device=device)) + prob

    if not expected_load:
        return torch.zeros((), device=device), 0, {}

    loads = torch.stack(list(expected_load.values()))
    caps = []
    for key in expected_load:
        meta = queue_meta.get(str(key), {}) if isinstance(queue_meta, Mapping) else {}
        try:
            target_concurrency = float(meta.get("target_concurrency", baseline_concurrency))
        except (TypeError, ValueError):
            if require_metadata:
                raise ValueError(f"Invalid target_concurrency metadata for queue key {key!r}")
            target_concurrency = baseline_concurrency
        if target_concurrency <= 0:
            if require_metadata:
                raise ValueError(f"Non-positive target_concurrency metadata for queue key {key!r}")
            target_concurrency = baseline_concurrency
        adaptive_cap = max(1.0, float(cap) * target_concurrency / max(float(baseline_concurrency), 1e-9))
        caps.append(adaptive_cap)
    caps_t = torch.tensor(caps, dtype=loads.dtype, device=device)
    excess = torch.relu(loads - caps_t)
    penalty = torch.mean(excess * excess)
    with torch.no_grad():
        stats = {
            "max_expected_load": float(torch.max(loads).item()),
            "mean_adaptive_cap": float(torch.mean(caps_t).item()),
            "keys_over_cap": float((loads > caps_t).sum().item()),
        }
    return penalty, 1, stats
