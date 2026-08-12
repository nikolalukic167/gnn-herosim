"""Phase 3: harvest (state, ect_pull action, soft ECT targets) for policy distillation.

Enabled when ``ECT_PULL_DISTILL_DIR`` points at an output directory. Each decision
writes a PyG ``Data`` frame (dim24, pull-ledger injected) plus a JSONL sidecar row.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING

import torch
from torch_geometric.data import Data

if TYPE_CHECKING:
    from src.placement.infrastructure import Node, Platform, Task
    from src.placement.model import SystemState

_LOCK = threading.Lock()
_TASK_TYPES_CACHE: Optional[Dict[str, Any]] = None
_FRAME_COUNTER = 0


def distill_enabled() -> bool:
    return bool(os.environ.get("ECT_PULL_DISTILL_DIR", "").strip())


def distill_tau() -> float:
    return float(os.environ.get("ECT_PULL_DISTILL_TAU", "0.25"))


def reset_frame_counter(start: int = 0) -> None:
    """Reset harvest frame index (call between multi-seed runs / append resume)."""
    global _FRAME_COUNTER
    if int(start) < 0:
        raise ValueError(f"FAIL LOUD: frame counter start={start} < 0")
    with _LOCK:
        _FRAME_COUNTER = int(start)


def next_frame_index() -> int:
    with _LOCK:
        return int(_FRAME_COUNTER)


def _distill_dir() -> Path:
    raw = os.environ.get("ECT_PULL_DISTILL_DIR", "").strip()
    if not raw:
        raise RuntimeError("ECT_PULL_DISTILL_DIR is empty")
    path = Path(raw)
    path.mkdir(parents=True, exist_ok=True)
    (path / "frames").mkdir(parents=True, exist_ok=True)
    return path


def _load_task_types_data() -> Dict[str, Any]:
    global _TASK_TYPES_CACHE
    if _TASK_TYPES_CACHE is not None:
        return _TASK_TYPES_CACHE
    root = Path(__file__).resolve().parents[3]
    path = root / "data" / "nofs-ids" / "task-types.json"
    if not path.is_file():
        raise FileNotFoundError(f"FAIL LOUD: task-types.json missing at {path}")
    _TASK_TYPES_CACHE = json.loads(path.read_text())
    return _TASK_TYPES_CACHE


def _boltzmann_from_costs(costs: Sequence[float], tau: float) -> List[float]:
    import math

    tau_safe = max(float(tau), 1e-6)
    scaled = [-c / tau_safe for c in costs]
    m = max(scaled)
    exps = [math.exp(v - m) for v in scaled]
    z = sum(exps)
    if z <= 0:
        n = max(1, len(costs))
        return [1.0 / n] * len(costs)
    return [e / z for e in exps]


def _inject_pull_ledger(
    graph: Data,
    pulls_committed: Mapping[str, int],
    queue_snapshot: Mapping[str, int],
) -> None:
    """Rewrite dim24 cold_count/pull_remaining/shared_fate to match seq_reforward_pull."""
    from src.policy.gnn.seq_decode import (
        _refresh_pull_dependent_platform_features,
        _unit_pull_from_platform_row,
    )

    meta = getattr(graph, "queue_key_to_platform_meta", None)
    if not isinstance(meta, dict) or not meta:
        raise RuntimeError("FAIL LOUD: distill harvest graph missing queue_key_to_platform_meta")
    if int(graph.platform_features.size(-1)) < 16:
        raise RuntimeError(
            f"FAIL LOUD: distill requires dim24 (>=16 plat dims); got "
            f"{int(graph.platform_features.size(-1))}"
        )

    base_cold_count_by_node: Dict[str, float] = {}
    unit_pull_by_node: Dict[str, float] = {}
    n_platforms_by_node: Dict[str, int] = {}
    platform_features = graph.platform_features
    for _qk, info in meta.items():
        node_name = str(info["node_name"])
        pos = int(info["platform_pos"])
        n_platforms_by_node[node_name] = n_platforms_by_node.get(node_name, 0) + 1
        if node_name not in base_cold_count_by_node:
            cold = float(platform_features[pos, 14].item())
            base_cold_count_by_node[node_name] = cold
            unit_pull_by_node[node_name] = _unit_pull_from_platform_row(
                platform_features[pos], cold
            )

    _refresh_pull_dependent_platform_features(
        graph,
        queue_snapshot,
        meta,
        base_cold_count_by_node=base_cold_count_by_node,
        pulls_committed=dict(pulls_committed),
        unit_pull_by_node=unit_pull_by_node,
        n_platforms_by_node=n_platforms_by_node,
    )


def maybe_log_ect_pull_decision(
    scheduler: Any,
    *,
    system_state: "SystemState",
    task: "Task",
    candidates: Sequence[Tuple["Node", "Platform"]],
    candidate_ect: Sequence[float],
    chosen: Tuple["Node", "Platform"],
    pulls_committed_before: Mapping[str, int],
) -> None:
    """Dump one dim24 decision frame aligned with the teacher's pre-commit ledger."""
    if not distill_enabled():
        return
    if len(candidates) != len(candidate_ect):
        raise RuntimeError(
            f"FAIL LOUD: candidates={len(candidates)} != ect_scores={len(candidate_ect)}"
        )
    if not candidates:
        raise RuntimeError("FAIL LOUD: empty candidate set for distill harvest")

    os.environ["INFERENCE_FEATURE_LAYOUT"] = "dim24"
    from src.policy.tabular.feature_builder import build_pyg_inference_graph

    queue_snapshot = scheduler._capture_full_queue_snapshot()
    temporal_state = scheduler._capture_temporal_state_for_replicas(list(candidates))
    graph, mapping = build_pyg_inference_graph(
        [task],
        system_state,
        queue_snapshot,
        nodes=list(scheduler.nodes.items),
        task_types_data=_load_task_types_data(),
        queue_norm_mode=os.environ.get("GNN_QUEUE_NORM_MODE", "adaptive"),
        temporal_state=temporal_state,
    )
    if graph is None or mapping is None:
        raise RuntimeError("FAIL LOUD: build_pyg_inference_graph returned None during distill harvest")

    graph.queue_snapshot = dict(queue_snapshot)
    _inject_pull_ledger(graph, pulls_committed_before, queue_snapshot)

    chosen_node, chosen_plat = chosen
    chosen_placement = (int(chosen_node.id), int(chosen_plat.id))
    task_map = mapping.get(0) or mapping.get("0") or []
    if not task_map:
        raise RuntimeError("FAIL LOUD: empty task_logit_to_placement[0] in distill harvest")

    y_idx = None
    for i, placement in enumerate(task_map):
        if tuple(placement) == chosen_placement:
            y_idx = i
            break
    if y_idx is None:
        # Fall back to queue-key match (node_name:plat_id).
        chosen_key = f"{chosen_node.node_name}:{chosen_plat.id}"
        keys = getattr(graph, "task_logit_to_queue_key", {}).get(0, [])
        for i, key in enumerate(keys):
            if str(key) == chosen_key:
                y_idx = i
                break
    if y_idx is None:
        raise RuntimeError(
            f"FAIL LOUD: teacher choice {chosen_placement} not in logit map "
            f"(n_candidates={len(task_map)})"
        )

    # Align soft targets to logit order (not candidate list order).
    ect_by_placement: Dict[Tuple[int, int], float] = {}
    for (node, plat), ect in zip(candidates, candidate_ect):
        ect_by_placement[(int(node.id), int(plat.id))] = float(ect)

    teacher_ect: List[float] = []
    for placement in task_map:
        key = (int(placement[0]), int(placement[1]))
        if key not in ect_by_placement:
            raise RuntimeError(
                f"FAIL LOUD: logit placement {key} missing from teacher ECT map"
            )
        teacher_ect.append(ect_by_placement[key])

    tau = distill_tau()
    soft = _boltzmann_from_costs(teacher_ect, tau)
    graph.y = torch.tensor([int(y_idx)], dtype=torch.long)
    graph.teacher_ect = torch.tensor(teacher_ect, dtype=torch.float32)
    graph.teacher_soft = torch.tensor(soft, dtype=torch.float32)
    graph.distill_tau = float(tau)
    graph.dataset_id = (
        f"ect_pull_distill/{os.environ.get('ECT_PULL_DISTILL_RUN_ID', 'run')}"
        f"/task{int(getattr(task, 'id', -1))}"
    )
    graph.env_now = float(scheduler.env.now)
    graph.pulls_committed = dict(pulls_committed_before)

    global _FRAME_COUNTER
    with _LOCK:
        out = _distill_dir()
        idx = _FRAME_COUNTER
        _FRAME_COUNTER += 1
        frame_path = out / "frames" / f"frame_{idx:06d}.pt"
        torch.save(graph, frame_path)
        row = {
            "frame_idx": idx,
            "frame_path": str(frame_path),
            "env_now": float(scheduler.env.now),
            "task_id": int(getattr(task, "id", -1)),
            "task_type": task.type["name"],
            "y": int(y_idx),
            "chosen_node_id": int(chosen_node.id),
            "chosen_platform_id": int(chosen_plat.id),
            "chosen_queue_key": f"{chosen_node.node_name}:{chosen_plat.id}",
            "n_logits": len(task_map),
            "tau": tau,
            "pulls_committed": dict(pulls_committed_before),
            "min_ect": float(min(teacher_ect)),
            "chosen_ect": float(teacher_ect[y_idx]),
            "inference_feature_layout": "dim24",
        }
        with (out / "trajectories.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
