"""Seed simulation state from a live scheduling audit snapshot."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

if False:  # TYPE_CHECKING
    from simpy import Environment

    from src.placement.infrastructure import Node, Platform
    from src.placement.model import SimulationData, SimulationPolicy


def _approx_comm(task_type: Mapping[str, Any]) -> float:
    state_size_map = task_type.get("stateSize", {})
    if not isinstance(state_size_map, dict) or not state_size_map:
        return 0.0
    app_state = next(iter(state_size_map.values()))
    if not isinstance(app_state, dict):
        return 0.0
    input_size = float(app_state.get("input", 0) or 0)
    output_size = float(app_state.get("output", 0) or 0)
    storage_throughput = 100.0 * 1024.0 * 1024.0
    storage_latency = 0.001
    return (
        (input_size / storage_throughput + storage_latency)
        + (output_size / storage_throughput + storage_latency)
    )


def build_live_snapshot_seed(snapshot: Mapping[str, Any]) -> Dict[str, Any]:
    """Convert an audit snapshot JSON object into simulation seed data."""
    full_queue = snapshot.get("full_queue_snapshot") or {}
    platform_info: Dict[str, Dict[str, Any]] = {}
    replicas_by_type: Dict[str, Set[Tuple[str, int]]] = {"dnn1": set(), "dnn2": set()}

    for task in snapshot.get("tasks", []):
        task_type = str(task.get("task_type", ""))
        if task_type not in replicas_by_type:
            continue
        for candidate in task.get("candidates", []):
            node_name = str(candidate.get("node_name", ""))
            platform_id = int(candidate.get("platform_id", -1))
            if not node_name or platform_id < 0:
                continue
            replicas_by_type[task_type].add((node_name, platform_id))
            qkey = f"{node_name}:{platform_id}"
            platform_info[qkey] = {
                "node_name": node_name,
                "platform_id": platform_id,
                "initialized": bool(candidate.get("initialized", True)),
                "queue_length": int(full_queue.get(qkey, candidate.get("queue_length", 0)) or 0),
                "current_task_remaining": float(candidate.get("current_task_remaining", 0) or 0),
                "comm_remaining": float(candidate.get("comm_remaining", 0) or 0),
                "cold_start_remaining": float(candidate.get("cold_start_remaining", 0) or 0),
                "task_type_hint": task_type,
            }

    for qkey, queue_len in full_queue.items():
        if qkey in platform_info:
            platform_info[qkey]["queue_length"] = int(queue_len or 0)
            continue
        if not qkey or ":" not in qkey:
            continue
        node_name, plat_str = qkey.rsplit(":", 1)
        try:
            platform_id = int(plat_str)
        except (TypeError, ValueError):
            continue
        platform_info[qkey] = {
            "node_name": node_name,
            "platform_id": platform_id,
            "initialized": True,
            "queue_length": int(queue_len or 0),
            "current_task_remaining": 0.0,
            "comm_remaining": 0.0,
            "cold_start_remaining": 0.0,
            "task_type_hint": "dnn1",
        }

    replicas_payload: Dict[str, List[Dict[str, Any]]] = {}
    for task_type, keys in replicas_by_type.items():
        specs: List[Dict[str, Any]] = []
        for node_name, platform_id in sorted(keys):
            qkey = f"{node_name}:{platform_id}"
            spec = dict(platform_info.get(qkey, {}))
            spec.setdefault("node_name", node_name)
            spec.setdefault("platform_id", platform_id)
            spec.setdefault("initialized", True)
            spec.setdefault("queue_length", 0)
            spec.setdefault("task_type_hint", task_type)
            specs.append(spec)
        replicas_payload[task_type] = specs

    return {
        "replicas_by_type": replicas_payload,
        "platforms": list(platform_info.values()),
    }


def _seed_platform_state(
    plat_map: Dict[Tuple[str, int], Tuple[Any, Any]],
    simulation_data: Any,
    spec: Mapping[str, Any],
) -> None:
    node_name = str(spec.get("node_name", ""))
    platform_id = int(spec.get("platform_id", -1))
    key = (node_name, platform_id)
    if key not in plat_map:
        return

    _node, plat = plat_map[key]
    if not plat.initialized.triggered:
        plat.initialized.succeed()
    if bool(spec.get("initialized", True)):
        task_type_name = str(spec.get("task_type_hint", "dnn1"))
        plat.previous_task = type("Task", (), {"type": {"name": task_type_name}})()
    else:
        plat.previous_task = None

    queue_len = int(spec.get("queue_length", 0) or 0)
    current_remaining = float(spec.get("current_task_remaining", 0) or 0)
    comm_remaining = float(spec.get("comm_remaining", 0) or 0)
    if queue_len <= 0 and current_remaining <= 0.0 and comm_remaining <= 0.0:
        return

    task_type_name = str(spec.get("task_type_hint", "dnn1"))
    task_type = simulation_data.task_types.get(task_type_name)
    if task_type is None:
        return

    plat_type = plat.type["shortName"]
    execution = float(task_type.get("executionTime", {}).get(plat_type, 0.0) or 0.0)
    comm = _approx_comm(task_type)

    virtual_count = queue_len
    if current_remaining > 0.0 or comm_remaining > 0.0:
        virtual_count = max(virtual_count, 1)

    plat.seed_virtual_warmup(task_type, task_type_name, virtual_count)
    plat.virtual_warmup_total_time = (
        current_remaining + comm_remaining + queue_len * (execution + comm)
    )


def apply_live_snapshot_seed(
    nodes: Any,
    simulation_data: Any,
    env: Any,
    simulation_policy: Any,
    seed_data: Mapping[str, Any],
) -> Dict[str, Set[Tuple[Any, Any]]]:
    """Create replicas and queue/temporal backlog from a live snapshot."""
    del env, simulation_policy  # reserved for future temporal task materialization

    initial_replicas: Dict[str, Set[Tuple[Any, Any]]] = {
        task_type: set() for task_type in simulation_data.task_types
    }
    plat_map: Dict[Tuple[str, int], Tuple[Any, Any]] = {}
    for node in nodes.items:
        for plat in node.platforms.items:
            plat_map[(node.node_name, plat.id)] = (node, plat)

    for task_type, specs in (seed_data.get("replicas_by_type") or {}).items():
        if task_type not in initial_replicas:
            continue
        for spec in specs:
            key = (str(spec.get("node_name", "")), int(spec.get("platform_id", -1)))
            if key not in plat_map:
                continue
            node, plat = plat_map[key]
            initial_replicas[task_type].add((node, plat))
            if not plat.initialized.triggered:
                plat.initialized.succeed()
            if bool(spec.get("initialized", True)):
                plat.previous_task = type("Task", (), {"type": {"name": task_type}})()
            else:
                plat.previous_task = None

    seen: Set[Tuple[str, int]] = set()
    for spec in seed_data.get("platforms") or []:
        key = (str(spec.get("node_name", "")), int(spec.get("platform_id", -1)))
        if key in seen:
            continue
        seen.add(key)
        _seed_platform_state(plat_map, simulation_data, spec)

    return initial_replicas
