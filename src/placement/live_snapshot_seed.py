"""Seed simulation state from a live scheduling audit snapshot."""

from __future__ import annotations

import math
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

    # Snapshots captured after the P3 extension carry the FULL per-type replica state
    # (live_audit._replicas_by_type_payload); seed from it so horizon arrivals from any
    # client node find the replicas the live autoscaler had provisioned. Older
    # snapshots fall back to the batch tasks' candidate union — sufficient for t=0
    # sweeps, structurally incomplete for horizon continuation.
    captured = snapshot.get("replicas_by_type") or {}
    replicas_by_type: Dict[str, Set[Tuple[str, int]]] = (
        {str(t): set() for t in captured} if captured else {"dnn1": set(), "dnn2": set()}
    )
    # peer_affinity_warm_v1: a captured replica may be marked `candidate: false` -- it is
    # a busy platform the live autoscaler had provisioned, replayed with its backlog so the
    # cluster's load is what the snapshot saw, but NOT offered to the sweep (and so not a
    # candidate edge in the cache). Default true keeps every older snapshot bit-identical.
    candidate_flags: Dict[Tuple[str, str, int], bool] = {}
    for task_type, specs in captured.items():
        for spec in specs:
            node_name = str(spec.get("node_name", ""))
            platform_id = int(spec.get("platform_id", -1))
            if not node_name or platform_id < 0:
                continue
            replicas_by_type[str(task_type)].add((node_name, platform_id))
            candidate_flags[(str(task_type), node_name, platform_id)] = bool(
                spec.get("candidate", True)
            )
            qkey = f"{node_name}:{platform_id}"
            platform_info.setdefault(
                qkey,
                {
                    "node_name": node_name,
                    "platform_id": platform_id,
                    "initialized": bool(spec.get("initialized", True)),
                    "queue_length": int(
                        full_queue.get(qkey, spec.get("queue_length", 0)) or 0
                    ),
                    # Present only in HEROSIM_INFLIGHT_CAPTURE=service_end_v1 captures.
                    "current_task_remaining": float(spec.get("current_task_remaining", 0.0) or 0.0),
                    "comm_remaining": 0.0,
                    "cold_start_remaining": 0.0,
                    "task_type_hint": str(task_type),
                    # 0.0 -> _seed_platform_state falls back to its exec+comm formula
                    "queue_drain_seconds": float(spec.get("queue_drain_seconds", 0.0) or 0.0),
                },
            )

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
                "queue_drain_seconds": float(
                    candidate.get(
                        "queue_drain_seconds",
                        platform_info.get(qkey, {}).get("queue_drain_seconds", 0.0),
                    )
                    or 0.0
                ),
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
            spec["candidate"] = candidate_flags.get((task_type, node_name, platform_id), True)
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

    task_type_name = str(spec.get("task_type_hint", "dnn1"))
    task_type = simulation_data.task_types.get(task_type_name)
    backlog = seeded_backlog_seconds(spec, task_type, plat.type["shortName"])
    if backlog is None:
        return

    queue_len = int(spec.get("queue_length", 0) or 0)
    virtual_count = queue_len + int(spec.get("synthetic_queue_length", 0) or 0)
    if float(spec.get("current_task_remaining", 0) or 0) > 0.0 or float(spec.get("comm_remaining", 0) or 0) > 0.0:
        virtual_count = max(virtual_count, 1)
    if float(spec.get("synthetic_backlog_seconds", 0) or 0) > 0.0:
        virtual_count = max(virtual_count, 1)

    plat.seed_virtual_warmup(task_type, task_type_name, virtual_count)
    plat.virtual_warmup_total_time = backlog


def seeded_backlog_seconds(
    spec: Mapping[str, Any], task_type: Optional[Mapping[str, Any]], plat_type: str
) -> Optional[float]:
    """The backlog clock a seeded platform replays: `current_task_remaining + comm_remaining`
    plus the snapshot's measured drain, else `queue_length x (execution + comm)` of the hint
    type. None when nothing is seeded (an idle platform, or an unknown hint type).

    One definition for the co-sim replay (`_seed_platform_state`) and the partial_state_v4
    backlog column (prepare_graphs_cache), so the feature a model is trained on is the clock
    its labels were simulated on.

    `synthetic_backlog_seconds` (backlog_corpus_v1, written by make_warm_corpus
    --synthetic-backlog-*) is added on top: fake queued work the snapshot never had."""
    synthetic = float(spec.get("synthetic_backlog_seconds", 0) or 0)
    base = _captured_backlog_seconds(spec, task_type, plat_type)
    if synthetic <= 0.0 or task_type is None:
        return base
    return (base or 0.0) + synthetic


def _poisson(rng: Any, lam: float) -> int:
    if lam <= 0.0:
        return 0
    threshold, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= threshold:
            return k
        k += 1


def inject_synthetic_backlog(
    seed_block: Dict[str, Any],
    rng: Any,
    *,
    mean_seconds: float,
    busy_prob: float,
    task_seconds: float,
) -> Dict[str, Any]:
    """backlog_corpus_v1: put fake queued work on the seed's platforms, in place.

    Each platform is busy with probability `busy_prob`; a busy one gets
    k = 1 + Poisson(mean_seconds / task_seconds - 1) fake queued tasks, each draining
    Gamma(2, task_seconds / 2) seconds, so counts and seconds relate the way a live queue's
    do (a live queued task pays execution plus its peer exchange, ~4.5 s). The draw order is
    the sorted queue key, so a (seed, snapshot) pair always gives the same backlog.
    `mean_seconds <= 0` injects nothing. Returns the record written into provenance.
    """
    record: Dict[str, Any] = {
        "mean_seconds": float(mean_seconds), "busy_prob": float(busy_prob),
        "task_seconds": float(task_seconds), "per_key": {},
    }
    if mean_seconds <= 0.0:
        return record
    platforms = seed_block.get("platforms") or []
    by_key = {f"{p['node_name']}:{int(p['platform_id'])}": p for p in platforms}
    draws: Dict[str, Tuple[int, float]] = {}
    for qkey in sorted(by_key):
        if rng.random() >= busy_prob:
            continue
        k = 1 + _poisson(rng, max(mean_seconds / task_seconds - 1.0, 0.0))
        seconds = sum(rng.gammavariate(2.0, task_seconds / 2.0) for _ in range(k))
        draws[qkey] = (k, seconds)
    for qkey, spec in by_key.items():
        k, seconds = draws.get(qkey, (0, 0.0))
        spec["synthetic_queue_length"] = k
        spec["synthetic_backlog_seconds"] = seconds
    for specs in (seed_block.get("replicas_by_type") or {}).values():
        for spec in specs:
            k, seconds = draws.get(f"{spec['node_name']}:{int(spec['platform_id'])}", (0, 0.0))
            spec["synthetic_queue_length"] = k
            spec["synthetic_backlog_seconds"] = seconds
    record["per_key"] = {q: [k, s] for q, (k, s) in sorted(draws.items())}
    return record


def _captured_backlog_seconds(
    spec: Mapping[str, Any], task_type: Optional[Mapping[str, Any]], plat_type: str
) -> Optional[float]:
    queue_len = int(spec.get("queue_length", 0) or 0)
    current_remaining = float(spec.get("current_task_remaining", 0) or 0)
    comm_remaining = float(spec.get("comm_remaining", 0) or 0)
    if queue_len <= 0 and current_remaining <= 0.0 and comm_remaining <= 0.0:
        return None
    if task_type is None:
        return None
    execution = float(task_type.get("executionTime", {}).get(plat_type, 0.0) or 0.0)
    comm = _approx_comm(task_type)
    # peer_affinity_warm_v1: a snapshot that measured its own drain (live_audit.
    # platform_queue_drain_seconds -- execution + I/O + latency + the peer transfers the
    # queued tasks will actually pay) replays that clock instead of the exec+comm formula,
    # which under HEROSIM_PEER_EXCHANGE=1 understates a deep queue's drain ~100x.
    measured_drain = float(spec.get("queue_drain_seconds", 0.0) or 0.0)
    if measured_drain > 0.0:
        return current_remaining + comm_remaining + measured_drain
    return current_remaining + comm_remaining + queue_len * (execution + comm)


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
            if not bool(spec.get("candidate", True)):
                # Occupied live, not offered to the sweep: seeded below like every other
                # platform, kept out of `replicas`, and flagged so the determined
                # orchestrator does not list it as free capacity either.
                plat.snapshot_reserved = True
            else:
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
