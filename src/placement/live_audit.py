"""Live oracle-audit snapshot capture, shared across scheduler families.

The knative_network(_batch) schedulers carry their own copy of this capture (the
original implementation); this module is the policy-agnostic version so the GNN and
MLP serve paths can write the same snapshot schema. The schema must stay identical
across policies — `scripts_cosim/live_snapshot_cosim_oracle.py` and
`live_snapshot_oracle_audit.py` consume it by shape, and a collapse-moment snapshot
from an MLP arm has to replay through exactly the pipeline a Knative snapshot does.

Env contract (same variables the knative capture reads):
  LIVE_AUDIT_SNAPSHOT_PATH   append-target JSONL; capture is off when unset
  LIVE_AUDIT_MAX_SNAPSHOTS   default 500
  LIVE_AUDIT_STRIDE          default 1, keyed on the batch's first task id
  LIVE_AUDIT_MIN_BATCH_SIZE  default 4
  LIVE_AUDIT_MIN_CANDIDATES  default 4
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, TYPE_CHECKING

from src.placement.scheduling_cost import network_latency_between

if TYPE_CHECKING:
    from src.placement.infrastructure import Node, Platform, Task
    from src.placement.model import SystemState

_STORAGE_THROUGHPUT = 100.0 * 1024.0 * 1024.0
_STORAGE_LATENCY = 0.001


def _candidate_payload(
    scheduler: Any,
    task: "Task",
    node: "Node",
    platform: "Platform",
) -> Dict[str, Any]:
    queue_key = f"{node.node_name}:{platform.id}"
    temporal = scheduler._capture_temporal_state_for_replicas([(node, platform)]).get(
        queue_key, {}
    )
    task_type = task.type
    platform_type = platform.type["shortName"]
    state_size = task_type.get("stateSize", {})
    app_name = task.application.type.get("name", "") if task.application else ""
    app_state = state_size.get(app_name, {}) if isinstance(state_size, dict) else {}
    input_size = float(app_state.get("input", 0) or 0)
    output_size = float(app_state.get("output", 0) or 0)

    return {
        "node_id": int(node.id),
        "node_name": node.node_name,
        "platform_id": int(platform.id),
        "platform_type": platform_type,
        "queue_key": queue_key,
        "queue_length": int(len(platform.queue.items)),
        "initialized": bool(platform.initialized.triggered),
        "current_task_remaining": float(temporal.get("current_task_remaining", 0.0) or 0.0),
        "cold_start_remaining": float(temporal.get("cold_start_remaining", 0.0) or 0.0),
        "comm_remaining": float(temporal.get("comm_remaining", 0.0) or 0.0),
        "execution_time": float(
            task_type.get("executionTime", {}).get(platform_type, 0.0) or 0.0
        ),
        "cold_start_time": float(
            task_type.get("coldStartDuration", {}).get(platform_type, 0.0) or 0.0
        ),
        "energy": float(task_type.get("energy", {}).get(platform_type, 0.0) or 0.0),
        "network_latency": float(
            network_latency_between(task.node_name, node, scheduler.nodes.items) or 0.0
        ),
        "communications_time": (input_size / _STORAGE_THROUGHPUT + _STORAGE_LATENCY)
        + (output_size / _STORAGE_THROUGHPUT + _STORAGE_LATENCY),
    }


def _task_payload(scheduler: Any, system_state: "SystemState", task: "Task") -> Dict[str, Any]:
    replicas = system_state.replicas.get(task.type["name"], set())
    valid_replicas = scheduler._get_valid_replicas(replicas, task)
    return {
        "task_id": int(task.id),
        "task_type": task.type["name"],
        "source_node": task.node_name,
        "qos": task.application.qos if task.application else {},
        "candidate_count": len(valid_replicas),
        "candidates": [
            _candidate_payload(scheduler, task, node, platform)
            for node, platform in valid_replicas
        ],
    }


def _batch_qualifies(
    scheduler: Any, system_state: "SystemState", batch_tasks: List["Task"]
) -> bool:
    min_batch_size = int(os.environ.get("LIVE_AUDIT_MIN_BATCH_SIZE", "4"))
    if len(batch_tasks) < min_batch_size:
        return False
    min_candidates = int(os.environ.get("LIVE_AUDIT_MIN_CANDIDATES", "4"))
    for task in batch_tasks:
        replicas = system_state.replicas.get(task.type["name"], set())
        candidate_count = len(scheduler._get_valid_replicas(replicas, task))
        if candidate_count == 0:
            return False
        if min_candidates and candidate_count < min_candidates:
            return False
    return True


def _replicas_by_type_payload(
    system_state: "SystemState",
) -> Dict[str, List[Dict[str, Any]]]:
    """The full replica set per task type, as the live autoscaler has it right now."""
    payload: Dict[str, List[Dict[str, Any]]] = {}
    for task_type, replicas in system_state.replicas.items():
        specs: List[Dict[str, Any]] = []
        for node, platform in sorted(
            replicas, key=lambda np: (np[0].node_name, np[1].id)
        ):
            specs.append(
                {
                    "node_name": str(node.node_name),
                    "node_id": int(node.id),
                    "platform_id": int(platform.id),
                    "initialized": bool(platform.initialized.triggered),
                    "queue_length": int(platform.queue_length()),
                }
            )
        payload[str(task_type)] = specs
    return payload


def maybe_capture_batch_live_audit_snapshot(
    scheduler: Any,
    system_state: "SystemState",
    batch_tasks: List["Task"],
    policy_name: str,
) -> None:
    """Append one batch snapshot to LIVE_AUDIT_SNAPSHOT_PATH, subject to the filters.

    The host scheduler must provide `_get_valid_replicas`,
    `_capture_temporal_state_for_replicas`, `_capture_full_queue_snapshot`, `env`,
    and `nodes` — the GNN scheduler family does.
    """
    output_path = os.environ.get("LIVE_AUDIT_SNAPSHOT_PATH")
    if not output_path or not batch_tasks:
        return

    written = int(getattr(scheduler, "_audit_snapshots_written", 0))
    max_snapshots = int(os.environ.get("LIVE_AUDIT_MAX_SNAPSHOTS", "500"))
    if written >= max_snapshots:
        return

    stride = max(1, int(os.environ.get("LIVE_AUDIT_STRIDE", "1")))
    if batch_tasks[0].id % stride != 0:
        return

    if not _batch_qualifies(scheduler, system_state, batch_tasks):
        return

    snapshot = {
        "snapshot_id": written,
        "time": float(scheduler.env.now),
        "policy": policy_name,
        "horizon": len(batch_tasks),
        "trigger_task_id": int(batch_tasks[0].id),
        "chosen": None,
        "full_queue_snapshot": scheduler._capture_full_queue_snapshot(),
        "tasks": [_task_payload(scheduler, system_state, task) for task in batch_tasks],
        # P3 horizon continuation needs the FULL per-type replica state, not just the
        # batch tasks' candidate lists: a horizon arrival from any client node must find
        # the replicas the live autoscaler had actually provisioned at capture time.
        # Snapshots without this field predate it and only support t=0 sweeps.
        "replicas_by_type": _replicas_by_type_payload(system_state),
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "a") as f:
        f.write(json.dumps(snapshot, separators=(",", ":")) + "\n")
    scheduler._audit_snapshots_written = written + 1
