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
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from src.placement.live_snapshot_seed import _approx_comm
from src.placement.scheduling_cost import network_latency_between

if TYPE_CHECKING:
    from src.placement.infrastructure import Node, Platform, Task
    from src.placement.model import SystemState

_STORAGE_THROUGHPUT = 100.0 * 1024.0 * 1024.0
_STORAGE_LATENCY = 0.001


def orchestrator_of(scheduler: Any) -> Any:
    """The orchestrator behind a scheduler, via the `orchestrator_ref` every node carries
    once the orchestrator has started (Orchestrator.__init__). None before that."""
    for node in getattr(scheduler, "nodes", None).items if getattr(scheduler, "nodes", None) else []:
        orch = getattr(node, "orchestrator_ref", None)
        if orch is not None:
            return orch
    return None


def platform_queue_drain_seconds(
    platform: "Platform", orchestrator: Any, memo: Optional[Dict[str, float]] = None,
    exec_scale: float = 1.0,
) -> float:
    """Seconds until `platform`'s backlog as it stands now has been served.

    `memo` (queue_key -> seconds) makes one snapshot walk each busy queue once: a batch
    snapshot asks for every candidate of every task AND every replica, and a 16k-deep
    queue walked 100 times per snapshot stalled the capture run (measured 2026-09-13).

    peer_affinity_warm_v1 (2026-09-13): a live snapshot replayed through
    `live_snapshot_seed` compresses a queue into a virtual backlog with one total time.
    The seed's own formula is `queue_length x (execution + storage I/O)`, which is what a
    queued task costs when nothing else is charged -- but under HEROSIM_PEER_EXCHANGE=1
    every queued task also pays its peer transfers (seconds each at 200 MB), so a 16k-deep
    live queue would drain in the co-sim ~100x faster than it drains live and the label
    would rank queues on the wrong clock. This walks the real queue and charges each
    queued task what `Platform.platform_process` will charge it: execution on this
    platform, the same storage I/O approximation, the source->platform network latency,
    and the peer-exchange transfer for every peer whose node is already known (placed or
    planned; an unknown peer contributes 0 here and is a live rendezvous, not a transfer).
    Any virtual backlog already seeded is carried through unchanged.
    """
    memo_key = f"{platform.node.node_name}:{platform.id}"
    if memo is not None and memo_key in memo:
        return memo[memo_key]
    total = float(getattr(platform, "virtual_warmup_total_time", 0.0) or 0.0)
    node = platform.node
    plat_type = platform.type["shortName"]
    network_map = getattr(node, "network_map", None) or {}
    peer_on = os.environ.get("HEROSIM_PEER_EXCHANGE", "0") == "1"
    peer_table = (getattr(orchestrator, "peer_exchange", None) or {}) if orchestrator is not None else {}
    task_by_id = (getattr(orchestrator, "task_by_id", None) or {}) if orchestrator is not None else {}
    # Platform._payload_transfer_time is linear in the payload for a fixed route
    # (hops x bytes / bottleneck, or bytes / node bandwidth), so price each peer node once.
    per_byte: Dict[str, float] = {}

    def _latency_to(other_node_name: str) -> float:
        entry = network_map.get(other_node_name)
        if entry is None:
            return 0.0
        return float(entry.get("latency", 0.0)) if isinstance(entry, dict) else float(entry)

    def _transfer(other_node_name: str, payload: float) -> float:
        if other_node_name not in per_byte:
            per_byte[other_node_name] = float(platform._payload_transfer_time(other_node_name, 1.0))
        return per_byte[other_node_name] * payload

    for task in platform.queue.items:
        task_type = task.type
        total += float(task_type.get("executionTime", {}).get(plat_type, 0.0) or 0.0) * exec_scale
        total += _approx_comm(task_type)
        if getattr(task, "node_name", None) and task.node_name != node.node_name:
            total += _latency_to(task.node_name)
        if not peer_on:
            continue
        for peer_id, payload in (peer_table.get(int(task.id)) or {}).items():
            peer = task_by_id.get(peer_id)
            if peer is None:
                continue
            peer_platform = getattr(peer, "platform", None)
            peer_node_name = (
                peer_platform.node.node_name
                if peer_platform is not None
                else getattr(peer, "planned_node_name", None)
            )
            if peer_node_name is None or peer_node_name == node.node_name:
                continue
            total += _transfer(peer_node_name, float(payload)) + _latency_to(peer_node_name)
    if memo is not None:
        memo[memo_key] = float(total)
    return float(total)


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
        "queue_drain_seconds": platform_queue_drain_seconds(
            platform, orchestrator_of(scheduler), getattr(scheduler, "_drain_memo", None)
        ),
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
    orchestrator: Any = None,
    memo: Optional[Dict[str, float]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """The full replica set per task type, as the live autoscaler has it right now.

    With an orchestrator, every spec also carries `platform_type` and
    `queue_drain_seconds` (see platform_queue_drain_seconds) so a snapshot can be replayed
    as a warm co-sim state; without one the payload is the pre-2026-09-13 shape."""
    payload: Dict[str, List[Dict[str, Any]]] = {}
    for task_type, replicas in system_state.replicas.items():
        specs: List[Dict[str, Any]] = []
        for node, platform in sorted(
            replicas, key=lambda np: (np[0].node_name, np[1].id)
        ):
            spec = {
                "node_name": str(node.node_name),
                "node_id": int(node.id),
                "platform_id": int(platform.id),
                "initialized": bool(platform.initialized.triggered),
                "queue_length": int(platform.queue_length()),
            }
            if orchestrator is not None:
                spec["platform_type"] = str(platform.type["shortName"])
                spec["queue_drain_seconds"] = platform_queue_drain_seconds(platform, orchestrator, memo)
            specs.append(spec)
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

    scheduler._drain_memo = {}  # one queue walk per platform per snapshot
    try:
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
            "replicas_by_type": _replicas_by_type_payload(
                system_state, orchestrator_of(scheduler), scheduler._drain_memo
            ),
        }
    finally:
        scheduler._drain_memo = None

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "a") as f:
        f.write(json.dumps(snapshot, separators=(",", ":")) + "\n")
    scheduler._audit_snapshots_written = written + 1
