"""Shared expected-completion-time (ECT) helpers for placement policies."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.placement.infrastructure import Node, Platform, Task

_STORAGE_THROUGHPUT = 100.0 * 1024.0 * 1024.0
_STORAGE_LATENCY = 0.001


def platform_temporal_state(platform: "Platform", env_now: float) -> Dict[str, float]:
    """Remaining work on a platform from its current task (matches Knative capture)."""
    current_task_remaining = 0.0
    cold_start_remaining = 0.0
    comm_remaining = 0.0

    if platform.current_task is not None:
        current_task = platform.current_task

        if current_task.cold_started and not hasattr(current_task, "started_time"):
            cold_start_duration = current_task.type["coldStartDuration"].get(
                platform.type["shortName"], 0.0
            )
            elapsed_cold_start = env_now - current_task.arrived_time
            cold_start_remaining = max(0.0, cold_start_duration - elapsed_cold_start)

        if hasattr(current_task, "started_time") and current_task.started_time is not None:
            exec_duration = current_task.type["executionTime"].get(
                platform.type["shortName"], 0.0
            )
            elapsed_exec = env_now - current_task.started_time
            current_task_remaining = max(0.0, exec_duration - elapsed_exec)

            if current_task.application:
                state_size_map = current_task.type.get("stateSize", {})
                app_name = current_task.application.type.get("name", "")
                if isinstance(state_size_map, dict) and app_name in state_size_map:
                    output_size = state_size_map[app_name].get("output", 0)
                    if isinstance(output_size, (int, float)) and output_size > 0:
                        comm_remaining = (output_size / _STORAGE_THROUGHPUT) + _STORAGE_LATENCY

    return {
        "current_task_remaining": current_task_remaining,
        "cold_start_remaining": cold_start_remaining,
        "comm_remaining": comm_remaining,
    }


def communications_time_for_task(task: "Task", platform_short_name: str) -> float:
    """Input + output I/O time for placing task on a platform type."""
    if not task.application:
        return 0.0
    app_name = task.application.type.get("name", "")
    state_size = task.type.get("stateSize", {})
    if not isinstance(state_size, dict) or app_name not in state_size:
        return 0.0
    app_state = state_size[app_name]
    input_size = float(app_state.get("input", 0) or 0)
    output_size = float(app_state.get("output", 0) or 0)
    return (
        (input_size / _STORAGE_THROUGHPUT + _STORAGE_LATENCY)
        + (output_size / _STORAGE_THROUGHPUT + _STORAGE_LATENCY)
    )


def incoming_cold_start_time(task: "Task", platform: "Platform") -> float:
    """Cold start for the incoming task (live Knative warm/cold semantics)."""
    if platform.current_task is not None:
        return 0.0
    warm = (
        platform.previous_task is not None
        and platform.previous_task.type["name"] == task.type["name"]
    )
    if warm:
        return 0.0
    return float(
        task.type["coldStartDuration"].get(platform.type["shortName"], 0.0) or 0.0
    )


def network_latency_between(
    source_node_name: str,
    target_node: "Node",
    nodes: Optional[Mapping[Any, Any]] = None,
) -> float:
    if target_node.node_name == source_node_name:
        return 0.0
    if hasattr(target_node, "network_map") and source_node_name in target_node.network_map:
        entry = target_node.network_map[source_node_name]
    elif nodes is not None:
        source_node = next(
            (node for node in nodes if getattr(node, "node_name", None) == source_node_name),
            None,
        )
        if source_node is None or not hasattr(source_node, "network_map"):
            return 0.0
        entry = source_node.network_map.get(target_node.node_name, 0.0)
    else:
        return 0.0
    if isinstance(entry, dict):
        return float(entry.get("latency", 0.0) or 0.0)
    return float(entry or 0.0)


def expected_completion_for_candidate(
    task: "Task",
    node: "Node",
    platform: "Platform",
    env_now: float,
    *,
    added_in_batch: int = 0,
    nodes: Optional[Mapping[Any, Any]] = None,
) -> float:
    """ECT for placing task on (node, platform)."""
    platform_type = platform.type["shortName"]
    exec_time = float(task.type["executionTime"].get(platform_type, 0.0) or 0.0)

    temporal = platform_temporal_state(platform, env_now)
    queue_work = (len(platform.queue.items) + added_in_batch) * exec_time
    current_work = (
        temporal["cold_start_remaining"]
        + temporal["current_task_remaining"]
        + temporal["comm_remaining"]
    )
    cold_start = incoming_cold_start_time(task, platform)
    comm_time = communications_time_for_task(task, platform_type)
    network = network_latency_between(task.node_name, node, nodes)

    return current_work + queue_work + cold_start + exec_time + comm_time + network


def expected_completion_from_snapshot_candidate(
    candidate: Dict[str, Any],
    queued_before: int,
    combo_added_before: int,
) -> float:
    """ECT from audit snapshot candidate payload (matches candidate_cost total)."""
    exec_time = float(candidate.get("execution_time", 0.0) or 0.0)
    queue_time = (
        float(candidate.get("current_task_remaining", 0.0) or 0.0)
        + float(candidate.get("comm_remaining", 0.0) or 0.0)
        + queued_before * exec_time
        + combo_added_before * exec_time
    )
    cold = (
        0.0
        if candidate.get("initialized", True)
        else float(candidate.get("cold_start_time", 0.0) or 0.0)
    )
    return (
        queue_time
        + cold
        + exec_time
        + float(candidate.get("network_latency", 0.0) or 0.0)
        + float(candidate.get("communications_time", 0.0) or 0.0)
    )
