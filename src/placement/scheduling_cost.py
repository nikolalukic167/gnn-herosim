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


def node_contention_wait(
    node: "Node",
    platform: "Platform",
    *,
    added_on_node: int = 0,
    added_unit_exec: float = 0.0,
) -> float:
    """Expected wait for one of the node's shared execution slots (node_contention_v3).

    Zero when the node has no slot pool, which is node_disk_v2 physics. Otherwise the
    work queued on *sibling* platforms is spread over the available slots — this is what
    makes the cost of a placement depend on what else sits on the same node.

    ``added_on_node`` counts batch-mates already assigned to this node but not yet
    enqueued, charged at ``added_unit_exec`` each — the node-level analogue of
    ``added_in_batch`` in ``queue_work``.
    """
    slots = getattr(node, "compute_slots", None)
    if slots is None:
        return 0.0
    backlog = 0.0
    for sibling in node.platforms.items:
        if sibling is platform:
            continue
        sibling_type = sibling.type["shortName"]
        for queued in sibling.queue.items:
            backlog += float(
                queued.type["executionTime"].get(sibling_type, 0.0) or 0.0
            )
        # The seeded queue depth lives in the compressed warmup backlog, not in
        # queue.items -- it is the bulk of a sibling's hold on the node's slots.
        backlog += float(getattr(sibling, "virtual_warmup_total_time", 0.0) or 0.0)
    backlog += added_on_node * float(added_unit_exec)
    return backlog / float(slots.capacity)


def transfer_time(
    task: "Task",
    bandwidth_mbps: Optional[float],
) -> float:
    """Transmission time for a task's input over a pipe of the given bandwidth.

    Zero when there is no pipe, which is node_disk_v2 physics.

    ``bandwidth_mbps`` is MB/s, matching ``NetworkDescription.bandwidth`` and
    ``_STORAGE_THROUGHPUT`` — value x 1024**2 is bytes/s.

    Note what is and is not additive here. The transmission itself is a function of
    ``(task, pipe)`` alone, so it stays a pointwise term. What breaks additivity is the
    *wait* for the pipe (``ingress_wait`` / ``link_wait``), which depends on where the
    batch-mates went.

    Shared verbatim by the simulator and the ECT cost model — ``infrastructure.py``
    imports it rather than re-deriving it, because two copies of one formula is exactly
    what produced the train/serve MP mismatch.
    """
    if not bandwidth_mbps:
        return 0.0
    if not task.application:
        return 0.0
    app_name = task.application.type.get("name", "")
    state_size = task.type.get("stateSize", {})
    if not isinstance(state_size, dict) or app_name not in state_size:
        return 0.0
    input_size = float(state_size[app_name].get("input", 0) or 0)
    if input_size <= 0:
        return 0.0
    return input_size / (float(bandwidth_mbps) * 1024.0 * 1024.0)


# network_contention_v1 spelling, kept so existing callers and tests read naturally.
ingress_transfer_time = transfer_time


def ingress_wait(
    node: "Node",
    task: "Task",
    *,
    added_on_node: int = 0,
) -> float:
    """Expected wait for the destination node's shared ingress pipe (network_contention_v1).

    Zero when the node has no pipe. Unlike ``node_contention_wait`` there is no standing
    backlog to inspect — transfers are transient, so the wait a scheduler can anticipate
    is the one created by ``added_on_node``: batch-mates already committed to this node
    this round but not yet in flight. That is exactly the non-additive term.
    """
    pipe = getattr(node, "ingress_pipe", None)
    if pipe is None or added_on_node <= 0:
        return 0.0
    unit = transfer_time(task, getattr(node, "ingress_bandwidth_mbps", None))
    return (added_on_node * unit) / float(pipe.capacity)


def link_transfer_cost(
    fabric: Any,
    task: "Task",
    source_name: str,
    node: "Node",
) -> float:
    """Transmission cost summed over every hop of the task's route (link_contention_v1).

    Store-and-forward: each hop holds the link for a full transmission, so the additive
    part of the route cost grows with hop count. Still a function of
    ``(task, source, destination)`` alone — pointwise, and not the interesting term.
    """
    if fabric is None or source_name == node.node_name:
        return 0.0
    return sum(
        transfer_time(task, bandwidth)
        for _key, bandwidth in fabric.hops(source_name, node.node_name)
    )


def link_wait(
    fabric: Any,
    task: "Task",
    source_name: str,
    node: "Node",
    *,
    added_on_links: Optional[Mapping[str, int]] = None,
) -> float:
    """Expected wait on shared links along the route (link_contention_v1).

    ``added_on_links`` maps a link key to how many batch-mates already committed this
    round will cross it. Like ``ingress_wait`` there is no standing backlog to inspect —
    transfers are transient — so the anticipatable wait is the one the rest of the batch
    creates.

    This is the term that distinguishes the mechanism from every one before it. The
    ingress pipe is indexed by the destination node, so its wait is a function of
    destination occupancy and one integer repairs it. A link is crossed by paths to
    *many* destinations, so this wait can be non-zero between two tasks that share no
    node at all, and no node-occupancy count can express it.
    """
    if fabric is None or not added_on_links or source_name == node.node_name:
        return 0.0
    total = 0.0
    for key, bandwidth in fabric.hops(source_name, node.node_name):
        crossing = int(added_on_links.get(key, 0))
        if crossing > 0:
            total += crossing * transfer_time(task, bandwidth)
    return total


def expected_completion_for_candidate(
    task: "Task",
    node: "Node",
    platform: "Platform",
    env_now: float,
    *,
    added_in_batch: int = 0,
    added_on_node: int = 0,
    added_on_links: Optional[Mapping[str, int]] = None,
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
    contention = node_contention_wait(
        node, platform, added_on_node=added_on_node, added_unit_exec=exec_time
    )
    # network_contention_v1: only remote tasks traverse the destination node's ingress
    # pipe, mirroring the simulation, which pays both terms inside the same
    # `task.node_name != self.node.node_name` branch.
    if task.node_name != node.node_name:
        ingress = transfer_time(
            task, getattr(node, "ingress_bandwidth_mbps", None)
        ) + ingress_wait(node, task, added_on_node=added_on_node)
        # link_contention_v1: same guard, since the simulation charges the route inside
        # the same remote-only branch.
        fabric = getattr(node, "fabric", None)
        link_cost = link_transfer_cost(
            fabric, task, task.node_name, node
        ) + link_wait(
            fabric, task, task.node_name, node, added_on_links=added_on_links
        )
    else:
        ingress = 0.0
        link_cost = 0.0

    return (
        current_work
        + queue_work
        + cold_start
        + exec_time
        + comm_time
        + network
        + contention
        + ingress
        + link_cost
    )


def node_cold_platform_count(node: "Node") -> int:
    """Absolute cold (not-yet-initialized) platforms on a node — FilterStore depth proxy."""
    return sum(1 for p in node.platforms.items if not p.initialized.triggered)


def filterstore_pull_wait_sec(
    node: "Node",
    platform: "Platform",
    *,
    unit_pull_sec: Optional[float] = None,
    extra_committed_pulls: int = 0,
    use_marginal_ordinal: bool = True,
) -> float:
    """
    Schedule-time FilterStore pull wait for placing on ``platform``.

    Marginal ordinal (default): (committed_pulls + 1) × T_pull — cost of one more
    pull on this node's FilterStore. Absolute cold_count × T_pull
    (use_marginal_ordinal=False) matches CACHE 5.6 feature magnitudes but
    over-penalizes an unused scarce node vs a depth-2 remote pile.
    Warm/initialized platforms → 0.
    """
    from src.placement.warmth import DEFAULT_T_PULL_S, estimated_pull_remaining_sec

    if platform.initialized.triggered:
        return 0.0
    if extra_committed_pulls < 0:
        raise ValueError(f"extra_committed_pulls must be >= 0, got {extra_committed_pulls}")
    unit = DEFAULT_T_PULL_S if unit_pull_sec is None else float(unit_pull_sec)
    if unit < 0:
        raise ValueError(f"unit_pull_sec must be >= 0, got {unit}")
    if use_marginal_ordinal:
        ordinal = float(extra_committed_pulls + 1)
        return estimated_pull_remaining_sec(ordinal, unit)
    cold = float(node_cold_platform_count(node) + extra_committed_pulls)
    return estimated_pull_remaining_sec(cold, unit)


def expected_completion_with_filterstore_pull(
    task: "Task",
    node: "Node",
    platform: "Platform",
    env_now: float,
    *,
    added_in_batch: int = 0,
    extra_committed_pulls: int = 0,
    unit_pull_sec: Optional[float] = None,
    nodes: Optional[Mapping[Any, Any]] = None,
) -> float:
    """ECT + FilterStore pull serialization (physics-aware residual baseline)."""
    base = expected_completion_for_candidate(
        task,
        node,
        platform,
        env_now,
        added_in_batch=added_in_batch,
        nodes=nodes,
    )
    pull_wait = filterstore_pull_wait_sec(
        node,
        platform,
        unit_pull_sec=unit_pull_sec,
        extra_committed_pulls=extra_committed_pulls,
    )
    return base + pull_wait


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
    # Contention terms are absent from older snapshot payloads, so they default to 0.0
    # and this stays identical on any snapshot that predates them. Kept here so the
    # snapshot form does not silently fall behind expected_completion_for_candidate the
    # way it did through node_contention_v3.
    return (
        queue_time
        + cold
        + exec_time
        + float(candidate.get("network_latency", 0.0) or 0.0)
        + float(candidate.get("communications_time", 0.0) or 0.0)
        + float(candidate.get("node_contention_time", 0.0) or 0.0)
        + float(candidate.get("ingress_time", 0.0) or 0.0)
    )
