"""Observable service state and versioned scheduling availability estimates.

No event is scheduled here. Unknown waits stay explicit; known work is an
estimate, not a prediction of when a resource or an unarrived peer will release.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from types import SimpleNamespace

LEGACY = "backlog_only_v1"
AVAILABILITY = "availability_v2"
CONTRACT_ENV = "HEROSIM_PG_DRAIN_CONTRACT"


def drain_contract():
    value = os.environ.get(CONTRACT_ENV, LEGACY)
    if value not in (LEGACY, AVAILABILITY):
        raise ValueError(f"unsupported {CONTRACT_ENV}={value!r}")
    return value


@dataclass
class ServiceState:
    phase: str = "idle"
    deadline: float | None = None
    pending: dict[str, float] = field(default_factory=dict)
    unresolved: tuple[str, ...] = ()
    event: object | None = None
    task_id: int | None = None
    virtual: bool = False
    unpriced_stages: set[str] = field(default_factory=set)


def begin_service(platform, task=None, *, virtual_seconds=None):
    pending = {}
    if virtual_seconds is not None:
        pending["virtual_backlog"] = float(virtual_seconds)
    elif task is not None:
        pending["execution"] = float(task.type["executionTime"][platform.type["shortName"]])
    platform.service_state = ServiceState(
        phase="dequeued", pending=pending,
        task_id=int(task.id) if task is not None else None,
        virtual=virtual_seconds is not None,
        unpriced_stages={"input_transfer", "output_transfer"} if task is not None else set(),
    )


def service_phase(platform, phase, *, seconds=None, event=None, unresolved=()):
    state = platform.service_state
    state.phase = phase
    state.pending.pop(phase, None)
    state.unpriced_stages.discard(phase)
    state.deadline = None if seconds is None else float(platform.env.now) + float(seconds)
    state.event = event
    state.unresolved = tuple(unresolved)


def end_service(platform):
    platform.service_state = ServiceState()


@dataclass(frozen=True)
class AvailabilityEstimate:
    """Known/estimated work, with unpriced future stages and waits kept separate.

    In particular, output I/O is not priced until the runtime selects its store.
    `known_work_s` must not be interpreted as an exact availability timestamp.
    """
    queued_work_s: float
    current_service_s: float
    virtual_backlog_s: float
    phase: str
    unresolved: tuple[str, ...]
    busy: bool
    unpriced_stages: tuple[str, ...] = ()

    @property
    def known_work_s(self):
        return self.queued_work_s + self.current_service_s + self.virtual_backlog_s

    @property
    def has_unresolved_wait(self):
        return bool(self.unresolved)


def platform_availability(platform, orchestrator, memo=None):
    from src.placement.live_audit import platform_queue_drain_seconds

    virtual = float(getattr(platform, "virtual_warmup_total_time", 0.0) or 0.0)
    # The legacy field deliberately includes the original virtual aggregate.
    queued = max(0.0, platform_queue_drain_seconds(platform, orchestrator, memo) - virtual)
    state = getattr(platform, "service_state", None)
    current = 0.0
    unresolved = []
    phase = "idle"
    busy = virtual > 0
    if state is not None and state.phase != "idle":
        phase = state.phase
        busy = True
        remaining = sum(state.pending.values())
        if state.deadline is not None:
            remaining += max(0.0, state.deadline - float(platform.env.now))
        if state.event is None or not state.event.triggered:
            unresolved.extend(state.unresolved)
        if state.virtual:
            virtual = virtual + remaining if phase == "fast_forward_warmup" else remaining
            # Fast-forward tasks still in the queue are represented by this timer.
            if phase == "fast_forward_warmup":
                represented = {id(t) for t in platform._warmup_tasks}
                view = SimpleNamespace(
                    id=platform.id, node=platform.node, type=platform.type,
                    virtual_warmup_total_time=0.0,
                    queue=SimpleNamespace(items=[t for t in platform.queue.items if id(t) not in represented]),
                    _payload_transfer_time=platform._payload_transfer_time,
                )
                queued = platform_queue_drain_seconds(view, orchestrator)
        else:
            current = remaining
    elif getattr(platform, "current_task", None) is not None:
        busy = True
        unresolved.append("missing_service_observation")
    initialized = getattr(platform, "initialized", None)
    if initialized is not None and not initialized.triggered:
        unresolved.append("initialization")
        busy = True
    if os.environ.get("HEROSIM_PEER_EXCHANGE", "0") == "1" and orchestrator is not None:
        table = getattr(orchestrator, "peer_exchange", {}) or {}
        arrived = getattr(orchestrator, "task_by_id", {}) or {}
        for task in platform.queue.items:
            for peer_id in table.get(int(task.id), {}):
                peer = arrived.get(peer_id)
                if peer is None or (getattr(peer, "platform", None) is None
                                    and getattr(peer, "planned_node_name", None) is None):
                    unresolved.append("queued_peer_rendezvous")
                    break
    return AvailabilityEstimate(queued, current, virtual, phase,
                                tuple(sorted(set(unresolved))), busy or queued > 0,
                                tuple(sorted(state.unpriced_stages)) if state is not None else ())
