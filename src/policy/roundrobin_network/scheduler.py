"""
Per-arrival network-aware round-robin scheduler.

Uses the same orchestrator/autoscaler stack as knative_network (KnativeNetworkOrchestrator,
KnativeNetworkAutoscaler) so baseline comparisons match Knative/HRC regime: one task per
scheduling wakeup, network-filtered replicas, kn-autoscale with source_node_name.
Only placement differs: least scheduled_count among valid initialized replicas.
"""

from __future__ import annotations

from typing import Generator, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from src.placement.infrastructure import Node, Platform, Task

from src.placement.model import SystemState
from src.policy.knative_network.scheduler import KnativeScheduler


class RoundRobinNetworkScheduler(KnativeScheduler):
    def placement(self, system_state: SystemState, task: Task) -> Generator:
        if False:
            yield

        replicas: Set[Tuple[Node, Platform]] = system_state.replicas[task.type["name"]]
        valid_replicas = self._get_valid_replicas(replicas, task)
        if not valid_replicas:
            raise ValueError(f"No valid replicas for task {task.id}")

        initialized_replicas = [r for r in valid_replicas if r[1].initialized.triggered]
        candidates = initialized_replicas if initialized_replicas else valid_replicas

        state = system_state.scheduler_state
        if not hasattr(state, "scheduled_count") or state.scheduled_count is None:
            state.scheduled_count = {tn: {} for tn in self.data.task_types}
        if task.type["name"] not in state.scheduled_count:
            state.scheduled_count[task.type["name"]] = {}

        chosen = min(
            candidates,
            key=lambda couple: state.scheduled_count[task.type["name"]].get(
                (couple[0].id, couple[1].id), 0
            ),
        )
        key = (chosen[0].id, chosen[1].id)
        state.scheduled_count[task.type["name"]][key] = (
            state.scheduled_count[task.type["name"]].get(key, 0) + 1
        )
        return chosen
