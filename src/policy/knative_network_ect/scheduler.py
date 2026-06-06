"""Network-aware Knative scheduler using expected completion time (ECT)."""

from __future__ import annotations

from typing import Generator, TYPE_CHECKING

from src.placement.model import SystemState
from src.placement.scheduling_cost import expected_completion_for_candidate
from src.policy.knative_network.scheduler import KnativeScheduler as KnativeNetworkScheduler

if TYPE_CHECKING:
    from src.placement.infrastructure import Task


class KnativeECTScheduler(KnativeNetworkScheduler):
    """Knative network placement by minimum expected completion time."""

    def placement(self, system_state: SystemState, task: "Task") -> Generator:
        if False:
            yield

        replicas = system_state.replicas[task.type["name"]]
        valid_replicas = self._get_valid_replicas(replicas, task)
        if not valid_replicas:
            raise ValueError(f"No valid replicas for task {task.id}")

        initialized_replicas = [r for r in valid_replicas if r[1].initialized.triggered]
        candidates = initialized_replicas if initialized_replicas else valid_replicas

        batch_added = getattr(self, "_ect_batch_added", None)

        def ect_key(couple):
            node, platform = couple
            key = f"{node.node_name}:{platform.id}"
            added = batch_added.get(key, 0) if batch_added is not None else 0
            return (
                expected_completion_for_candidate(
                    task,
                    node,
                    platform,
                    self.env.now,
                    added_in_batch=added,
                    nodes=self.nodes.items,
                ),
                key,
            )

        chosen = min(candidates, key=ect_key)
        if batch_added is not None:
            key = f"{chosen[0].node_name}:{chosen[1].id}"
            batch_added[key] = batch_added.get(key, 0) + 1
        return chosen
