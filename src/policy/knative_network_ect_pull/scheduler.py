"""Knative network ECT + FilterStore pull cost (physics-aware residual baseline).

Uses CACHE 5.6 observables (node cold_count × T_pull) as an explicit placement
cost. Proves whether pull-obs features close oracle_split residual when used
directly — before another learned retrain.
"""

from __future__ import annotations

from typing import Generator, TYPE_CHECKING

from src.placement.model import SystemState
from src.placement.scheduling_cost import expected_completion_with_filterstore_pull
from src.policy.knative_network.scheduler import KnativeScheduler as KnativeNetworkScheduler

if TYPE_CHECKING:
    from src.placement.infrastructure import Task


class KnativeECTPullScheduler(KnativeNetworkScheduler):
    """Min ECT + estimated FilterStore pull wait (cold_count × T_pull)."""

    def placement(self, system_state: SystemState, task: "Task") -> Generator:
        if False:
            yield

        replicas = system_state.replicas[task.type["name"]]
        valid_replicas = self._get_valid_replicas(replicas, task)
        if not valid_replicas:
            raise ValueError(f"No valid replicas for task {task.id}")

        # Do NOT prefer already-initialized only — oracle_split is all-cold;
        # restricting to warm collapses the action space once the first pull finishes.
        candidates = valid_replicas

        batch_added = getattr(self, "_ect_batch_added", None)
        # Per-node pulls committed in this decision window (feature roll-forward).
        pulls_committed = getattr(self, "_ect_pull_committed", None)
        if pulls_committed is None:
            pulls_committed = {}
            self._ect_pull_committed = pulls_committed

        def ect_pull_key(couple):
            node, platform = couple
            key = f"{node.node_name}:{platform.id}"
            added = batch_added.get(key, 0) if batch_added is not None else 0
            extra = int(pulls_committed.get(str(node.node_name), 0))
            return (
                expected_completion_with_filterstore_pull(
                    task,
                    node,
                    platform,
                    self.env.now,
                    added_in_batch=added,
                    extra_committed_pulls=extra,
                    nodes=self.nodes.items,
                ),
                key,
            )

        chosen = min(candidates, key=ect_pull_key)
        node, platform = chosen
        if not platform.initialized.triggered:
            name = str(node.node_name)
            pulls_committed[name] = int(pulls_committed.get(name, 0)) + 1
        if batch_added is not None:
            key = f"{node.node_name}:{platform.id}"
            batch_added[key] = batch_added.get(key, 0) + 1
        return chosen
