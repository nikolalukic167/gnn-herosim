"""Knative network ECT + FilterStore pull cost (physics-aware residual baseline).

Uses CACHE 5.6 observables (node cold_count × T_pull) as an explicit placement
cost. Proves whether pull-obs features close oracle_split residual when used
directly — before another learned retrain.

Phase 3: when ``ECT_PULL_DISTILL_DIR`` is set, each decision dumps a dim24
PyG frame + soft ECT targets for policy distillation.
"""

from __future__ import annotations

from typing import Generator, List, Tuple, TYPE_CHECKING

from src.placement.model import SystemState
from src.placement.scheduling_cost import expected_completion_with_filterstore_pull
from src.policy.knative_network.scheduler import KnativeScheduler as KnativeNetworkScheduler
from src.policy.knative_network_ect_pull.distill_log import (
    distill_enabled,
    maybe_log_ect_pull_decision,
)

if TYPE_CHECKING:
    from src.placement.infrastructure import Node, Platform, Task


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

        # Snapshot ledger BEFORE the choice (state the teacher scored under).
        pulls_before = {str(k): int(v) for k, v in pulls_committed.items()}

        def ect_pull_score(couple: Tuple["Node", "Platform"]) -> float:
            node, platform = couple
            key = f"{node.node_name}:{platform.id}"
            added = batch_added.get(key, 0) if batch_added is not None else 0
            extra = int(pulls_committed.get(str(node.node_name), 0))
            return float(
                expected_completion_with_filterstore_pull(
                    task,
                    node,
                    platform,
                    self.env.now,
                    added_in_batch=added,
                    extra_committed_pulls=extra,
                    nodes=self.nodes.items,
                )
            )

        scored: List[Tuple[float, str, Tuple["Node", "Platform"]]] = []
        for couple in candidates:
            node, platform = couple
            key = f"{node.node_name}:{platform.id}"
            scored.append((ect_pull_score(couple), key, couple))

        chosen_ect, _chosen_key, chosen = min(scored, key=lambda row: (row[0], row[1]))
        node, platform = chosen

        if distill_enabled():
            maybe_log_ect_pull_decision(
                self,
                system_state=system_state,
                task=task,
                candidates=[row[2] for row in scored],
                candidate_ect=[row[0] for row in scored],
                chosen=chosen,
                pulls_committed_before=pulls_before,
            )
            # Silence unused in non-debug paths.
            _ = chosen_ect

        if not platform.initialized.triggered:
            name = str(node.node_name)
            pulls_committed[name] = int(pulls_committed.get(name, 0)) + 1
        if batch_added is not None:
            key = f"{node.node_name}:{platform.id}"
            batch_added[key] = batch_added.get(key, 0) + 1
        return chosen
