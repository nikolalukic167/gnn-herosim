"""Coordinate refinement of a GNN plan with the existing peer-greedy rule."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from src.placement.live_snapshot_seed import _approx_comm
from src.placement.model import SystemState
from src.policy.gnn.scheduler import GNNScheduler
from src.policy.peer_greedy_network.scheduler import PeerGreedyNetworkCDScheduler


class GNNSeededCDScheduler(PeerGreedyNetworkCDScheduler):
    _policy_label = "gnn_seeded_cd"
    _live_audit_policy_name = "gnn_seeded_cd"

    def set_models(self, models: dict):
        GNNScheduler.set_models(self, models)

    def _prefix_inference(
        self,
        batch_tasks: List,
        system_state: SystemState,
        queue_snapshot: Dict[str, int],
        temporal_state: Optional[Dict[str, Dict[str, float]]],
    ) -> Dict[int, Tuple[int, int]]:
        seed = GNNScheduler._prefix_inference(
            self, batch_tasks, system_state, queue_snapshot, temporal_state
        )
        if set(seed) != set(range(len(batch_tasks))):
            raise RuntimeError("gnn_seeded_cd: GNN did not place every decodable task")

        orch = self._pg_orchestrator()
        replicas = {}
        planned = {}
        for idx, task in enumerate(batch_tasks):
            valid = self._get_valid_replicas(
                system_state.replicas.get(task.type["name"], set()), task
            )
            matches = [
                (node, platform) for node, platform in valid
                if (node.id, platform.id) == tuple(seed[idx])
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"gnn_seeded_cd: task {task.id} seed {seed[idx]} is not one valid replica"
                )
            replicas[idx] = matches[0]
            planned[int(task.id)] = matches[0][0].node_name

        committed_service: Dict[str, float] = {}
        service_of: Dict[int, Tuple[str, float]] = {}
        placements = dict(seed)
        for idx, task in enumerate(batch_tasks):
            node, platform = replicas[idx]
            peer_nodes = self._pg_peer_nodes(task, orch, planned)
            execution = float(
                task.type["executionTime"].get(platform.type["shortName"], 0.0) or 0.0
            )
            service = (
                execution + _approx_comm(task.type)
                + self._pg_exchange_seconds(node, platform, peer_nodes)
            )
            key = f"{node.node_name}:{platform.id}"
            committed_service[key] = committed_service.get(key, 0.0) + service
            service_of[int(task.id)] = (key, service)

        self._pg_refine(
            batch_tasks, system_state, orch, {}, committed_service,
            planned, placements, service_of,
        )
        self.pg_batches += 1
        return placements
