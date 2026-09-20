"""peer_greedy_live_v1: a hand rule for peer-aware placement, served live.

The learned arms' only measured placement gain is co-location -- about 1 s of peer exchange
per task -- and their whole deficit is the wait they pay to see a peer group before deciding
(`unsaturated_edge_v1`, `batch_window_edge_v1`). The natural question is whether a two-line
rule that uses the same information gets the same gain, and what it costs without the wait.

Score of placing task i on candidate (node n, platform p), in SECONDS, every term charged by
the physics the simulator itself applies (`Platform.platform_process`):

    S(n, p) = drain(p)              seconds until p's standing backlog is served
                                    (`platform_queue_drain_seconds`, the audit's estimator)
            + cold(p) + exec_i(p)   incoming cold start and execution on p's device type
            + latency(src_i -> n)   the network hop the task pays to reach n
            + X_i(n)                exchange with every peer whose node is already KNOWN:
                                    `Platform._payload_transfer_time` + latency per remote
                                    peer, 0 for a co-located one; unknown peers contribute 0

The argmin is taken over Knative's own candidate set (network-reachable replicas, initialized
ones preferred) with Knative's tie-break, so with X == 0 and a flat drain estimate this IS
shortest-queue. X is the only peer-aware term and it is the term the exchange physics charges;
there is no free constant: a task joins a partner's node unless the queue there is longer, in
seconds, than the transfer it would save. `knative_network_ect` differs in the queue term
(`len(queue) x exec`, ~0.1 s per queued task, blind to the ~5 s of exchange each queued task
carries) and reads +12.8 % behind shortest-queue on the study environments -- which is why the
rule is built on the drain estimator and not on ECT.

Three flavours, three registry names, so the arm name carries the serving configuration:

  peer_greedy_network        per arrival, no wait (KnativeNetwork stack; item-1 competitor)
  drain_greedy_network       the same with X == 0 -- the ablation of the exchange term
  peer_greedy_network_batch  the GNN stack's peer-group batching (cell window, 16 s in the
                             study), greedy over the batch in task-id order with in-batch
                             commitments charged (B3-style: committed peers exact, uncommitted
                             ignored), `planned_node_name` pre-pass exactly as masked_topo

The peer flavours fail loud unless HEROSIM_PEER_EXCHANGE=1: a peer-aware rule served against
peer-free physics would score a term the simulator never charges.
"""
from __future__ import annotations

import os
from typing import Dict, Generator, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING

from src.placement.live_audit import orchestrator_of, platform_queue_drain_seconds
from src.placement.live_snapshot_seed import _approx_comm
from src.placement.model import SystemState
from src.placement.scheduling_cost import incoming_cold_start_time, network_latency_between
from src.policy.gnn.scheduler import GNNScheduler, PREFIX_DECODE_MODE
from src.policy.knative_network.scheduler import KnativeScheduler as KnativeNetworkScheduler

if TYPE_CHECKING:
    from src.placement.infrastructure import Node, Platform, Task


PEER_GREEDY_COUNTERS = (
    "pg_decisions", "pg_partners_known", "pg_partners_unknown", "pg_joined_partner",
    "pg_moved_by_exchange", "pg_batches",
)


def _require_peer_physics(policy_name: str) -> None:
    if os.environ.get("HEROSIM_PEER_EXCHANGE", "0") != "1":
        raise RuntimeError(
            f"FAIL LOUD: {policy_name} scores peer exchange but HEROSIM_PEER_EXCHANGE is not 1; "
            "the simulator would never charge the term the rule optimises"
        )


class _PeerGreedyCore:
    """The score and its counters. Mixed into two scheduler stacks; holds no SimPy state."""

    exchange_on: bool = True
    _policy_label: str = "peer_greedy"

    def _pg_init(self) -> None:
        self.pg_decisions = 0
        self.pg_partners_known = 0
        self.pg_partners_unknown = 0
        self.pg_joined_partner = 0
        self.pg_moved_by_exchange = 0
        self.pg_batches = 0
        if self.exchange_on:
            _require_peer_physics(self._policy_label)

    def _pg_orchestrator(self):
        orch = orchestrator_of(self)
        if orch is None:
            raise RuntimeError(f"{self._policy_label}: no node carries an orchestrator_ref yet")
        return orch

    def _pg_peer_nodes(self, task: "Task", orch, planned: Dict[int, str]) -> List[Tuple[str, float]]:
        """[(node_name, bytes)] for every peer of `task` whose node is already known: placed
        (platform set), planned by a batch pre-pass, or committed earlier in this batch."""
        table = getattr(orch, "peer_exchange", None) or {}
        peers = table.get(int(task.id)) or {}
        known: List[Tuple[str, float]] = []
        for peer_id in sorted(peers):
            node_name = planned.get(int(peer_id))
            if node_name is None:
                peer = orch.task_by_id.get(int(peer_id))
                if peer is not None:
                    plat = getattr(peer, "platform", None)
                    node_name = (plat.node.node_name if plat is not None
                                 else getattr(peer, "planned_node_name", None))
            if node_name is None:
                self.pg_partners_unknown += 1
                continue
            self.pg_partners_known += 1
            known.append((node_name, float(peers[peer_id])))
        return known

    @staticmethod
    def _pg_exchange_seconds(node: "Node", platform: "Platform",
                             peer_nodes: Sequence[Tuple[str, float]]) -> float:
        """What `Platform._peer_exchange_time` will charge at this task's input stage for the
        peers whose node is known -- the same method, not a re-derivation."""
        if not peer_nodes:
            return 0.0
        network_map = getattr(node, "network_map", None) or {}
        total = 0.0
        for peer_node_name, payload in peer_nodes:
            if peer_node_name == node.node_name:
                continue
            entry = network_map.get(peer_node_name)
            if entry is None:
                raise RuntimeError(
                    f"{node.node_name} has no network_map entry for {peer_node_name}; the "
                    "physics would fail on this placement too (server mesh reachability)"
                )
            latency = float(entry.get("latency", 0.0)) if isinstance(entry, dict) else float(entry)
            total += platform._payload_transfer_time(peer_node_name, payload) + latency
        return total

    def _pg_choose(
        self,
        task: "Task",
        candidates: Sequence[Tuple["Node", "Platform"]],
        orch,
        *,
        memo: Dict[str, float],
        committed_service: Dict[str, float],
        planned: Dict[int, str],
        nodes,
    ) -> Tuple["Node", "Platform", float]:
        """Argmin of S over `candidates`; returns the couple and the service seconds the task
        adds to its platform's backlog (for in-batch commitments)."""
        peer_nodes = self._pg_peer_nodes(task, orch, planned) if self.exchange_on else []
        comm = _approx_comm(task.type)
        scored = []
        for node, platform in candidates:
            key = f"{node.node_name}:{platform.id}"
            plat_type = platform.type["shortName"]
            drain = platform_queue_drain_seconds(platform, orch, memo) + committed_service.get(key, 0.0)
            exec_s = float(task.type["executionTime"].get(plat_type, 0.0) or 0.0)
            base = (drain + incoming_cold_start_time(task, platform) + exec_s
                    + network_latency_between(task.node_name, node, nodes))
            exch = self._pg_exchange_seconds(node, platform, peer_nodes)
            scored.append((base + exch, base, exch, exec_s + comm, node, platform))
        best = min(scored, key=lambda s: (s[0], s[4].id, s[5].id))
        self.pg_decisions += 1
        if peer_nodes:
            best_without = min(scored, key=lambda s: (s[1], s[4].id, s[5].id))
            if (best_without[4].id, best_without[5].id) != (best[4].id, best[5].id):
                self.pg_moved_by_exchange += 1
            if any(pn == best[4].node_name for pn, _b in peer_nodes):
                self.pg_joined_partner += 1
        # service this task adds to the chosen backlog: its own exec + I/O + its exchange
        return best[4], best[5], best[3] + best[2]


class PeerGreedyNetworkScheduler(_PeerGreedyCore, KnativeNetworkScheduler):
    """Per-arrival peer-aware greedy on the knative_network stack (no batching, no wait)."""

    _policy_label = "peer_greedy_network"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pg_init()

    def placement(self, system_state: SystemState, task: "Task") -> Generator:
        if False:
            yield
        replicas = system_state.replicas[task.type["name"]]
        valid = self._get_valid_replicas(replicas, task)
        if not valid:
            raise ValueError(f"No valid replicas for task {task.id}")
        initialized = [r for r in valid if r[1].initialized.triggered]
        candidates = initialized if initialized else valid
        node, platform, _svc = self._pg_choose(
            task, candidates, self._pg_orchestrator(),
            memo={}, committed_service={}, planned={}, nodes=self.nodes.items,
        )
        return node, platform


class DrainGreedyNetworkScheduler(PeerGreedyNetworkScheduler):
    """The ablation: the same rule with the exchange term switched off."""

    exchange_on = False
    _policy_label = "drain_greedy_network"


class PeerGreedyNetworkBatchScheduler(_PeerGreedyCore, GNNScheduler):
    """The rule on the learned arms' serving stack: peer-group batching at the cell's window,
    masked_topo's batch path (planned_node_name pre-pass, deferred tasks to the autoscaler),
    with the greedy in place of the decoder. Requires the same env the gnn arms are served
    with (GNN_DECODE_MODE=masked_topo, GNN_BATCH_BY_PEER_GROUP=1) so the two stacks differ in
    the decision alone."""

    _policy_label = "peer_greedy_network_batch"
    _live_audit_policy_name = "peer_greedy_network_batch"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._decode_mode != PREFIX_DECODE_MODE or not self.batch_by_peer_group:
            raise RuntimeError(
                "FAIL LOUD: peer_greedy_network_batch must be served exactly as the learned "
                "arms are: export GNN_DECODE_MODE=masked_topo and GNN_BATCH_BY_PEER_GROUP=1"
            )
        self._pg_init()

    def set_models(self, models: dict):
        if models:
            raise RuntimeError(
                "FAIL LOUD: peer_greedy_network_batch is a hand rule and takes no model; "
                f"got models with keys {sorted(models)}"
            )

    def _prefix_inference(
        self,
        batch_tasks: List["Task"],
        system_state: SystemState,
        queue_snapshot: Dict[str, int],
        temporal_state: Optional[Dict[str, Dict[str, float]]],
    ) -> Dict[int, Tuple[int, int]]:
        orch = self._pg_orchestrator()
        memo: Dict[str, float] = {}
        committed_service: Dict[str, float] = {}
        planned: Dict[int, str] = {}
        placements: Dict[int, Tuple[int, int]] = {}
        order = sorted(range(len(batch_tasks)), key=lambda i: int(batch_tasks[i].id))
        for idx in order:
            task = batch_tasks[idx]
            valid = self._get_valid_replicas(system_state.replicas.get(task.type["name"], set()), task)
            if not valid:
                raise RuntimeError(
                    f"peer_greedy_network_batch: task {task.id} reached the decoder without a "
                    "network-accessible replica; the batch path defers those before decoding"
                )
            initialized = [r for r in valid if r[1].initialized.triggered]
            candidates = initialized if initialized else valid
            node, platform, service = self._pg_choose(
                task, candidates, orch,
                memo=memo, committed_service=committed_service, planned=planned,
                nodes=self.nodes.items,
            )
            key = f"{node.node_name}:{platform.id}"
            committed_service[key] = committed_service.get(key, 0.0) + service
            planned[int(task.id)] = node.node_name
            placements[idx] = (node.id, platform.id)
        self.pg_batches += 1
        # the batch path counts what the decoder saw; the rule keeps the same books
        table = getattr(orch, "peer_exchange", None) or {}
        ids: Set[int] = {int(t.id) for t in batch_tasks}
        pairs_in = sum(1 for i in ids for j in (table.get(i) or {}) if j in ids) // 2
        outside = sum(1 for i in ids for j in (table.get(i) or {}) if j not in ids)
        self.prefix_pairs_in_batch += pairs_in
        self.prefix_peers_outside_batch += outside
        return placements
