"""A HEFT-style list scheduler served on the peer-group batch stack (uncut_groups_s0_v1, descriptive reference row).

HEFT (Topcuoglu et al., 2002) has two phases: rank the tasks by upward rank, then give each task, in decreasing rank, the processor with the
earliest finish time. Here a "processor" is an eligible replica and the earliest finish time is the registered greedy's score S
(`_PeerGreedyCore._pg_choose`): queue drain (including the service already committed in this batch) + cold start + execution + ingress latency +
the exchange time to the partners already placed. So this policy is the registered one-pass greedy with ONE change: the order the tasks are
placed in is decreasing upward rank instead of task id. No refine pass, no search, no learned component.

Upward rank, computed once per batch over the batch's tasks:
    rank_u(i) = mean_exec(i) + max over successors j of ( mean_comm(i, j) + rank_u(j) ),      rank_u(i) = mean_exec(i) with no successor.
mean_exec(i) is the mean of the execution-time table over the task's candidate platforms; a successor of i is a batch task j with a larger id
(ids follow the workload's topological order) that is i's exchange partner or has i among its dependencies; mean_comm(i, j) is the mean over
(sender node, receiver platform) pairs of the transfer time and link latency of the pair's exchange payload (the same two methods the
registered scorer uses), over at most `_RANK_NODES` nodes on each side (the first by node name) so the ranking stays bounded. Ties break by id.
Every candidate score and every transfer-time evaluation is metered as a plan evaluation, so the decision time charged by
`HEROSIM_DECISION_COST_MS` covers the ranking as well as the placement.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple, TYPE_CHECKING

from src.placement.decision_cost import meter as _meter
from src.policy.peer_greedy_network.scheduler import PeerGreedyNetworkBatchScheduler

if TYPE_CHECKING:
    from src.placement.infrastructure import Node, Platform, Task

_RANK_NODES = 8


def upward_ranks(
    ids: Sequence[int],
    mean_exec: Mapping[int, float],
    successors: Mapping[int, Sequence[int]],
    mean_comm: Mapping[Tuple[int, int], float],
) -> Dict[int, float]:
    """rank_u for every id. Every successor of i must have a larger id than i (ids are topologically ordered), which is what makes one pass
    in decreasing id order enough; anything else fails loud."""
    rank: Dict[int, float] = {}
    for i in sorted(ids, reverse=True):
        best = 0.0
        for j in successors.get(i, ()):
            if j <= i:
                raise ValueError(f"FAIL LOUD: successor {j} of task {i} does not have a larger id")
            best = max(best, float(mean_comm.get((i, j), 0.0)) + rank[j])
        rank[i] = float(mean_exec[i]) + best
    return rank


def rank_order(ids: Sequence[int], rank: Mapping[int, float]) -> List[int]:
    """Decreasing upward rank, ties by id."""
    return sorted(ids, key=lambda i: (-rank[i], i))


class HeftNetworkBatchScheduler(PeerGreedyNetworkBatchScheduler):
    """The registered batched greedy placed in decreasing upward-rank order (see the module docstring)."""

    _policy_label = "heft_network_batch"
    _live_audit_policy_name = "heft_network_batch"

    def _heft_mean_comm(self, cands_i, cands_j, payload: float) -> float:
        if payload <= 0.0:
            return 0.0
        senders = sorted({n.node_name for n, _p in cands_i})[:_RANK_NODES]
        receivers: Dict[str, "Platform"] = {}
        for n, p in sorted(cands_j, key=lambda c: (c[0].node_name, c[1].id)):
            receivers.setdefault(n.node_name, p)
        recv = [receivers[k] for k in sorted(receivers)[:_RANK_NODES]]
        total, count = 0.0, 0
        for platform in recv:
            for sender in senders:
                count += 1
                if sender != platform.node.node_name:
                    total += platform._payload_transfer_time(sender, payload) + platform.peer_link_latency(sender, context="heft rank estimate")
        _meter(self, count)
        return total / count if count else 0.0

    def _heft_ranks(self, batch_tasks: Sequence["Task"], cands: Mapping[int, Sequence[Tuple["Node", "Platform"]]], orch) -> Dict[int, float]:
        ids = [int(t.id) for t in batch_tasks]
        by_id = {int(t.id): (k, t) for k, t in enumerate(batch_tasks)}
        table = getattr(orch, "peer_exchange", None) or {}
        mean_exec: Dict[int, float] = {}
        for tid, (k, task) in by_id.items():
            costs = [self._pg_exec_table(task.type, p.type["shortName"]) for _n, p in cands[k]]
            mean_exec[tid] = sum(costs) / len(costs)
        successors: Dict[int, List[int]] = {}
        mean_comm: Dict[Tuple[int, int], float] = {}
        for i in ids:
            for j in ids:
                if j <= i:
                    continue
                partner = j in (table.get(i) or {}) or i in (table.get(j) or {})
                child = i in {int(d.id) for d in by_id[j][1].dependencies}
                if not (partner or child):
                    continue
                successors.setdefault(i, []).append(j)
                payload = float((table.get(i) or {}).get(j) or (table.get(j) or {}).get(i) or 0.0)
                mean_comm[(i, j)] = self._heft_mean_comm(cands[by_id[i][0]], cands[by_id[j][0]], payload)
        return upward_ranks(ids, mean_exec, successors, mean_comm)

    def _pg_batch_pass(
        self,
        batch_tasks: List["Task"],
        system_state,
        orch,
        *,
        memo: Dict[str, float],
        committed_service: Dict[str, float],
        planned: Dict[int, str],
        placements: Dict[int, Tuple[int, int]],
        service_of: Dict[int, Tuple[str, float]],
        refine: bool,
    ) -> int:
        if refine:
            raise RuntimeError("FAIL LOUD: heft_network_batch has no refine pass")
        cands: Dict[int, Sequence[Tuple["Node", "Platform"]]] = {}
        for k, task in enumerate(batch_tasks):
            valid = self._pg_restrict(task, self._get_valid_replicas(system_state.replicas.get(task.type["name"], set()), task))
            if not valid:
                raise RuntimeError(
                    f"{self._policy_label}: task {task.id} reached the decoder without a network-accessible replica; "
                    "the batch path defers those before decoding"
                )
            initialized = [r for r in valid if r[1].initialized.triggered]
            cands[k] = initialized if initialized else valid
        rank = self._heft_ranks(batch_tasks, cands, orch)
        position = {int(t.id): k for k, t in enumerate(batch_tasks)}
        for tid in rank_order([int(t.id) for t in batch_tasks], rank):
            idx = position[tid]
            task = batch_tasks[idx]
            node, platform, service = self._pg_choose(
                task, cands[idx], orch,
                memo=memo, committed_service=committed_service, planned=planned,
                nodes=self.nodes.items,
            )
            key = f"{node.node_name}:{platform.id}"
            committed_service[key] = committed_service.get(key, 0.0) + service
            planned[tid] = node.node_name
            placements[idx] = (node.id, platform.id)
            service_of[tid] = (key, service)
        return 0
