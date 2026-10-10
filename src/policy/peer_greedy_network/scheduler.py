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

from src.placement.live_audit import inflight_remaining_seconds, orchestrator_of, platform_queue_drain_seconds
from src.placement.live_snapshot_seed import _approx_comm
from src.placement.model import SystemState
from src.placement.scheduling_cost import incoming_cold_start_time, network_latency_between
from src.policy.gnn.scheduler import GNNScheduler, PREFIX_DECODE_MODE
from src.policy.knative_network.scheduler import KnativeScheduler as KnativeNetworkScheduler

if TYPE_CHECKING:
    from src.placement.infrastructure import Node, Platform, Task


PEER_GREEDY_COUNTERS = (
    "pg_decisions", "pg_partners_known", "pg_partners_unknown", "pg_joined_partner",
    "pg_moved_by_exchange", "pg_batches", "pg_cd_passes", "pg_cd_moves", "pg_forced",
    "pg_partners_blinded", "pg_inflight_charged", "pg_inflight_seconds",
    "pg_ext_batches", "pg_ext_charged", "pg_ext_seconds",
    "pg_declared_batches", "pg_declared_pruned", "pg_declared_sub_batched", "pg_declared_groups",
)

# cd_gap_v1 D1 (2026-09-25): a DISCLOSED probe knob for the batched flavours only. A partner outside
# the batch being decided is never priced, even when it is already placed -- exactly the information
# the served GNN decoder has (`prefix_serving.py`: peers outside the batch are invisible by contract).
PG_BATCH_BLIND_ENV = "HEROSIM_PG_BATCH_BLIND"

# burst_ladder_v1 (2026-09-26): a DISCLOSED control knob. The drain term never charges the task a
# platform is serving now (`platform_queue_drain_seconds` walks only the queue), and that task's
# remaining service includes its input + peer-exchange stage, so under a burst a busy platform reads
# ~idle. 1 adds `live_audit.inflight_remaining_seconds` -- the service end the platform recorded -- to
# the drain. 0 (the default) is the registered rule, byte-identical.
PG_INFLIGHT_ENV = "HEROSIM_PG_INFLIGHT"
# accel pull-hold control (cd_pull): price the wait an image pull holding the candidate node's local storage adds at the
# task's output write, max(0, hold - time to that write); needs the pull ledger (HEROSIM_PULL_LEDGER=1)
PG_PULL_HOLD_ENV = "HEROSIM_PG_PULL_HOLD"

# joint_burst_v2 (2026-09-20): a DISCLOSED probe knob, never a registered arm's default. The
# exchange term X is multiplied by this factor in the score (the service a task adds to its
# backlog stays unscaled). 1.0 is the rule. The x2 arm asks whether MORE co-location than the
# rule buys is live-valid -- the direction the group-optimum label pushes a learned arm in.
PG_EXCHANGE_SCALE_ENV = "HEROSIM_PG_EXCHANGE_SCALE"

# peak_controls_v1 (2026-09-30): the co-sim label's queueing externality added to the rule's score, for
# the batched flavours. A candidate that adds `a` seconds to a backlog of `B` seconds is charged
# (lambda_p / 2)[(B + a)^2 - B^2], lambda_p = rate x the batch's type share / |that type's replicas|,
# summed over the types p serves -- drift_label.py's lambda_p, with this env's value as the rate
# (tasks/s). Unset = the registered rule, byte-identical.
PG_EXT_RATE_ENV = "HEROSIM_PG_EXT_RATE"


def _pg_ext_rate() -> Optional[float]:
    raw = os.environ.get(PG_EXT_RATE_ENV, "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"FAIL LOUD: {PG_EXT_RATE_ENV}={raw!r} is not a float") from exc
    if not value > 0.0:
        raise ValueError(f"FAIL LOUD: {PG_EXT_RATE_ENV} must be > 0, got {value}")
    return value


# hidden_exec_s0_v1: what the rule knows about execution time. `table` (default) reads the
# task-type table; `oracle` also reads exec_physics' hidden node constants and current co-execution.
PG_EXEC_KNOWLEDGE_ENV = "HEROSIM_PG_EXEC_KNOWLEDGE"


def _pg_exec_oracle() -> bool:
    from src.placement.exec_physics import TABLE_V0, resolve_exec_physics

    raw = os.environ.get(PG_EXEC_KNOWLEDGE_ENV, "table").strip() or "table"
    if raw not in ("table", "oracle"):
        raise ValueError(f"FAIL LOUD: {PG_EXEC_KNOWLEDGE_ENV}={raw!r}; expected table or oracle")
    if raw == "oracle" and resolve_exec_physics() == TABLE_V0:
        raise ValueError(f"FAIL LOUD: {PG_EXEC_KNOWLEDGE_ENV}=oracle under table_v0 has nothing to know")
    return raw == "oracle"


def _pg_exchange_scale() -> float:
    raw = os.environ.get(PG_EXCHANGE_SCALE_ENV, "1.0").strip() or "1.0"
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"FAIL LOUD: {PG_EXCHANGE_SCALE_ENV}={raw!r} is not a float") from exc
    if value < 0.0:
        raise ValueError(f"FAIL LOUD: {PG_EXCHANGE_SCALE_ENV} must be >= 0, got {value}")
    return value


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
    _pg_batched: bool = False
    _pg_batch_ids: Set[int] = frozenset()
    # cd_declared: {task id -> {(node_id, platform_id)}} while a declared-pruning group is being decided, else None (inert)
    _pg_allowed: Optional[Dict[int, Set[Tuple[int, int]]]] = None

    def _pg_restrict(self, task, valid):
        """The declared slate's top-5 for this task, or `valid` unchanged when no slate is being applied."""
        if self._pg_allowed is None:
            return valid
        allowed = self._pg_allowed[int(task.id)]
        return [r for r in valid if (r[0].id, r[1].id) in allowed]

    def _pg_init(self) -> None:
        self.pg_declared_batches = 0
        self.pg_declared_pruned = 0
        self.pg_declared_sub_batched = 0
        self.pg_declared_groups = 0
        self.pg_decisions = 0
        self.pg_partners_known = 0
        self.pg_partners_unknown = 0
        self.pg_joined_partner = 0
        self.pg_moved_by_exchange = 0
        self.pg_batches = 0
        self.pg_cd_passes = 0
        self.pg_cd_moves = 0
        self.pg_exchange_scale = _pg_exchange_scale()
        self.pg_exec_oracle = _pg_exec_oracle()
        # forced_placements: {task_id -> (node_id, platform_id)} injected by the orchestrator from
        # config["infrastructure"]["forced_placements"] (rollout_imitation_v1's label engine forces
        # one task to a candidate and lets the rule choose everything else). Empty = normal rule.
        self.forced_placements: Dict[int, Tuple[int, int]] = {}
        self.pg_forced = 0
        self._pg_capture_path = os.environ.get("HEROSIM_PG_CAPTURE_PATH") or None
        self.pg_partners_blinded = 0
        raw_blind = os.environ.get(PG_BATCH_BLIND_ENV, "0").strip() or "0"
        if raw_blind not in ("0", "1"):
            raise ValueError(f"FAIL LOUD: {PG_BATCH_BLIND_ENV} must be 0 or 1, got {raw_blind!r}")
        self.pg_batch_blind = raw_blind == "1"
        raw_inflight = os.environ.get(PG_INFLIGHT_ENV, "0").strip() or "0"
        if raw_inflight not in ("0", "1"):
            raise ValueError(f"FAIL LOUD: {PG_INFLIGHT_ENV} must be 0 or 1, got {raw_inflight!r}")
        self.pg_inflight = raw_inflight == "1"
        self.pg_inflight_charged = 0
        self.pg_inflight_seconds = 0.0
        raw_pull = os.environ.get(PG_PULL_HOLD_ENV, "0").strip() or "0"
        if raw_pull not in ("0", "1"):
            raise ValueError(f"FAIL LOUD: {PG_PULL_HOLD_ENV} must be 0 or 1, got {raw_pull!r}")
        self.pg_pull_hold = raw_pull == "1"
        if self.pg_pull_hold:
            from src.placement.infrastructure import PULL_LEDGER

            if not PULL_LEDGER:
                raise ValueError(f"FAIL LOUD: {PG_PULL_HOLD_ENV}=1 reads the pull ledger; set HEROSIM_PULL_LEDGER=1")
        self.pg_pull_charged = 0
        self.pg_pull_seconds = 0.0
        self.pg_ext_rate = _pg_ext_rate()
        self._pg_ext_lambda: Dict[str, float] = {}
        self.pg_ext_batches = 0
        self.pg_ext_charged = 0
        self.pg_ext_seconds = 0.0
        if self.pg_ext_rate is not None and not self._pg_batched:
            raise RuntimeError(
                f"FAIL LOUD: {PG_EXT_RATE_ENV} is defined for the batched flavours only (lambda_p is "
                f"split by the batch's type mix); {self._policy_label} decides per arrival"
            )
        if self.pg_batch_blind and not self._pg_batched:
            raise RuntimeError(
                f"FAIL LOUD: {PG_BATCH_BLIND_ENV}=1 is defined for the batched flavours only; "
                f"{self._policy_label} decides per arrival and has no batch to be blind outside of"
            )
        if self.exchange_on:
            _require_peer_physics(self._policy_label)

    def _pg_capture(self, task_id: int, candidate_feature_rows) -> None:
        """Append one decision. `candidate_feature_rows` are pre-formatted rows built in _pg_choose:
        [node_id, plat_id, node_name, plat_str, drain, cold, exec, latency, exchange]."""
        import json as _json
        rec = {"task_id": task_id, "candidates": candidate_feature_rows}
        with open(self._pg_capture_path, "a") as fh:
            fh.write(_json.dumps(rec) + "\n")

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
            if node_name is None and self.pg_batch_blind and int(peer_id) not in self._pg_batch_ids:
                self.pg_partners_blinded += 1
                continue
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
        total = 0.0
        for peer_node_name, payload in peer_nodes:
            if peer_node_name == node.node_name:
                continue
            total += platform._payload_transfer_time(peer_node_name, payload) + platform.peer_link_latency(
                peer_node_name, context="peer-greedy estimate")
        return total

    def _pg_xf(self, platform) -> float:
        if not self.pg_exec_oracle:
            return 1.0
        from src.placement.exec_physics import expected_factor

        return expected_factor(platform)

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
        feats = []  # rollout_imitation_v1: per-candidate score terms captured for training
        for node, platform in candidates:
            key = f"{node.node_name}:{platform.id}"
            plat_type = platform.type["shortName"]
            xf = self._pg_xf(platform)
            drain = (platform_queue_drain_seconds(platform, orch, memo, exec_scale=xf)
                     + committed_service.get(key, 0.0))
            if self.pg_inflight:
                remaining = inflight_remaining_seconds(platform)
                if remaining:
                    drain += remaining
                    self.pg_inflight_charged += 1
                    self.pg_inflight_seconds += remaining
            exec_s = float(task.type["executionTime"].get(plat_type, 0.0) or 0.0) * xf
            cold = incoming_cold_start_time(task, platform)
            lat = network_latency_between(task.node_name, node, nodes)
            base = drain + cold + exec_s + lat
            exch = self._pg_exchange_seconds(node, platform, peer_nodes)
            if getattr(self, "pg_pull_hold", False):
                hold_key = f"pull_hold:{node.node_name}"
                if hold_key not in memo:
                    from src.placement.snapshot_fidelity import node_pull_hold_seconds

                    memo[hold_key] = node_pull_hold_seconds(self, node, float(self.env.now))
                # the hold blocks the task at its output write, after drain, cold start, ingress, exchange and execution
                pull_wait = max(0.0, memo[hold_key] - (drain + cold + lat + exch + exec_s))
                if pull_wait > 0.0:
                    base += pull_wait
                    self.pg_pull_charged += 1
                    self.pg_pull_seconds += pull_wait
            ext = 0.0
            if self.pg_ext_rate is not None:
                lam = self._pg_ext_lambda.get(key, 0.0)
                added = exec_s + comm + exch
                ext = 0.5 * lam * ((drain + added) ** 2 - drain ** 2)
            if self._pg_capture_path:
                feats.append([int(node.id), int(platform.id), node.node_name, str(platform.id),
                              float(drain), float(cold), float(exec_s), float(lat), float(exch)])
            scored.append((base + self.pg_exchange_scale * exch + ext, base, exch, exec_s + comm, node, platform, ext))
        best = min(scored, key=lambda s: (s[0], s[4].id, s[5].id))
        if best[6] > 0.0:
            self.pg_ext_charged += 1
            self.pg_ext_seconds += best[6]
        if self._pg_capture_path and feats:
            self._pg_capture(int(task.id), feats)
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
        # capture (candidate set + score-term features) happens inside _pg_choose when
        # HEROSIM_PG_CAPTURE_PATH is set; forced runs never capture (path unset there).
        forced = None
        if self.forced_placements:
            forced = self.forced_placements.get(int(task.id))
            if forced is None:
                forced = self.forced_placements.get(str(task.id))
        if forced is not None:
            fn, fp = int(forced[0]), int(forced[1])
            match = next((c for c in valid if c[0].id == fn and c[1].id == fp), None)
            if match is None:
                raise RuntimeError(
                    f"{self._policy_label}: forced placement ({fn},{fp}) for task {task.id} is not "
                    f"a valid replica; valid={[(n.id, p.id) for n, p in valid]}"
                )
            self.pg_forced += 1
            return match[0], match[1]
        node, platform, _svc = self._pg_choose(
            task, candidates, self._pg_orchestrator(),
            memo={}, committed_service={}, planned={}, nodes=self.nodes.items,
        )
        return node, platform


class DrainGreedyNetworkScheduler(PeerGreedyNetworkScheduler):
    """The ablation: the same rule with the exchange term switched off."""

    exchange_on = False
    _policy_label = "drain_greedy_network"


class PeerGreedyLookaheadNetworkScheduler(PeerGreedyNetworkScheduler):
    """lookahead_mp_v1 P0: the immediate rule, plus a price for partners that have NOT arrived.

    The physics charges this task, at its input stage, the exchange to EVERY partner: it waits
    for an unarrived one to be placed, then pays against where that partner lands
    (Platform._peer_rendezvous_events / _peer_exchange_time). The rule prices such partners at 0
    because their node is unknown, so this task's node choice already fixes a cost the rule never
    sees. This arm predicts the partner's node -- the node its bytes-heaviest already-known partner
    (other than this task) runs on, a two-step hand estimate -- and prices it with the rule's own
    _pg_exchange_seconds at weight 1.0. A partner with no known partner of its own stays unpriced,
    exactly as in the rule (pg_lookahead_blind).

    Disclosed: pg_joined_partner here counts joining a known OR a predicted partner's node."""

    _policy_label = "peer_greedy_lookahead_network"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pg_lookahead_priced = 0
        self.pg_lookahead_blind = 0

    @staticmethod
    def _pg_node_of(orch, peer_id: int, planned: Dict[int, str]) -> Optional[str]:
        node_name = planned.get(int(peer_id))
        if node_name is not None:
            return node_name
        peer = orch.task_by_id.get(int(peer_id))
        if peer is None:
            return None
        plat = getattr(peer, "platform", None)
        return plat.node.node_name if plat is not None else getattr(peer, "planned_node_name", None)

    def _pg_predict_node(self, task: "Task", orch, peer_id: int,
                         planned: Dict[int, str]) -> Optional[str]:
        table = getattr(orch, "peer_exchange", None) or {}
        weight: Dict[str, float] = {}
        for q, b in (table.get(int(peer_id)) or {}).items():
            if int(q) == int(task.id):
                continue
            node_name = self._pg_node_of(orch, int(q), planned)
            if node_name is not None:
                weight[node_name] = weight.get(node_name, 0.0) + float(b)
        if not weight:
            return None
        return max(sorted(weight), key=lambda n: weight[n])

    def _pg_peer_nodes(self, task: "Task", orch, planned: Dict[int, str]) -> List[Tuple[str, float]]:
        known = super()._pg_peer_nodes(task, orch, planned)
        peers = (getattr(orch, "peer_exchange", None) or {}).get(int(task.id)) or {}
        for peer_id in sorted(peers):
            if self._pg_node_of(orch, int(peer_id), planned) is not None:
                continue
            node_name = self._pg_predict_node(task, orch, int(peer_id), planned)
            if node_name is None:
                self.pg_lookahead_blind += 1
                continue
            self.pg_lookahead_priced += 1
            known.append((node_name, float(peers[peer_id])))
        return known


class PeerGreedyOracleNetworkScheduler(PeerGreedyLookaheadNetworkScheduler):
    """lookahead_mp_v1 P0 headroom bound: an unarrived partner's node is where it ACTUALLY ran in
    the paired rule run of the same environment (HEROSIM_PG_ORACLE_NODES -> {task_id: node_name},
    written from that run's taskResults). No deployable policy has this knowledge; it bounds what
    lookahead can buy along the rule's trajectory. Disclosed: this arm's own decisions shift that
    trajectory, so a partner may not land where the rule run put it."""

    _policy_label = "peer_greedy_oracle_network"

    def __init__(self, *args, **kwargs):
        import json

        path = os.environ.get("HEROSIM_PG_ORACLE_NODES")
        if not path or not os.path.exists(path):
            raise RuntimeError(
                f"FAIL LOUD: peer_greedy_oracle_network needs HEROSIM_PG_ORACLE_NODES=<nodes.json>, got {path!r}"
            )
        self._oracle_nodes = {int(k): str(v) for k, v in json.load(open(path)).items()}
        super().__init__(*args, **kwargs)

    def _pg_predict_node(self, task: "Task", orch, peer_id: int,
                         planned: Dict[int, str]) -> Optional[str]:
        node_name = self._oracle_nodes.get(int(peer_id))
        if node_name is None:
            raise RuntimeError(
                f"FAIL LOUD: oracle map has no node for task {peer_id}; it must come from the "
                "paired rule run of this exact environment"
            )
        return node_name


class PeerGreedySelfPredictNetworkScheduler(PeerGreedyLookaheadNetworkScheduler):
    """lookahead_mp_v1 P0b: the hand COORDINATION control. An unarrived partner's node is predicted
    as the rule's own argmin for that partner if it arrived now -- its type and client (read from
    its workload event), the current queues, and exchange to ITS already-known partners (this task
    excluded: its node is what is being decided). No learning, no message passing. If this recovers
    most of the oracle's headroom, lookahead is hand-buildable here and P1 does not start.

    The full event list comes from KnativeOrchestrator (pg_event_index): the gateway pops from
    time_series.events, so the next arrival is in neither task_by_id nor the remaining list."""

    _policy_label = "peer_greedy_selfpredict_network"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pg_event_index = None
        self._pg_state = None
        self._pg_memo: Dict[str, float] = {}

    def placement(self, system_state: SystemState, task: "Task") -> Generator:
        self._pg_state = system_state
        self._pg_memo = {}
        return (yield from super().placement(system_state, task))

    def _pg_predict_node(self, task: "Task", orch, peer_id: int,
                         planned: Dict[int, str]) -> Optional[str]:
        from types import SimpleNamespace

        if self.pg_event_index is None or self._pg_state is None:
            raise RuntimeError(
                "FAIL LOUD: peer_greedy_selfpredict_network has no event index / system state; "
                "it must run under KnativeOrchestrator"
            )
        event = self.pg_event_index[int(peer_id)]
        (function_name,) = tuple(event["application"]["dag"])
        task_type = orch.data.task_types[function_name]
        partner = SimpleNamespace(id=int(peer_id), type=task_type, node_name=event["node_name"])
        valid = self._get_valid_replicas(self._pg_state.replicas.get(task_type["name"]) or set(), partner)
        if not valid:
            return None
        candidates = [r for r in valid if r[1].initialized.triggered] or valid
        known: List[Tuple[str, float]] = []
        for q, b in sorted(((getattr(orch, "peer_exchange", None) or {}).get(int(peer_id)) or {}).items()):
            if int(q) == int(task.id):
                continue
            node_name = self._pg_node_of(orch, int(q), planned)
            if node_name is not None:
                known.append((node_name, float(b)))
        best = None
        for node, platform in candidates:
            plat_type = platform.type["shortName"]
            xf = self._pg_xf(platform)
            score = (platform_queue_drain_seconds(platform, orch, self._pg_memo, exec_scale=xf)
                     + incoming_cold_start_time(partner, platform)
                     + float(task_type["executionTime"].get(plat_type, 0.0) or 0.0) * xf
                     + network_latency_between(partner.node_name, node, self.nodes.items)
                     + self.pg_exchange_scale * self._pg_exchange_seconds(node, platform, known))
            key = (score, node.id, platform.id)
            if best is None or key < best[0]:
                best = (key, node.node_name)
        return best[1]


class PeerGreedyLearnedNetworkScheduler(PeerGreedyNetworkScheduler):
    """rollout_imitation_v1: the immediate rule's candidate set and score-term features, but the
    candidate is chosen by a LEARNED scorer -- an MLP over the rule's own terms
    [drain, cold, exec, latency, exchange] -- trained on the one-step group-local rollout label,
    NOT the hand-weighted sum. Served exactly as the rule (per arrival, no wait, same stack), so a
    live win over `peer_greedy_network` is one step of policy improvement realised in the closed
    loop, not a serving-stack artefact.

    The scorer and its feature normalisation load from HEROSIM_ROLLOUT_SCORER (a .pt state_dict)
    and its `.contract.json` sidecar; fail loud if either is missing (a checkpoint without a
    contract is not evidence, CLAUDE.md) or if the feature order in the contract is not the order
    _pg_choose builds below."""

    _policy_label = "peer_greedy_learned_network"
    _live_audit_policy_name = "peer_greedy_learned_network"
    _FEATURE_ORDER = ("drain", "cold", "exec", "latency", "exchange")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._load_rollout_scorer()

    def _load_rollout_scorer(self) -> None:
        import json
        import torch
        import torch.nn as nn

        path = os.environ.get("HEROSIM_ROLLOUT_SCORER")
        if not path:
            raise RuntimeError(
                "FAIL LOUD: peer_greedy_learned_network needs HEROSIM_ROLLOUT_SCORER=<scorer.pt>"
            )
        contract_path = os.path.splitext(path)[0] + ".contract.json"
        if not os.path.exists(path) or not os.path.exists(contract_path):
            raise RuntimeError(
                f"FAIL LOUD: scorer or contract missing ({path}, {contract_path}); "
                "a checkpoint without a contract is not evidence"
            )
        contract = json.load(open(contract_path))
        order = tuple(contract.get("feature_order", ()))
        if order != self._FEATURE_ORDER:
            raise RuntimeError(
                f"FAIL LOUD: scorer contract feature_order {order} != served order "
                f"{self._FEATURE_ORDER}; a mismatch scores the wrong term per candidate"
            )
        self._sc_mean = [float(x) for x in contract["feature_mean"]]
        self._sc_sd = [float(x) for x in contract["feature_sd"]]
        hidden = int(contract.get("hidden", 16))
        ncol = len(self._FEATURE_ORDER)
        net = nn.Sequential(
            nn.Linear(ncol, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        net.load_state_dict(torch.load(path, map_location="cpu"))
        net.eval()
        self._scorer = net
        self._torch = torch

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
        if getattr(self, "pg_pull_hold", False):
            raise ValueError(f"FAIL LOUD: {PG_PULL_HOLD_ENV}=1 is priced by the hand rule only; the learned scorer has no such term")
        peer_nodes = self._pg_peer_nodes(task, orch, planned) if self.exchange_on else []
        comm = _approx_comm(task.type)
        rows = []  # (node, platform, service_seconds, feats5 in _FEATURE_ORDER)
        for node, platform in candidates:
            key = f"{node.node_name}:{platform.id}"
            plat_type = platform.type["shortName"]
            drain = platform_queue_drain_seconds(platform, orch, memo) + committed_service.get(key, 0.0)
            exec_s = float(task.type["executionTime"].get(plat_type, 0.0) or 0.0)
            cold = incoming_cold_start_time(task, platform)
            lat = network_latency_between(task.node_name, node, nodes)
            exch = self._pg_exchange_seconds(node, platform, peer_nodes)
            rows.append((node, platform, exec_s + comm + exch, [drain, cold, exec_s, lat, exch]))
        torch = self._torch
        norm = [[(r[3][j] - self._sc_mean[j]) / self._sc_sd[j] for j in range(len(self._sc_mean))]
                for r in rows]
        with torch.no_grad():
            scores = self._scorer(torch.tensor(norm, dtype=torch.float32)).squeeze(1).tolist()
        best_i = min(range(len(rows)), key=lambda i: (scores[i], rows[i][0].id, rows[i][1].id))
        self.pg_decisions += 1
        best = rows[best_i]
        # Behavioural instrument (added 2026-09-21, post-close). The rule's pg_joined_partner /
        # pg_moved_by_exchange counters live in _PeerGreedyCore._pg_choose, which this override
        # replaces -- so before this they read 0 for the learned arm as an INSTRUMENT GAP, not as
        # "never co-locates". We restore ONLY pg_joined_partner (did the chosen node host a known
        # partner) -- exact and transferable, and the co-location rate the decomposition needs.
        # pg_moved_by_exchange is deliberately NOT restored: the rule's version removes an ADDITIVE
        # exchange term, which is clean; an MLP has no additive term, and neutralising the exchange
        # feature perturbs a nonlinear input that can flip the argmin even when exchange is constant
        # across candidates (verified: a constant column still moves the argmin under the ReLU). A
        # learned scorer has no honest term-ablation analog, so it stays 0 by construction here.
        if peer_nodes and any(pn == best[0].node_name for pn, _b in peer_nodes):
            self.pg_joined_partner += 1
        return best[0], best[1], best[2]


class PeerGreedyNetworkBatchScheduler(_PeerGreedyCore, GNNScheduler):
    """The rule on the learned arms' serving stack: peer-group batching at the cell's window,
    masked_topo's batch path (planned_node_name pre-pass, deferred tasks to the autoscaler),
    with the greedy in place of the decoder. Requires the same env the gnn arms are served
    with (GNN_DECODE_MODE=masked_topo, GNN_BATCH_BY_PEER_GROUP=1) so the two stacks differ in
    the decision alone."""

    _policy_label = "peer_greedy_network_batch"
    _live_audit_policy_name = "peer_greedy_network_batch"
    _pg_batched = True

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

    def _pg_batch_pass(
        self,
        batch_tasks: List["Task"],
        system_state: SystemState,
        orch,
        *,
        memo: Dict[str, float],
        committed_service: Dict[str, float],
        planned: Dict[int, str],
        placements: Dict[int, Tuple[int, int]],
        service_of: Dict[int, Tuple[str, float]],
        refine: bool,
    ) -> int:
        """One greedy pass over the batch in task-id order. First pass (`refine=False`): every
        task is placed with the partners committed so far. A refine pass: every task is
        re-chosen with EVERY other partner's node known and its own previous service removed
        from the backlog it sat on. Returns the number of tasks that moved."""
        moved = 0
        order = sorted(range(len(batch_tasks)), key=lambda i: int(batch_tasks[i].id))
        for idx in order:
            task = batch_tasks[idx]
            tid = int(task.id)
            valid = self._pg_restrict(task, self._get_valid_replicas(system_state.replicas.get(task.type["name"], set()), task))
            if not valid:
                raise RuntimeError(
                    f"{self._policy_label}: task {task.id} reached the decoder without a "
                    "network-accessible replica; the batch path defers those before decoding"
                )
            initialized = [r for r in valid if r[1].initialized.triggered]
            candidates = initialized if initialized else valid
            if refine:
                prev_key, prev_service = service_of[tid]
                committed_service[prev_key] -= prev_service
                planned.pop(tid, None)
            node, platform, service = self._pg_choose(
                task, candidates, orch,
                memo=memo, committed_service=committed_service, planned=planned,
                nodes=self.nodes.items,
            )
            key = f"{node.node_name}:{platform.id}"
            committed_service[key] = committed_service.get(key, 0.0) + service
            planned[tid] = node.node_name
            if refine and placements[idx] != (node.id, platform.id):
                moved += 1
            placements[idx] = (node.id, platform.id)
            service_of[tid] = (key, service)
        return moved

    def _prefix_inference(
        self,
        batch_tasks: List["Task"],
        system_state: SystemState,
        queue_snapshot: Dict[str, int],
        temporal_state: Optional[Dict[str, Dict[str, float]]],
    ) -> Dict[int, Tuple[int, int]]:
        from src.placement import declared_slate

        if declared_slate.serving_slate() is None:
            return self._pg_decide(batch_tasks, system_state)
        return self._pg_decide_declared(batch_tasks, system_state)

    def _pg_decide_declared(self, batch_tasks: List["Task"], system_state: SystemState) -> Dict[int, Tuple[int, int]]:
        """cd_declared (GNN_SERVE_CANDIDATE_SLATE=declared_pruning_v1): the rule over exactly the slate the learned arms are served --
        each task's top-5 candidates by standalone cost, a batch above 100,000 pruned plans cut into sub-batches of <= 4 tasks
        (declared_slate.slate, the function GNNScheduler._prefix_inference calls) -- each sub-batch decided on its own, as the
        learned arms decide theirs. A descriptive arm: it measures what the pruning costs, not a bar."""
        from src.placement import declared_slate
        from src.placement.live_audit import _candidate_payload

        payloads = []
        for task in batch_tasks:
            valid = self._get_valid_replicas(system_state.replicas.get(task.type["name"], set()), task)
            payloads.append({"task_id": int(task.id), "candidates": [_candidate_payload(self, task, n, p) for n, p in valid]})
        ids = {int(t.id) for t in batch_tasks}
        peers = getattr(self._pg_orchestrator(), "peer_exchange", None) or {}
        pairs = [(int(t.id), int(j)) for t in batch_tasks for j in (peers.get(int(t.id)) or {}) if int(j) in ids]
        sl = declared_slate.slate(payloads, pairs)
        self.pg_declared_batches += 1
        self.pg_declared_pruned += int(sl.pruned)
        self.pg_declared_sub_batched += int(sl.sub_batched)
        placements: Dict[int, Tuple[int, int]] = {}
        try:
            for group in sl.groups:
                self._pg_allowed = {
                    int(batch_tasks[i].id): {(int(c["node_id"]), int(c["platform_id"])) for c in sl.kept[i]} for i in group
                }
                sub = self._pg_decide([batch_tasks[i] for i in group], system_state)
                for k, i in enumerate(group):
                    placements[i] = sub[k]
                self.pg_declared_groups += 1
        finally:
            self._pg_allowed = None
        return placements

    def _pg_decide(self, batch_tasks: List["Task"], system_state: SystemState) -> Dict[int, Tuple[int, int]]:
        orch = self._pg_orchestrator()
        memo: Dict[str, float] = {}
        committed_service: Dict[str, float] = {}
        planned: Dict[int, str] = {}
        placements: Dict[int, Tuple[int, int]] = {}
        service_of: Dict[int, Tuple[str, float]] = {}
        self._pg_batch_ids = frozenset(int(t.id) for t in batch_tasks)
        if self.pg_ext_rate is not None:
            self._pg_ext_lambda = self._pg_ext_lambdas(batch_tasks, system_state)
            self.pg_ext_batches += 1
        self._pg_batch_pass(
            batch_tasks, system_state, orch, memo=memo, committed_service=committed_service,
            planned=planned, placements=placements, service_of=service_of, refine=False,
        )
        self._pg_refine(batch_tasks, system_state, orch, memo, committed_service, planned, placements, service_of)
        self.pg_batches += 1
        # the batch path counts what the decoder saw; the rule keeps the same books
        table = getattr(orch, "peer_exchange", None) or {}
        ids: Set[int] = {int(t.id) for t in batch_tasks}
        pairs_in = sum(1 for i in ids for j in (table.get(i) or {}) if j in ids) // 2
        outside = sum(1 for i in ids for j in (table.get(i) or {}) if j not in ids)
        self.prefix_pairs_in_batch += pairs_in
        self.prefix_peers_outside_batch += outside
        return placements

    def _pg_ext_lambdas(self, batch_tasks: List["Task"], system_state: SystemState) -> Dict[str, float]:
        """lambda_p per `node:platform` key: the rate split by the batch's type mix, then evenly over the
        replicas the pass would consider for that type (initialized ones if any, else all valid)."""
        share: Dict[str, float] = {}
        for task in batch_tasks:
            share[task.type["name"]] = share.get(task.type["name"], 0.0) + 1.0 / len(batch_tasks)
        lam: Dict[str, float] = {}
        for task in batch_tasks:
            name = task.type["name"]
            if name not in share:
                continue
            valid = self._pg_restrict(task, self._get_valid_replicas(system_state.replicas.get(name, set()), task))
            initialized = [r for r in valid if r[1].initialized.triggered]
            pool = initialized if initialized else valid
            if not pool:
                share.pop(name)
                continue
            per_replica = self.pg_ext_rate * share.pop(name) / len(pool)
            for node, platform in pool:
                key = f"{node.node_name}:{platform.id}"
                lam[key] = lam.get(key, 0.0) + per_replica
        return lam

    def _pg_refine(self, batch_tasks, system_state, orch, memo, committed_service, planned,
                   placements, service_of) -> None:
        """The 1-pass rule refines nothing (the registered `peer_greedy_network_batch`)."""
        return None


class PeerGreedyNetworkCDScheduler(PeerGreedyNetworkBatchScheduler):
    """joint_burst_v2 (2026-09-20): the batched rule plus coordinate descent on its own score.

    The id-order greedy anchors the group on the first task with X == 0 (no partner known yet)
    and never revisits an early task once the later partners' nodes are known -- the myopia a
    joint decoder could exploit. Offline on the 48 burst held-out groups the 1-pass rule sits
    +73 % above the sweep optimum with exchange 41.5 s vs the optimum's 25.8 s. This flavour
    re-runs the pass up to `HEROSIM_PG_CD_PASSES` (default 3) more times, each task re-chosen
    with every other partner's planned node known and its own service removed from the backlog
    it sat on, stopping at the first pass that moves nothing. Same information as the rule,
    same seat as the learned arms; registered as the honest bar for any arm that claims to
    have learned joint structure.
    """

    _policy_label = "peer_greedy_network_cd"
    _live_audit_policy_name = "peer_greedy_network_cd"

    def _pg_refine(self, batch_tasks, system_state, orch, memo, committed_service, planned,
                   placements, service_of) -> None:
        raw = os.environ.get("HEROSIM_PG_CD_PASSES", "3").strip() or "3"
        passes = int(raw)
        if passes < 1:
            raise ValueError(f"FAIL LOUD: HEROSIM_PG_CD_PASSES must be >= 1, got {raw!r}")
        for _ in range(passes):
            moved = self._pg_batch_pass(
                batch_tasks, system_state, orch, memo=memo, committed_service=committed_service,
                planned=planned, placements=placements, service_of=service_of, refine=True,
            )
            self.pg_cd_passes += 1
            self.pg_cd_moves += moved
            if moved == 0:
                break


class PeerGreedyNetworkCDRandomSeedScheduler(PeerGreedyNetworkBatchScheduler):
    """r1_attribution_v1 CD<-random: the control for the "worse basin" reading of CD<-GNN / CD<-Twin.

    The batch's starting plan is drawn uniformly at random over each task's candidates in the declared slate (GNN_SERVE_CANDIDATE_SLATE=
    declared_pruning_v1: top-5 by standalone cost, sub-batches as the learned arms), seeded per batch from the cell seed
    (HEROSIM_CD_RANDOM_SEED) and the batch's task ids. The batch is then refined by exactly the code CD<-GNN uses
    (GNN_CD_REFINE=apply: GnnCdRefiner, 3 passes, in GNNScheduler's batch path), so the arms differ in the seed plan alone."""

    _policy_label = "peer_greedy_network_cd_random_seed"
    _live_audit_policy_name = "peer_greedy_network_cd_random_seed"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from src.policy.gnn.scheduler import _cd_refine_mode

        if _cd_refine_mode() != "apply":
            raise RuntimeError("FAIL LOUD: cd_random_seed refines its random plan; export GNN_CD_REFINE=apply")
        raw = os.environ.get("HEROSIM_CD_RANDOM_SEED", "").strip()
        if not raw:
            raise RuntimeError("FAIL LOUD: cd_random_seed needs HEROSIM_CD_RANDOM_SEED (the cell seed)")
        self._cd_random_seed = int(raw)
        self.pg_random_seed_batches = 0

    def _prefix_inference(self, batch_tasks, system_state, queue_snapshot, temporal_state):
        import random

        from src.placement import declared_slate
        from src.placement.live_audit import _candidate_payload

        valid_of = [self._get_valid_replicas(system_state.replicas.get(t.type["name"], set()), t) for t in batch_tasks]
        kept: Dict[int, Set[Tuple[int, int]]] = {i: {(n.id, p.id) for n, p in valid_of[i]} for i in range(len(batch_tasks))}
        if declared_slate.serving_slate() is not None:
            payloads = [{"task_id": int(t.id), "candidates": [_candidate_payload(self, t, n, p) for n, p in valid_of[i]]}
                        for i, t in enumerate(batch_tasks)]
            ids = {int(t.id) for t in batch_tasks}
            peers = getattr(self._pg_orchestrator(), "peer_exchange", None) or {}
            pairs = [(int(t.id), int(j)) for t in batch_tasks for j in (peers.get(int(t.id)) or {}) if int(j) in ids]
            sl = declared_slate.slate(payloads, pairs)
            self.pg_declared_batches += 1
            self.pg_declared_pruned += int(sl.pruned)
            self.pg_declared_sub_batched += int(sl.sub_batched)
            self.pg_declared_groups += len(sl.groups)
            kept = {i: {(int(c["node_id"]), int(c["platform_id"])) for c in sl.kept[i]} for i in range(len(batch_tasks))}
        rng = random.Random(f"{self._cd_random_seed}:{sorted(int(t.id) for t in batch_tasks)}")
        placements: Dict[int, Tuple[int, int]] = {}
        for i in range(len(batch_tasks)):
            options = sorted(kept[i])
            if not options:
                raise RuntimeError(f"FAIL LOUD: cd_random_seed: task {batch_tasks[i].id} has no candidate in its slate")
            placements[i] = rng.choice(options)
        self.pg_random_seed_batches += 1
        return placements


class GnnCdRefiner(_PeerGreedyCore):
    """cd_gap_v1 D5: the CD greedy's refine passes, started from a plan someone else decoded.

    Held by the GNN scheduler (`GNN_CD_REFINE=shadow|apply`). Seeds CD's own books -- committed
    service per platform, planned nodes -- with the seed plan, then runs `_pg_batch_pass(refine=True)`
    (the same method CD runs, bound here, not copied) until a pass moves nothing or the pass budget
    is spent. Returns the refined plan and what moved."""

    _policy_label = "gnn_cd_refine"
    _pg_batched = True
    _pg_batch_pass = PeerGreedyNetworkBatchScheduler._pg_batch_pass

    def __init__(self, host) -> None:
        self._host = host
        self._pg_init()

    @property
    def nodes(self):
        return self._host.nodes

    def _get_valid_replicas(self, replicas, task):
        return self._host._get_valid_replicas(replicas, task)

    def _pg_orchestrator(self):
        return self._host._orchestrator()

    def refine(self, batch_tasks: List["Task"], system_state: SystemState,
               seed: Dict[int, Tuple[int, int]], passes: int) -> Tuple[Dict[int, Tuple[int, int]], Dict[str, int]]:
        orch = self._pg_orchestrator()
        memo: Dict[str, float] = {}
        committed_service: Dict[str, float] = {}
        planned: Dict[int, str] = {}
        service_of: Dict[int, Tuple[str, float]] = {}
        placements: Dict[int, Tuple[int, int]] = dict(seed)
        self._pg_batch_ids = frozenset(int(t.id) for t in batch_tasks)
        by_id = {}
        for idx, task in enumerate(batch_tasks):
            valid = self._get_valid_replicas(system_state.replicas.get(task.type["name"], set()), task)
            couple = next(((n, p) for n, p in valid if (n.id, p.id) == tuple(seed[idx])), None)
            if couple is None:
                raise RuntimeError(f"FAIL LOUD: seed placement {seed[idx]} of task {task.id} is not a valid replica")
            by_id[idx] = couple
            planned[int(task.id)] = couple[0].node_name
        for idx in sorted(range(len(batch_tasks)), key=lambda i: int(batch_tasks[i].id)):
            task = batch_tasks[idx]
            node, platform, service = self._pg_choose(task, [by_id[idx]], orch, memo=memo,
                                                      committed_service=committed_service, planned=planned,
                                                      nodes=self.nodes.items)
            key = f"{node.node_name}:{platform.id}"
            committed_service[key] = committed_service.get(key, 0.0) + service
            service_of[int(task.id)] = (key, service)
        for _ in range(passes):
            moved = self._pg_batch_pass(batch_tasks, system_state, orch, memo=memo,
                                        committed_service=committed_service, planned=planned,
                                        placements=placements, service_of=service_of, refine=True)
            if moved == 0:
                break
        seed_load: Dict[Tuple[int, int], int] = {}
        for v in seed.values():
            seed_load[tuple(v)] = seed_load.get(tuple(v), 0) + 1
        new_load: Dict[Tuple[int, int], int] = {}
        for v in placements.values():
            new_load[tuple(v)] = new_load.get(tuple(v), 0) + 1
        info = {"moved": 0, "node_change": 0, "platform_only": 0, "unstack": 0}
        for idx, before in seed.items():
            after = placements[idx]
            if tuple(after) == tuple(before):
                continue
            info["moved"] += 1
            info["node_change" if after[0] != before[0] else "platform_only"] += 1
            if seed_load[tuple(before)] >= 2 and new_load.get(tuple(after), 0) < seed_load[tuple(before)]:
                info["unstack"] += 1
        return placements, info


class PeerGreedyLearnedNetworkBatchScheduler(PeerGreedyNetworkBatchScheduler):
    """rollout_imitation_v1 cross-study arm: the LEARNED rollout scorer served in the BATCHED seat
    (peer-group batching at the cell window, masked_topo path) instead of per arrival -- the
    burst/batched analogue of `peer_greedy_learned_network`, so the rollout scorer can be gated in
    the SAME seat gnnedge0 wins in. The scorer is the same MLP over the rule's own terms
    [drain, cold, exec, latency, exchange]; the batch pass calls _pg_choose exactly as the batched
    greedy does, so the learned scoring replaces the hand sum and nothing else changes.

    Disclosed mismatch: the scorer was trained on the immediate rule's per-arrival states (no
    in-batch commitment), and here `drain` carries the committed_service of peers placed earlier in
    the batch -- a mild train/serve shift within the feature's own meaning, the price of gating a
    per-arrival-trained scorer in the batched seat."""

    _policy_label = "peer_greedy_learned_network_batch"
    _live_audit_policy_name = "peer_greedy_learned_network_batch"
    _FEATURE_ORDER = PeerGreedyLearnedNetworkScheduler._FEATURE_ORDER
    _load_rollout_scorer = PeerGreedyLearnedNetworkScheduler._load_rollout_scorer
    _pg_choose = PeerGreedyLearnedNetworkScheduler._pg_choose

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._load_rollout_scorer()
