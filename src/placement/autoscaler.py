"""
Copyright 2024 b<>com

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import logging
import os
import math
from abc import abstractmethod
from collections import defaultdict
from typing import Dict, Generator, List, Set, Tuple, TYPE_CHECKING, Optional, Union, Any

from simpy.core import Environment, SimTime
from simpy.events import Process
from simpy.resources.store import Store

if TYPE_CHECKING:
    from src.placement.infrastructure import Node, Platform

from src.placement.scaleout import (
    KPA,
    KpaConfig,
    KpaScaler,
    new_scaleout_stats,
    pending_in_flight,
    platform_in_flight,
    scaleout_mode,
)
from src.placement.warmth import NODE_DISK_V2
from src.placement.physics_audit import AUDIT as _AUDIT
from src.placement.model import (
    PlatformVector,
    ScaleEvent,
    SimulationData,
    SimulationPolicy,
    SystemState,
    TaskType, SystemEvent,
)

logger = logging.getLogger(__name__)



def replica_platform_types_allowed() -> Optional[frozenset]:
    """`HEROSIM_REPLICA_PLATFORM_TYPES`: the platform `shortName`s the autoscaler may host a
    replica on. Unset (the default) means no restriction, so every run recorded before
    2026-09-13 is bit-identical.

    Why this exists (audit 2026-09-13). The `peer_affinity_v1` co-sim corpora contain
    `xavierGpu` and `xavierDla` platform ROWS -- 9 and 9 per dataset -- but in 516 of 516
    datasets neither is ever a replica, so neither is ever a candidate the model ranks. Live,
    the same cell running 450,729 tasks scales up long enough that `xavierGpu` becomes a
    replica and **7.6 % of live candidates are of a platform type the checkpoint never saw as
    a candidate**. That is not only an unseen one-hot column: `node_caps[n] = alpha x max
    single candidate demand on n`, and the GPU demand is 1.739 against a corpus maximum of
    0.213, so ONE such candidate inflates that node's cap ~8x. Measured on the production
    traces: **46.4 % of live node caps exceed the largest cap any training dataset contains**,
    and on a smoke the roomiest node goes from holding 5.2 tasks (cache) to 32.5 (live, as
    served) and back to 3.2 once these candidates are removed -- i.e. the capacity mask, the
    decoder's only concentration control before `GNN_PREFIX_PLATFORM_CAP`, stops binding in
    17 of 23 batches.

    Restricting the autoscaler is the intervention that restores the corpus's action space
    for EVERY arm at once (the reactive baselines included), which is what makes a gate run
    under it a like-for-like comparison rather than a handicap on one policy.
    """
    raw = os.environ.get("HEROSIM_REPLICA_PLATFORM_TYPES", "").strip()
    if not raw:
        return None
    types = frozenset(t.strip() for t in raw.split(",") if t.strip())
    if not types:
        raise ValueError(
            "HEROSIM_REPLICA_PLATFORM_TYPES is set but lists no platform type; unset it to "
            "allow every type rather than declaring an empty allow-list"
        )
    return types


def replica_platform_type_allowed(short_name: str) -> bool:
    allowed = replica_platform_types_allowed()
    return True if allowed is None else str(short_name) in allowed


class Autoscaler:
    def __init__(
            self,
            env: Environment,
            mutex: Store,
            data: SimulationData,
            policy: SimulationPolicy,
    ):
        self.env = env
        self.mutex = mutex
        self.data = data
        self.policy = policy
        self.reconcile_interval = policy.reconcile_interval

        self.scale_events: List[ScaleEvent] = []
        self.system_status_events: List[SystemEvent] = []
        self.run: Process

        # kpa_scaleout_v1: HEROSIM_SCALEOUT. Legacy paths read these with getattr so an autoscaler built
        # without __init__ (tests) stays on legacy.
        self.scaleout = scaleout_mode()
        self.kpa: Optional[KpaScaler] = None
        self.scaleout_stats: Optional[Dict[str, Any]] = None
        self._replica_born: Dict[Tuple[str, int, int], SimTime] = {}
        self._replica_cause: Dict[Tuple[str, int, int], str] = {}
        # A1: arrived, not yet placed tasks per type (fed by the orchestrator's gateway)
        self._kpa_pending: Dict[str, List[Any]] = {}
        if self.scaleout == KPA:
            self.kpa = KpaScaler(KpaConfig.from_env(policy.queue_length))
            self.scaleout_stats = new_scaleout_stats()

    def kpa_note_arrival(self, task: Any) -> None:
        """Called by the orchestrator for every task it hands to the scheduler (kpa only)."""
        if getattr(self, "kpa", None) is not None:
            self._kpa_pending.setdefault(task.type["name"], []).append(task)

    def scaleout_summary(self) -> Optional[Dict[str, Any]]:
        """The `scaleOut` block of a kpa run's stats; None under legacy so its stats are unchanged."""
        if getattr(self, "kpa", None) is None:
            return None
        return {**self.kpa.config.describe(), **self.scaleout_stats}

    def autoscaler_process(self):
        if getattr(self, "scaleout", None) == KPA:
            yield from self._kpa_autoscaler_process()
            return

        logging.info(
            f"[ {self.env.now} ] Orchestrator Autoscaler started with policy"
            f" {self.policy}"
        )

        last_force_scale_up: Dict[str, SimTime] = {
            function_name: 0.0 for function_name in self.data.task_types
        }

        while True:
            # Per-function scaling decision
            system_state: SystemState = yield self.mutex.get()
            replicas: Dict[str, Set[Tuple[Node, Platform]]] = system_state.replicas

            for function_name, function_replicas in replicas.items():
                force_scale_up = True

                scaling_difference: PlatformVector[float] = yield self.env.process(
                    self.scaling_level(
                        system_state, self.data.task_types[function_name]
                    )
                )


                for hardware_target, hardware_scaling in scaling_difference.items():
                    if hardware_scaling < 0:
                        # Scale down
                        count = abs(math.floor(hardware_scaling))
                        # logging.error(f"[ {self.env.now} ] Scaling down {function_name} by {count} (currently {len(function_replicas)})")
                        stop = yield self.env.process(
                            self.scale_down(
                                count, system_state, function_name, hardware_target
                            )
                        )
                        # Do not force scale up
                        force_scale_up = False

                    elif hardware_scaling > 0:
                        # Scale up
                        count = abs(math.ceil(hardware_scaling))
                        # logging.error(f"[ {self.env.now} ] Scaling up {function_name} by {count} (currently {len(function_replicas)})")
                        stop = yield self.env.process(
                            self.scale_up(
                                count, system_state, function_name, hardware_target
                            )
                        )
                        # Successfully scaled up on hardware target
                        if not isinstance(stop, StopIteration):
                            force_scale_up = False

                    else:
                        # Correct scaling level, do nothing
                        force_scale_up = False
                        # pass

                # Force scale up on any hardware type if necessary
                if force_scale_up and (
                        (self.env.now - last_force_scale_up[function_name])
                        > self.policy.keep_alive
                ):
                    stop = yield self.env.process(
                        self.create_first_replica(
                            system_state, self.data.task_types[function_name]
                        )
                    )
                    last_force_scale_up[function_name] = self.env.now

            self.log_system_status(replicas)

            # Release mutex
            yield self.mutex.put(system_state)

            # Next event
            self.env.step()

            # Wake Autoscaler up once per second
            yield self.env.timeout(self.reconcile_interval)

    def _kpa_autoscaler_process(self) -> Generator:
        """KPA reconcile (HEROSIM_SCALEOUT=kpa): sample in-flight concurrency per function every tick,
        size it to the stable/panic decision on any hardware type. Same tick, mutex and status
        logging as the legacy loop; the legacy forced first-replica fallback is not used, because
        reachability-triggered creation still runs from the schedulers."""
        stats = self.scaleout_stats
        while True:
            system_state: SystemState = yield self.mutex.get()
            replicas: Dict[str, Set[Tuple[Node, Platform]]] = system_state.replicas

            for function_name, function_replicas in replicas.items():
                now = self.env.now
                self.kpa.observe(
                    function_name,
                    now,
                    sum(platform_in_flight(platform) for _, platform in function_replicas)
                    + pending_in_flight(self._kpa_pending.get(function_name)),
                )
                current = len(function_replicas)
                ready = sum(
                    1 for _, platform in function_replicas if platform.initialized.triggered
                )
                decision = self.kpa.decide(function_name, now, current, ready)
                if _AUDIT is not None:
                    _AUDIT.kpa_tick(self.env, function_name, self.kpa.functions[function_name].samples[-1][1],
                                    current, ready, decision,
                                    sum(1 for (fn, _n, _p), c in self._replica_cause.items()
                                        if fn == function_name and c == "load"))
                stats["panic_entries"] += int(decision.entered_panic)
                stats["panic_ticks"] += int(decision.panicking)

                if decision.desired > current:
                    yield self.env.process(
                        self.scale_up(
                            decision.desired - current,
                            system_state,
                            function_name,
                            "any",
                            cause="load",
                        )
                    )
                elif decision.desired < current:
                    yield self.env.process(
                        self._kpa_scale_down(
                            current - decision.desired, system_state, function_name
                        )
                    )

            self.log_system_status(replicas)

            yield self.mutex.put(system_state)

            self.env.step()

            yield self.env.timeout(self.reconcile_interval)

    def _kpa_scale_down(
            self, count: int, system_state: SystemState, function_name: str
    ) -> Generator:
        """Remove up to `count` idle, initialised replicas, longest idle first. The decision already
        holds the last replica until a full stable window without traffic, so no keep-alive applies."""
        if False:
            yield
        function_replicas = system_state.replicas[function_name]

        def idle_reference(platform: Platform) -> SimTime:
            return platform.idle_since if math.isfinite(platform.idle_since) else platform.last_allocated

        def protected(node: Node, platform: Platform) -> bool:
            # A3: a reachability-created replica that has not started a task since its creation is waiting
            # for the postponed task that created it; it is not idle. Bounded by one stable window.
            key = (function_name, node.id, platform.id)
            born = self._replica_born.get(key)
            if born is None or self._replica_cause.get(key) != "reachability":
                return False
            if getattr(platform, "last_started", -math.inf) >= born:
                return False
            return self.env.now - born < self.kpa.config.stable_window

        candidates = sorted(
            (
                (node, platform)
                for node, platform in function_replicas
                if platform.initialized.triggered
                and platform_in_flight(platform) == 0
                and not getattr(platform, "rendezvous_procs", None)
                and not protected(node, platform)
            ),
            key=lambda c: (idle_reference(c[1]), c[0].id, c[1].id),
        )
        contention = getattr(system_state.scheduler_state, "average_contention", None) or {}
        for node, platform in candidates[:count]:
            (contention.get(function_name) or {}).pop((node.id, platform.id), None)
            if self._release_replica(system_state, function_name, (node, platform)):
                self.scaleout_stats["scale_downs"] += 1
                if not function_replicas:
                    self.scaleout_stats["scale_to_zero"] += 1

    def scale_up(
            self,
            count: int,
            system_state: SystemState,
            function_name: str,
            hardware_target: str,
            cause: str = "reachability",
    ) -> Generator:
        """`cause` is recorded under kpa only: "load" from the KPA decision, "reachability" from every
        create_first_replica path (a task's source reaches no replica of its type)."""
        # Get current function replicas
        function_replicas = system_state.replicas[function_name]
        kpa = getattr(self, "scaleout", None) == KPA

        # Scale up by `count` replicas
        for _ in range(count):
            replicas_count = len(function_replicas)
            # Filter out nodes by task requirements
            couples_suitable: Set[Tuple[Node, Platform]] = set()

            available_resources: Dict[Node, Set[Platform]] = (
                system_state.available_resources
            )
            # peer_affinity_v1 stage 3 (2026-09-11): HEROSIM_SERVER_ONLY_REPLICAS=1 keeps
            # every replica on a server node. The peer_affinity corpora host replicas on
            # servers only (replicas.per_client = 0; the exchange physics needs a network
            # map between the two execution nodes, which clients do not have to every
            # server), and a live episode with no replica plan would otherwise fall back
            # to the source client the moment the reachable servers are full — which is
            # what a 2,650-arrival/s production trace does within its first second.
            # Off by default; every other run is bit-identical.
            server_only = os.environ.get("HEROSIM_SERVER_ONLY_REPLICAS", "0") == "1"
            memory_refused = 0
            for node, platforms in available_resources.items():
                if server_only and str(node.node_name).startswith("client_node"):
                    continue
                for platform in platforms:
                    if not replica_platform_type_allowed(platform.type["shortName"]):
                        continue
                    if (
                            hardware_target != "any"
                            and platform.type["shortName"] != hardware_target
                    ):
                        continue
                    if (
                            platform.type["shortName"]
                            not in self.data.task_types[function_name]["platforms"]
                    ):
                        continue
                    if (
                            node.memory
                            < self.data.task_types[function_name]["memoryRequirements"][
                        platform.type["shortName"]
                    ]
                    ):
                        continue
                    # kpa: a replica is only placed where the node's free memory holds it
                    if kpa and (
                            node.available_memory
                            < self.data.task_types[function_name]["memoryRequirements"][
                        platform.type["shortName"]
                    ]
                    ):
                        memory_refused += 1
                        continue
                    couples_suitable.add((node, platform))

            if kpa:
                self.scaleout_stats["memory_cap_refusals"] += memory_refused
            if _AUDIT is not None and memory_refused:
                _AUDIT.memory_refusal(self.env, function_name, memory_refused)

            # No suitable resources for replica creation
            if not couples_suitable:
                if kpa:
                    self.scaleout_stats["scale_up_failures_by_cause"][cause] += 1
                    if memory_refused:
                        self.scaleout_stats["memory_cap_blocked"] += 1
                # logging.error(state.average_hardware_contention[function_name])
                # Next step
                return StopIteration(
                    f"Autoscaler could not create a {hardware_target} replica for"
                    f" {function_name} (currently {replicas_count} replica)"
                )

            logging.info(
                f"[ {self.env.now} ] Autoscaler scaling up {function_name} (currently"
                f" {replicas_count})"
            )

            # Resources selection (Node, Platform)
            new_replica: Tuple[Node, Platform]
            new_replica = yield self.env.process(
                self.create_replica(
                    couples_suitable, self.data.task_types[function_name]
                )
            )

            logging.info(f"[ {self.env.now} ] {new_replica}")

            try:
                # Remove selected platform from available resources on the node
                available_resources[new_replica[0]].remove(new_replica[1])

                # Update node availability
                new_replica[0].available_platforms -= 1

                # Allocate task memory requirements from node's available memory
                new_replica[0].available_memory -= self.data.task_types[function_name][
                    "memoryRequirements"
                ][new_replica[1].type["shortName"]]
                if kpa and new_replica[0].available_memory < 0:
                    raise RuntimeError(
                        f"kpa: {new_replica} for {function_name} overcommits node memory "
                        f"(available {new_replica[0].available_memory})"
                    )

                # Add function replica to the pool so it can be considered by the Scheduler
                function_replicas.add(new_replica)

                # Initialize replica (pull image) asynchronously
                # The platform_process will wait on initialized event before processing tasks
                self.env.process(
                    self.initialize_replica(
                        new_replica,
                        function_replicas,
                        self.data.task_types[function_name],
                        system_state,
                    )
                )

                # Statistics
                new_replica[1].last_allocated = self.env.now

                event: ScaleEvent = {
                    "name": function_name,
                    "timestamp": self.env.now,
                    "action": "up",
                    "count": len(function_replicas),
                    "average_queue_length": sum(
                        [replica[1].queue_length() for replica in function_replicas]
                    ) / len(function_replicas),
                }
                if kpa:
                    event["cause"] = cause
                    self.scaleout_stats["scale_ups_by_cause"][cause] += 1
                    born_key = (function_name, new_replica[0].id, new_replica[1].id)
                    self._replica_born[born_key] = self.env.now
                    self._replica_cause[born_key] = cause
                self.scale_events.append(event)
                if _AUDIT is not None:
                    _AUDIT.replica_up(self.env, function_name, new_replica[0], new_replica[1], cause,
                                      self.data.task_types[function_name]["memoryRequirements"][
                                          new_replica[1].type["shortName"]])
            except KeyError:
                """
                logging.error(
                    f"[ {self.env.now} ] Autoscaler tried to scale up "
                    f"{function_name}, but {new_replica} was already allocated"
                )

                logging.error(
                    f"[ {self.env.now} ] Last allocation time: "
                    f"{new_replica[1].last_allocated} "
                    " -- Last removal time: "
                    f"{new_replica[1].last_removed}"
                )

                logging.error(
                    f"[ {self.env.now} ] {system_state.available_resources}"
                )
                logging.error(
                    f"{new_replica[1].initialized} // {new_replica[0].available_platforms}"
                )
                """
                pass

    def scale_down(
            self,
            count: int,
            system_state: SystemState,
            function_name: str,
            hardware_target: str,
    ) -> Generator[Any, Any, Optional[Union[StopIteration, Tuple[Node, Platform]]]]:
        """Scale down replicas for a given function"""
        # Get current function replicas
        function_replicas = system_state.replicas[function_name]

        # Filter replicas according to hardware target
        suitable_replicas = set(
            filter(
                lambda replica: (replica[1].type["shortName"] == hardware_target or hardware_target == 'any'),
                function_replicas,
            )
        )

        # Scale down
        for _ in range(count):
            replicas_count = len(function_replicas)
            # print(f"[ {self.env.now} ] Attempting to scale down {function_name} (currently {replicas_count} replicas)")

            """
            # Check if we need to scale down based on queue length
            avg_queue_length = sum(len(replica[1].queue.items) for replica in function_replicas) / len(function_replicas)
            if avg_queue_length > self.policy.queue_length * 0.5:  # Only scale down if queue is less than 50% full
                print(f"Cannot scale down {function_name} due to high queue utilization ({avg_queue_length:.2f})")
                # return None

            # Check if any tasks are currently running on replicas
            tasks_running = any(len(replica[1].queue.items) > 0 for replica in function_replicas)
            if tasks_running:
                print(f"Cannot scale down {function_name} while tasks are running")
                # return None
            """

            removed_replica: Tuple[Node, Platform]
            removed_replica = yield self.env.process(
                self.remove_replica(
                    suitable_replicas, self.data.task_types[function_name], system_state
                )
            )

            # Could not scale down (tasks in queue on all replicas)
            # if not removed_replica:
                # print(f"Autoscaler could not scale down {function_name} (currently {replicas_count})")
                # return None

            logging.info(
                f"[ {self.env.now} ] Autoscaler scaling down {function_name} (currently"
                f" {replicas_count})"
            )

            logging.info(f"[ {self.env.now} ] {removed_replica}")
            if removed_replica:
                print(f"[ {self.env.now} ] removed: {removed_replica}")

            return self._release_replica(system_state, function_name, removed_replica)

    def _release_replica(
            self,
            system_state: SystemState,
            function_name: str,
            removed_replica: Tuple[Node, Platform],
            already_removed: bool = False,
    ) -> Optional[Tuple[Node, Platform]]:
        """Return a removed replica's platform and memory to the pool and record the scale-down event.

        `already_removed`: the replica left `system_state.replicas` earlier (a drain, see
        GNN KnativeAutoscaler.evict_idle_for) and only its platform and memory are still held."""
        function_replicas = system_state.replicas[function_name]
        try:
            # Remove replica from function replicas
            # FIXME: Sometimes raises KeyError ... (double remove)
            if not already_removed:
                function_replicas.remove(removed_replica)

            # Reset platform to uninitialized state. An initialized event that never fired
            # already means "uninitialized" and may have a waiter (platform_process parks on
            # it); replacing it would strand that waiter forever when scale-up fires the new
            # one (selfpredict_burst_v1, 2026-09-24: 301 tasks queued behind a dead worker).
            if removed_replica[1].initialized.triggered:
                removed_replica[1].initialized = removed_replica[1].env.event()

            if getattr(self.env, "warmth_physics", None) == NODE_DISK_V2:
                removed_replica[1].previous_task = None

            # Release replica into available resources
            available_resources: Dict[Node, Set[Platform]] = (
                system_state.available_resources
            )
            available_resources[removed_replica[0]].add(removed_replica[1])

            # Update node availability
            removed_replica[0].available_platforms += 1

            # Reclaim node memory
            removed_replica[0].available_memory += self.data.task_types[
                function_name
            ]["memoryRequirements"][removed_replica[1].type["shortName"]]

            # Statistics
            removed_replica[1].last_removed = self.env.now
            if _AUDIT is not None:
                _AUDIT.replica_down(self.env, function_name, removed_replica[0], removed_replica[1],
                                    self.data.task_types[function_name]["memoryRequirements"][
                                        removed_replica[1].type["shortName"]],
                                    already_removed, platform_in_flight(removed_replica[1]))
            if getattr(self, "kpa", None) is not None:
                born_key = (function_name, removed_replica[0].id, removed_replica[1].id)
                born = self._replica_born.pop(born_key, None)
                self._replica_cause.pop(born_key, None)
                if born is not None:
                    self.scaleout_stats["replica_lifetimes_closed"] += 1
                    if self.env.now - born < self.kpa.config.stable_window:
                        self.scaleout_stats["replica_lifetimes_within_stable_window"] += 1

            event: ScaleEvent = {
                "name": function_name,
                "timestamp": self.env.now,
                "action": "down",
                "count": len(function_replicas),
                "average_queue_length": (
                    sum(
                        [
                            replica[1].queue_length()
                            for replica in function_replicas
                        ]
                    )
                    / len(function_replicas)
                    if function_replicas
                    else 0.0
                ),
                "platform_type": removed_replica[1].type["shortName"]
            }
            self.scale_events.append(event)
            return removed_replica
        except KeyError:
            logging.debug(
                f"[ {self.env.now} ] Replica {removed_replica} was already removed"
            )
            return None

    @abstractmethod
    def scaling_level(
            self, system_state: SystemState, task_type: TaskType
    ) -> Generator[Any, Any, PlatformVector[float]]:
        """Determine the scaling level for a task type"""
        pass

    @abstractmethod
    def create_first_replica(
            self, system_state: SystemState, task_type: TaskType
    ) -> Generator:
        pass

    @abstractmethod
    def create_first_replica_on_node(
            self, system_state: SystemState, task_type: TaskType, node_name: str
    ) -> Generator:
        pass

    @abstractmethod
    def create_replica(
            self, couples_suitable: Set[Tuple[Node, Platform]], task_type: TaskType
    ) -> Generator:
        pass

    @abstractmethod
    def initialize_replica(
            self,
            new_replica: Tuple[Node, Platform],
            function_replicas: Set[Tuple[Node, Platform]],
            task_type: TaskType,
            state: SystemState,
    ) -> Generator:
        pass

    @abstractmethod
    def remove_replica(
            self,
            couples_suitable: Set[Tuple[Node, Platform]],
            task_type: TaskType,
            state: SystemState,
    ) -> Generator:
        pass

    def log_system_status(self, replicas: Dict[str, Set[Tuple[Node, Platform]]]):
        for function_name, function_replicas in replicas.items():
            count_by_platform_type = defaultdict(int)

            for _, platform in function_replicas:
                count_by_platform_type[platform.type['shortName']] += 1

            event: ScaleEvent = {
                "name": function_name,
                "timestamp": self.env.now,
                "count": len(function_replicas),
                "average_queue_length": (
                    sum(
                        [
                            replica[1].queue_length()
                            for replica in function_replicas
                        ]
                    )
                    / len(function_replicas)
                    if function_replicas
                    else 0.0
                ),
            }

            for platform_type, count in count_by_platform_type.items():
                event[platform_type] = count
            self.system_status_events.append(event)
