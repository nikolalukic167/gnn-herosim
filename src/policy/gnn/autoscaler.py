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
import math
import os

from typing import Generator, Set, Tuple, TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from src.placement.infrastructure import Node

from src.policy.gnn.model import KnativeSchedulerState, KnativeSystemState

if TYPE_CHECKING:
    from src.placement.infrastructure import Node, Platform, Task

from src.placement.model import (
    DurationSecond,
    PlatformVector,
    SchedulerState,
    SizeGigabyte,
    SpeedMBps,
    SystemState,
    TaskType, TimeSeries,
)

from src.placement.autoscaler import Autoscaler, replica_platform_type_allowed
from src.placement.infrastructure import FIDELITY, STARVED_RENDEZVOUS
from src.placement.warmth import (
    PLATFORM_REUSE_V1,
    image_pull_disk_hit,
    needs_image_pull,
)



class KnativeAutoscaler(Autoscaler):





    def scaling_level(self, system_state: KnativeSystemState, task_type: TaskType):
        """Calculate scaling level - matches knative_network autoscaler.

        HEROSIM_SCALEOUT=legacy only; kpa sizes replicas in Autoscaler._kpa_autoscaler_process."""
        # Scheduling functions called in a Simpy Process must be Generators
        # No-op as per https://stackoverflow.com/a/68628599/9568489
        if False:
            yield

        # Knative default values (cf. https://knative.dev/docs/serving/autoscaling/concurrency/)
        # Lambda is 1 (cf. https://notes.crmarsh.com/isolates-microvms-and-webassembly)
        state: KnativeSchedulerState = system_state.scheduler_state
        target_concurrencies: PlatformVector = state.target_concurrencies[
            task_type["name"]
        ]
        function_concurrencies = state.average_contention[task_type["name"]].values()
        function_replicas: Set[Tuple[Node, Platform]] = system_state.replicas[
            task_type["name"]
        ]

        replica_count = len(function_replicas)

        # Per-function concurrency level
        # Use TOTAL concurrency across all replicas (not average per replica)
        # This matches Knative's autoscaling formula:
        #   desired_replicas = ceil(total_concurrency / target_concurrency_per_replica)
        total_concurrency: float = sum(function_concurrencies) if function_concurrencies else 0.0

        # Result > 0 means scaling up
        # Result < 0 means scaling down
        # Result == 0 means current scaling level is adequate
        # Formula: desired = ceil(total / target), scaling_diff = desired - current
        concurrency_results: PlatformVector = {
            platform_type["shortName"]: (
                math.ceil(
                    total_concurrency / target_concurrencies[platform_type["shortName"]]
                )
                - replica_count
            )
            for platform_type in self.data.platform_types.values()
        }

        return concurrency_results

    def evict_idle_for(
        self, system_state: SystemState, task_type: TaskType, source_node_name: str
    ) -> Optional[Tuple[Node, Platform]]:
        """Free one replica of another function so a starved `task_type` can claim its platform.

        A new replica needs a platform that hosts no replica, and scale-down only removes a type's own surplus, so a
        type can starve forever while other types' replicas hold every compatible platform its source reaches (the
        scheduler detects this, GNNScheduler._starved_spin, and only then calls here). Scale-to-zero in Knative frees
        such a pod; this does the same for one replica: on a node the source reaches, on a platform `task_type` can run
        on with enough memory once freed, never a function's last replica. An idle one (empty queue, no running task,
        initialised) is released now, longest idle first, ties by (node id, platform id), and returned. When none is
        idle, the least-loaded one is drained instead: it leaves `system_state.replicas` at once, so nothing new is
        placed on it, and once its queue empties its platform is released and the starved type's replica is created there;
        returns None meanwhile (the caller retries). At most
        one drain per (starved type, source) at a time.
        """
        name = task_type["name"]
        server_only = os.environ.get("HEROSIM_SERVER_ONLY_REPLICAS", "0") == "1"
        idle, busy = [], []
        for other, replicas in system_state.replicas.items():
            if other == name or len(replicas) < 2:
                continue
            mem_other = self.data.task_types[other]["memoryRequirements"]
            for node, platform in replicas:
                short = platform.type["shortName"]
                if short not in task_type["platforms"] or not replica_platform_type_allowed(short):
                    continue
                if server_only and str(node.node_name).startswith("client_node"):
                    continue
                if node.node_name != source_node_name and (
                    str(node.node_name).startswith("client_node")
                    or source_node_name not in (getattr(node, "network_map", None) or {})
                ):
                    continue
                if not platform.initialized.triggered:
                    continue
                if node.available_memory + mem_other[short] < task_type["memoryRequirements"][short]:
                    continue
                load = len(platform.queue.items) + (1 if platform.current_task else 0) + len(getattr(platform, "inflight", ()))
                if load == 0:
                    idle.append((platform.idle_since, node.id, platform.id, other, node, platform))
                else:
                    busy.append((load, node.id, platform.id, other, node, platform))
        state = system_state.scheduler_state
        if idle:
            _, _, _, other, node, platform = min(idle, key=lambda c: (c[0], c[1], c[2]))
            (getattr(state, "average_contention", {}).get(other) or {}).pop((node.id, platform.id), None)
            released = self._release_replica(system_state, other, (node, platform))
            if released:
                self.starved_evictions = getattr(self, "starved_evictions", 0) + 1
                print(f"[ {self.env.now} ] evicted: {released} ({other}) for starved {name} from {source_node_name}")
            return released
        draining = self.__dict__.setdefault("_draining", set())
        if not busy or (name, source_node_name) in draining:
            return None
        load, _, _, other, node, platform = min(busy, key=lambda c: (c[0], c[1], c[2]))
        system_state.replicas[other].remove((node, platform))
        (getattr(state, "average_contention", {}).get(other) or {}).pop((node.id, platform.id), None)
        draining.add((name, source_node_name))
        print(f"[ {self.env.now} ] draining: {(node, platform)} ({other}, load {load}) for starved {name} "
              f"from {source_node_name}")
        self.env.process(self._release_when_drained(system_state, other, (node, platform), (name, source_node_name)))
        return None

    def _release_starved_rendezvous(self, replica, key) -> bool:
        """Break the drain deadlock: the draining platform's task waits (rendezvous) for peers that are themselves the
        starved type from the starved source, so neither can proceed. Plan those peers onto the drained node, where
        their replica is created once the platform empties, and resume the wait; the exchange is charged against that
        node. Only when every unplaced peer is such a starved task; otherwise the wait stays."""
        node, platform = replica
        waits = [(platform.rendezvous_task, platform.run)] if getattr(platform, "rendezvous_task", None) else []
        waits += list((getattr(platform, "rendezvous_procs", None) or {}).items())  # HEROSIM_REPLICA_RELEASE=1
        released = False
        for task, proc in waits:
            released |= self._release_one_rendezvous(node, platform, task, proc, key)
        return released

    def _release_one_rendezvous(self, node, platform, task, proc, key) -> bool:
        orchestrator = getattr(node, "orchestrator_ref", None)
        if orchestrator is None:
            return False
        unplaced = []
        for peer_id in sorted((orchestrator.peer_exchange or {}).get(task.id) or {}):
            peer = orchestrator.task_by_id.get(peer_id)
            if peer is None:
                return False
            if getattr(peer, "platform", None) is not None or getattr(peer, "planned_node_name", None) is not None:
                continue
            if peer.type["name"] != key[0] or peer.node_name != key[1]:
                return False
            unplaced.append(peer)
        if not unplaced:
            return False
        for peer in unplaced:
            peer.planned_node_name = node.node_name
        proc.interrupt(STARVED_RENDEZVOUS)
        self.starved_rendezvous_releases = getattr(self, "starved_rendezvous_releases", 0) + 1
        print(f"[ {self.env.now} ] rendezvous released: task {task.id} on {platform} planned "
              f"{[p.id for p in unplaced]} ({key[0]}) onto {node.node_name}")
        return True

    def _release_when_drained(self, system_state, function_name, replica, key) -> Generator:
        platform = replica[1]
        while platform.queue.items or platform.current_task or getattr(platform, "inflight", None):
            self._release_starved_rendezvous(replica, key)
            yield self.env.timeout(0.1)
        released = self._release_replica(system_state, function_name, replica, already_removed=True)
        self._draining.discard(key)
        if released:
            self.starved_evictions = getattr(self, "starved_evictions", 0) + 1
            print(f"[ {self.env.now} ] evicted: {released} ({function_name}) drained for starved {key[0]} "
                  f"from {key[1]}")
            yield self.env.process(
                self.create_first_replica(system_state, self.data.task_types[key[0]], source_node_name=key[1])
            )

    def create_first_replica(
        self, 
        system_state: SystemState, 
        task_type: TaskType,
        source_node_name: Optional[str] = None
    ):
        """
        Create the first replica for a task type.
        
        Args:
            system_state: Current system state
            task_type: Task type to create replica for
            source_node_name: Optional source node name to check network connectivity.
                            If provided, only creates replicas on nodes that can reach this node.
        """
        # Filter available resources by network connectivity if source_node_name is provided
        original_available_resources = system_state.available_resources
        filtered_resources = None
        
        if source_node_name:
            # Filter to only nodes that have network connectivity to the source
            nodes_with_connectivity: Set[Node] = set()
            
            for node, platforms in system_state.available_resources.items():
                can_reach_source = False
                
                # Local placement: same node as source (always valid)
                if node.node_name == source_node_name:
                    can_reach_source = True
                # Server node: check if it has network_map entry for source
                elif not node.node_name.startswith('client_node'):
                    if hasattr(node, 'network_map') and source_node_name in node.network_map:
                        can_reach_source = True
                
                if can_reach_source:
                    nodes_with_connectivity.add(node)
            
            if nodes_with_connectivity:
                # Create filtered resources dict
                filtered_resources = {
                    node: platforms 
                    for node, platforms in system_state.available_resources.items()
                    if node in nodes_with_connectivity
                }
                # Temporarily replace available_resources
                system_state.available_resources = filtered_resources
                
                logging.info(
                    f"[ {self.env.now} ] Creating {task_type['name']} replica: "
                    f"{len(nodes_with_connectivity)} nodes with connectivity to {source_node_name} "
                    f"(out of {len(original_available_resources)} total nodes)"
                )
            else:
                logging.warning(
                    f"[ {self.env.now} ] No nodes with connectivity to {source_node_name} "
                    f"for {task_type['name']} replica creation"
                )
        
        try:
            # Collect available hardware types from (possibly filtered) resources
            available_hardware: Set[str] = set()
            resources_to_check = filtered_resources if filtered_resources else original_available_resources
            for _, platforms in resources_to_check.items():
                for platform in platforms:
                    if platform.type["shortName"] in task_type["platforms"]:
                        if not replica_platform_type_allowed(platform.type["shortName"]):
                            continue
                        available_hardware.add(platform.type["shortName"])

            if not available_hardware:
                logging.error(
                    f"[ {self.env.now} ] No compatible hardware available for {task_type['name']} "
                    f"on nodes with connectivity to {source_node_name if source_node_name else 'any node'}"
                )
                return StopIteration(
                    f"No compatible hardware for {task_type['name']} on connected nodes"
                )

            stop = None
            # Try each available hardware type. `available_hardware` is a set, so its
            # iteration order is not reproducible across processes (PYTHONHASHSEED) —
            # sort so which hardware type gets scaled up first is deterministic.
            for platform_name in sorted(available_hardware):
                stop = yield self.env.process(
                    self.scale_up(
                        1,
                        system_state,
                        task_type["name"],
                        self.data.platform_types[platform_name]["shortName"],
                    )
                )

                if not isinstance(stop, StopIteration):
                    # Resource found, stop iterating
                    break

            return stop
        finally:
            # Always restore original available_resources
            if filtered_resources is not None:
                system_state.available_resources = original_available_resources

    def create_replica(
        self, couples_suitable: Set[Tuple[Node, Platform]], task_type: TaskType
    ):
        # Scaling functions that do not yield values must still be Generators
        # No-op as per https://stackoverflow.com/a/68628599/9568489
        if False:
            yield

        """
        # Knative only allocates CPUs
        filtered_couples = set(filter(
            lambda couple: couple[1].type["hardware"] == "cpu",
            couples_suitable
        ))
        """

        # Align with knative_network: prefer server nodes first, then clients.
        # Server-hosted replicas can typically serve more sources.
        server_couples = [
            c for c in couples_suitable if not c[0].node_name.startswith("client_node")
        ]
        client_couples = [
            c for c in couples_suitable if c[0].node_name.startswith("client_node")
        ]
        candidates = server_couples if server_couples else client_couples

        # Select a replica on the most available node. `couples_suitable` is a set, so
        # `candidates`' order is not reproducible across processes (PYTHONHASHSEED) —
        # tie-break deterministically on replica identity.
        available_couple = max(
            candidates,
            key=lambda couple: (couple[0].available_platforms, -couple[0].id, -couple[1].id),
        )

        return available_couple

    def initialize_replica(
        self,
        new_replica: Tuple[Node, Platform],
        function_replicas: Set[Tuple[Node, Platform]],
        task_type: TaskType,
        system_state: KnativeSystemState,
    ):
        node: Node = new_replica[0]
        platform: Platform = new_replica[1]

        physics = getattr(self.env, "warmth_physics", PLATFORM_REUSE_V1)
        retrieval_duration: DurationSecond = 0.0

        # warmth: skip entire pull branch when needs_image_pull is False.
        # Hold FilterStore for the full pull timeout (determined parity).
        if FIDELITY:
            # one record per call, not per platform: this function can run twice for one platform (the
            # "double initialize" below) and both calls queue for the node's storage
            pull_call = {"platform": platform, "fn": task_type["name"], "init_start": self.env.now,
                         "end": None, "done": False}
            calls = node.__dict__.setdefault("_fid_pull_calls", [])
            calls[:] = [c for c in calls if not c["done"]]
            calls.append(pull_call)
        if needs_image_pull(physics, platform, node, task_type):
            node_storage = yield node.storage.get(
                lambda storage: not storage.type["remote"]
            )
            if needs_image_pull(
                physics, platform, node, task_type, active_storage=node_storage
            ):
                logging.info(
                    f"[ {self.env.now} ] 💾 {node} needs to pull image for {task_type}"
                )

                retrieval_size: SizeGigabyte = task_type["imageSize"][
                    platform.type["shortName"]
                ]
                retrieval_speed: SpeedMBps = min(
                    node_storage.type["throughput"]["write"], node.network["bandwidth"]
                )
                retrieval_duration += (
                    retrieval_size / (retrieval_speed / 1024)
                    + node_storage.type["latency"]["write"]
                )

                stored = node_storage.store_function(platform.type["shortName"], task_type)

                if not stored:
                    logging.error(
                        f"[ {self.env.now} ] 💾 {node_storage} has no available capacity to"
                        f" cache image for {self}"
                    )

                if FIDELITY:
                    pull_call["end"] = self.env.now + retrieval_duration
                yield self.env.timeout(retrieval_duration)
            yield node.storage.put(node_storage)
        else:
            retrieval_duration = 0.0

        platform.storage_time += retrieval_duration

        # Update state
        # FIXME: Move to state update methods
        state: KnativeSchedulerState = system_state.scheduler_state
        # Knative policy
        state.average_contention[task_type["name"]][
            (new_replica[0].id, new_replica[1].id)
        ] = 1.0

        # FIXME: Double initialize bug...
        try:
            # Set platform to ready state
            platform.initialized.succeed()
        except RuntimeError:
            """
            logging.error(
                f"[ {self.env.now} ] Autoscaler tried to initialize "
                f"{new_replica[1]} ({new_replica[0]}) but it was already initialized."
            )

            logging.error(
                f"[ {self.env.now} ] Last allocation time: "
                f"{new_replica[1].last_allocated} "
                " -- Last removal time: "
                f"{new_replica[1].last_removed}"
            )
            """
            pass

        # Statistics (Node)
        node.cache_hits += int(image_pull_disk_hit(physics, platform, node, task_type))
        if FIDELITY:
            pull_call["done"] = True

    def remove_replica(
        self,
        function_replicas: Set[Tuple[Node, Platform]],
        task_type: TaskType,
        system_state: KnativeSystemState,
    ):
        # Scaling functions that do not yield values must still be Generators
        # No-op as per https://stackoverflow.com/a/68628599/9568489&
        if False:
            yield

        # Sort function replicas by in-flight requests count
        sorted_replicas = sorted(
            function_replicas,
            # Total key: the queue length alone ties for every eligible candidate
            # (scale-down only removes empty queues), so stable sort would fall back
            # to Set iteration order, which PYTHONHASHSEED does not pin.
            key=lambda couple: (len(couple[1].queue.items), couple[0].id, couple[1].id),
        )

        # Mark replica for removal if its task queue is empty
        # Return None if no replica can be removed
        removed_couple = next(
            (
                replica
                for replica in sorted_replicas
                if not replica[1].queue.items
                and not replica[1].current_task
                and not getattr(replica[1], "inflight", None)
                and (self.env.now - replica[1].idle_since) > self.policy.keep_alive
            ),
            None,
        )

        if removed_couple:
            # Update state
            # FIXME: Move to state update methods
            state: SchedulerState = system_state.scheduler_state
            try:
                # Knative policy
                del state.average_contention[task_type["name"]][
                    (removed_couple[0].id, removed_couple[1].id)
                ]
            except KeyError:
                """
                logging.error(
                    f"[ {self.env.now} ] Autoscaler tried to scale down "
                    f"{task_type['name']}, but {removed_couple[1]} was already removed"
                )
                """
                pass

        return removed_couple
