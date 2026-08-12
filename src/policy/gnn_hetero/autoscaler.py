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

from typing import Set, Tuple, TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from src.placement.infrastructure import Node

from src.policy.gnn_hetero.model import KnativeSchedulerState, KnativeSystemState

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

from src.placement.autoscaler import Autoscaler
from src.placement.warmth import (
    PLATFORM_REUSE_V1,
    image_pull_disk_hit,
    needs_image_pull,
)



class KnativeAutoscaler(Autoscaler):





    def scaling_level(self, system_state: KnativeSystemState, task_type: TaskType):
        """Calculate scaling level - matches knative_network autoscaler."""
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
            # Try each available hardware type
            for platform_name in available_hardware:
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

        # Select a replica on the most available node
        available_couple = max(
            candidates,
            key=lambda couple: couple[0].available_platforms,
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
            function_replicas, key=lambda couple: len(couple[1].queue.items)
        )

        # Mark replica for removal if its task queue is empty
        # Return None if no replica can be removed
        removed_couple = next(
            (
                replica
                for replica in sorted_replicas
                if not replica[1].queue.items
                and not replica[1].current_task
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
