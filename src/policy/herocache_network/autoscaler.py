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

from typing import Generator, Optional, Set, Tuple, TYPE_CHECKING

from src.policy.herocache_network.model import HRCSchedulerState, HRCSystemState

if TYPE_CHECKING:
    from src.placement.infrastructure import Node, Platform

from src.placement.model import (
    DurationSecond,
    PlatformVector,
    SchedulerState,
    SizeGigabyte,
    SpeedMBps,
    SystemState,
    TaskType,
)

from src.placement.autoscaler import Autoscaler


class HRCAutoscaler(Autoscaler):
    """Knative-network autoscaling; HRC scheduler handles placement separately."""

    def scaling_level(
        self, system_state: HRCSystemState, task_type: TaskType
    ) -> Generator:
        if False:
            yield

        state: HRCSchedulerState = system_state.scheduler_state
        target_concurrencies: PlatformVector = state.target_concurrencies[
            task_type["name"]
        ]
        function_concurrencies = state.average_contention[task_type["name"]].values()
        function_replicas: Set[Tuple[Node, Platform]] = system_state.replicas[
            task_type["name"]
        ]

        replica_count = len(function_replicas)
        total_concurrency: float = (
            sum(function_concurrencies) if function_concurrencies else 0.0
        )

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
        source_node_name: Optional[str] = None,
    ) -> Generator:
        available_hardware: Set[str] = set()

        source_node = None
        if source_node_name:
            for node in system_state.available_resources.keys():
                if node.node_name == source_node_name:
                    source_node = node
                    break

            if source_node is None:
                logging.warning(
                    f"[ {self.env.now} ] Autoscaler: Source node {source_node_name} not found in available_resources. "
                    f"Available nodes: {[n.node_name for n in system_state.available_resources.keys()][:5]}"
                )
            else:
                logging.info(
                    f"[ {self.env.now} ] Autoscaler: Creating replica for {task_type['name']} from {source_node_name}, "
                    f"source node network_map has {len(source_node.network_map)} connections"
                )

        for node, platforms in system_state.available_resources.items():
            if source_node_name is not None:
                is_reachable = False

                if node.node_name == source_node_name:
                    is_reachable = True
                elif source_node is not None:
                    if node.node_name in source_node.network_map:
                        is_reachable = True
                    elif source_node_name.startswith("client_node"):
                        is_reachable = False
                    elif not source_node_name.startswith(
                        "client_node"
                    ) and not node.node_name.startswith("client_node"):
                        is_reachable = True
                else:
                    if source_node_name.startswith("client_node"):
                        is_reachable = False
                    elif not node.node_name.startswith("client_node"):
                        is_reachable = True
                    else:
                        is_reachable = False

                if not is_reachable:
                    logging.debug(
                        f"[ {self.env.now} ] Autoscaler: Skipping unreachable node {node.node_name} "
                        f"for task from {source_node_name}"
                    )
                    continue

            for platform in platforms:
                if platform.type["shortName"] in task_type["platforms"]:
                    available_hardware.add(platform.type["shortName"])

        stop = None
        if not available_hardware:
            logging.warning(
                f"[ {self.env.now} ] Autoscaler: No available hardware for {task_type['name']} from {source_node_name}"
            )
            if source_node_name:
                logging.warning(
                    f"[ {self.env.now} ]   Source node {source_node_name} network_map: "
                    f"{list(source_node.network_map.keys())[:5] if source_node else 'N/A'}"
                )
        else:
            logging.info(
                f"[ {self.env.now} ] Autoscaler: Found {len(available_hardware)} available hardware types "
                f"for {task_type['name']}: {list(available_hardware)}"
            )

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
                break

        return stop

    def create_replica(
        self, couples_suitable: Set[Tuple[Node, Platform]], task_type: TaskType
    ) -> Generator:
        if False:
            yield

        server_couples = [
            c for c in couples_suitable if not c[0].node_name.startswith("client_node")
        ]
        client_couples = [
            c for c in couples_suitable if c[0].node_name.startswith("client_node")
        ]
        candidates = server_couples if server_couples else client_couples

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
        system_state: HRCSystemState,
    ) -> Generator:
        node: Node = new_replica[0]
        platform: Platform = new_replica[1]

        warm_function: bool = (
            platform.previous_task is not None
            and platform.previous_task.type["name"] == task_type["name"]
        )

        retrieval_duration: DurationSecond = 0.0

        if not warm_function:
            logging.info(
                f"[ {self.env.now} ] 💾 {node} needs to pull image for {task_type}"
            )

            retrieval_size: SizeGigabyte = task_type["imageSize"][
                platform.type["shortName"]
            ]
            node_storage = yield node.storage.get(
                lambda storage: not storage.type["remote"]
            )
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

            yield node.storage.put(node_storage)

        state: HRCSchedulerState = system_state.scheduler_state
        state.average_contention[task_type["name"]][
            (new_replica[0].id, new_replica[1].id)
        ] = 1.0

        yield self.env.timeout(retrieval_duration)

        platform.storage_time += retrieval_duration

        try:
            yield platform.initialized.succeed()
        except RuntimeError:
            pass

        node.cache_hits += 0

    def remove_replica(
        self,
        function_replicas: Set[Tuple[Node, Platform]],
        task_type: TaskType,
        system_state: HRCSystemState,
    ) -> Generator:
        if False:
            yield

        sorted_replicas = sorted(
            function_replicas, key=lambda couple: len(couple[1].queue.items)
        )

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
            state: SchedulerState = system_state.scheduler_state
            try:
                del state.average_contention[task_type["name"]][
                    (removed_couple[0].id, removed_couple[1].id)
                ]
            except KeyError:
                pass

        return removed_couple
