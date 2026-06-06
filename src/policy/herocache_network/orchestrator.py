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

import logging
from typing import TYPE_CHECKING, Dict, Set, Tuple

from src.policy.herocache_network.model import HRCSchedulerState, HRCSystemState

if TYPE_CHECKING:
    from src.placement.infrastructure import Node, Platform

from src.placement.orchestrator import Orchestrator


class HRCOrchestrator(Orchestrator):
    def initialize_state(self) -> HRCSystemState:
        scheduler_state = HRCSchedulerState(
            average_contention={task_type: {} for task_type in self.data.task_types},
            panic_contention={task_type: {} for task_type in self.data.task_types},
            target_concurrencies={
                task_type: {
                    platform_type["shortName"]: self.policy.queue_length
                    for platform_type in self.data.platform_types.values()
                }
                for task_type in self.data.task_types
            },
        )
        available_resources: Dict[Node, Set[Platform]] = {
            node: {platform for platform in set(node.platforms.items)}
            for node in set(self.nodes.items)
        }
        replicas: Dict[str, Set[Tuple[Node, Platform]]] = {
            task_type: set() for task_type in self.data.task_types
        }
        system_state = HRCSystemState(
            scheduler_state=scheduler_state,
            available_resources=available_resources,
            replicas=replicas,
        )

        return system_state

    def monitor_process(self):
        logging.info(f"[ {self.env.now} ] Orchestrator Monitor started")

        while True:
            system_state: HRCSystemState = yield self.mutex.get()
            replicas: Dict[str, Set[Tuple[Node, Platform]]] = system_state.replicas
            state: HRCSchedulerState = system_state.scheduler_state

            for function_name, function_replicas in replicas.items():
                for node, platform in function_replicas:
                    state.average_contention[function_name][
                        (node.id, platform.id)
                    ] = len(platform.queue.items)

            yield self.mutex.put(system_state)

            yield self.env.timeout(1)
