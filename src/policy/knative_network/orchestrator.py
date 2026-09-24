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
import math
from typing import TYPE_CHECKING, Dict, Set, Tuple

from src.policy.knative.model import KnativeSchedulerState, KnativeSystemState

if TYPE_CHECKING:
    from src.placement.infrastructure import Node, Platform

from src.placement.orchestrator import Orchestrator


class KnativeOrchestrator(Orchestrator):
    def __init__(self, *args, **kwargs):
        # `infrastructure` is passed as a kwarg (simulation.py) but the base __init__ does not
        # accept it, so pop it first — mirroring DeterminedOrchestrator.
        self.infrastructure = kwargs.pop("infrastructure", None)
        super().__init__(*args, **kwargs)
        # No-op unless a caller injects forced placements (rollout_imitation_v1's label engine):
        # copy config["infrastructure"]["forced_placements"] onto a scheduler that supports it.
        # Absent -> normal reactive/rule behaviour, untouched.
        fp = self.infrastructure.get("forced_placements") if isinstance(self.infrastructure, dict) else None
        if fp and hasattr(self.scheduler, "forced_placements"):
            self.scheduler.forced_placements = fp
        # lookahead_mp_v1 P0b: a scheduler that predicts unarrived partners needs the FULL event
        # list; the gateway pops from time_series.events, so hand over a shallow copy now.
        if hasattr(self.scheduler, "pg_event_index"):
            self.scheduler.pg_event_index = list(self.time_series.events)

    def initialize_state(self) -> KnativeSystemState:
        scheduler_state = KnativeSchedulerState(
            average_contention={task_type: {} for task_type in self.data.task_types},
            panic_contention={task_type: {} for task_type in self.data.task_types},
            target_concurrencies={
                task_type: {
                    # platform_type["shortName"]: self.policy.queue_length if platform_type["hardware"] == "cpu" else 0.0
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
        from src.placement.replica_seeding import integrate_initial_replicas

        integrate_initial_replicas(
            replicas=replicas,
            available_resources=available_resources,
            initial_replicas=self.initial_replicas,
            task_types=self.data.task_types,
            average_contention=scheduler_state.average_contention,
            label="KnativeOrchestrator",
        )
        system_state = KnativeSystemState(
            scheduler_state=scheduler_state,
            available_resources=available_resources,
            replicas=replicas,
            tasks=self.task_archive,
            time_series=self.time_series
        )

        return system_state

    def monitor_process(self):
        # TODO: State initialization and update methods should be made abstract
        # and moved to policy package
        logging.info(f"[ {self.env.now} ] Orchestrator Monitor started")

        while True:
            system_state: KnativeSystemState = yield self.mutex.get()
            replicas: Dict[str, Set[Tuple[Node, Platform]]] = system_state.replicas
            state: KnativeSchedulerState = system_state.scheduler_state

            for function_name, function_replicas in replicas.items():
                for node, platform in function_replicas:
                    state.average_contention[function_name][
                        (node.id, platform.id)
                    ] = len(platform.queue.items)

            yield self.mutex.put(system_state)

            yield self.env.timeout(1)
