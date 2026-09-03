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
import os
from typing import TYPE_CHECKING, Dict, Set, Tuple

from src.policy.gnn.model import KnativeSchedulerState, KnativeSystemState

if TYPE_CHECKING:
    from src.placement.infrastructure import Node, Platform

from src.placement.orchestrator import Orchestrator


class GNNOrchestrator(Orchestrator):
    """GNN Orchestrator - simplified to match knative_network structure.
    
    The GNN model is passed via the models parameter and forwarded to the scheduler.
    """
    
    def __init__(self, *args, models=None, **kwargs):
        """Initialize orchestrator with optional GNN models."""
        # scheduler_config / device_type_mapping are GNN-only; keep initial_replicas
        # so scarce-preinit live stubs seed replica sets (do NOT pop them).
        self.scheduler_config = kwargs.pop('scheduler_config', None)
        kwargs.pop('device_type_mapping', None)

        # Store models temporarily - will be overwritten by parent's __init__
        _models = models
        print(f"[GNN Orchestrator] __init__ called with models={models is not None}", flush=True)
        if models:
            print(f"[GNN Orchestrator] models type: {type(models)}, keys: {list(models.keys()) if isinstance(models, dict) else 'N/A'}", flush=True)

        # Call parent init (which sets self.models = None since models not in kwargs)
        super().__init__(*args, **kwargs)

        # Re-set self.models after parent init
        self.models = _models
        print(f"[GNN Orchestrator] After super().__init__, self.models restored: {self.models is not None}", flush=True)
        # Stub may set scheduler.batch_size=N for determined refs; GNN/MLP must stay
        # within [2,4] or they silently fall back to shortest-queue.
        if self.scheduler_config and "batch_timeout" in self.scheduler_config:
            self.scheduler.batch_timeout = float(self.scheduler_config["batch_timeout"])
        if self.scheduler_config and "batch_size" in self.scheduler_config:
            cfg_bs = int(self.scheduler_config["batch_size"])
            from src.policy.gnn.scheduler import MAX_BATCH_SIZE_FOR_GNN, MIN_BATCH_SIZE_FOR_GNN

            if cfg_bs > MAX_BATCH_SIZE_FOR_GNN or cfg_bs < MIN_BATCH_SIZE_FOR_GNN:
                print(
                    f"[GNN Orchestrator] Ignoring infrastructure scheduler.batch_size={cfg_bs} "
                    f"(outside GNN range [{MIN_BATCH_SIZE_FOR_GNN},{MAX_BATCH_SIZE_FOR_GNN}]); "
                    f"keeping GNN_BATCH_SIZE={self.scheduler.batch_size}",
                    flush=True,
                )
            else:
                # The cell config wins over the env var. That used to happen silently: an
                # episode run with GNN_BATCH_SIZE=1 against a config carrying batch_size=4
                # served 4-task batches and reproduced the batch_size=4 result to the last
                # digit (measured 2026-09-03, bb_core8_bw1p5/cell01). An explicitly
                # exported value that disagrees with the config is a misconfigured
                # experiment, not a preference to be overridden.
                env_raw = os.environ.get("GNN_BATCH_SIZE")
                if env_raw is not None and int(env_raw) != cfg_bs:
                    raise ValueError(
                        f"FAIL LOUD: GNN_BATCH_SIZE={env_raw} was exported but the cell config "
                        f"declares scheduler.batch_size={cfg_bs}, which takes precedence. "
                        f"Edit the config (or unset the variable) so the served batch size "
                        f"is the one the experiment names."
                    )
                self.scheduler.batch_size = cfg_bs
    
    def initialize_state(self) -> KnativeSystemState:
        """Initialize system state - matches knative_network."""
        # Initialize scheduler state
        scheduler_state = KnativeSchedulerState(
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
        # Initialize available resources to all Tuple[Node, Platform]
        available_resources: Dict[Node, Set[Platform]] = {
            node: {platform for platform in set(node.platforms.items)}
            for node in set(self.nodes.items)
        }
        # Initialize function replicas to empty sets
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
            label="GNNOrchestrator",
        )
        system_state = KnativeSystemState(
            scheduler_state=scheduler_state,
            available_resources=available_resources,
            replicas=replicas,
            tasks=self.task_archive,
            time_series=self.time_series
        )

        # Pass models to scheduler if available
        print(f"[GNN Orchestrator] initialize_state called, models={self.models is not None}", flush=True)
        print(f"[GNN Orchestrator] scheduler has set_models: {hasattr(self.scheduler, 'set_models')}", flush=True)
        if self.models:
            print(f"[GNN Orchestrator] models keys: {list(self.models.keys()) if isinstance(self.models, dict) else 'not a dict'}", flush=True)
            if hasattr(self.scheduler, 'set_models'):
                self.scheduler.set_models(self.models)
                print("[GNN Orchestrator] Models passed to scheduler", flush=True)
            else:
                raise RuntimeError(
                    "GNNOrchestrator: scheduler missing set_models — cannot load GNN/MLP"
                )
        else:
            print("[GNN Orchestrator] WARNING: self.models is None or empty!", flush=True)

        return system_state

    def monitor_process(self):
        """Monitor process - matches knative_network (simple direct queue count)."""
        logging.info(f"[ {self.env.now} ] GNN Orchestrator Monitor started")

        while True:
            system_state: KnativeSystemState = yield self.mutex.get()
            replicas: Dict[str, Set[Tuple[Node, Platform]]] = system_state.replicas
            state: KnativeSchedulerState = system_state.scheduler_state

            # Count queue depth for autoscaling (direct assignment like knative_network)
            for function_name, function_replicas in replicas.items():
                for node, platform in function_replicas:
                    state.average_contention[function_name][
                        (node.id, platform.id)
                    ] = len(platform.queue.items)

            yield self.mutex.put(system_state)

            # Wake Monitor up once per second
            yield self.env.timeout(1)
