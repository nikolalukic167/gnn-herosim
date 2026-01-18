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
from collections import Counter
from timeit import default_timer
from typing import Set, Tuple, TYPE_CHECKING, List, Generator, Any, Dict

if TYPE_CHECKING:
    from src.placement.infrastructure import Node, Platform, Task

from src.placement.model import SchedulerState, SystemState

from src.placement.scheduler import Scheduler


class RoundRobinScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = 5
        self.batch_timeout = 0.1

    def scheduler_process(self) -> Generator:
        if False:
            yield

        print(
            f"[ {self.env.now} ] RoundRobinNetwork Scheduler started with policy"
            f" {self.policy} (batch_size={self.batch_size})"
        )

        while True:
            batch_tasks = yield self.env.process(self._collect_task_batch())
            
            if not batch_tasks:
                yield self.env.timeout(0.1)
                continue

            print(f"[ {self.env.now} ] RoundRobin: Processing batch of {len(batch_tasks)} tasks")
            yield self.env.process(self._process_task_batch(batch_tasks))

    def _collect_task_batch(self) -> Generator[Any, Any, List[Task]]:
        """
        Collect tasks into a batch with timeout to avoid blocking.
        
        Uses polling-based approach to avoid dangling get events that can
        cause tasks to be silently consumed.
        """
        batch: List[Task] = []
        
        def task_filter(queued_task):
            return all(dependency.finished for dependency in queued_task.dependencies)
        
        # First task: wait indefinitely (blocking is expected)
        task: Task = yield self.tasks.get(task_filter)
        batch.append(task)
        
        # Set deadline for collecting rest of batch
        batch_deadline = self.env.now + self.batch_timeout
        
        # Collect remaining tasks using polling (avoids dangling get events)
        while len(batch) < self.batch_size:
            remaining_time = batch_deadline - self.env.now
            if remaining_time <= 0:
                break
            
            # Check if there are any ready tasks in the queue
            ready_tasks = [t for t in self.tasks.items if task_filter(t)]
            
            if ready_tasks:
                # Get the first ready task immediately
                task = yield self.tasks.get(task_filter)
                batch.append(task)
            else:
                # No ready tasks - wait a small step and check again
                step_time = min(0.01, remaining_time)
                yield self.env.timeout(step_time)
        
        return batch

    def _ensure_replicas_for_batch(
        self, 
        batch_tasks: List[Task], 
        system_state: SystemState
    ) -> Generator[Any, Any, Dict[str, int]]:
        """
        Pre-scan batch and ensure enough replicas exist for all task types.
        Waits for initialization to complete before returning.
        """
        # Count how many tasks of each type are in the batch
        task_type_counts: Dict[str, int] = Counter(task.type["name"] for task in batch_tasks)
        replicas_created: Dict[str, int] = {}
        
        # Track newly created replicas so we can wait for initialization
        new_replicas: List[Tuple[Node, Platform]] = []
        
        for task_type_name, needed_count in task_type_counts.items():
            current_replicas = system_state.replicas.get(task_type_name, set())
            existing_replica_ids = {(n.id, p.id) for n, p in current_replicas}
            current_count = len(current_replicas)
            
            # Calculate how many additional replicas we need
            deficit = needed_count - current_count
            
            if deficit <= 0:
                replicas_created[task_type_name] = 0
                continue
            
            print(f"[ {self.env.now} ] RoundRobin: Pre-creating {deficit} replica(s) for {task_type_name} "
                  f"(have {current_count}, need {needed_count} for batch)")
            
            # Get task type data
            task_type = self.data.task_types.get(task_type_name)
            if not task_type:
                logging.warning(f"[ {self.env.now} ] Unknown task type: {task_type_name}")
                replicas_created[task_type_name] = 0
                continue
            
            created = 0
            for _ in range(deficit):
                stop = yield self.env.process(
                    self.autoscaler.create_first_replica(system_state, task_type)
                )
                if not isinstance(stop, StopIteration):
                    created += 1
                else:
                    logging.warning(
                        f"[ {self.env.now} ] RoundRobin: Could not create more replicas for {task_type_name} "
                        f"(created {created}/{deficit})"
                    )
                    break
            
            # Find newly created replicas
            for node, plat in system_state.replicas.get(task_type_name, set()):
                if (node.id, plat.id) not in existing_replica_ids:
                    new_replicas.append((node, plat))
            
            replicas_created[task_type_name] = created
            if created > 0:
                print(f"[ {self.env.now} ] RoundRobin: Pre-created {created} replica(s) for {task_type_name}")
        
        # Wait for all newly created replicas to initialize
        if new_replicas:
            print(f"[ {self.env.now} ] RoundRobin: Waiting for {len(new_replicas)} replica(s) to initialize...")
            init_events = [plat.initialized for _, plat in new_replicas]
            yield self.env.all_of(init_events)
            print(f"[ {self.env.now} ] RoundRobin: All {len(new_replicas)} replica(s) initialized")
        
        return replicas_created

    def _process_task_batch(self, batch_tasks: List[Task]) -> Generator:
        """Process a batch using round-robin placement strategy."""
        print(f"[ {self.env.now} ] RoundRobin: Processing {len(batch_tasks)} tasks in batch")
        
        # Track scheduling time for the batch
        batch_start = default_timer()
        
        system_state: SystemState = yield self.mutex.get()
        
        # Pre-create replicas and wait for initialization
        replicas_created = yield self.env.process(
            self._ensure_replicas_for_batch(batch_tasks, system_state)
        )
        if any(count > 0 for count in replicas_created.values()):
            print(f"[ {self.env.now} ] RoundRobin: Pre-created replicas: {replicas_created}")
        
        # Track used platforms within this batch to ensure unique placements
        used_platforms: Set[Tuple[int, int]] = set()
        
        for task in batch_tasks:
            task_start = default_timer()
            task_replicas = system_state.replicas.get(task.type["name"], set())

            if not task_replicas:
                logging.warning(f"[ {self.env.now} ] RoundRobin: No replicas for {task}")
                task.postponed_count += 1
                
                if task.postponed_count > 9999999:
                    self._mark_task_failed(task, "No replicas available")
                    continue
                
                yield self.tasks.put(task)
                stop = yield self.env.process(
                    self.autoscaler.create_first_replica(system_state, task.type)
                )
                
                if isinstance(stop, StopIteration):
                    logging.warning(f"[ {self.env.now} ] RoundRobin: Failed to create replica for {task.type['name']}")
                
                # CRITICAL: Force simulation time to advance
                self.env.step()
                continue

            # Get valid replicas (network-aware)
            valid_replicas = self._get_valid_replicas(task_replicas, task)
            
            if not valid_replicas:
                task.postponed_count += 1
                
                if task.postponed_count > 9999999:
                    self._mark_task_failed(task, "No valid replicas with network connectivity")
                    continue
                
                yield self.tasks.put(task)
                stop = yield self.env.process(
                    self.autoscaler.create_first_replica(system_state, task.type)
                )
                
                if isinstance(stop, StopIteration):
                    logging.warning(f"[ {self.env.now} ] RoundRobin: Failed to create replica for {task.type['name']}")
                
                # CRITICAL: Force simulation time to advance
                self.env.step()
                continue

            # Filter: unused AND initialized replicas
            available_replicas = [
                (node, plat) for node, plat in valid_replicas 
                if (node.id, plat.id) not in used_platforms
                and plat.initialized.triggered
            ]
            
            if not available_replicas:
                task.postponed_count += 1
                
                if task.postponed_count > 9999999:
                    self._mark_task_failed(task, "No unique initialized replicas available")
                    continue
                
                yield self.tasks.put(task)
                stop = yield self.env.process(
                    self.autoscaler.create_first_replica(system_state, task.type)
                )
                
                if isinstance(stop, StopIteration):
                    logging.warning(f"[ {self.env.now} ] RoundRobin: Failed to create replica for {task.type['name']}")
                
                # CRITICAL: Force simulation time to advance
                self.env.step()
                continue

            # Round-robin placement: find least scheduled replica
            state: SchedulerState = system_state.scheduler_state
            
            # Ensure scheduled_count is initialized
            if not hasattr(state, 'scheduled_count'):
                state.scheduled_count = {task.type["name"]: {} for task_type in self.data.task_types}
            if task.type["name"] not in state.scheduled_count:
                state.scheduled_count[task.type["name"]] = {}
            
            target_node, target_platform = min(
                available_replicas,
                key=lambda couple: state.scheduled_count[task.type["name"]].get(
                    (couple[0].id, couple[1].id), 0
                ),
            )

            # Update scheduler state
            state.scheduled_count[task.type["name"]][
                (target_node.id, target_platform.id)
            ] = state.scheduled_count[task.type["name"]].get(
                (target_node.id, target_platform.id), 0
            ) + 1

            # Mark this platform as used
            used_platforms.add((target_node.id, target_platform.id))

            task.execution_node = target_node.node_name
            task.execution_platform = str(target_platform.id)
            
            # Track scheduling decision time
            task_end = default_timer()
            task.roundrobin_decision_time = task_end - task_start

            node: Node = yield self.nodes.get(lambda node: node.id == target_node.id)
            task.node = node
            node.unused = False
            
            platform: Platform = yield node.platforms.get(lambda platform: platform.id == target_platform.id)
            task.platform = platform

            end = default_timer()
            elapsed_clock_time = end - task_start
            node.wall_clock_scheduling_time += elapsed_clock_time

            yield platform.queue.put(task)
            yield task.scheduled.succeed()

            yield node.platforms.put(platform)
            yield self.nodes.put(node)

            print(f"RoundRobin placed task {task.id} on ({target_node.node_name}, platform {target_platform.id}) "
                  f"[queue: {len(target_platform.queue.items)}]")

        yield self.mutex.put(system_state)
        
        batch_end = default_timer()
        batch_time = (batch_end - batch_start) * 1000  # ms
        print(f"[ {self.env.now} ] RoundRobin: Batch processing complete for {len(batch_tasks)} tasks "
              f"(scheduling time: {batch_time:.2f}ms)")

    def _mark_task_failed(self, task: Task, reason: str):
        """Mark a task as failed and trigger all events."""
        logging.error(f"[ {self.env.now} ] Task {task.id} failed: {reason}")
        task.finished = True
        task.failed = True
        task.failure_reason = reason
        if not task.scheduled.triggered:
            task.scheduled.succeed()
        if not task.arrived.triggered:
            task.arrived.succeed()
        if not task.started.triggered:
            task.started.succeed()
        if not task.done.triggered:
            task.done.succeed()

    def placement(self, system_state: SystemState, task: Task) -> Generator:
        # This method is not used when scheduler_process is overridden
        # but kept for compatibility
        if False:
            yield
        return None

    def _get_valid_replicas(self, replicas: Set[Tuple[Node, Platform]], task: Task) -> List[Tuple[Node, Platform]]:
        """Get valid replicas based on network topology."""
        valid_replicas = []
        for node, platform in replicas:
            if node.node_name == task.node_name:
                valid_replicas.append((node, platform))
            elif not node.node_name.startswith('client_node'):
                if hasattr(node, 'network_map') and task.node_name in node.network_map:
                    valid_replicas.append((node, platform))
        return valid_replicas
