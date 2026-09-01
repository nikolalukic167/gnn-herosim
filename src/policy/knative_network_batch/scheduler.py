"""Network-aware Knative scheduler with timeout-based batching."""

from __future__ import annotations

import json
import logging
import os
from timeit import default_timer
from typing import Any, Dict, Generator, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from src.placement.infrastructure import Node, Platform, Task

from src.placement.live_audit import _replicas_by_type_payload
from src.placement.model import SystemState
from src.policy.knative_network.scheduler import KnativeScheduler as KnativeNetworkScheduler


class KnativeBatchScheduler(KnativeNetworkScheduler):
    """Batch wrapper around the existing Knative network shortest-queue rule."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = int(os.environ.get("KNATIVE_BATCH_SIZE", "4"))
        self.batch_timeout = float(os.environ.get("KNATIVE_BATCH_TIMEOUT", "0.002"))

    def scheduler_process(self) -> Generator:
        if False:
            yield

        logging.info(
            f"[ {self.env.now} ] Knative batch network scheduler started "
            f"(batch_size={self.batch_size}, timeout={self.batch_timeout})"
        )

        while True:
            batch_tasks = yield self.env.process(self._collect_task_batch())
            if not batch_tasks:
                yield self.env.timeout(0.001)
                continue
            yield self.env.process(self._process_task_batch(batch_tasks))

    def _collect_task_batch(self) -> Generator[Any, Any, List[Task]]:
        batch: List[Task] = []

        def task_filter(queued_task):
            return all(dependency.finished for dependency in queued_task.dependencies)

        task: Task = yield self.tasks.get(task_filter)
        batch.append(task)

        timeout_remaining = self.batch_timeout
        poll_interval = min(0.001, self.batch_timeout) if self.batch_timeout > 0 else 0.0

        while len(batch) < self.batch_size and timeout_remaining > 0:
            ready_tasks = [t for t in self.tasks.items if task_filter(t)]
            if ready_tasks:
                task = yield self.tasks.get(task_filter)
                batch.append(task)
            else:
                wait_time = min(poll_interval, timeout_remaining)
                yield self.env.timeout(wait_time)
                timeout_remaining -= wait_time

        return batch

    def _process_task_batch(self, batch_tasks: List[Task]) -> Generator:
        batch_start = default_timer()
        system_state: Optional[SystemState] = yield self.mutex.get()
        if system_state is None:
            logging.error(f"[ {self.env.now} ] Knative batch: failed to get system state")
            yield self.mutex.put(None)
            return

        self._maybe_capture_batch_live_audit_snapshot(system_state, batch_tasks)

        for task in batch_tasks:
            task_start = default_timer()
            replicas: Set[Tuple[Node, Platform]] = system_state.replicas[task.type["name"]]
            valid_replicas = self._get_valid_replicas(replicas, task)

            if not valid_replicas:
                logging.warning(
                    f"[ {self.env.now} ] Knative batch: no network-accessible replica for {task}"
                )
                task.postponed_count += 1
                yield self.tasks.put(task)
                yield self.env.process(
                    self.autoscaler.create_first_replica(
                        system_state, task.type, source_node_name=task.node_name
                    )
                )
                continue

            sched_node, sched_platform = yield self.env.process(
                self.placement(system_state, task)
            )
            task.execution_node = sched_node.node_name
            task.execution_platform = str(sched_platform.id)

            node: Node = yield self.nodes.get(lambda node: node.id == sched_node.id)
            task.node = node
            node.unused = False
            platform: Platform = yield node.platforms.get(
                lambda platform: platform.id == sched_platform.id
            )
            task.platform = platform

            elapsed_clock_time = default_timer() - task_start
            node.wall_clock_scheduling_time += elapsed_clock_time

            yield platform.queue.put(task)
            yield task.scheduled.succeed()
            yield node.platforms.put(platform)
            yield self.nodes.put(node)

        yield self.mutex.put(system_state)
        batch_time = (default_timer() - batch_start) * 1000.0
        logging.debug(
            f"[ {self.env.now} ] Knative batch scheduled {len(batch_tasks)} tasks "
            f"in {batch_time:.2f}ms"
        )

    def _audit_batch_qualifies(
        self,
        system_state: SystemState,
        batch_tasks: List[Task],
    ) -> bool:
        min_batch_size = int(os.environ.get("LIVE_AUDIT_MIN_BATCH_SIZE", "4"))
        if len(batch_tasks) < min_batch_size:
            return False

        min_candidates = int(os.environ.get("LIVE_AUDIT_MIN_CANDIDATES", "4"))
        for task in batch_tasks:
            payload = self._audit_task_payload(system_state, task)
            candidate_count = len(payload.get("candidates", []))
            if candidate_count == 0:
                return False
            if min_candidates and candidate_count < min_candidates:
                return False
        return True

    def _maybe_capture_batch_live_audit_snapshot(
        self,
        system_state: SystemState,
        batch_tasks: List[Task],
    ) -> None:
        output_path = os.environ.get("LIVE_AUDIT_SNAPSHOT_PATH")
        if not output_path or not batch_tasks:
            return

        max_snapshots = int(os.environ.get("LIVE_AUDIT_MAX_SNAPSHOTS", "500"))
        if self._audit_snapshots_written >= max_snapshots:
            return

        stride = max(1, int(os.environ.get("LIVE_AUDIT_STRIDE", "1")))
        if batch_tasks[0].id % stride != 0:
            return

        if not self._audit_batch_qualifies(system_state, batch_tasks):
            return

        snapshot = {
            "snapshot_id": self._audit_snapshots_written,
            "time": float(self.env.now),
            "policy": "knative_network_batch",
            "horizon": len(batch_tasks),
            "trigger_task_id": int(batch_tasks[0].id),
            "chosen": None,
            "full_queue_snapshot": self._capture_full_queue_snapshot(),
            "tasks": [
                self._audit_task_payload(system_state, task)
                for task in batch_tasks
            ],
            # Shared schema with src/placement/live_audit.py — the P3 horizon sweep
            # needs the full per-type replica state, not just batch candidates.
            "replicas_by_type": _replicas_by_type_payload(system_state),
        }

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "a") as f:
            f.write(json.dumps(snapshot, separators=(",", ":")) + "\n")
        self._audit_snapshots_written += 1
