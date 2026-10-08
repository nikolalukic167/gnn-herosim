"""Deferring a task whose source reaches no replica, shared by every scheduler family (kpa_scaleout_v1 A4).

A deferred task goes straight back into the scheduler queue. When no replica can be created for it (every
compatible platform its source reaches is taken, or under kpa the node's free memory cannot hold one), the
scheduler re-defers the same task without the clock moving: the frozen-clock hang. After DEFER_SPIN_LIMIT
deferrals at one instant the scheduler asks the autoscaler to free a replica of another type
(`evict_idle_for`, GNN-family autoscaler) and otherwise retries after DEFER_RETRY_S.
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Dict, Generator, Iterable, Optional, Tuple

if TYPE_CHECKING:
    from src.placement.infrastructure import Task
    from src.placement.model import SystemState

DEFER_SPIN_LIMIT = 50
DEFER_RETRY_S = float(os.environ.get("HEROSIM_DEFER_RETRY_S", "1.0"))
STARVED_LOG_INTERVAL_S = 60.0


class StarvedForeverError(RuntimeError):
    """A task's source reaches no node with a platform type its function can run on: no capacity will ever free."""


def log_starved(owner: Any, now: float, key: Tuple[str, ...], message: str, level: int = logging.ERROR) -> bool:
    """Emit a starved-task line at most once per `STARVED_LOG_INTERVAL_S` simulated seconds per `key`, carrying the
    number of lines suppressed since. A stuck run retries every `DEFER_RETRY_S` per task (and 50 times per instant
    before that) and used to write the line each time: 45 GB of one message. State lives on `owner`."""
    seen: Dict[Tuple[str, ...], list] = owner.__dict__.setdefault("_starved_log", {})
    entry = seen.get(key)
    if entry is not None and now - entry[0] < STARVED_LOG_INTERVAL_S:
        entry[1] += 1
        return False
    suppressed = entry[1] if entry is not None else 0
    seen[key] = [now, 0]
    logging.log(level, message + (f" [{suppressed} identical lines suppressed]" if suppressed else ""))
    return True


def servable_somewhere(nodes: Iterable[Any], task_type: Any, source_node_name: str, server_only: bool,
                       type_allowed: Any) -> bool:
    """True when some node the source reaches hosts a platform type `task_type` can run on, whatever is free now.
    The same reachability create_first_replica applies (the source's own node, or a non-client node whose
    network_map lists the source), ignoring occupancy."""
    for node in nodes:
        name = str(node.node_name)
        is_client = name.startswith("client_node")
        if server_only and is_client:
            continue
        if name != source_node_name and (is_client or source_node_name not in (getattr(node, "network_map", None) or {})):
            continue
        for platform in node.platforms.items:
            short = platform.type["shortName"]
            if short in task_type["platforms"] and type_allowed(short):
                return True
    return False


class StarvedDeferMixin:
    """Needs `env`, `tasks` and `autoscaler`; call `_init_starved_defer()` from `__init__`."""

    def _init_starved_defer(self) -> None:
        self._defer_spin: Dict[int, Tuple[float, int]] = {}
        self._starved_tasks: set = set()
        self.deferred_spin_waits = 0

    def _starved_spin(self, task: "Task") -> bool:
        """True once `task` has been deferred more than DEFER_SPIN_LIMIT times at one simulated instant, and on every
        later retry of that task: it has proved starved, so a retry goes straight to the eviction attempt instead of
        spinning another DEFER_SPIN_LIMIT times (about 50 error lines per task per simulated second)."""
        tid = int(task.id)
        if tid in self._starved_tasks:
            return True
        prev, n = self._defer_spin.get(tid, (None, 0))
        n = n + 1 if prev == self.env.now else 1
        self._defer_spin[tid] = (self.env.now, n)
        if n > DEFER_SPIN_LIMIT:
            self._starved_tasks.add(tid)
            count = getattr(self.autoscaler, "count_starved", None)
            if count is not None:
                count("starved_tasks")
            return True
        return False

    def _check_servable(self, task: "Task") -> None:
        from src.placement.autoscaler import replica_platform_type_allowed
        nodes = getattr(self, "nodes", None)
        if nodes is None:
            return
        server_only = os.environ.get("HEROSIM_SERVER_ONLY_REPLICAS", "0") == "1"
        if not servable_somewhere(list(nodes.items), task.type, task.node_name, server_only, replica_platform_type_allowed):
            raise StarvedForeverError(
                f"[ {self.env.now} ] {task} from {task.node_name}: no node it reaches has a platform type "
                f"{sorted(task.type['platforms'])} of {task.type['name']} (server-only replicas: {server_only}); "
                f"no capacity will ever free, so the run would spin forever. The topology cannot serve this "
                f"workload: remove the task's client from the workload or repair the topology."
            )

    def _requeue_after(self, task: "Task", delay: float) -> Generator:
        yield self.env.timeout(delay)
        task.postponed_count += 1
        yield self.tasks.put(task)

    def _defer(self, task: "Task", system_state: "SystemState") -> Generator:
        retry = int(task.id) in self._starved_tasks
        if self._starved_spin(task):
            self._check_servable(task)
            if retry:
                # a starved task's retry skipped the spin that used to attempt creation: capacity may have freed
                made = yield self.env.process(
                    self.autoscaler.create_first_replica(system_state, task.type, source_node_name=task.node_name)
                )
                if not isinstance(made, StopIteration):
                    task.postponed_count += 1
                    yield self.tasks.put(task)
                    return
            evict = getattr(self.autoscaler, "evict_idle_for", None)
            if evict is not None and evict(system_state, task.type, task.node_name):
                self._defer_spin.pop(int(task.id), None)
                task.postponed_count += 1
                yield self.tasks.put(task)
                yield self.env.process(
                    self.autoscaler.create_first_replica(system_state, task.type, source_node_name=task.node_name)
                )
                return
            self.deferred_spin_waits += 1
            log_starved(
                self, self.env.now, ("retry", task.type["name"], task.node_name),
                f"[ {self.env.now} ] starved: no replica can be created for {task}; retrying in {DEFER_RETRY_S}s",
                logging.WARNING,
            )
            self.env.process(self._requeue_after(task, DEFER_RETRY_S))
            return
        task.postponed_count += 1
        yield self.tasks.put(task)
        yield self.env.process(
            self.autoscaler.create_first_replica(system_state, task.type, source_node_name=task.node_name)
        )
