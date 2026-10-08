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
from typing import TYPE_CHECKING, Dict, Generator, Tuple

if TYPE_CHECKING:
    from src.placement.infrastructure import Task
    from src.placement.model import SystemState

DEFER_SPIN_LIMIT = 50
DEFER_RETRY_S = float(os.environ.get("HEROSIM_DEFER_RETRY_S", "1.0"))


class StarvedDeferMixin:
    """Needs `env`, `tasks` and `autoscaler`; call `_init_starved_defer()` from `__init__`."""

    def _init_starved_defer(self) -> None:
        self._defer_spin: Dict[int, Tuple[float, int]] = {}
        self.deferred_spin_waits = 0

    def _starved_spin(self, task: "Task") -> bool:
        """True once `task` has been deferred more than DEFER_SPIN_LIMIT times at one simulated instant."""
        prev, n = self._defer_spin.get(int(task.id), (None, 0))
        n = n + 1 if prev == self.env.now else 1
        self._defer_spin[int(task.id)] = (self.env.now, n)
        return n > DEFER_SPIN_LIMIT

    def _requeue_after(self, task: "Task", delay: float) -> Generator:
        yield self.env.timeout(delay)
        task.postponed_count += 1
        yield self.tasks.put(task)

    def _defer(self, task: "Task", system_state: "SystemState") -> Generator:
        if self._starved_spin(task):
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
            logging.warning(
                f"[ {self.env.now} ] starved: no replica can be created for {task}; retrying in {DEFER_RETRY_S}s"
            )
            self.env.process(self._requeue_after(task, DEFER_RETRY_S))
            return
        task.postponed_count += 1
        yield self.tasks.put(task)
        yield self.env.process(
            self.autoscaler.create_first_replica(system_state, task.type, source_node_name=task.node_name)
        )
