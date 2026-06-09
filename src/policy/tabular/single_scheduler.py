"""
Regime B (per-arrival) XGBoost scheduler — mirrors knative_network control loop.

One task per wakeup; no batch collection window; live SimPy queue state.
"""

from __future__ import annotations

import logging
from timeit import default_timer
from typing import Dict, Generator, List, Optional, Set, Tuple, TYPE_CHECKING

import numpy as np
import xgboost as xgb

if TYPE_CHECKING:
    from src.placement.infrastructure import Node, Platform, Task

from src.placement.model import SystemState
from src.policy.knative_network.scheduler import KnativeScheduler as KnativeNetworkScheduler
from src.policy.tabular.constants import FEATURE_COLUMN_NAMES, FEATURE_DIM
from src.policy.tabular.feature_builder import (
    build_inference_feature_bundle,
    edge_row_features,
)


class XGBoostSingleScheduler(KnativeNetworkScheduler):
    """Per-arrival tabular scheduler: Knative loop with XGBoost edge argmax."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.xgb_model: Optional[xgb.Booster] = None
        self.task_types_data: Optional[Dict] = None
        self.xgb_decisions = 0
        self.xgb_fallback_decisions = 0
        self._last_inference_time_s = 0.0

    def set_models(self, models: dict):
        if models is None:
            raise ValueError("XGBoostSingleScheduler requires models dict")
        if "task_types_data" not in models or models["task_types_data"] is None:
            raise ValueError("XGBoostSingleScheduler requires task_types_data in models")
        self.task_types_data = models["task_types_data"]

        if "xgb_model" in models:
            self.xgb_model = models["xgb_model"]
        elif "xgb_model_path" in models:
            path = models["xgb_model_path"]
            booster = xgb.Booster()
            booster.load_model(str(path))
            self.xgb_model = booster
            logging.info("[XGB Single] Loaded booster from %s", path)
        else:
            raise ValueError("models must include xgb_model or xgb_model_path")

    def scheduler_process(self):
        logging.info(
            "[ %s ] XGBoost Single Scheduler started (policy %s)",
            self.env.now,
            self.policy,
        )

        while True:
            task: Task = yield self.tasks.get(
                lambda queued_task: all(
                    dependency.finished for dependency in queued_task.dependencies
                )
            )

            logging.info("[ %s ] Scheduler woken up (task %s)", self.env.now, task.id)

            system_state: SystemState = yield self.mutex.get()
            replicas: Set[Tuple[Node, Platform]] = system_state.replicas[task.type["name"]]
            valid_replicas = self._get_valid_replicas(replicas, task)

            if not valid_replicas:
                logging.warning(
                    "[ %s ] No network-accessible replica for %s (total=%s)",
                    self.env.now,
                    task,
                    len(replicas),
                )
                task.postponed_count += 1
                yield self.tasks.put(task)
                yield self.env.process(
                    self.autoscaler.create_first_replica(
                        system_state, task.type, source_node_name=task.node_name
                    )
                )
                yield self.mutex.put(system_state)
                continue

            start = default_timer()
            sched_node, sched_platform = self._place_with_xgb(system_state, task, valid_replicas)
            self._last_inference_time_s = default_timer() - start
            task.gnn_decision_time = self._last_inference_time_s

            task.execution_node = sched_node.node_name
            task.execution_platform = str(sched_platform.id)

            node: Node = yield self.nodes.get(lambda n: n.id == sched_node.id)
            task.node = node
            node.unused = False
            platform: Platform = yield node.platforms.get(
                lambda p: p.id == sched_platform.id
            )
            task.platform = platform
            yield self.mutex.put(system_state)

            node.wall_clock_scheduling_time += self._last_inference_time_s
            yield platform.queue.put(task)
            yield task.scheduled.succeed()
            yield node.platforms.put(platform)
            yield self.nodes.put(node)

    def _place_with_xgb(
        self,
        system_state: SystemState,
        task: "Task",
        valid_replicas: List[Tuple[Node, Platform]],
    ) -> Tuple[Node, Platform]:
        if self.xgb_model is None:
            raise RuntimeError("[XGB Single] Model not loaded")

        queue_snapshot = self._capture_full_queue_snapshot()
        bundle = build_inference_feature_bundle(
            [task],
            system_state,
            queue_snapshot,
            nodes=list(self.nodes.items),
            task_types_data=self.task_types_data,
        )
        if bundle is None:
            logging.warning("[XGB Single] No feature bundle — shortest-queue fallback")
            self.xgb_fallback_decisions += 1
            return self._shortest_queue(valid_replicas)

        candidates = bundle.task_logit_to_placement.get(0, [])
        if not candidates:
            logging.warning("[XGB Single] Empty candidates — shortest-queue fallback")
            self.xgb_fallback_decisions += 1
            return self._shortest_queue(valid_replicas)

        rows = []
        for logit_idx in range(len(candidates)):
            rows.append(edge_row_features(bundle, task_idx=0, logit_idx=logit_idx))

        dmat = xgb.DMatrix(np.asarray(rows, dtype=np.float32), feature_names=FEATURE_COLUMN_NAMES)
        scores = self.xgb_model.predict(dmat)
        best_idx = int(np.argmax(scores))
        node_id, plat_id = candidates[best_idx]

        replica_map = {(int(n.id), int(p.id)): (n, p) for n, p in valid_replicas}
        chosen = replica_map.get((int(node_id), int(plat_id)))
        if chosen is None:
            logging.warning(
                "[XGB Single] Chosen (%s, %s) not in valid replicas — SQ fallback",
                node_id,
                plat_id,
            )
            self.xgb_fallback_decisions += 1
            return self._shortest_queue(valid_replicas)

        self.xgb_decisions += 1
        return chosen

    @staticmethod
    def _shortest_queue(valid_replicas: List[Tuple[Node, Platform]]) -> Tuple[Node, Platform]:
        initialized = [r for r in valid_replicas if r[1].initialized.triggered]
        candidates = initialized if initialized else valid_replicas
        return min(candidates, key=lambda couple: len(couple[1].queue.items))

    def placement(self, system_state: SystemState, task: Task) -> Generator:
        if False:
            yield
        raise RuntimeError("XGBoostSingleScheduler uses scheduler_process directly")
