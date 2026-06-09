"""
Regime A (batch) XGBoost scheduler — mirrors GNNScheduler control loop.

Uses the same batch collection, graph feature construction, and sequential decode
with queue roll-forward as GNNScheduler; replaces GIN inference with XGBoost edge scores.
"""

from __future__ import annotations

import logging
from timeit import default_timer
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch
import xgboost as xgb

if TYPE_CHECKING:
    from src.placement.infrastructure import Node, Platform, Task

from src.policy.gnn.scheduler import GNNScheduler
from src.policy.tabular.constants import FEATURE_COLUMN_NAMES, FEATURE_DIM
from src.policy.tabular.graph_extraction import resolve_platform_pos


class XGBoostBatchScheduler(GNNScheduler):
    """Batch tabular scheduler: GNNScheduler loop with XGBoost edge ranking."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.xgb_model: Optional[xgb.Booster] = None
        self.xgb_decisions = 0
        self.xgb_fallback_decisions = 0

    def set_models(self, models: dict):
        super().set_models(models)
        if models is None:
            return
        if "xgb_model" in models:
            self.xgb_model = models["xgb_model"]
            logging.info("[XGB Batch] Loaded XGBoost booster")
        elif "xgb_model_path" in models:
            path = models["xgb_model_path"]
            booster = xgb.Booster()
            booster.load_model(str(path))
            self.xgb_model = booster
            logging.info("[XGB Batch] Loaded XGBoost booster from %s", path)
        else:
            logging.warning("[XGB Batch] No xgb_model or xgb_model_path in models dict")

    def _gnn_inference(
        self,
        batch_tasks: List["Task"],
        system_state,
        queue_snapshot: Dict[str, int],
    ) -> Optional[Dict[int, Tuple[int, int]]]:
        if self.xgb_model is None:
            logging.error("[XGB Batch] Model not loaded")
            return None

        try:
            graph, task_logit_to_placement = self._build_inference_graph(
                batch_tasks, system_state, queue_snapshot
            )
            if graph is None or task_logit_to_placement is None:
                return None

            logits_per_task = self._xgb_logits_from_graph(graph, task_logit_to_placement)
            task_logit_to_queue_key = getattr(graph, "_task_logit_to_queue_key", None)

            placements = self._decode_placements(
                logits_per_task,
                task_logit_to_placement,
                len(batch_tasks),
                queue_snapshot,
                task_logit_to_queue_key,
            )
            return placements
        except Exception as exc:
            logging.exception("[XGB Batch] Inference error: %s", exc)
            return None

    def _xgb_logits_from_graph(
        self,
        graph,
        task_logit_to_placement: Dict[int, List[Tuple[int, int]]],
    ) -> List[torch.Tensor]:
        n_tasks = int(graph.n_tasks)
        task_features = graph.task_features.detach().cpu().numpy()
        platform_features = graph.platform_features.detach().cpu().numpy()
        edge_attr_all = graph.edge_attr.detach().cpu().numpy()
        n_directed = edge_attr_all.shape[0] // 2
        edge_attr = edge_attr_all[:n_directed]

        task_queue_map = getattr(graph, "_task_logit_to_queue_key", None) or getattr(
            graph, "task_logit_to_queue_key", {}
        )

        logits_per_task: List[torch.Tensor] = []
        edge_offset = 0

        for t_idx in range(n_tasks):
            candidates = task_logit_to_placement.get(t_idx, [])
            queue_keys = task_queue_map.get(t_idx, [])
            if not candidates:
                logits_per_task.append(torch.empty(0))
                continue

            rows = []
            for logit_idx, (node_id, plat_id) in enumerate(candidates):
                queue_key = str(queue_keys[logit_idx]) if logit_idx < len(queue_keys) else ""
                plat_pos = resolve_platform_pos(
                    graph, int(node_id), int(plat_id), queue_key
                )
                global_edge_idx = edge_offset + logit_idx
                feat = np.concatenate(
                    [
                        task_features[t_idx],
                        platform_features[plat_pos],
                        edge_attr[global_edge_idx],
                    ]
                )
                if feat.shape[0] != FEATURE_DIM:
                    raise ValueError(f"Feature dim mismatch: {feat.shape[0]} != {FEATURE_DIM}")
                rows.append(feat)

            edge_offset += len(candidates)
            dmat = xgb.DMatrix(np.asarray(rows, dtype=np.float32), feature_names=FEATURE_COLUMN_NAMES)
            scores = self.xgb_model.predict(dmat)
            logits_per_task.append(torch.tensor(scores, dtype=torch.float32))

        return logits_per_task

    def _select_placement_pure_gnn(
        self,
        task: "Task",
        task_idx: int,
        placements: Optional[Dict[int, Tuple[int, int]]],
        available_replicas: List[Tuple["Node", "Platform"]],
    ) -> Tuple[Optional["Node"], Optional["Platform"]]:
        node, plat = super()._select_placement_pure_gnn(
            task, task_idx, placements, available_replicas
        )
        if node is not None and plat is not None and placements and task_idx in placements:
            self.xgb_decisions += 1
        elif node is not None and plat is not None:
            self.xgb_fallback_decisions += 1
        return node, plat
