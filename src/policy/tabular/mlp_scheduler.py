"""
Regime A (batch) MLP scheduler — mirrors XGBoostBatchScheduler control loop.

Replaces XGBoost scoring with a single batched [N_edges, 22] → [N_edges] forward
pass through PointwiseEdgeMLP.  All other logic (graph build, decode, roll-forward)
is inherited from GNNScheduler via XGBoostBatchScheduler.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from src.placement.infrastructure import Node, Platform, Task

from src.placement.queue_features import (
    QUEUE_FEATURE_CONTRACT_ENV,
    require_matching_queue_feature_contract,
    resolve_queue_feature_contract,
    validate_queue_feature_contract,
)
from src.policy.tabular.constants import FEATURE_DIM
from src.policy.tabular.feature_builder import (
    build_inference_feature_bundle,
    InferenceFeatureBundle,
    CE_REDUCED_EDGE_INDICES,
    CE_REDUCED_PLATFORM_INDICES,
    CE_REDUCED_TASK_FEATURE_DIM,
    _inference_feature_layout,
    _uses_candidate_relative_layout,
)
from src.policy.tabular.mlp_model import PointwiseEdgeMLP
from src.policy.tabular.reduced_features import (
    FULL_PLATFORM_QUEUE_DIM,
    candidate_relative_queue_columns,
)
from src.policy.tabular.scheduler import XGBoostBatchScheduler


class MLPBatchScheduler(XGBoostBatchScheduler):
    """Batch MLP scheduler: GNNScheduler loop with PointwiseEdgeMLP edge scoring."""

    _live_audit_policy_name = "mlp_batch"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mlp_model: Optional[PointwiseEdgeMLP] = None
        self.mlp_decisions = 0
        self.mlp_fallback_decisions = 0

    def set_models(self, models: dict):
        # Call GNNScheduler's set_models (which wires gnn_model/device/task_types_data)
        # but skip XGB-specific loading.  We call super() which chains to GNNScheduler.
        # Then load MLP on top.
        super().set_models(models)
        if models is None:
            return
        if "mlp_model" in models:
            self.mlp_model = models["mlp_model"]
            logging.info("[MLP Batch] Loaded MLP model (pre-built)")
        elif "mlp_model_path" in models:
            path = models["mlp_model_path"]
            checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
            hidden_dim = checkpoint.get("hidden_dim", 64)
            input_dim = checkpoint.get("input_dim", FEATURE_DIM)
            self.mlp_model = PointwiseEdgeMLP(input_dim=input_dim, hidden_dim=hidden_dim)
            self.mlp_model.load_state_dict(checkpoint["model_state_dict"])
            self.mlp_model.eval()
            self.mlp_model.to(self.device)
            layout = checkpoint.get("inference_feature_layout")
            if layout:
                # Mirror the GNN loader: a declared-but-different layout is a hard error,
                # never a silent override — the columns change meaning, not shape.
                trained_layout = str(layout).strip().lower()
                declared_layout = os.environ.get("INFERENCE_FEATURE_LAYOUT", "").strip().lower()
                if declared_layout and declared_layout != trained_layout:
                    raise ValueError(
                        f"[MLP Batch] checkpoint {path} was trained with "
                        f"inference_feature_layout={trained_layout!r} but this run declares "
                        f"INFERENCE_FEATURE_LAYOUT={declared_layout!r}. Serving the wrong "
                        f"layout corrupts every score without changing any tensor shape."
                    )
                os.environ["INFERENCE_FEATURE_LAYOUT"] = trained_layout
            else:
                if int(input_dim) == 25:
                    inferred_layout = "dim25cr"
                elif int(input_dim) == 24:
                    inferred_layout = "dim24"
                elif int(input_dim) == 22:
                    inferred_layout = "dim22"
                elif int(input_dim) == FEATURE_DIM:
                    inferred_layout = "atomic21"
                elif checkpoint.get("reduced_features") or int(input_dim) == 11:
                    inferred_layout = "ce_reduced"
                else:
                    raise RuntimeError(
                        f"[MLP Batch] FAIL LOUD: cannot infer inference_feature_layout "
                        f"from input_dim={input_dim} (checkpoint missing inference_feature_layout)"
                    )
                declared_layout = os.environ.get("INFERENCE_FEATURE_LAYOUT", "").strip().lower()
                if declared_layout and declared_layout != inferred_layout:
                    raise ValueError(
                        f"[MLP Batch] checkpoint {path} has no inference_feature_layout; its "
                        f"input_dim={input_dim} implies {inferred_layout!r}, but this run "
                        f"declares INFERENCE_FEATURE_LAYOUT={declared_layout!r}. Refusing to "
                        f"silently override the declaration."
                    )
                os.environ["INFERENCE_FEATURE_LAYOUT"] = inferred_layout
            # Checkpoints without the field predate the contract split (legacy_v0). A
            # declared-but-different contract is a hard error: dim7/dim13 would silently
            # change meaning under the model.
            trained_contract = checkpoint.get("queue_feature_contract")
            declared = os.environ.get(QUEUE_FEATURE_CONTRACT_ENV, "").strip()
            if trained_contract and declared:
                require_matching_queue_feature_contract(
                    trained_contract, declared, model_label=f"MLP checkpoint {path}"
                )
            elif trained_contract:
                os.environ[QUEUE_FEATURE_CONTRACT_ENV] = validate_queue_feature_contract(
                    trained_contract
                )
            logging.info(
                "[MLP Batch] Loaded MLP model from %s (input_dim=%s, queue_feature_contract=%s)",
                path,
                input_dim,
                resolve_queue_feature_contract(),
            )
        else:
            raise RuntimeError(
                "[MLP Batch] FAIL LOUD: models dict missing mlp_model and mlp_model_path — "
                "refusing silent shortest-queue fallback. "
                f"keys={sorted(models.keys()) if isinstance(models, dict) else models!r}"
            )
        if self.mlp_model is None:
            raise RuntimeError(
                "[MLP Batch] FAIL LOUD: mlp_model still None after set_models"
            )

    # ------------------------------------------------------------------
    # Override _gnn_inference: build feature bundle, batched MLP forward
    # ------------------------------------------------------------------

    def _gnn_inference(
        self,
        batch_tasks: List["Task"],
        system_state,
        queue_snapshot: Dict[str, int],
        temporal_state=None,
    ) -> Optional[Dict[int, Tuple[int, int]]]:
        if self.mlp_model is None:
            logging.error("[MLP Batch] Model not loaded")
            return None

        try:
            norm_mode = os.environ.get("GNN_QUEUE_NORM_MODE", "adaptive").strip().lower()
            bundle = build_inference_feature_bundle(
                batch_tasks,
                system_state,
                queue_snapshot,
                nodes=list(self.nodes.items),
                task_types_data=self.task_types_data,
                queue_norm_mode=norm_mode,
            )
            if bundle is None:
                logging.warning("[MLP Batch] Empty feature bundle (no feasible edges)")
                return None

            logits_per_task = self._mlp_logits_from_bundle(bundle)
            task_logit_to_queue_key = bundle.task_logit_to_queue_key

            placements = self._decode_placements(
                logits_per_task,
                bundle.task_logit_to_placement,
                bundle.n_tasks,
                queue_snapshot,
                task_logit_to_queue_key,
            )
            return placements
        except Exception as exc:
            logging.exception("[MLP Batch] Inference error: %s", exc)
            return None

    @staticmethod
    def build_feature_matrix(
        bundle: InferenceFeatureBundle,
    ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """Assemble the [N_total_edges, D] serving matrix and its per-task row spans.

        Split out of `_mlp_logits_from_bundle` so the serving-side feature layout can be
        asserted directly (scripts_cosim/test_mlp_serving_layout.py) instead of only
        through a model forward. Static and side-effect free for the same reason.
        """
        n_tasks = bundle.n_tasks
        total_edges = int(bundle.edge_attr_directed.shape[0])

        if total_edges == 0:
            raise ValueError("[MLP Batch] Zero candidate edges in bundle")

        # Build index arrays: for each directed edge row, which task and plat_pos?
        task_idx_arr = np.empty(total_edges, dtype=np.int32)
        plat_pos_arr = np.empty(total_edges, dtype=np.int32)
        task_boundaries: List[Tuple[int, int]] = []

        edge_row = 0
        for t_idx in range(n_tasks):
            candidates = bundle.task_logit_to_placement.get(t_idx, [])
            queue_keys = bundle.task_logit_to_queue_key.get(t_idx, [])
            n_cands = len(candidates)
            start = edge_row
            for l_idx in range(n_cands):
                qk = queue_keys[l_idx]
                meta = bundle.queue_key_to_platform_meta.get(qk)
                if meta is None or "platform_pos" not in meta:
                    raise ValueError(
                        f"[MLP Batch] platform_pos missing for queue_key={qk!r}"
                    )
                task_idx_arr[edge_row] = t_idx
                plat_pos_arr[edge_row] = int(meta["platform_pos"])
                edge_row += 1
            task_boundaries.append((start, edge_row))

        if edge_row != total_edges:
            raise ValueError(
                f"[MLP Batch] Edge count mismatch: iterated {edge_row}, expected {total_edges}"
            )

        # Vectorised feature assembly (full or ce-reduced layout)
        task_feats = bundle.task_features[task_idx_arr]
        plat_feats = bundle.platform_features[plat_pos_arr]
        edge_feats = bundle.edge_attr_directed
        layout = _inference_feature_layout()
        if layout in ("ce_reduced", "reduced_ce", "reduced1060"):
            task_feats = task_feats[:, :CE_REDUCED_TASK_FEATURE_DIM]
            plat_feats = plat_feats[:, CE_REDUCED_PLATFORM_INDICES]
            edge_feats = edge_feats[:, CE_REDUCED_EDGE_INDICES]

        parts = [task_feats, plat_feats, edge_feats]
        if _uses_candidate_relative_layout(layout):
            # Set-relative columns, computed per task's candidate group over the SAME
            # normalized queue column the training extractor reads. task_boundaries
            # already delimits the groups. Shared formula — see
            # reduced_features.candidate_relative_queue_columns.
            cand_rel = np.zeros((total_edges, 3), dtype=np.float32)
            for start, end in task_boundaries:
                if end > start:
                    cand_rel[start:end] = candidate_relative_queue_columns(
                        plat_feats[start:end, FULL_PLATFORM_QUEUE_DIM]
                    )
            parts.append(cand_rel)

        feat_matrix = np.concatenate(parts, axis=1).astype(np.float32)
        if not np.isfinite(feat_matrix).all():
            raise ValueError("[MLP Batch] Non-finite values in feature matrix")
        return feat_matrix, task_boundaries

    def _mlp_logits_from_bundle(
        self,
        bundle: InferenceFeatureBundle,
    ) -> List[torch.Tensor]:
        """Vectorised [N_total_edges, D] → [N_total_edges] forward pass.

        One torch.from_numpy + one model forward on GPU/CPU; returns a List[Tensor] of
        per-task score vectors.
        """
        feat_matrix, task_boundaries = self.build_feature_matrix(bundle)

        expected_dim = int(self.mlp_model.input_dim)
        if feat_matrix.shape[1] != expected_dim:
            raise ValueError(
                f"[MLP Batch] Feature dim mismatch: {feat_matrix.shape[1]} != {expected_dim}"
            )

        x = torch.from_numpy(feat_matrix).to(self.device)
        with torch.no_grad():
            scores = self.mlp_model(x)  # [N_total]

        if not torch.isfinite(scores).all():
            raise ValueError("[MLP Batch] NaN/Inf in MLP scores")

        scores_cpu = scores.cpu()
        logits_per_task: List[torch.Tensor] = []
        for start, end in task_boundaries:
            if end > start:
                logits_per_task.append(scores_cpu[start:end])
            else:
                logits_per_task.append(torch.empty(0))

        return logits_per_task

    def _select_placement_pure_gnn(
        self,
        task: "Task",
        task_idx: int,
        placements,
        available_replicas,
    ):
        node, plat = super()._select_placement_pure_gnn(
            task, task_idx, placements, available_replicas
        )
        if node is not None and plat is not None and placements and task_idx in placements:
            self.mlp_decisions += 1
        elif node is not None and plat is not None:
            self.mlp_fallback_decisions += 1
        return node, plat
