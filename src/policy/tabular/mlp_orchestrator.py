"""MLP batch orchestrator — forwards MLP model handle to the scheduler."""

from __future__ import annotations

import logging

from src.policy.tabular.orchestrator import XGBoostBatchOrchestrator


class MLPBatchOrchestrator(XGBoostBatchOrchestrator):
    """Same wiring as XGBoostBatchOrchestrator; expects models['mlp_model'] or
    models['mlp_model_path']."""

    def __init__(self, *args, models=None, **kwargs):
        super().__init__(*args, models=models, **kwargs)
        logging.info(
            "[MLP Batch Orchestrator] initialized (models present=%s)",
            self.models is not None,
        )
