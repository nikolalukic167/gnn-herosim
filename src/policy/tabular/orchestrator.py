"""XGBoost batch orchestrator — forwards tabular model handle to the scheduler."""

from __future__ import annotations

import logging

from src.policy.gnn.orchestrator import GNNOrchestrator


class XGBoostBatchOrchestrator(GNNOrchestrator):
    """Same wiring as GNNOrchestrator; expects models['xgb_model'] or models['xgb_model_path']."""

    def __init__(self, *args, models=None, **kwargs):
        super().__init__(*args, models=models, **kwargs)
        logging.info(
            "[XGB Batch Orchestrator] initialized (models present=%s)",
            self.models is not None,
        )
