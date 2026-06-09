"""XGBoost single-task (Regime B) orchestrator — forwards model to per-arrival scheduler."""

from __future__ import annotations

import logging

from src.policy.knative_network.orchestrator import KnativeOrchestrator as KnativeNetworkOrchestrator
from src.policy.knative.model import KnativeSystemState


class XGBoostSingleOrchestrator(KnativeNetworkOrchestrator):
    """Knative-network orchestrator with XGB model wiring for Regime B."""

    def __init__(self, *args, models=None, **kwargs):
        # Knative autoscaler does not accept models; keep them for scheduler only.
        stored_models = models
        super().__init__(*args, models=None, **kwargs)
        self.models = stored_models

    def initialize_state(self) -> KnativeSystemState:
        system_state = super().initialize_state()
        if self.models:
            if not hasattr(self.scheduler, "set_models"):
                raise RuntimeError("XGBoostSingleScheduler missing set_models — cannot load ranker")
            self.scheduler.set_models(self.models)
            logging.info("[XGB Single Orchestrator] Models passed to scheduler")
        else:
            raise RuntimeError("XGBoostSingleOrchestrator requires models dict with xgb_model_path")
        return system_state
