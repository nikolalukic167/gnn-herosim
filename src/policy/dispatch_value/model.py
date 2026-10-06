"""Outcome-aware dispatch uses the matched ready-set architecture."""
from src.policy.dispatch_state.model import ReadySetNet

CONTRACT = "mixed_dispatch_action_value_v1"

__all__ = ["CONTRACT", "ReadySetNet"]
