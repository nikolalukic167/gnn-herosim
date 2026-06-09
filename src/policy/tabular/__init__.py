"""Tabular placement policies (Regime A batch + Regime B single).

Import schedulers from their modules directly to avoid circular imports with gnn.scheduler.
"""

__all__ = [
    "XGBoostBatchScheduler",
    "XGBoostSingleScheduler",
]
