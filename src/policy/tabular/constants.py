"""Shared constants for tabular placement models (Option B edge features)."""

from __future__ import annotations

TASK_FEATURE_DIM = 3
PLATFORM_FEATURE_DIM = 14
EDGE_FEATURE_DIM = 5
FEATURE_DIM = TASK_FEATURE_DIM + PLATFORM_FEATURE_DIM + EDGE_FEATURE_DIM  # 22

FEATURE_COLUMN_NAMES = [f"x_{i}" for i in range(FEATURE_DIM)]

# Platform feature layout (must match prepare_graphs_cache*.py / GNNScheduler).
PLATFORM_QUEUE_NORM_DIM = 7

# Regime A batch scheduler mirrors GNNScheduler batch bounds.
MIN_BATCH_SIZE_FOR_ML = 2
MAX_BATCH_SIZE_FOR_ML = 4
