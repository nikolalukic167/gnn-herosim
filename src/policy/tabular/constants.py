"""Shared constants for tabular placement models (Option B edge features)."""

from __future__ import annotations

TASK_FEATURE_DIM = 2
PLATFORM_FEATURE_DIM = 14
EDGE_FEATURE_DIM = 5
FEATURE_DIM = TASK_FEATURE_DIM + PLATFORM_FEATURE_DIM + EDGE_FEATURE_DIM  # 21

FEATURE_COLUMN_NAMES = [f"x_{i}" for i in range(FEATURE_DIM)]

# Graph cache version (must match prepare_graphs_cache_seq.py metadata).
CACHE_VERSION = "7.1-atomic21"

# Platform feature layout (must match prepare_graphs_cache*.py / GNNScheduler).
# Dim 7: raw queue length; dim 8: per-platform is_cold; dim 13: reserved (0.0).
PLATFORM_QUEUE_RAW_DIM = 7
PLATFORM_IS_COLD_DIM = 8
PLATFORM_RESERVED_DIM = 13

# Regime A batch scheduler mirrors GNNScheduler batch bounds.
MIN_BATCH_SIZE_FOR_ML = 2
MAX_BATCH_SIZE_FOR_ML = 4
