#!/usr/bin/env python3
"""The MLP serving matrix must lay out columns exactly as the training extractor did.

`verify_cache_live_feature_parity.py --layout dim25cr` proves the cache and live *graphs*
agree. This file covers the step after that: `MLPBatchScheduler.build_feature_matrix`
assembles [task | platform | edge | candidate-relative] rows from the bundle, and if its
candidate grouping or column order drifted from
`reduced_features._extract_dim22_rows_for_task`, every score would be wrong with no shape
error to announce it — the same silent-corruption shape as the atomic21/dim22 layout
confound (tests/test_inference_layout_contract.py).

Run:
    pipenv run python3 -m pytest scripts_cosim/test_mlp_serving_layout.py -q
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.policy.tabular.feature_builder import InferenceFeatureBundle  # noqa: E402
from src.policy.tabular.mlp_scheduler import MLPBatchScheduler  # noqa: E402
from src.policy.tabular.reduced_features import (  # noqa: E402
    DIM22_FEATURE_DIM,
    DIM25CR_FEATURE_DIM,
    FULL_PLATFORM_QUEUE_DIM,
    candidate_relative_queue_columns,
)

N_TASKS = 3
N_PLATFORMS = 5
# Deliberately ragged: tasks see different candidate counts (sparse topologies produce
# single-candidate tasks, which is the degenerate case the feature must not invent a
# choice for).
CANDIDATES = {0: [0, 3, 1], 1: [2, 4], 2: [0]}


def _bundle() -> InferenceFeatureBundle:
    rng = np.random.default_rng(0)
    platform_features = rng.random((N_PLATFORMS, 14)).astype(np.float32)
    # Distinct, non-monotone queue values so ranks and z-scores are informative.
    platform_features[:, FULL_PLATFORM_QUEUE_DIM] = np.array(
        [0.10, 0.90, 0.30, 0.10, 0.55], dtype=np.float32
    )

    edge_rows, placements, queue_keys, meta = [], {}, {}, {}
    for t_idx in range(N_TASKS):
        placements[t_idx], queue_keys[t_idx] = [], []
        for pos in CANDIDATES[t_idx]:
            qk = f"node{pos}:0"
            edge_rows.append(rng.random(5))
            placements[t_idx].append((pos, 0))
            queue_keys[t_idx].append(qk)
            meta[qk] = {"platform_pos": pos, "node_id": pos, "platform_id": 0,
                        "node_name": f"node{pos}"}

    return InferenceFeatureBundle(
        n_tasks=N_TASKS,
        n_platforms=N_PLATFORMS,
        task_features=rng.random((N_TASKS, 3)).astype(np.float32),
        platform_features=platform_features,
        edge_attr_directed=np.asarray(edge_rows, dtype=np.float32),
        task_logit_to_placement=placements,
        task_logit_to_queue_key=queue_keys,
        queue_key_to_platform_meta=meta,
    )


@pytest.fixture(autouse=True)
def _clean_layout_env():
    saved = os.environ.pop("INFERENCE_FEATURE_LAYOUT", None)
    yield
    os.environ.pop("INFERENCE_FEATURE_LAYOUT", None)
    if saved is not None:
        os.environ["INFERENCE_FEATURE_LAYOUT"] = saved


def test_dim22_width_unchanged():
    """The GNN-era layout must not grow columns because dim25cr exists."""
    os.environ["INFERENCE_FEATURE_LAYOUT"] = "dim22"
    matrix, boundaries = MLPBatchScheduler.build_feature_matrix(_bundle())
    assert matrix.shape == (6, DIM22_FEATURE_DIM)
    assert boundaries == [(0, 3), (3, 5), (5, 6)]


def test_dim25cr_appends_three_columns_after_dim22():
    """dim25cr is dim22 with 3 columns appended — the first 22 must be untouched."""
    bundle = _bundle()
    os.environ["INFERENCE_FEATURE_LAYOUT"] = "dim22"
    base, _ = MLPBatchScheduler.build_feature_matrix(bundle)
    os.environ["INFERENCE_FEATURE_LAYOUT"] = "dim25cr"
    cr, _ = MLPBatchScheduler.build_feature_matrix(bundle)

    assert cr.shape == (6, DIM25CR_FEATURE_DIM)
    np.testing.assert_array_equal(cr[:, :DIM22_FEATURE_DIM], base)


def test_candidate_relative_columns_are_grouped_per_task():
    """Groups are the task's own candidate set, not the whole batch."""
    bundle = _bundle()
    os.environ["INFERENCE_FEATURE_LAYOUT"] = "dim25cr"
    matrix, boundaries = MLPBatchScheduler.build_feature_matrix(bundle)

    q_col = bundle.platform_features[:, FULL_PLATFORM_QUEUE_DIM]
    for t_idx, (start, end) in enumerate(boundaries):
        expected = candidate_relative_queue_columns(q_col[CANDIDATES[t_idx]])
        np.testing.assert_allclose(
            matrix[start:end, DIM22_FEATURE_DIM:], expected, atol=1e-6
        )

    # Task 2 has a single candidate: no choice, so no set-relative signal.
    np.testing.assert_array_equal(matrix[5, DIM22_FEATURE_DIM:], np.zeros(3, np.float32))
    # Task 0's shortest-queue candidate (pos 0, tied at 0.10 with pos 3) sits at delta 0.
    assert matrix[0, DIM22_FEATURE_DIM] == pytest.approx(0.0)


def test_batchwide_grouping_would_be_detected():
    """Guard the guard: pooling all tasks into one group must NOT match.

    If it did, this file would pass while the serving path computed a batch-relative
    feature the trainer never produced.
    """
    bundle = _bundle()
    os.environ["INFERENCE_FEATURE_LAYOUT"] = "dim25cr"
    matrix, _ = MLPBatchScheduler.build_feature_matrix(bundle)

    q_col = bundle.platform_features[:, FULL_PLATFORM_QUEUE_DIM]
    all_positions = [p for t in range(N_TASKS) for p in CANDIDATES[t]]
    batchwide = candidate_relative_queue_columns(q_col[all_positions])
    assert not np.allclose(matrix[:, DIM22_FEATURE_DIM:], batchwide, atol=1e-6)


def test_finite_everywhere():
    os.environ["INFERENCE_FEATURE_LAYOUT"] = "dim25cr"
    matrix, _ = MLPBatchScheduler.build_feature_matrix(_bundle())
    assert np.isfinite(matrix).all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
