#!/usr/bin/env python3
"""Properties of the P5b candidate-relative queue columns (program_verdict_v1).

These three columns are the whole intervention: they give the pointwise MLP the
set-relative view of a task's candidate queues that message passing gives the GNN for
free. Two things must hold or the control measures the wrong thing:

  * they are a function of the candidate SET, not of the order candidates happen to be
    enumerated in (the enumeration order is an artifact of platform iteration), and
  * they are computed by exactly one function, imported by both the training extractor
    and the live scheduler.

The second is guarded by `verify_cache_live_feature_parity.py --layout dim25cr`; this
file pins the first, plus the degenerate cases that would otherwise emit NaN into a
feature matrix (single candidate, all-equal queues).

Run:
    pipenv run python3 -m pytest scripts_cosim/test_candidate_relative_features.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.policy.tabular.reduced_features import (  # noqa: E402
    CANDIDATE_RELATIVE_FEATURE_DIM,
    DIM22_FEATURE_DIM,
    DIM25CR_FEATURE_DIM,
    candidate_relative_queue_columns,
)

DELTA, RANK, ZSCORE = 0, 1, 2


def test_width_and_dtype():
    out = candidate_relative_queue_columns([0.1, 0.4, 0.2])
    assert out.shape == (3, CANDIDATE_RELATIVE_FEATURE_DIM)
    assert out.dtype == np.float32
    assert DIM25CR_FEATURE_DIM == DIM22_FEATURE_DIM + CANDIDATE_RELATIVE_FEATURE_DIM


def test_shift_invariance():
    """Adding a constant to every candidate changes no column.

    Queue normalization is adaptive, so the absolute level of the whole set already moves
    run to run. Only the within-set structure is meant to be new information.
    """
    q = np.array([0.10, 0.55, 0.20, 0.55, 0.90])
    base = candidate_relative_queue_columns(q)
    for shift in (0.25, -0.05, 100.0):
        np.testing.assert_allclose(
            candidate_relative_queue_columns(q + shift), base, rtol=0, atol=1e-6
        )


def test_rank_and_zscore_are_scale_invariant_but_delta_is_not():
    q = np.array([0.1, 0.2, 0.4])
    base = candidate_relative_queue_columns(q)
    scaled = candidate_relative_queue_columns(q * 3.0)
    np.testing.assert_allclose(scaled[:, RANK], base[:, RANK], atol=1e-6)
    np.testing.assert_allclose(scaled[:, ZSCORE], base[:, ZSCORE], atol=1e-6)
    # The delta column intentionally keeps its units: "3 tasks deeper than the best
    # candidate" is the quantity the collapse detector (chosen_queue_vs_min) is built on.
    np.testing.assert_allclose(scaled[:, DELTA], base[:, DELTA] * 3.0, atol=1e-6)


def test_permutation_equivariance():
    """Reordering candidates permutes the rows and changes nothing else."""
    q = np.array([0.3, 0.1, 0.7, 0.1])
    base = candidate_relative_queue_columns(q)
    perm = np.array([2, 0, 3, 1])
    permuted = candidate_relative_queue_columns(q[perm])
    np.testing.assert_allclose(permuted, base[perm], atol=1e-6)


def test_ties_share_a_value():
    """Equal queues must get equal features regardless of enumeration order."""
    out = candidate_relative_queue_columns([0.5, 0.1, 0.5, 0.1])
    np.testing.assert_allclose(out[0], out[2], atol=1e-6)
    np.testing.assert_allclose(out[1], out[3], atol=1e-6)
    # Average ranks: two at 0.5 -> ranks {0,1} -> 0.5; two at 0.5 -> ranks {2,3} -> 2.5.
    np.testing.assert_allclose(out[1][RANK], 0.5 / 3.0, atol=1e-6)
    np.testing.assert_allclose(out[0][RANK], 2.5 / 3.0, atol=1e-6)


def test_minimum_is_zero_delta_and_rank_zero():
    out = candidate_relative_queue_columns([0.8, 0.2, 0.5])
    assert out[1][DELTA] == pytest.approx(0.0)
    assert out[1][RANK] == pytest.approx(0.0)
    assert out[0][RANK] == pytest.approx(1.0)  # the deepest candidate


def test_single_candidate_is_all_zeros():
    """One feasible platform is no choice at all — the columns must not invent one."""
    out = candidate_relative_queue_columns([0.42])
    np.testing.assert_array_equal(out, np.zeros((1, CANDIDATE_RELATIVE_FEATURE_DIM), np.float32))


def test_all_equal_queues_are_finite():
    """std == 0 would make the z-score NaN and poison the whole feature matrix."""
    out = candidate_relative_queue_columns([0.3, 0.3, 0.3])
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out[:, DELTA], 0.0, atol=1e-6)
    np.testing.assert_allclose(out[:, ZSCORE], 0.0, atol=1e-6)
    # All tied -> every candidate carries the same average rank.
    assert len(set(out[:, RANK].tolist())) == 1


def test_empty_group():
    out = candidate_relative_queue_columns([])
    assert out.shape == (0, CANDIDATE_RELATIVE_FEATURE_DIM)


def test_non_finite_input_fails_loud():
    with pytest.raises(ValueError):
        candidate_relative_queue_columns([0.1, np.nan, 0.3])


def test_zscore_is_standardized():
    q = np.array([0.1, 0.2, 0.3, 0.4])
    z = candidate_relative_queue_columns(q)[:, ZSCORE]
    assert z.mean() == pytest.approx(0.0, abs=1e-6)
    assert z.std() == pytest.approx(1.0, abs=1e-6)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
