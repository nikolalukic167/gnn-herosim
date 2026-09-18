#!/usr/bin/env python3
"""Numerics of the C3 probe (peer_only_v1 AMENDMENT 8). No torch, no cache, no GPU."""
from __future__ import annotations

import numpy as np

from scripts_cosim.peer_only_v1_oversmoothing import (
    mean_pairwise_cosine_distance,
    ridge_r2,
)


def test_identical_rows_have_zero_pairwise_distance():
    block = np.tile(np.array([[1.0, 2.0, 3.0]]), (5, 1))
    assert mean_pairwise_cosine_distance(block) == 0.0


def test_orthogonal_rows_are_maximally_separated_for_this_measure():
    assert abs(mean_pairwise_cosine_distance(np.eye(3)) - 1.0) < 1e-12


def test_a_uniform_rescale_does_not_count_as_collapse():
    """Cosine on purpose: the GIN may rescale the whole block, which loses no information."""
    block = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    assert abs(mean_pairwise_cosine_distance(block)
               - mean_pairwise_cosine_distance(block * 1e-4)) < 1e-9


def test_dead_rows_are_dropped_not_counted_as_a_direction():
    block = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    assert abs(mean_pairwise_cosine_distance(block) - 1.0) < 1e-12


def test_too_few_live_rows_is_none_not_zero():
    assert mean_pairwise_cosine_distance(np.array([[1.0, 2.0]])) is None
    assert mean_pairwise_cosine_distance(np.array([[1.0, 0.0], [0.0, 0.0]])) is None


def test_ridge_recovers_a_linear_target_it_can_see():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 4))
    y = 3.0 * x[:, 0] - 2.0 * x[:, 1] + 0.5
    assert ridge_r2(x[:100], y[:100], x[100:], y[100:]) > 0.99


def test_ridge_recovers_nothing_from_noise_and_never_goes_negative():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(200, 4))
    y = rng.normal(size=200)
    r2 = ridge_r2(x[:100], y[:100], x[100:], y[100:])
    assert 0.0 <= r2 < 0.3


def test_ridge_refuses_a_sample_too_small_to_score():
    x = np.ones((3, 2))
    y = np.ones(3)
    assert ridge_r2(x, y, x, y) == 0.0


def test_ridge_standardises_on_train_moments_only():
    """A test block on a different scale must not leak its own moments into the fit."""
    rng = np.random.default_rng(2)
    xtr = rng.normal(size=(200, 3))
    ytr = xtr[:, 0] * 2.0
    xte = rng.normal(size=(200, 3)) * 5.0 + 10.0
    yte = xte[:, 0] * 2.0
    assert ridge_r2(xtr, ytr, xte, yte) > 0.95
