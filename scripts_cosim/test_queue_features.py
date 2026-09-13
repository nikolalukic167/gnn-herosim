#!/usr/bin/env python3
"""Contract tests for the queue-scale platform features (dim 7 divisor, dim 13 usage ratio).

Guards the two properties the 2026-08-13 live post-mortem depends on:

1. `legacy_v0` reproduces the pre-split formulas exactly, so every existing cache and
   checkpoint (873/v5.5, regime B distill, ect_pull) keeps its meaning.
2. `scale_invariant_v1` is invariant to a uniform scaling of queue depth, which is what
   went wrong live: training p90 depth was 26-70 while live was ~14.5k, and the cap of 100
   turned dim7 into a ~145 (vs ~1) input with no clipping anywhere.

Run: pipenv run python3 -m pytest scripts_cosim/test_queue_features.py -q
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.placement.queue_features import (  # noqa: E402
    DEFAULT_QUEUE_FEATURE_CONTRACT,
    LEGACY_QUEUE_NORM_CAP,
    QUEUE_FEATURE_CONTRACT_ENV,
    QUEUE_FEATURE_CONTRACT_LEGACY_V0,
    QUEUE_FEATURE_CONTRACT_SCALE_INVARIANT_V1,
    InvalidQueueFeatureContractError,
    QueueFeatureContractMismatchError,
    queue_depth_norm,
    require_matching_queue_feature_contract,
    resolve_queue_feature_contract,
    usage_ratio_feature,
    validate_queue_feature_contract,
)

LEGACY = QUEUE_FEATURE_CONTRACT_LEGACY_V0
V1 = QUEUE_FEATURE_CONTRACT_SCALE_INVARIANT_V1

# Depth distribution shaped like a contention_v2 snapshot: mostly idle, p90 well under the cap.
TRAIN_DEPTHS = [0] * 150 + list(range(1, 51)) + [60, 65, 70, 75]
# Live shape measured on sparse_p35 s42: a deep pile plus a mid band plus idle platforms.
LIVE_DEPTHS = [0] * 56 + [200] * 150 + [14500] * 34


def _legacy_norm_reference(values, mode):
    """The formulas as they existed before src/placement/queue_features.py."""
    vals = sorted(int(v) for v in values)
    if not vals:
        return 50.0
    if mode in ("adaptive", "scheduler_adaptive"):
        idx = int(len(vals) * 0.9)
        p90 = vals[min(idx, len(vals) - 1)]
        return float(min(max(1.0, p90), 100.0))
    if mode == "adaptive_nonzero":
        nz = [v for v in vals if v > 0]
        if not nz:
            return 1.0
        idx = int(len(nz) * 0.9)
        p90 = nz[min(idx, len(nz) - 1)]
        return float(min(max(1.0, p90), 100.0))
    return 50.0


@pytest.mark.parametrize("mode", ["adaptive", "scheduler_adaptive", "adaptive_nonzero"])
@pytest.mark.parametrize("depths", [TRAIN_DEPTHS, LIVE_DEPTHS, [0] * 40, []])
def test_legacy_matches_pre_split_reference(mode, depths):
    assert queue_depth_norm(depths, mode, LEGACY) == pytest.approx(
        _legacy_norm_reference(depths, mode)
    )


def test_legacy_usage_ratio_matches_pre_split_reference():
    for depth, tc in ((0, 5.0), (39, 5.0), (14500, 5.0), (14500, 58.4), (7, 0.0)):
        expected = (depth / tc / 5.0) if tc > 0 else 0.0
        assert usage_ratio_feature(depth, tc, LEGACY) == pytest.approx(expected)


def test_default_contract_is_legacy():
    assert DEFAULT_QUEUE_FEATURE_CONTRACT == LEGACY


def test_cap_is_inactive_on_training_depths_so_v1_is_a_noop_there():
    """The uncap only changes anything outside the training range."""
    legacy = queue_depth_norm(TRAIN_DEPTHS, "adaptive", LEGACY)
    v1 = queue_depth_norm(TRAIN_DEPTHS, "adaptive", V1)
    assert legacy < LEGACY_QUEUE_NORM_CAP
    assert legacy == pytest.approx(v1)


def test_cap_is_active_live_and_v1_removes_it():
    assert queue_depth_norm(LIVE_DEPTHS, "adaptive", LEGACY) == pytest.approx(
        LEGACY_QUEUE_NORM_CAP
    )
    assert queue_depth_norm(LIVE_DEPTHS, "adaptive", V1) > LEGACY_QUEUE_NORM_CAP


@pytest.mark.parametrize("factor", [10, 400, 5000])
def test_v1_dim7_is_scale_invariant(factor):
    base_norm = queue_depth_norm(TRAIN_DEPTHS, "adaptive", V1)
    scaled = [d * factor for d in TRAIN_DEPTHS]
    scaled_norm = queue_depth_norm(scaled, "adaptive", V1)
    for depth in (0, 25, 50, 75):
        assert depth / base_norm == pytest.approx((depth * factor) / scaled_norm)


@pytest.mark.parametrize("factor", [400])
def test_legacy_dim7_blows_up_under_the_same_scaling(factor):
    """The regression this whole change exists for."""
    base = max(TRAIN_DEPTHS) / queue_depth_norm(TRAIN_DEPTHS, "adaptive", LEGACY)
    scaled_depths = [d * factor for d in TRAIN_DEPTHS]
    scaled = max(scaled_depths) / queue_depth_norm(scaled_depths, "adaptive", LEGACY)
    assert scaled / base > 100


def test_v1_divisor_does_not_collapse_when_almost_everything_is_idle():
    """p90-over-all degenerates to 1.0 at >=90% idle, which re-exposes raw depth."""
    depths = [0] * 200 + [3000] * 5
    assert queue_depth_norm(depths, "adaptive", LEGACY) == pytest.approx(1.0)
    assert queue_depth_norm(depths, "adaptive", V1) == pytest.approx(3000.0)


def test_v1_usage_ratio_is_monotone_and_compressed():
    ratios = [0.0, 1.0, 5.0, 100.0, 2900.0]
    feats = [usage_ratio_feature(r * 5.0, 5.0, V1) for r in ratios]
    assert feats == sorted(feats)
    # Exactly 5x target concurrency reads 1.0 by construction.
    assert usage_ratio_feature(25.0, 5.0, V1) == pytest.approx(1.0)
    # Live regime lands a few multiples above the training maximum, not ~300x.
    train_max = usage_ratio_feature(79.0, 5.0, V1)
    live = usage_ratio_feature(14500.0, 5.0, V1)
    assert 1.0 < live / train_max < 5.0
    legacy_ratio = usage_ratio_feature(14500.0, 5.0, LEGACY) / usage_ratio_feature(
        79.0, 5.0, LEGACY
    )
    assert legacy_ratio > 100


def test_v1_usage_ratio_handles_degenerate_target_concurrency():
    assert usage_ratio_feature(100.0, 0.0, V1) == 0.0
    assert usage_ratio_feature(100.0, float("nan"), V1) == 0.0
    with pytest.raises(ValueError):
        usage_ratio_feature(-1.0, 5.0, V1)


def test_fixed_mode_ignores_contract():
    for contract in (LEGACY, V1):
        assert queue_depth_norm(
            TRAIN_DEPTHS, "fixed", contract, fixed_factor=50.0
        ) == pytest.approx(50.0)
    with pytest.raises(ValueError):
        queue_depth_norm(TRAIN_DEPTHS, "fixed", V1, fixed_factor=0.0)


def test_unknown_contract_fails_loudly():
    with pytest.raises(InvalidQueueFeatureContractError):
        validate_queue_feature_contract("scale_invariant_v2")
    with pytest.raises(InvalidQueueFeatureContractError):
        queue_depth_norm([1, 2, 3], "adaptive", "nonsense")


def test_resolution_precedence(monkeypatch):
    monkeypatch.delenv(QUEUE_FEATURE_CONTRACT_ENV, raising=False)
    assert resolve_queue_feature_contract() == LEGACY
    monkeypatch.setenv(QUEUE_FEATURE_CONTRACT_ENV, V1)
    assert resolve_queue_feature_contract() == V1
    assert resolve_queue_feature_contract(LEGACY) == LEGACY
    monkeypatch.setenv(QUEUE_FEATURE_CONTRACT_ENV, "bogus")
    with pytest.raises(InvalidQueueFeatureContractError):
        resolve_queue_feature_contract()


def test_serving_mismatch_fails_loudly():
    require_matching_queue_feature_contract(V1, V1, model_label="ckpt")
    require_matching_queue_feature_contract(None, V1, model_label="pre-split ckpt")
    with pytest.raises(QueueFeatureContractMismatchError):
        require_matching_queue_feature_contract(LEGACY, V1, model_label="ckpt")
    with pytest.raises(QueueFeatureContractMismatchError):
        require_matching_queue_feature_contract(V1, LEGACY, model_label="ckpt")


def test_log_divisor_constant_is_consistent():
    from src.placement.queue_features import (
        USAGE_RATIO_LOG_BASELINE,
        USAGE_RATIO_LOG_DIVISOR,
    )

    assert USAGE_RATIO_LOG_DIVISOR == pytest.approx(math.log1p(USAGE_RATIO_LOG_BASELINE))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
