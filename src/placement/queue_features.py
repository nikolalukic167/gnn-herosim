"""Queue-scale platform features — single source of truth for cache and live inference.

Platform queue depth reaches the models twice: as a normalized depth (platform dim 7)
and as a usage ratio against target concurrency (platform dim 13). Both formulas used to
live in duplicate in `src/notebooks/prepare_graphs_cache.py` and
`src/policy/tabular/feature_builder.py`, which kept the *formula* in parity but left the
*magnitudes* unguarded.

Measured 2026-08-13 on the 873/v5.5 deploy checkpoints:

  training (contention_v2)   per-dataset p90 queue depth 26-70, cap 100 binds 0/200
                             dim7 p90 1.00 / max 1.88 ; dim13 p90 0.33 / max 3.16
  live (sparse_p35 s42 GNN)  averageExecutionTime 0.035s vs averageQueueTime 503s
                             => chosen-platform depth ~14.5k tasks
                             dim7 ~145 (cap always binds) ; dim13 ~50-990

Nothing clips, so the models silently consumed queue inputs 100-300x outside the training
manifold and stopped ranking queues at all: `chosen_queue_vs_min` over 117 live cells shows
the GNN picking platforms with a median of +411 extra queued tasks (p95 +6399) versus the
shortest available one. Knative wins those cells because its rule is scale-free.

Two contracts exist so that checkpoints trained under the old features are never served the
new ones:

  legacy_v0           the historical formulas, kept bit-exact for every pre-2026-08-13
                      cache and checkpoint (873/v5.5, regime B distill, ect_pull, ...).
  scale_invariant_v1  dim7 divisor uncapped, so depth is expressed relative to the p90 of
                      the current snapshot and a uniform 400x deeper infrastructure yields
                      identical features; dim13 log1p-compressed, so genuine overload stays
                      ordered and visible without growing linearly forever.

Uncapping dim7 is a no-op on the contention_v2 corpus (its p90 divisors are 26-67, all
below the old cap), so v1 changes training values only through dim13.
"""

from __future__ import annotations

import math
import os
from typing import Iterable, Optional, Sequence

QUEUE_FEATURE_CONTRACT_LEGACY_V0 = "legacy_v0"
QUEUE_FEATURE_CONTRACT_SCALE_INVARIANT_V1 = "scale_invariant_v1"
VALID_QUEUE_FEATURE_CONTRACTS = frozenset(
    {QUEUE_FEATURE_CONTRACT_LEGACY_V0, QUEUE_FEATURE_CONTRACT_SCALE_INVARIANT_V1}
)
DEFAULT_QUEUE_FEATURE_CONTRACT = QUEUE_FEATURE_CONTRACT_LEGACY_V0
QUEUE_FEATURE_CONTRACT_ENV = "QUEUE_FEATURE_CONTRACT"

QUEUE_NORM_MODE_ADAPTIVE = "adaptive"
QUEUE_NORM_MODE_SCHEDULER_ADAPTIVE = "scheduler_adaptive"
QUEUE_NORM_MODE_ADAPTIVE_NONZERO = "adaptive_nonzero"
QUEUE_NORM_MODE_FIXED = "fixed"

# Historical ceiling on the adaptive divisor. Retained by legacy_v0 only: it never binds in
# training (p90 <= 70) and always binds live (p90 ~ 14.5k), which is precisely the asymmetry
# that put dim7 ~145x out of distribution.
LEGACY_QUEUE_NORM_CAP = 100.0
# Divisor used when a snapshot carries no platforms at all.
QUEUE_NORM_EMPTY_FALLBACK = 50.0
# Historical divisor for the raw queue_depth/target_concurrency ratio (legacy_v0 only).
LEGACY_USAGE_RATIO_DIVISOR = 5.0
# usage_ratio = queue_depth / target_concurrency is compressed as log1p(x)/log1p(5.0), so a
# platform queued at exactly 5x its target concurrency reads 1.0. On contention_v2 this keeps
# the training range wide (p90 0.55, max 1.57 vs legacy 0.33/3.16) while pulling the live
# regime from 50-990 down to ~3-5.
USAGE_RATIO_LOG_BASELINE = 5.0
USAGE_RATIO_LOG_DIVISOR = math.log1p(USAGE_RATIO_LOG_BASELINE)


class InvalidQueueFeatureContractError(ValueError):
    """Raised when a queue feature contract name is not recognized."""


class QueueFeatureContractMismatchError(ValueError):
    """Raised when a checkpoint's training contract differs from the serving contract."""


def validate_queue_feature_contract(contract: str) -> str:
    normalized = str(contract).strip().lower()
    if normalized not in VALID_QUEUE_FEATURE_CONTRACTS:
        raise InvalidQueueFeatureContractError(
            f"Unknown queue feature contract {contract!r}; "
            f"expected one of {sorted(VALID_QUEUE_FEATURE_CONTRACTS)}"
        )
    return normalized


def resolve_queue_feature_contract(explicit: Optional[str] = None) -> str:
    """Explicit argument wins, then $QUEUE_FEATURE_CONTRACT, then legacy_v0."""
    if explicit is not None and str(explicit).strip():
        return validate_queue_feature_contract(explicit)
    from_env = os.environ.get(QUEUE_FEATURE_CONTRACT_ENV, "").strip()
    if from_env:
        return validate_queue_feature_contract(from_env)
    return DEFAULT_QUEUE_FEATURE_CONTRACT


def require_matching_queue_feature_contract(
    trained_contract: Optional[str], serving_contract: str, *, model_label: str
) -> None:
    """Fail loudly rather than serve a checkpoint features it was never trained on."""
    serving = validate_queue_feature_contract(serving_contract)
    if trained_contract is None:
        return
    trained = validate_queue_feature_contract(trained_contract)
    if trained != serving:
        raise QueueFeatureContractMismatchError(
            f"{model_label} was trained under queue feature contract {trained!r} but the "
            f"run resolves to {serving!r}. Set {QUEUE_FEATURE_CONTRACT_ENV}={trained} "
            "(or load a checkpoint trained under the serving contract); dim7/dim13 "
            "differ between contracts and a mismatch silently corrupts queue ranking."
        )


def _p90(sorted_values: Sequence[float]) -> float:
    idx = int(len(sorted_values) * 0.9)
    return float(sorted_values[min(idx, len(sorted_values) - 1)])


def queue_depth_norm(
    queue_values: Iterable[float],
    queue_norm_mode: str,
    contract: str = DEFAULT_QUEUE_FEATURE_CONTRACT,
    *,
    fixed_factor: float = QUEUE_NORM_EMPTY_FALLBACK,
) -> float:
    """Divisor for platform dim 7 (normalized queue depth).

    `adaptive`/`scheduler_adaptive` use the p90 over all platforms, `adaptive_nonzero` over
    busy platforms only. Under scale_invariant_v1 the divisor is uncapped, and an all-platform
    p90 that collapses to zero (>=90% of platforms idle) falls back to the busy-platform p90
    so the divisor cannot degenerate to 1.0 and re-expose raw depth.
    """
    contract = validate_queue_feature_contract(contract)
    cap = LEGACY_QUEUE_NORM_CAP if contract == QUEUE_FEATURE_CONTRACT_LEGACY_V0 else math.inf

    values = sorted(float(v) for v in queue_values)
    if not values:
        return float(QUEUE_NORM_EMPTY_FALLBACK)

    mode = str(queue_norm_mode).strip().lower()
    if mode in (QUEUE_NORM_MODE_ADAPTIVE, QUEUE_NORM_MODE_SCHEDULER_ADAPTIVE):
        divisor = _p90(values)
        if divisor <= 0.0 and contract == QUEUE_FEATURE_CONTRACT_SCALE_INVARIANT_V1:
            non_zero = [v for v in values if v > 0.0]
            divisor = _p90(non_zero) if non_zero else 0.0
    elif mode == QUEUE_NORM_MODE_ADAPTIVE_NONZERO:
        non_zero = [v for v in values if v > 0.0]
        if not non_zero:
            return 1.0
        divisor = _p90(non_zero)
    elif mode == QUEUE_NORM_MODE_FIXED:
        if float(fixed_factor) <= 0.0:
            raise ValueError(
                f"fixed queue norm factor must be > 0, got {fixed_factor}"
            )
        return float(fixed_factor)
    else:
        valid_modes = (
            QUEUE_NORM_MODE_ADAPTIVE,
            QUEUE_NORM_MODE_SCHEDULER_ADAPTIVE,
            QUEUE_NORM_MODE_ADAPTIVE_NONZERO,
            QUEUE_NORM_MODE_FIXED,
        )
        raise ValueError(
            f"Unknown queue_norm_mode {queue_norm_mode!r}; expected one of {valid_modes}"
        )

    return float(min(max(1.0, divisor), cap))


def usage_ratio_feature(
    queue_depth: float,
    target_concurrency: float,
    contract: str = DEFAULT_QUEUE_FEATURE_CONTRACT,
) -> float:
    """Platform dim 13: how overloaded this platform is versus its target concurrency.

    legacy_v0 divides the raw ratio by 5.0 and therefore grows linearly without bound
    (~990 live vs 3.16 max in training). scale_invariant_v1 compresses with log1p so the
    ordering survives but the live regime lands ~3-5x above the training maximum instead
    of ~300x.
    """
    contract = validate_queue_feature_contract(contract)
    tc = float(target_concurrency)
    if not math.isfinite(tc) or tc <= 0.0:
        return 0.0
    depth = float(queue_depth)
    if depth < 0.0:
        raise ValueError(f"queue_depth must be >= 0, got {depth}")
    ratio = depth / tc
    if contract == QUEUE_FEATURE_CONTRACT_LEGACY_V0:
        return ratio / LEGACY_USAGE_RATIO_DIVISOR
    return math.log1p(ratio) / USAGE_RATIO_LOG_DIVISOR
