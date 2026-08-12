#!/usr/bin/env python3
"""Unit tests for FilterStore pull observables (node_cold_count / estimated_pull_remaining_sec).

Fail loud if shared_fate saturates while absolute cold count still separates scarce vs remote.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.placement.warmth import (  # noqa: E402
    DEFAULT_T_PULL_S,
    estimate_unit_pull_sec,
    estimated_pull_remaining_sec,
    normalize_estimated_pull_remaining_sec,
    unit_pull_sec_from_task_priors,
)


def _approx(a: float, b: float, tol: float = 1e-4) -> bool:
    return abs(float(a) - float(b)) <= tol


def test_unit_pull_matches_theory() -> None:
    # dnn1 3.057 GB @ min(171, 100) MB/s + 0.00012 write latency
    t = estimate_unit_pull_sec(
        3.057,
        storage_write_mbps=171.0,
        network_bandwidth_mbps=100.0,
        write_latency_s=0.00012,
    )
    if not _approx(t, DEFAULT_T_PULL_S, tol=1e-3):
        raise AssertionError(f"T_pull mismatch: {t} vs theory {DEFAULT_T_PULL_S}")


def test_estimated_scales_with_cold_count() -> None:
    unit = DEFAULT_T_PULL_S
    for n in (1, 4, 12):
        rem = estimated_pull_remaining_sec(n, unit)
        if not _approx(rem, n * unit):
            raise AssertionError(f"remaining({n})={rem} != {n * unit}")
        norm = normalize_estimated_pull_remaining_sec(rem)
        if not _approx(norm, rem / 100.0):
            raise AssertionError(f"norm({n})={norm}")


def test_shared_fate_saturation_vs_cold_count() -> None:
    """Scarce node0 (12/12 cold) and remote (1/1 cold) both have shared_fate=1.0."""
    scarce_cold = 12
    scarce_total = 12
    remote_cold = 1
    remote_total = 1
    shared_scarce = scarce_cold / scarce_total
    shared_remote = remote_cold / remote_total
    if shared_scarce != 1.0 or shared_remote != 1.0:
        raise AssertionError("fixture broken: expected both shared_fate=1.0")
    if scarce_cold == remote_cold:
        raise AssertionError("fixture broken: cold counts must differ")
    rem_scarce = estimated_pull_remaining_sec(scarce_cold, DEFAULT_T_PULL_S)
    rem_remote = estimated_pull_remaining_sec(remote_cold, DEFAULT_T_PULL_S)
    if not (rem_scarce > rem_remote * 10):
        raise AssertionError(
            f"pull remaining must separate scarce vs remote: {rem_scarce} vs {rem_remote}"
        )
    # Oracle residual target ≈ T_baseline; contended last ≈ 12×T_pull
    if rem_remote > 40.0 or rem_scarce < 300.0:
        raise AssertionError(
            f"unexpected magnitude rem_remote={rem_remote} rem_scarce={rem_scarce}"
        )


def test_priors_resolve_dnn1() -> None:
    priors = {
        "dnn1": {"imageSize": {"rpiCpu": 3.057, "xavierGpu": 3.057}},
        "dnn2": {"imageSize": {"rpiCpu": 1.0}},
    }
    t = unit_pull_sec_from_task_priors(priors, "rpiCpu")
    if not _approx(t, DEFAULT_T_PULL_S, tol=1e-3):
        raise AssertionError(f"priors T_pull={t} != {DEFAULT_T_PULL_S}")
    missing = unit_pull_sec_from_task_priors({}, "rpiCpu")
    if missing != DEFAULT_T_PULL_S:
        raise AssertionError(f"fallback T_pull={missing}")


def test_fail_loud_negative() -> None:
    try:
        estimated_pull_remaining_sec(-1, 1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for negative cold_count")


def test_filterstore_pull_wait_matches_observable() -> None:
    """Absolute feature magnitude vs marginal placement cost (physics baseline)."""
    from src.placement.scheduling_cost import filterstore_pull_wait_sec

    class _Plat:
        def __init__(self, init: bool) -> None:
            self.initialized = type("E", (), {"triggered": init})()

    class _Node:
        def __init__(self, cold: int, warm: int = 0) -> None:
            plats = [_Plat(False) for _ in range(cold)] + [
                _Plat(True) for _ in range(warm)
            ]
            self.platforms = type("S", (), {"items": plats})()

    scarce = _Node(12)
    remote = _Node(1)
    rem_scarce = estimated_pull_remaining_sec(12, DEFAULT_T_PULL_S)
    rem_remote = estimated_pull_remaining_sec(1, DEFAULT_T_PULL_S)
    # Absolute mode = CACHE 5.6 feature (over-penalizes unused scarce).
    got_scarce = filterstore_pull_wait_sec(
        scarce, scarce.platforms.items[0], use_marginal_ordinal=False
    )
    got_remote = filterstore_pull_wait_sec(
        remote, remote.platforms.items[0], use_marginal_ordinal=False
    )
    if not _approx(got_scarce, rem_scarce):
        raise AssertionError(f"scarce wait {got_scarce} != {rem_scarce}")
    if not _approx(got_remote, rem_remote):
        raise AssertionError(f"remote wait {got_remote} != {rem_remote}")
    if filterstore_pull_wait_sec(scarce, _Plat(True)) != 0.0:
        raise AssertionError("warm platform must have zero pull wait")
    # Marginal: first pull on scarce == first pull on remote (== T_pull).
    m_scarce = filterstore_pull_wait_sec(scarce, scarce.platforms.items[0])
    m_remote = filterstore_pull_wait_sec(remote, remote.platforms.items[0])
    if not _approx(m_scarce, DEFAULT_T_PULL_S) or not _approx(m_remote, DEFAULT_T_PULL_S):
        raise AssertionError(f"marginal first pull {m_scarce}/{m_remote} != T_pull")
    m_scarce_2 = filterstore_pull_wait_sec(
        scarce, scarce.platforms.items[1], extra_committed_pulls=1
    )
    if not _approx(m_scarce_2, 2 * DEFAULT_T_PULL_S):
        raise AssertionError(f"marginal second pull {m_scarce_2}")
    # Absolute feature would wrongly prefer depth-2 remote over unused scarce.
    if not (got_scarce > 2 * got_remote):
        raise AssertionError("absolute feature must show scarce >> 2× remote")


def main() -> None:
    tests = [
        test_unit_pull_matches_theory,
        test_estimated_scales_with_cold_count,
        test_shared_fate_saturation_vs_cold_count,
        test_priors_resolve_dnn1,
        test_fail_loud_negative,
        test_filterstore_pull_wait_matches_observable,
    ]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"ALL {len(tests)} TESTS PASSED")


if __name__ == "__main__":
    main()
