"""Contract tests for src/placement/temporal_features.py.

Guards the two bugs the module was created to kill, and the one property that must not
regress: cache and live compute dims 9-11 from the *same* function, so they cannot drift
again the way they did for four copies.

Companion suites: `test_queue_features.py`, `test_topology_features.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.placement.temporal_features import (  # noqa: E402
    COLD_START_FRACTION,
    COMM_FRACTION,
    mean_execution_time,
    temporal_remainders,
)

# Real numbers from data/nofs-ids/task-types.json, rpiCpu column. `cnn` is the outlier that
# caused Bug 2: it is 3.09s where the dispatched types are 0.003s and 0.168s.
PRIORS = {
    "dnn1": {"executionTime": {"rpiCpu": 0.00290875, "xavierGpu": 0.020835}},
    "dnn2": {"executionTime": {"rpiCpu": 0.16842, "xavierGpu": 0.0362488}},
    "rf": {"executionTime": {"rpiCpu": 0.00422375, "xavierGpu": 0.0004975}},
    "cnn": {"executionTime": {"rpiCpu": 3.0858438, "xavierGpu": 0.1036875}},
}
VOCAB = ("dnn1", "dnn2")

DNN_MEAN_RPI = (0.00290875 + 0.16842) / 2  # 0.085664...
ALL_MEAN_RPI = (0.00290875 + 0.16842 + 0.00422375 + 3.0858438) / 4  # 0.815349...


# ----------------------------------------------------------------- Bug 1: granularity


def test_estimate_fires_per_platform_not_per_snapshot():
    """A queued platform with no recorded remainder gets an estimate.

    The cache builders used to gate this on the *snapshot* having any temporal data at all
    (`if temporal_state: ... else: estimate`), so a platform like this one trained on 0.0
    while live served an estimate. That is the divergence; this is the assertion that keeps
    it closed.
    """
    current, _, _ = temporal_remainders(
        queue_depth=3,
        recorded={},  # snapshot has data for OTHER platforms, none for this one
        platform_type="rpiCpu",
        task_types_data=PRIORS,
        task_types_vocab=VOCAB,
    )
    assert current == pytest.approx(DNN_MEAN_RPI)


def test_recorded_value_always_wins():
    current, cold, comm = temporal_remainders(
        queue_depth=5,
        recorded={
            "current_task_remaining": 1.25,
            "cold_start_remaining": 0.5,
            "comm_remaining": 0.25,
        },
        platform_type="rpiCpu",
        task_types_data=PRIORS,
        task_types_vocab=VOCAB,
    )
    assert (current, cold, comm) == (1.25, 0.5, 0.25)


def test_idle_platform_is_never_estimated():
    """No queued work means no committed work — estimating there would invent load."""
    assert temporal_remainders(
        queue_depth=0,
        recorded={},
        platform_type="rpiCpu",
        task_types_data=PRIORS,
        task_types_vocab=VOCAB,
    ) == (0.0, 0.0, 0.0)


def test_recorded_zero_with_queue_is_treated_as_absent():
    """0.0 is how the capture records 'nothing known', which is what live has always done."""
    current, _, _ = temporal_remainders(
        queue_depth=1,
        recorded={"current_task_remaining": 0.0},
        platform_type="rpiCpu",
        task_types_data=PRIORS,
        task_types_vocab=VOCAB,
    )
    assert current == pytest.approx(DNN_MEAN_RPI)


# ------------------------------------------------------------------- Bug 2: task types


def test_mean_is_over_vocab_only_not_every_prior():
    """`rf`/`cnn` are in task-types.json but no corpus dispatches them.

    Live used to iterate every key, and cnn's 3.09s on rpiCpu inflated the estimate 9.5x.
    """
    assert mean_execution_time("rpiCpu", PRIORS, VOCAB) == pytest.approx(DNN_MEAN_RPI)
    assert mean_execution_time("rpiCpu", PRIORS, tuple(PRIORS)) == pytest.approx(ALL_MEAN_RPI)
    assert ALL_MEAN_RPI / DNN_MEAN_RPI > 9.0, "the outlier this test guards against is gone"


def test_estimate_uses_the_vocab_mean():
    current, _, _ = temporal_remainders(
        queue_depth=2,
        recorded=None,
        platform_type="rpiCpu",
        task_types_data=PRIORS,
        task_types_vocab=VOCAB,
    )
    assert current == pytest.approx(DNN_MEAN_RPI)
    assert current < 0.1, "regressed to averaging in cnn"


def test_platform_types_with_no_positive_entry_are_skipped():
    """Skipped, not counted as zero — otherwise the mean is dragged toward 0."""
    priors = {
        "dnn1": {"executionTime": {"rpiCpu": 0.4}},
        "dnn2": {"executionTime": {"rpiCpu": 0.0}},  # unsupported
    }
    assert mean_execution_time("rpiCpu", priors, ("dnn1", "dnn2")) == pytest.approx(0.4)


def test_unknown_platform_type_yields_no_estimate():
    assert mean_execution_time("quantumFpga", PRIORS, VOCAB) == 0.0
    assert temporal_remainders(
        queue_depth=9,
        recorded={},
        platform_type="quantumFpga",
        task_types_data=PRIORS,
        task_types_vocab=VOCAB,
    ) == (0.0, 0.0, 0.0)


# ------------------------------------------------------------------------ derived dims


def test_cold_start_and_comm_are_fixed_fractions():
    """Pinned: every existing checkpoint was fitted with these ratios."""
    current, cold, comm = temporal_remainders(
        queue_depth=1,
        recorded={},
        platform_type="rpiCpu",
        task_types_data=PRIORS,
        task_types_vocab=VOCAB,
    )
    assert cold == pytest.approx(current * COLD_START_FRACTION)
    assert comm == pytest.approx(current * COMM_FRACTION)
    assert (COLD_START_FRACTION, COMM_FRACTION) == (0.1, 0.05)


# ----------------------------------------------------------------------- robustness


@pytest.mark.parametrize("bad", [None, "n/a", float("nan"), float("inf")])
def test_non_numeric_recorded_values_fall_back_rather_than_crash(bad):
    """Captured state carries nulls; the cache used `_safe_float`, live used bare `float()`.

    The tolerant version is correct for both — and a NaN reaching a feature vector would
    otherwise be caught much later by the non-finite guards, if at all.
    """
    current, _, _ = temporal_remainders(
        queue_depth=1,
        recorded={"current_task_remaining": bad},
        platform_type="rpiCpu",
        task_types_data=PRIORS,
        task_types_vocab=VOCAB,
    )
    assert current == pytest.approx(DNN_MEAN_RPI)


def test_no_priors_yields_no_estimate():
    assert temporal_remainders(
        queue_depth=4,
        recorded={},
        platform_type="rpiCpu",
        task_types_data=None,
        task_types_vocab=VOCAB,
    ) == (0.0, 0.0, 0.0)


def test_all_four_call_sites_import_the_shared_helper():
    """The formula had four copies; two bugs hid in the divergence between them.

    A new copy is how this comes back, so the import is asserted rather than trusted.
    """
    root = Path(__file__).resolve().parents[1]
    call_sites = [
        "src/policy/tabular/feature_builder.py",
        "src/notebooks/prepare_graphs_cache.py",
        "src/notebooks/prepare_graphs_cache_seq.py",
        "src/notebooks/prepare_graphs_ram.py",
    ]
    for rel in call_sites:
        text = (root / rel).read_text()
        assert "temporal_remainders(" in text, f"{rel} does not call the shared helper"
        assert "from src.placement.temporal_features import" in text, f"{rel} missing import"
        # The old snapshot-level gate, verbatim, must not reappear.
        assert "# Approximate: if queue > 0, estimate some remaining time" not in text, (
            f"{rel} still carries the old inline estimate"
        )
