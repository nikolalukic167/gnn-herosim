"""Skip-reason attribution: an empty combination list must name its actual cause.

The defect this pins (gate-tools 2026-08-27): `too_many_combinations` was asserted for
ANY empty combination list, on the assumption that the only other cause — zero candidates
— had returned earlier. `replica_overlap` breaks that assumption, because uniqueness
exhaustion also yields empty. 102 H2 datasets recorded `skip_threshold: 2000000` against
a real pre-uniqueness product of 16.

The arm coverage that matters here is `replica_overlap` — the arm that did not exist when
the attribution was written.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.executecosimulation import (  # noqa: E402
    classify_empty_combinations,
    generate_all_combinations_cartesian,
    generate_all_combinations_with_unique_replicas,
)


def _task(task_id, platforms):
    return {
        "task_id": task_id,
        "task_type": f"t{task_id}",
        "source_node": "client_node0",
        "feasible_platforms": [
            {"node_id": n, "platform_id": p, "node_name": f"server_node{n}"}
            for n, p in platforms
        ],
    }


# The H2 shape: replica_overlap collapses four task types onto one shared platform set,
# so four tasks see the same two distinct platforms. 2**4 = 16 combinations, 0 unique.
OVERLAP_TASKS = [_task(i, [(0, 104), (1, 108)]) for i in range(4)]


def test_replica_overlap_arm_is_uniqueness_exhaustion_not_the_threshold():
    """The arm the original assumption never saw."""
    assert generate_all_combinations_with_unique_replicas(OVERLAP_TASKS) == []

    reason = classify_empty_combinations(
        OVERLAP_TASKS, skip_threshold=2_000_000, allow_non_unique_replicas=False)

    assert reason["reason"] == "uniqueness_exhausted"
    assert reason["total_possible_pre_uniqueness"] == 16
    assert reason["n_tasks"] == 4
    assert reason["n_distinct_platforms"] == 2
    assert reason["pigeonhole"] is True


def test_the_h2_corpus_number_is_reproduced_exactly():
    """102 H2 datasets recorded `too_many_combinations` with skip_threshold 2000000
    against a pre-uniqueness product of 16. Pin both numbers."""
    reason = classify_empty_combinations(
        OVERLAP_TASKS, skip_threshold=2_000_000, allow_non_unique_replicas=False)
    assert reason["skip_threshold"] == 2_000_000
    assert reason["total_possible_pre_uniqueness"] == 16
    assert reason["total_possible_pre_uniqueness"] < reason["skip_threshold"]
    assert reason["reason"] != "too_many_combinations"


def test_a_genuine_over_limit_skip_keeps_its_label():
    """The fix must be inert where the old attribution was right."""
    tasks = [_task(i, [(0, 100 + i), (1, 200 + i), (2, 300 + i)]) for i in range(4)]
    assert generate_all_combinations_with_unique_replicas(tasks, skip_if_exceeds=10) == []

    reason = classify_empty_combinations(
        tasks, skip_threshold=10, allow_non_unique_replicas=False)
    assert reason["reason"] == "too_many_combinations"
    assert reason["skip_threshold"] == 10
    assert reason["total_possible_pre_uniqueness"] == 81


def test_never_skip_threshold_zero_is_not_attributed_to_the_threshold():
    """MAX_PLACEMENT_COMBINATIONS_SKIP unset means 0 means never skip, so an empty list
    can never be the threshold's doing — it used to be labelled one anyway."""
    reason = classify_empty_combinations(
        OVERLAP_TASKS, skip_threshold=0, allow_non_unique_replicas=False)
    assert reason["reason"] == "uniqueness_exhausted"
    assert reason["skip_threshold"] == 0


def test_uniqueness_exhaustion_without_pigeonhole_is_still_named():
    """Enough distinct platforms to go round, but no system of distinct representatives:
    three tasks, five platforms, and every task confined to the same two."""
    tasks = [_task(i, [(0, 104), (0, 105)]) for i in range(3)]
    tasks.append(_task(3, [(1, 108), (1, 109), (1, 110)]))
    assert generate_all_combinations_with_unique_replicas(tasks) == []

    reason = classify_empty_combinations(
        tasks, skip_threshold=2_000_000, allow_non_unique_replicas=False)
    assert reason["reason"] == "uniqueness_exhausted"
    assert reason["n_distinct_platforms"] >= reason["n_tasks"]
    assert reason["pigeonhole"] is False


def test_cartesian_mode_over_limit_still_reads_as_the_threshold():
    tasks = [_task(i, [(0, 100 + i), (1, 200 + i)]) for i in range(4)]
    assert generate_all_combinations_cartesian(tasks, skip_if_exceeds=4) == []

    reason = classify_empty_combinations(
        tasks, skip_threshold=4, allow_non_unique_replicas=True)
    assert reason["reason"] == "too_many_combinations"


def test_cartesian_mode_under_the_limit_is_unexplained_never_invented():
    """Without the uniqueness mask an empty list has no known cause. Report `unknown`
    rather than borrowing a reason it has not earned."""
    tasks = [_task(i, [(0, 100 + i)]) for i in range(4)]
    reason = classify_empty_combinations(
        tasks, skip_threshold=2_000_000, allow_non_unique_replicas=True)
    assert reason["reason"] == "unknown"


@pytest.mark.parametrize("allow_non_unique", [True, False])
def test_diagnostics_are_always_present_so_the_census_is_legible(allow_non_unique):
    reason = classify_empty_combinations(
        OVERLAP_TASKS, skip_threshold=16, allow_non_unique_replicas=allow_non_unique)
    for key in ("n_tasks", "n_distinct_platforms",
                "total_possible_pre_uniqueness", "skip_threshold"):
        assert key in reason, f"{key} missing — the census cannot tell the causes apart"
