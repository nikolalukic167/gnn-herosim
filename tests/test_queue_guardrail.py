"""serving_stability_v1 S3: the decode-time queue guardrail.

The guardrail is one letter away from GNN_PREFIX_PLATFORM_CAP, which deadlocks 3/16 seeds in
a drainable regime and is closed as a serving default. These tests pin the two properties the
registration leans on: it is a strict no-op when disabled, and it can never be the reason a
task is stranded.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.policy.gnn.seq_decode import (  # noqa: E402
    GnnDecodeRunStats, decode_masked_topo_placement,
)


def _world(n_tasks=3):
    """Three independent tasks, each choosing between a shallow and a deep replica."""
    tl = {t: [(1, 1), (2, 2)] for t in range(n_tasks)}
    return {
        "logits_per_task": [None] * n_tasks,
        "task_logit_to_placement": tl,
        "n_tasks": n_tasks,
        "dag_parents": {t: [] for t in range(n_tasks)},
        "node_caps": {1: 1e9, 2: 1e9},
        "demands": {t: [1.0, 1.0] for t in range(n_tasks)},
        "allow_replica_reuse": True,
    }


def _prefers_deep(t, chosen):
    """A model that always wants (2, 2) -- the deep one."""
    return [0.0, 10.0]


def test_disabled_guardrail_is_a_strict_no_op():
    combo = decode_masked_topo_placement(**_world(), score_fn=_prefers_deep)
    assert combo is not None
    assert all(p == (2, 2) for p in combo)


def test_zero_k_is_identical_to_not_passing_queue_depths():
    a = decode_masked_topo_placement(**_world(), score_fn=_prefers_deep)
    b = decode_masked_topo_placement(
        **_world(), score_fn=_prefers_deep,
        queue_of={(1, 1): 0.0, (2, 2): 500.0}, queue_guard_k=0.0)
    assert a == b


def test_the_guardrail_masks_a_much_deeper_replica():
    combo = decode_masked_topo_placement(
        **_world(), score_fn=_prefers_deep,
        queue_of={(1, 1): 1.0, (2, 2): 500.0}, queue_guard_k=3.0)
    assert all(p == (1, 1) for p in combo), "the deep replica should have been masked"


def test_the_guardrail_is_a_no_op_when_queues_are_even():
    """Depth-relative, not concentration-absolute: an even-but-concentrated batch is fine."""
    combo = decode_masked_topo_placement(
        **_world(), score_fn=_prefers_deep,
        queue_of={(1, 1): 10.0, (2, 2): 12.0}, queue_guard_k=3.0)
    assert all(p == (2, 2) for p in combo)


def test_the_threshold_is_relative_to_the_shallowest_plus_one():
    """k=3, shallowest 1 -> threshold 6: depth 6 survives, depth 7 does not."""
    ok = decode_masked_topo_placement(
        **_world(1), score_fn=_prefers_deep,
        queue_of={(1, 1): 1.0, (2, 2): 6.0}, queue_guard_k=3.0)
    assert ok[0] == (2, 2)
    masked = decode_masked_topo_placement(
        **_world(1), score_fn=_prefers_deep,
        queue_of={(1, 1): 1.0, (2, 2): 7.0}, queue_guard_k=3.0)
    assert masked[0] == (1, 1)


def test_the_guardrail_never_strands_a_task():
    """Every candidate over the threshold -> the guard switches itself off for that task."""
    world = _world(1)
    world["task_logit_to_placement"] = {0: [(2, 2)]}
    world["demands"] = {0: [1.0]}
    combo = decode_masked_topo_placement(
        **world, score_fn=lambda t, chosen: [1.0],
        queue_of={(2, 2): 900.0}, queue_guard_k=3.0)
    assert combo == ((2, 2),), "a task with only deep candidates must still be placed"


def test_an_unknown_depth_is_not_read_as_an_empty_queue():
    """A candidate absent from the snapshot must not be preferred as though depth were 0."""
    combo = decode_masked_topo_placement(
        **_world(1), score_fn=_prefers_deep,
        queue_of={(1, 1): 1.0}, queue_guard_k=3.0)
    assert combo[0] == (2, 2), "an unknown depth must not be masked"


def test_counters_record_what_the_guardrail_did():
    stats = GnnDecodeRunStats()
    decode_masked_topo_placement(
        **_world(3), score_fn=_prefers_deep, stats=stats,
        queue_of={(1, 1): 1.0, (2, 2): 500.0}, queue_guard_k=3.0)
    assert stats.queue_guard_decisions == 3
    assert stats.queue_guard_steps_active == 3
    assert stats.queue_guard_masked >= 3


def test_counters_stay_zero_when_the_guardrail_is_off():
    stats = GnnDecodeRunStats()
    decode_masked_topo_placement(**_world(3), score_fn=_prefers_deep, stats=stats)
    assert stats.queue_guard_decisions == 0
    assert stats.queue_guard_steps_active == 0
    assert stats.queue_guard_masked == 0
