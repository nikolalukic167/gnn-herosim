"""Unit tests for the §4 shared masked decoder (decode mode "masked_topo").

ROUTE_B_STAGE2_PREREGISTRATION.md (corrected 2026-08-26) §4 registers: DAG
topological order with task_id tie-break, a mask forbidding replica reuse and
node-capacity overflow, a placement-id tie rule on exact score ties, and — a
registered prohibition — NO relax path: an infeasible completion is a counted
failure, never an unmasked argmax. These tests pin each of those properties in
isolation; the corpus-level acceptance (1e-9 against the topological-order
masked greedy and the frozen stage-1 r_greedy_pct on all 408 datasets x
alpha in {2.0, 3.0}) lives in verify_route_b_scorer_agreement.py
--check-decoder and is run per B1, not here.

Also here: the B1 hetero static guard — src/policy/gnn_hetero/ carries a second
decoder copy with no KNOWN_DECODE_MODES registry, is out of scope for stage 2,
and must never grow a "masked_topo" spelling.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.policy.gnn.seq_decode import (  # noqa: E402
    KNOWN_DECODE_MODES,
    GnnDecodeRunStats,
    decode_masked_topo_placement,
    run_decode_with_timing,
    topological_task_order,
)

# diamond4: 0 -> {1, 2} -> 3 (parents map, batch-local ids)
DIAMOND = {0: [], 1: [0], 2: [0], 3: [1, 2]}


def test_mode_is_registered():
    assert "masked_topo" in KNOWN_DECODE_MODES


def test_topological_order_diamond():
    assert topological_task_order(4, DIAMOND) == [0, 1, 2, 3]


def test_topological_order_ready_set_ties_break_by_task_id():
    # 2 is a root alongside 0; after 0 commits, ready = {1, 2} -> lowest id first
    parents = {0: [], 1: [0], 2: [], 3: [1, 2]}
    assert topological_task_order(4, parents) == [0, 1, 2, 3]
    # reversed diamond: 3 is the root, 0 the sink
    parents = {3: [], 1: [3], 2: [3], 0: [1, 2]}
    assert topological_task_order(4, parents) == [3, 1, 2, 0]


def test_topological_order_cycle_raises():
    with pytest.raises(RuntimeError, match="cycle"):
        topological_task_order(2, {0: [1], 1: [0]})


def _decode(logits, candidates, caps, demands, parents=DIAMOND, stats=None):
    n = len(logits)
    return decode_masked_topo_placement(
        logits,
        {t: candidates[t] for t in range(n)},
        n,
        dag_parents=parents,
        node_caps=caps,
        demands=demands,
        stats=stats,
    )


def test_unconstrained_argmax_with_reuse_mask():
    # two nodes, one platform each; both tasks prefer platform (1, 11) — the
    # second task must take the other platform (no replica reuse), never share.
    cands = {0: [(1, 11), (2, 21)], 1: [(1, 11), (2, 21)]}
    combo = _decode(
        [[5.0, 1.0], [5.0, 1.0]],
        cands,
        caps={},
        demands={0: [1.0, 1.0], 1: [1.0, 1.0]},
        parents={0: [], 1: [0]},
    )
    assert combo == ((1, 11), (2, 21))


def test_capacity_mask_blocks_over_cap_placement():
    # node 1 holds one unit; task 0 fills it, task 1's preferred candidate on
    # node 1 (a DIFFERENT platform, so reuse alone would not block it) is over
    # cap and must be skipped.
    cands = {0: [(1, 11), (2, 21)], 1: [(1, 12), (2, 22)]}
    combo = _decode(
        [[5.0, 1.0], [5.0, 1.0]],
        cands,
        caps={1: 1.0},
        demands={0: [1.0, 1.0], 1: [1.0, 1.0]},
        parents={0: [], 1: [0]},
    )
    assert combo == ((1, 11), (2, 22))


def test_exact_score_tie_breaks_by_lowest_placement_id():
    cands = {0: [(2, 21), (1, 11), (1, 12)]}
    combo = _decode(
        [[3.0, 3.0, 3.0]],
        cands,
        caps={},
        demands={0: [1.0, 1.0, 1.0]},
        parents={0: []},
    )
    assert combo == ((1, 11),)


def test_no_relax_path_infeasible_is_counted_failure():
    # both candidates of task 1 are masked (one reused, one over cap): the whole
    # decode fails and the failure is counted — NOT relaxed to argmax.
    stats = GnnDecodeRunStats()
    cands = {0: [(1, 11)], 1: [(1, 11), (1, 12)]}
    combo = _decode(
        [[1.0], [9.0, 9.0]],
        cands,
        caps={1: 1.0},
        demands={0: [1.0], 1: [1.0, 1.0]},
        parents={0: [], 1: [0]},
        stats=stats,
    )
    assert combo is None
    assert stats.masked_topo_infeasible_tasks == 1
    assert stats.masked_topo_failed_decodes == 1
    assert stats.masked_topo_failed_decodes == stats.masked_topo_infeasible_tasks


def test_decode_order_is_topological_not_index_order():
    # task 0 is the CHILD of task 1. Node 1 has capacity for one unit and both
    # tasks prefer it. Under the registered topological order the parent (1)
    # commits first and takes node 1; the child lands on node 2. Index order
    # would give the child node 1 — the assertion distinguishes the two.
    cands = {0: [(1, 11), (2, 21)], 1: [(1, 12), (2, 22)]}
    combo = _decode(
        [[5.0, 1.0], [5.0, 1.0]],
        cands,
        caps={1: 1.0},
        demands={0: [1.0, 1.0], 1: [1.0, 1.0]},
        parents={0: [1], 1: []},
    )
    assert combo == ((2, 21), (1, 12))


def test_misaligned_demand_vector_fails_loud():
    with pytest.raises(RuntimeError, match="demand vector"):
        _decode(
            [[1.0, 2.0]],
            {0: [(1, 11), (2, 21)]},
            caps={},
            demands={0: [1.0]},
            parents={0: []},
        )


def test_run_decode_with_timing_requires_masked_inputs():
    with pytest.raises(RuntimeError, match="masked_topo"):
        run_decode_with_timing(
            "masked_topo",
            [[1.0]],
            {0: [(1, 11)]},
            1,
        )


def test_run_decode_with_timing_dispatches_masked_topo():
    combo = run_decode_with_timing(
        "masked_topo",
        [[1.0, 2.0]],
        {0: [(1, 11), (2, 21)]},
        1,
        dag_parents={0: []},
        node_caps={},
        demands={0: [1.0, 1.0]},
    )
    assert combo == ((2, 21),)


def test_hetero_static_guard_mode_name_absent():
    """src/policy/gnn_hetero/ is a second decoder copy with no KNOWN_DECODE_MODES
    registry and is out of scope for stage 2 (§4). If the mode name ever appears
    there, someone wired the registered decoder into the unregistered copy."""
    hetero = REPO_ROOT / "src" / "policy" / "gnn_hetero"
    assert hetero.is_dir(), f"{hetero} missing — guard needs updating"
    offenders = [
        path
        for path in hetero.rglob("*.py")
        if "masked_topo" in path.read_text(encoding="utf-8", errors="replace")
    ]
    assert not offenders, (
        f"'masked_topo' appears in {offenders} — the hetero decoder copy is out "
        "of scope for stage 2 and must not grow the registered mode"
    )


# =======================================================================================
# Per-step rescoring (score_fn) — the §2 information-tier hook.
#
# A T1/T2 arm conditions each step's scores on the prefix committed so far, but it must
# do so through THIS decoder: §4 requires A1-A4 to share one decode order, one mask and
# one tie rule. So score_fn changes only where a number comes from, never how the plan is
# assembled — and with score_fn absent the decoder must behave exactly as before, because
# B1's frozen 408+408-cell acceptance was measured on that path.
# =======================================================================================
def _decode_rescored(logits, candidates, caps, demands, score_fn, parents=DIAMOND, stats=None):
    n = len(logits)
    return decode_masked_topo_placement(
        logits,
        {t: candidates[t] for t in range(n)},
        n,
        dag_parents=parents,
        node_caps=caps,
        demands=demands,
        stats=stats,
        score_fn=score_fn,
    )


def test_score_fn_absent_is_identical_to_the_frozen_path():
    """The static path must not move: it is what the 408+408 acceptance measured."""
    cands = {0: [(1, 11), (2, 21)], 1: [(1, 11), (2, 21)]}
    args = dict(
        candidates=cands,
        caps={},
        demands={0: [1.0, 1.0], 1: [1.0, 1.0]},
        parents={0: [], 1: [0]},
    )
    logits = [[5.0, 1.0], [5.0, 1.0]]
    assert _decode(logits, **args) == _decode_rescored(logits, score_fn=None, **args)


def test_score_fn_replaces_the_static_logits():
    """A prefix-aware score must be able to overturn the static argmax."""
    cands = {0: [(1, 11), (2, 21)], 1: [(1, 11), (2, 21)]}
    # Static logits prefer (1, 11) for task 0; the score_fn prefers (2, 21).
    combo = _decode_rescored(
        [[5.0, 1.0], [5.0, 1.0]],
        cands,
        caps={},
        demands={0: [1.0, 1.0], 1: [1.0, 1.0]},
        score_fn=lambda t, committed: [1.0, 5.0],
        parents={0: [], 1: [0]},
    )
    assert combo == ((2, 21), (1, 11))


def test_score_fn_sees_the_committed_prefix_in_topological_order():
    """The whole point of T2: every step is handed the placements already committed."""
    seen = []

    def spy(task_idx, committed):
        seen.append((task_idx, dict(committed)))
        return [1.0, 0.0]

    cands = {t: [(1, 10 + t), (2, 20 + t)] for t in range(4)}
    _decode_rescored(
        [[0.0, 0.0]] * 4,
        cands,
        caps={},
        demands={t: [1.0, 1.0] for t in range(4)},
        score_fn=spy,
        parents=DIAMOND,
    )
    assert [t for t, _ in seen] == [0, 1, 2, 3], "not the §4 topological order"
    assert seen[0][1] == {}, "the root was scored against a non-empty prefix"
    # Every parent must be committed by the time its child is scored.
    for task_idx, committed in seen:
        for parent in DIAMOND[task_idx]:
            assert parent in committed, f"task {task_idx} scored before parent {parent}"


def test_score_fn_cannot_mutate_decoder_state():
    """`committed` is handed over as a copy; a careless callback must not corrupt it."""

    def vandal(task_idx, committed):
        committed.clear()
        committed[999] = (0, 0)
        return [1.0, 0.0]

    cands = {t: [(1, 10 + t), (2, 20 + t)] for t in range(4)}
    combo = _decode_rescored(
        [[0.0, 0.0]] * 4,
        cands,
        caps={},
        demands={t: [1.0, 1.0] for t in range(4)},
        score_fn=vandal,
        parents=DIAMOND,
    )
    assert combo == ((1, 10), (1, 11), (1, 12), (1, 13))


def test_score_fn_length_mismatch_fails_loud():
    """A misaligned score vector would silently score the wrong candidates."""
    stats = GnnDecodeRunStats()
    cands = {0: [(1, 11), (2, 21)], 1: [(1, 11), (2, 21)]}
    with pytest.raises(RuntimeError, match="score_fn returned"):
        _decode_rescored(
            [[0.0, 0.0], [0.0, 0.0]],
            cands,
            caps={},
            demands={0: [1.0, 1.0], 1: [1.0, 1.0]},
            score_fn=lambda t, committed: [1.0],
            parents={0: [], 1: [0]},
            stats=stats,
        )
    assert stats.masked_topo_score_fn_failures == 1
    assert stats.masked_topo_failed_decodes == 1


def test_score_fn_exception_is_fatal_not_absorbed():
    """Falling back to the static logits would reinstate the relax path §4 forbids."""
    stats = GnnDecodeRunStats()
    cands = {0: [(1, 11), (2, 21)], 1: [(1, 11), (2, 21)]}

    def boom(task_idx, committed):
        raise KeyError("missing demand")

    with pytest.raises(KeyError):
        _decode_rescored(
            [[5.0, 1.0], [5.0, 1.0]],
            cands,
            caps={},
            demands={0: [1.0, 1.0], 1: [1.0, 1.0]},
            score_fn=boom,
            parents={0: [], 1: [0]},
            stats=stats,
        )
    assert stats.masked_topo_score_fn_failures == 1


def test_rescored_steps_are_counted():
    """A dead callback must show up as 0, not pass as 'the static scores were fine'."""
    stats = GnnDecodeRunStats()
    cands = {t: [(1, 10 + t), (2, 20 + t)] for t in range(4)}
    _decode_rescored(
        [[0.0, 0.0]] * 4,
        cands,
        caps={},
        demands={t: [1.0, 1.0] for t in range(4)},
        score_fn=lambda t, committed: [1.0, 0.0],
        parents=DIAMOND,
        stats=stats,
    )
    assert stats.masked_topo_rescored_steps == 4
    assert stats.masked_topo_score_fn_failures == 0
    assert stats.to_dict()["masked_topo"]["rescored_steps"] == 4


def test_score_fn_infeasible_still_counted_not_relaxed():
    """Rescoring does not buy an escape from the no-relax rule."""
    stats = GnnDecodeRunStats()
    # Both tasks have only one candidate, and it is the same one: task 1 is stuck.
    cands = {0: [(1, 11)], 1: [(1, 11)]}
    combo = _decode_rescored(
        [[0.0], [0.0]],
        cands,
        caps={},
        demands={0: [1.0], 1: [1.0]},
        score_fn=lambda t, committed: [1.0],
        parents={0: [], 1: [0]},
        stats=stats,
    )
    assert combo is None
    assert stats.masked_topo_infeasible_tasks == 1
    assert stats.masked_topo_failed_decodes == 1


def test_run_decode_with_timing_forwards_score_fn():
    """The dispatcher must actually pass it through, and keep the mode name."""
    stats = GnnDecodeRunStats()
    cands = {0: [(1, 11), (2, 21)], 1: [(1, 11), (2, 21)]}
    combo = run_decode_with_timing(
        "masked_topo",
        [[5.0, 1.0], [5.0, 1.0]],
        cands,
        2,
        dag_parents={0: [], 1: [0]},
        node_caps={},
        demands={0: [1.0, 1.0], 1: [1.0, 1.0]},
        stats=stats,
        score_fn=lambda t, committed: [1.0, 5.0],
    )
    assert combo == ((2, 21), (1, 11)), "score_fn did not reach the decoder"
    assert stats.masked_topo_rescored_steps == 2


def test_rescoring_does_not_mint_a_second_decode_mode():
    """§4 requires ONE decoder across A1-A4; where scores come from is an arm property."""
    assert "masked_topo" in KNOWN_DECODE_MODES
    assert not any("rescore" in mode for mode in KNOWN_DECODE_MODES)
