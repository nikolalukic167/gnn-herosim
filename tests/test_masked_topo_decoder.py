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
